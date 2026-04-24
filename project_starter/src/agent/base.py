"""
BaseAgent: a ReAct agent with decorator-based observability.

Observability uses the same @observe / langfuse_context API as production Langfuse.
Swapping to real Langfuse requires only changing the import:
    from langfuse.decorators import observe, langfuse_context
"""

import asyncio
import json

import structlog
from litellm import acompletion, completion_cost
from pydantic import ValidationError

from src.agent.prompts import DEFAULT_SYSTEM_PROMPT
from src.config import settings
from src.observability.loop_detector import LoopDetector
from src.observability.observe import langfuse_context, observe
from src.tools.registry import registry

logger = structlog.get_logger()


class BaseAgent:
    """
    A ReAct agent with full observability:
    - Decorator-based tracing of every call (@observe)
    - Loop detection (exact, fuzzy, stagnation)
    - Per-run cost tracking
    - Async execution
    """

    def __init__(
        self,
        model: str | None = None,
        max_steps: int = 10,
        agent_name: str = "BaseAgent",
        verbose: bool = True,
        system_prompt: str | None = None,
        tools: list | None = None,
    ):
        self.model = model or settings.model_name
        self.max_steps = max_steps
        self.agent_name = agent_name
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.verbose = verbose

        self.tools = registry.get_all_tools() if tools is None else tools
        self.tools_schema = [tool.to_openai_schema() for tool in self.tools]
        self.loop_detector = LoopDetector()

    @observe
    async def run(self, user_query: str) -> dict:
        """
        Execute the ReAct (Reasoning + Acting) loop to answer a user query.

        The agent iteratively:
        1. Reasons about the current state/query.
        2. Decides to call tools or provide a final answer.
        3. Observes tool outputs and repeats until a solution is found
           or max_steps is reached.
        """
        langfuse_context.update_current_observation(input=user_query)
        self.loop_detector.reset()

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query},
        ]
        answer = "Agent reached max steps without a final answer."
        total_cost = 0.0
        step = 0

        for step in range(1, self.max_steps + 1):
            call_kwargs: dict = {"model": self.model, "messages": messages}
            if self.tools_schema:
                call_kwargs["tools"] = self.tools_schema
                call_kwargs["tool_choice"] = "auto"

            response = await acompletion(**call_kwargs)
            message = response.choices[0].message

            try:
                total_cost += completion_cost(completion_response=response)
            except Exception as e:
                logger.debug("cost_tracking_unavailable", error=str(e))

            if not message.tool_calls:
                answer = message.content or answer
                break

            # Append assistant turn (with tool calls) to message history
            assistant_msg: dict = {"role": "assistant", "content": message.content or ""}
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]
            messages.append(assistant_msg)

            tool_results = await asyncio.gather(*[
                self._execute_tool(tc.function.name, self._parse_args(tc.function.arguments))
                for tc in message.tool_calls
            ])
            for tc, result in zip(message.tool_calls, tool_results):
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        langfuse_context.update_current_observation(
            output=answer,
            cost_usd=total_cost,
        )
        return {"answer": answer, "metadata": {"total_steps": step}}

    def _parse_args(self, args_str: str) -> dict:
        """Parse tool call arguments, returning empty dict on malformed JSON."""
        try:
            return json.loads(args_str)
        except json.JSONDecodeError as e:
            logger.warning("tool_args_parse_failed", raw=args_str, error=str(e))
            return {}

    @observe("tool_call")
    async def _execute_tool(self, tool_name: str, arguments: dict) -> str:
        """Registry lookup + loop detection + asyncio.to_thread + error handling."""
        langfuse_context.update_current_observation(
            input={"tool": tool_name, "args": arguments}
        )

        loop_check = self.loop_detector.check_tool_call(tool_name, json.dumps(arguments))
        if loop_check.is_looping:
            logger.warning(
                "loop_detected",
                tool=tool_name,
                strategy=loop_check.strategy,
                message=loop_check.message,
            )
            result = f"SYSTEM: {loop_check.message} (Detection: {loop_check.strategy})"
            langfuse_context.update_current_observation(output=result)
            return result

        tool = registry.get_tool(tool_name)
        if not tool:
            logger.error("tool_not_found", tool=tool_name)
            result = f"Error: Tool '{tool_name}' not found."
            langfuse_context.update_current_observation(output=result)
            return result

        try:
            result = str(await asyncio.to_thread(tool.execute, **arguments))
        except ValidationError as e:
            logger.warning("tool_validation_failed", tool=tool_name, error=str(e))
            result = f"Error: Tool arguments validation failed. {e}"
        except Exception as e:
            logger.error("tool_execution_failed", tool=tool_name, error=str(e))
            result = f"Error: {type(e).__name__}: {e}"

        langfuse_context.update_current_observation(output=result)
        return result
