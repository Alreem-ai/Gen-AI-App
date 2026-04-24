import logging
from dataclasses import dataclass, field
# from litellm import completion_cost

logger = logging.getLogger(__name__)

@dataclass
class StepCost:
    step_number: int
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    is_tool_call: bool = False

@dataclass
class QueryCost:
    query: str
    steps: list[StepCost] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    def add_step(self, step: StepCost):
        self.steps.append(step)
        self.total_cost_usd += step.cost_usd
        self.total_input_tokens += step.input_tokens
        self.total_output_tokens += step.output_tokens

class CostTracker:
    """
    Tracks costs across agent executions.
    """
    def __init__(self):
        self.queries: list[QueryCost] = []
        self._current_query: QueryCost | None = None

    def start_query(self, query: str):
        self._current_query = QueryCost(query=query)

    def log_completion(self, step_number: int, response, is_tool_call: bool = False):
        """
        Log a completion response's cost.
        """
        if not self._current_query:
            return

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        model = getattr(response, "model", "unknown")

        try:
            from litellm import completion_cost
            cost = completion_cost(completion_response=response)
        except Exception:
            cost = 0.0

        step = StepCost(
            step_number=step_number,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            is_tool_call=is_tool_call,
        )
        self._current_query.add_step(step)

    def end_query(self):
        if self._current_query:
            self.queries.append(self._current_query)
            self._current_query = None

    def print_cost_breakdown(self):
        if not self.queries:
            print("No queries tracked yet.")
            return

        print("\n" + "=" * 60)
        print("COST BREAKDOWN")
        print("=" * 60)

        for q in self.queries:
            print(f"\nQuery: {q.query[:80]}")
            print(f"  Total steps : {len(q.steps)}")
            print(f"  Input tokens: {q.total_input_tokens}")
            print(f"  Output tokens: {q.total_output_tokens}")
            print(f"  Total cost  : ${q.total_cost_usd:.6f}")
            for s in q.steps:
                tag = "[tool]" if s.is_tool_call else "[llm] "
                print(f"    Step {s.step_number} {tag} {s.model} | in={s.input_tokens} out={s.output_tokens} ${s.cost_usd:.6f}")

        total = sum(q.total_cost_usd for q in self.queries)
        print(f"\nAll-time total: ${total:.6f}")
        print("=" * 60)

