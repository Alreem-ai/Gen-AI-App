"""
Multi-agent orchestration layer.

Strategy: Plan-and-Execute with Parallel Research
  1. Planner     — breaks the query into sub-tasks
  2. Researcher-1 + Researcher-2 — run in PARALLEL, each covering a different angle
  3. Analyst     — synthesizes both research outputs into structured insights
  4. Writer      — produces the final polished report
  5. URLs are extracted from research and appended as a guaranteed References section
"""

import asyncio
import re

import structlog

from src.agent.base import BaseAgent
from src.agent.prompts import (
    ANALYST_PROMPT,
    PLANNER_PROMPT,
    RESEARCHER_PROMPT,
    WRITER_PROMPT,
)
from src.config import settings

logger = structlog.get_logger()


def _extract_urls(text: str) -> list[str]:
    """Extract all unique http/https URLs from a block of text."""
    pattern = r'https?://[^\s\)\]\>\"\']+'
    found = re.findall(pattern, text)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for url in found:
        url = url.rstrip(".,;:")  # strip trailing punctuation
        if url not in seen:
            seen.add(url)
            unique.append(url)
    return unique


def _append_references(report: str, urls: list[str]) -> str:
    """Append a References section to the report if URLs were found."""
    if not urls:
        return report

    # Don't duplicate if writer already added a references section
    if "## References" in report or "## Sources" in report:
        return report

    refs = "\n\n---\n\n## References\n"
    for i, url in enumerate(urls, 1):
        refs += f"{i}. {url}\n"
    return report + refs


class OrchestratorAgent:
    """
    Plan-and-Execute multi-agent orchestrator.

    Pipeline:
        Planner → [Researcher-1 ∥ Researcher-2] → Analyst → Writer → References appended
    """

    def __init__(self, model: str = None, max_steps: int = 10):
        resolved_model = model or settings.model_name

        # Planner: reasons only, no web tools needed
        self.planner = BaseAgent(
            model=resolved_model,
            max_steps=max_steps,
            agent_name="Planner",
            system_prompt=PLANNER_PROMPT,
            tools=[],
        )

        # Two researchers run in parallel, each covering a different angle
        self.researcher_1 = BaseAgent(
            model=resolved_model,
            max_steps=max_steps,
            agent_name="Researcher-1",
            system_prompt=RESEARCHER_PROMPT,
        )
        self.researcher_2 = BaseAgent(
            model=resolved_model,
            max_steps=max_steps,
            agent_name="Researcher-2",
            system_prompt=RESEARCHER_PROMPT,
        )

        # Analyst: reasons over provided text, no web tools needed
        self.analyst = BaseAgent(
            model=resolved_model,
            max_steps=max_steps,
            agent_name="Analyst",
            system_prompt=ANALYST_PROMPT,
            tools=[],
        )

        # Writer: synthesizes only, no web tools needed
        self.writer = BaseAgent(
            model=resolved_model,
            max_steps=max_steps,
            agent_name="Writer",
            system_prompt=WRITER_PROMPT,
            tools=[],
        )

    async def run(self, query: str) -> dict:
        total_steps = 0

        # Step 1 — Planner breaks the query into a research plan
        logger.info("orchestrator.planning", query=query)
        plan_result = await self.planner.run(PLANNER_PROMPT.format(query=query))
        plan_output = plan_result["answer"]
        total_steps += plan_result["metadata"].get("total_steps", 0)
        logger.info("orchestrator.plan_ready")

        # Step 2 — Two Researchers run in PARALLEL, each covering a different angle
        logger.info("orchestrator.parallel_research_start")
        result_1, result_2 = await asyncio.gather(
            self.researcher_1.run(
                f"Research plan:\n{plan_output}\n\n"
                f"Original query: {query}\n\n"
                f"Focus on: facts, statistics, and primary sources."
            ),
            self.researcher_2.run(
                f"Research plan:\n{plan_output}\n\n"
                f"Original query: {query}\n\n"
                f"Focus on: recent developments, expert opinions, and real-world examples."
            ),
        )
        research_output_1 = result_1["answer"]
        research_output_2 = result_2["answer"]
        total_steps += result_1["metadata"].get("total_steps", 0)
        total_steps += result_2["metadata"].get("total_steps", 0)
        logger.info("orchestrator.parallel_research_done")

        # Extract all URLs from both research outputs — guaranteed reference collection
        all_urls = _extract_urls(research_output_1 + "\n" + research_output_2)
        logger.info("orchestrator.urls_collected", count=len(all_urls))

        # Step 3 — Analyst synthesizes both research outputs
        logger.info("orchestrator.analysis_start")
        analyst_result = await self.analyst.run(
            f"Original query: {query}\n\n"
            f"--- Researcher 1 (facts & statistics) ---\n{research_output_1}\n\n"
            f"--- Researcher 2 (recent developments & examples) ---\n{research_output_2}"
        )
        analyst_output = analyst_result["answer"]
        total_steps += analyst_result["metadata"].get("total_steps", 0)
        logger.info("orchestrator.analysis_done")

        # Step 4 — Writer produces the final report
        logger.info("orchestrator.writing_start")
        writer_result = await self.writer.run(
            f"Original query: {query}\n\n"
            f"--- Analysis ---\n{analyst_output}"
        )
        final_answer = writer_result["answer"]
        total_steps += writer_result["metadata"].get("total_steps", 0)
        logger.info("orchestrator.report_ready")

        # Guarantee a References section is always present in the output
        final_answer = _append_references(final_answer, all_urls)

        return {
            "answer": final_answer,
            "metadata": {
                "total_steps": total_steps,
                "pipeline": "planner → [researcher-1 ∥ researcher-2] → analyst → writer",
                "references_found": len(all_urls),
            },
        }
