"""
Simple observability layer that mirrors the Langfuse decorator API.

Usage (identical to production Langfuse):
    from src.observability.observe import observe, langfuse_context

    @observe
    async def run(self, query):
        langfuse_context.update_current_observation(input=query)
        ...
        langfuse_context.update_current_observation(output=result)

Swapping to real Langfuse later requires only changing the import:
    from langfuse.decorators import observe, langfuse_context
"""

import asyncio
import functools
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class Span:
    id: str
    name: str
    start_time: float
    level: int
    parent: Optional["Span"] = None
    children: List["Span"] = field(default_factory=list)
    end_time: Optional[float] = None
    input: Any = None
    output: Any = None
    metadata: dict = field(default_factory=dict)


_current_span: ContextVar[Optional[Span]] = ContextVar("current_span", default=None)


class SimpleObserver:
    @staticmethod
    def print_tree(span: Span):
        duration = (span.end_time - span.start_time) * 1000
        indent = "  " * span.level

        if span.level == 0:
            prefix, suffix = "=== TRACE: ", " ==="
        else:
            prefix, suffix = "|-- ", ""

        # Build display name: prefer agent/tool info stored in metadata
        display_name = span.name
        if "agent" in span.metadata:
            display_name = f"{span.metadata['agent']}"
        if isinstance(span.input, dict) and "tool" in span.input:
            display_name = f"{span.metadata.get('agent', '?')} → {span.input['tool']}"

        meta_parts = []
        if "tokens_in" in span.metadata:
            meta_parts.append(f"in={span.metadata['tokens_in']}")
        if "tokens_out" in span.metadata:
            meta_parts.append(f"out={span.metadata['tokens_out']}")
        if "cost_usd" in span.metadata:
            meta_parts.append(f"${span.metadata['cost_usd']:.4f}")
        meta_str = f" [{', '.join(meta_parts)}]" if meta_parts else ""

        print(f"{indent}{prefix}{display_name}{suffix} ({duration:.2f}ms){meta_str}")
        for child in span.children:
            SimpleObserver.print_tree(child)


def _make_span(span_name: str, func, args, kwargs) -> Span:
    parent = _current_span.get()
    level = parent.level + 1 if parent else 0
    captured_args = (
        args[1:]
        if args and hasattr(args[0], "__class__") and func.__name__ in dir(args[0].__class__)
        else args
    )
    span = Span(
        id=str(uuid.uuid4())[:8],
        name=span_name,
        start_time=time.time(),
        level=level,
        parent=parent,
        input={"args": captured_args, "kwargs": kwargs},
    )
    if parent:
        parent.children.append(span)
    return span


def _finish_span(span: Span):
    span.end_time = time.time()
    if span.level == 0:
        print("\n" + "-" * 60)
        SimpleObserver.print_tree(span)
        print("-" * 60 + "\n")


def observe(name=None):
    def decorator(func):
        span_name = name if isinstance(name, str) else func.__name__

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                span = _make_span(span_name, func, args, kwargs)
                token = _current_span.set(span)
                try:
                    result = await func(*args, **kwargs)
                    if span.output is None:
                        span.output = result
                    return result
                except Exception as e:
                    span.output = f"Error: {e}"
                    raise
                finally:
                    _finish_span(span)
                    _current_span.reset(token)
        else:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                span = _make_span(span_name, func, args, kwargs)
                token = _current_span.set(span)
                try:
                    result = func(*args, **kwargs)
                    if span.output is None:
                        span.output = result
                    return result
                except Exception as e:
                    span.output = f"Error: {e}"
                    raise
                finally:
                    _finish_span(span)
                    _current_span.reset(token)

        return wrapper

    # Support both @observe and @observe() and @observe("name")
    if callable(name):
        f = name
        name = None
        return decorator(f)
    return decorator


class LangfuseContext:
    """
    Stub that mirrors the real langfuse.decorators.langfuse_context API.
    Stores kwargs on the current span's metadata; 'input'/'output' also
    update the span's dedicated fields.
    """

    def update_current_observation(self, **kwargs):
        span = _current_span.get()
        if not span:
            return
        span.metadata.update(kwargs)
        if "input" in kwargs:
            span.input = kwargs["input"]
        if "output" in kwargs:
            span.output = kwargs["output"]


langfuse_context = LangfuseContext()
