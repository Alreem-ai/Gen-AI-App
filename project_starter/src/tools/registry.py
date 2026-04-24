import inspect
from typing import Any, Callable, Dict

from pydantic import BaseModel, create_model


class Tool:
    """A callable tool with schema."""

    def __init__(self, name: str, func: Callable, description: str):
        self.name = name
        self.func = func
        self.description = description
        self.model = self._create_pydantic_model(func)

    def _create_pydantic_model(self, func: Callable) -> type[BaseModel]:
        """
        Create a Pydantic model from function signature.
        Required params stay required.
        Optional params keep defaults.
        """
        sig = inspect.signature(func)
        fields = {}

        for name, param in sig.parameters.items():
            if name == "self":
                continue

            annotation = param.annotation
            if annotation == inspect.Parameter.empty:
                annotation = str

            default = param.default

            if default == inspect.Parameter.empty:
                fields[name] = (annotation, ...)
            else:
                fields[name] = (annotation, default)

        return create_model(f"{self.name}Schema", **fields)

    def to_openai_schema(self) -> dict:
        """
        Convert tool to OpenAI / OpenRouter compatible schema.
        Fixes strict-mode required array issues.
        """
        schema = self.model.model_json_schema()

        properties = schema.get("properties", {})

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,

                    # OpenRouter strict mode fix
                    "required": list(properties.keys()),

                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

    def execute(self, **kwargs) -> Any:
        """Validate args then execute tool."""
        validated_args = self.model(**kwargs)
        return self.func(**validated_args.model_dump())


class ToolRegistry:
    """Registry for managing tools."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._categories: Dict[str, list[str]] = {}

    def register(
        self,
        name: str,
        description: str,
        category: str = "general",
    ):
        """
        Decorator to register a tool.
        """

        def decorator(func: Callable):
            tool = Tool(
                name=name,
                func=func,
                description=description,
            )

            self._tools[name] = tool

            if category not in self._categories:
                self._categories[category] = []

            self._categories[category].append(name)

            return func

        return decorator

    def get_tool(self, name: str) -> Tool | None:
        """Return tool by name."""
        return self._tools.get(name)

    def get_all_tools(self) -> list[Tool]:
        """Return all tools."""
        return list(self._tools.values())

    def get_tools_by_category(self, category: str) -> list[Tool]:
        """Return tools by category."""
        names = self._categories.get(category, [])
        return [self._tools[name] for name in names]

    def execute_tool(self, name: str) -> Callable:
        """Return executable tool."""
        tool = self.get_tool(name)

        if not tool:
            raise ValueError(f"Tool '{name}' not found")

        return tool.execute


# Global registry instance
registry = ToolRegistry()
