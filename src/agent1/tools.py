from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar


class ToolProvider(ABC):
    """Base class for model-specific tool providers."""

    SUPPORTED_MODEL_PREFIXES: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def supports_model(cls, model: str) -> bool:
        """Return True if this provider supports the given model."""
        return model.lower().startswith(cls.SUPPORTED_MODEL_PREFIXES)

    @classmethod
    @abstractmethod
    def make(
        cls,
        names: list[str],
        *,
        model: str,
    ) -> list[dict[str, dict]]:
        """Create tool definitions for the given model."""
        raise NotImplementedError


class GeminiToolProvider(ToolProvider):
    """Tool provider implementation for Gemini-based models."""

    SUPPORTED_MODEL_PREFIXES: ClassVar[tuple[str, ...]] = (
        "gemini",
        "vertex_ai",
    )

    _TOOLS: ClassVar[dict[str, dict]] = {
        "googleMaps": {
            "enableWidget": "ENABLE_WIDGET",
        },
        "googleSearch": {},
        "urlContext": {},
        "enterpriseWebSearch": {},
        "codeExecution": {},
    }

    @classmethod
    def make(
        cls,
        names: list[str],
        *,
        model: str,
    ) -> list[dict[str, dict]]:
        """Build Gemini-compatible tool configurations."""
        if not cls.supports_model(model):
            raise ValueError(
                f"Model '{model}' is not supported by Gemini tools",
            )

        tools: list[dict[str, dict]] = []

        for name in names:
            if name not in cls._TOOLS:
                raise ValueError(
                    f"Unsupported Gemini tool: '{name}'",
                )

            tools.append(
                {
                    name: cls._TOOLS[name].copy(),
                },
            )

        return tools


class OpenAIToolProvider(ToolProvider):
    """Tool provider implementation for OpenAI-based models."""

    SUPPORTED_MODEL_PREFIXES: ClassVar[tuple[str, ...]] = (
        "gpt-",
        "openai",
    )

    @classmethod
    def make(
        cls,
        names: list[str],
        *,
        model: str,
    ) -> list[dict[str, dict]]:
        """Build OpenAI-compatible tool configurations."""
        if not cls.supports_model(model):
            raise ValueError(
                f"Model '{model}' is not supported by OpenAI tools",
            )

        raise NotImplementedError(
            "OpenAI tool provider is not implemented yet",
        )


class ToolFactory:
    """Factory for resolving tool providers based on model name."""

    @staticmethod
    def make(
        names: list[str],
        *,
        model: str,
    ) -> list[dict[str, dict]]:
        """Resolve tool names into provider-specific tool definitions."""
        if not isinstance(names, list) or not all(
            isinstance(name, str) for name in names
        ):
            raise TypeError("Tool names must be list[str]")

        for provider in ToolProvider.__subclasses__():
            if provider.supports_model(model):
                return provider.make(
                    names,
                    model=model,
                )

        raise ValueError(
            f"No tool provider supports model '{model}'",
        )