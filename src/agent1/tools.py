from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar


class ToolProvider(ABC):
    SUPPORTED_MODEL_PREFIXES: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def supports_model(cls, model: str) -> bool:
        return model.lower().startswith(cls.SUPPORTED_MODEL_PREFIXES)

    @classmethod
    @abstractmethod
    def make(
        cls,
        names: list[str],
        *,
        model: str,
    ) -> list[dict[str, dict]]:
        raise NotImplementedError


class GeminiToolProvider(ToolProvider):
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


class ToolFactory:
    @staticmethod
    def make(
        names: list[str],
        *,
        model: str,
    ) -> list[dict[str, dict]]:
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
