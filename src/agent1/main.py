from __future__ import annotations

import json
import logging
import tomllib
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Literal

import litellm
import yaml
from pydantic import BaseModel, ConfigDict

from .models import Message, MessageChunk
from .settings import AGENT1_DROP_PARAMS
from .tools import ToolFactory

log = logging.getLogger(__name__)
litellm.drop_params = AGENT1_DROP_PARAMS

ResponseFormat = Literal["text", "json_object", "json_schema"]


class Agent(BaseModel):
    """High-level wrapper around LiteLLM for chat-based agents.

    Handles message construction, tool resolution, response formatting,
    streaming, and TOML-based configuration.
    """

    instruction: str
    model: str

    name: str | None = None
    temperature: float | None = None
    response_format: ResponseFormat | None = None
    tools: list[str] | None = None

    model_config = ConfigDict(
        extra="allow",
        from_attributes=True,
        validate_assignment=True,
    )

    @property
    def resolved_tools(self) -> list[dict[str, dict]] | None:
        """Resolve tool names into LiteLLM-compatible tool definitions."""
        if not self.tools:
            return None

        return ToolFactory.make(
            self.tools,
            model=self.model,
        )

    def _build_params(
        self,
        messages: list[Message],
        *,
        data: dict[str, object] | None = None,
        text_mode: bool = False,
        **kwargs: object,
    ) -> dict[str, object]:
        """Construct LiteLLM completion parameters from agent configuration."""
        system = Message(role="system", content=self.instruction)
        all_messages = [system, *messages]

        params = self.model_dump(exclude_none=True)

        params["messages"] = [
            message.core(
                text_mode=text_mode,
                data=data,
            )
            for message in all_messages
        ]

        tools = self.resolved_tools
        if tools:
            params["tools"] = tools

        params.update(kwargs)

        return params

    def work(
        self,
        messages: list[Message],
        *,
        data: dict[str, object] | None = None,
        text_mode: bool = False,
        **kwargs: object,
    ) -> Message:
        """Execute a synchronous completion request."""
        log.debug(
            "Agent.work called | model=%s | messages=%d",
            self.model,
            len(messages),
        )

        params = self._build_params(
            messages,
            data=data,
            text_mode=text_mode,
            **kwargs,
        )

        raw = litellm.completion(**params)
        return Message.from_completion(raw)

    async def work_async(
        self,
        messages: list[Message],
        *,
        data: dict[str, object] | None = None,
        text_mode: bool = False,
        **kwargs: object,
    ) -> Message:
        """Execute an asynchronous completion request."""
        log.debug(
            "Agent.work_async called | model=%s | messages=%d",
            self.model,
            len(messages),
        )

        params = self._build_params(
            messages,
            data=data,
            text_mode=text_mode,
            **kwargs,
        )

        raw = await litellm.acompletion(**params)
        return Message.from_completion(raw)

    def stream(
        self,
        messages: list[Message],
        *,
        data: dict[str, object] | None = None,
        text_mode: bool = False,
        **kwargs: object,
    ) -> Iterator[Message | MessageChunk]:
        """Stream a synchronous completion response.

        Yields MessageChunk objects followed by the final assembled Message.
        """
        log.debug(
            "Agent.stream started | model=%s | messages=%d",
            self.model,
            len(messages),
        )

        params = self._build_params(
            messages,
            data=data,
            text_mode=text_mode,
            **kwargs,
        )
        params["stream"] = True

        chunks: list[MessageChunk] = []

        for raw in litellm.completion(**params):
            chunk = MessageChunk.from_completion(raw)
            chunks.append(chunk)
            yield chunk

        if chunks:
            yield Message.from_chunks(chunks)

        log.debug("Agent.stream completed")

    async def stream_async(
        self,
        messages: list[Message],
        *,
        data: dict[str, object] | None = None,
        text_mode: bool = False,
        **kwargs: object,
    ) -> AsyncIterator[Message | MessageChunk]:
        """Stream an asynchronous completion response.

        Yields MessageChunk objects followed by the final assembled Message.
        """
        log.debug(
            "Agent.stream_async started | model=%s | messages=%d",
            self.model,
            len(messages),
        )

        params = self._build_params(
            messages,
            data=data,
            text_mode=text_mode,
            **kwargs,
        )
        params["stream"] = True

        chunks: list[MessageChunk] = []

        result = await litellm.acompletion(**params)

        async for raw in result:
            chunk = MessageChunk.from_completion(raw)
            chunks.append(chunk)
            yield chunk

        if chunks:
            yield Message.from_chunks(chunks)

        log.debug("Agent.stream_async completed")

    @classmethod
    def load(cls, path: str | Path) -> Agent:
        """Load an Agent instance from a config file.

        Supported formats: .toml, .json, .yaml, .yml
        """
        file_path = Path(path).expanduser().resolve()
        suffix = file_path.suffix.lower()

        content = file_path.read_text(encoding="utf-8")

        if suffix == ".toml":
            data = tomllib.loads(content)

        elif suffix == ".json":
            data = json.loads(content)

        elif suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(content)

        else:
            supported = [".toml", ".json", ".yaml", ".yml"]
            raise ValueError(
                f"Unsupported config format '{suffix}'. "
                f"Supported formats: {', '.join(supported)}."
            )

        if not isinstance(data, dict):
            raise ValueError(
                "Config file must define a dictionary at the top level.",
            )

        return cls(**data)
