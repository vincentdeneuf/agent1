from __future__ import annotations

import logging
import tomllib
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Literal

import litellm
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

    def _build_messages(
        self,
        messages: list[Message],
        data: dict[str, object] | None,
        text_mode: bool,
    ) -> list[dict[str, object]]:
        """Build the full message list including system instruction.

        Optionally formats messages with provided data.
        """
        system = Message(role="system", content=self.instruction)
        all_messages = [system, *messages]

        if data:
            for message in all_messages:
                message.format(data)

        return [message.core(text_mode=text_mode) for message in all_messages]

    def _build_params(
        self,
        messages: list[Message],
        *,
        data: dict[str, object] | None = None,
        text_mode: bool = False,
        **kwargs: object,
    ) -> dict[str, object]:
        """Construct LiteLLM completion parameters from agent configuration."""
        params: dict[str, object] = {
            "model": self.model,
            "messages": self._build_messages(
                messages,
                data,
                text_mode,
            ),
        }

        if self.temperature is not None:
            params["temperature"] = self.temperature

        if self.response_format:
            if self.response_format == "json_schema":
                raise NotImplementedError(
                    "response_format='json_schema' is not implemented yet",
                )

            params["response_format"] = {
                "type": self.response_format,
            }

        tools = self.resolved_tools
        if tools:
            params["tools"] = tools

        params.update(kwargs)
        return params

    def _postwork(self, raw: object) -> Message:
        """Convert a raw LiteLLM response into a Message object."""
        message = Message.from_litellm(raw)

        if self.response_format == "json_object":
            message.data_from_content()

        return message

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
        return self._postwork(raw)

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
        return self._postwork(raw)

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
    def from_toml(cls, path: str | Path) -> Agent:
        """Create an Agent instance from a TOML configuration file."""
        file_path = Path(path).expanduser()

        if not file_path.is_absolute():
            file_path = file_path.resolve()

        if not file_path.exists():
            raise FileNotFoundError(
                f"Agent config not found: {file_path}",
            )

        data = tomllib.loads(
            file_path.read_text(encoding="utf-8"),
        )

        model: str = data["model"]
        if not isinstance(model, str) or not model.strip():
            raise ValueError(
                "Invalid or missing 'model' in config",
            )

        agent = cls(**data)

        log.debug(
            "Agent created from TOML | model=%s | path=%s",
            model,
            file_path,
        )

        return agent