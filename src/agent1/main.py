from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Literal

import litellm
import tomllib
from core.log import get_logger
from pydantic import BaseModel, ConfigDict

from .models import Message, MessageChunk
from .tools import ToolFactory

log = get_logger(__name__)

ResponseFormat = Literal["text", "json_object", "json_schema"]


class Agent(BaseModel):
    instruction: str
    model: str

    name: str | None = None
    temperature: float | None = None
    response_format: ResponseFormat | None = None

    model_config = ConfigDict(
        extra="allow",
        from_attributes=True,
        validate_assignment=True,
    )

    # =========================
    # Internal Builders
    # =========================

    def _build_messages(
        self,
        messages: list[Message],
        data: dict[str, object] | None,
        text_mode: bool,
    ) -> list[dict[str, object]]:
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

        params.update(kwargs)
        return params

    def _postprocess(self, raw: object) -> Message:
        message = Message.from_litellm(raw)

        if self.response_format == "json_object":
            message.data_from_content()

        return message

    # =========================
    # Sync
    # =========================

    def work(
        self,
        messages: list[Message],
        *,
        data: dict[str, object] | None = None,
        text_mode: bool = False,
        **kwargs: object,
    ) -> Message:
        params = self._build_params(
            messages,
            data=data,
            text_mode=text_mode,
            **kwargs,
        )

        raw = litellm.completion(**params)
        return self._postprocess(raw)

    # =========================
    # Async
    # =========================

    async def work_async(
        self,
        messages: list[Message],
        *,
        data: dict[str, object] | None = None,
        text_mode: bool = False,
        **kwargs: object,
    ) -> Message:
        params = self._build_params(
            messages,
            data=data,
            text_mode=text_mode,
            **kwargs,
        )

        raw = await litellm.acompletion(**params)
        return self._postprocess(raw)

    # =========================
    # Streaming
    # =========================

    def stream(
        self,
        messages: list[Message],
        *,
        data: dict[str, object] | None = None,
        text_mode: bool = False,
        **kwargs: object,
    ) -> Iterator[Message | MessageChunk]:
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

    async def stream_async(
        self,
        messages: list[Message],
        *,
        data: dict[str, object] | None = None,
        text_mode: bool = False,
        **kwargs: object,
    ) -> AsyncIterator[Message | MessageChunk]:
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

    @classmethod
    def from_toml(cls, path: str | Path) -> Agent:
        file_path = Path(path).expanduser()

        if not file_path.is_absolute():
            file_path = file_path.resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"Agent config not found: {file_path}")

        data = tomllib.loads(file_path.read_text(encoding="utf-8"))

        model: str = data["model"]
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Invalid or missing 'model' in config")

        config: dict[str, object] = {}

        for key, value in data.items():
            if not value:
                continue

            if key == "tools":
                config[key] = ToolFactory.make(
                    value,
                    model=model,
                )
            else:
                config[key] = value

        return cls(**config)
