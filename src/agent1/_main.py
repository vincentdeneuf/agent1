from __future__ import annotations

import os
import json
import logging
import tomllib
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, ClassVar, Literal, TypeAlias

import litellm
import yaml
from pydantic import BaseModel, ConfigDict, Field


log = logging.getLogger(__name__)


def _env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


AGENT1_DROP_PARAMS: bool = _env_bool("AGENT1_DROP_PARAMS", True)

litellm.drop_params = AGENT1_DROP_PARAMS


MessageRole = Literal["system", "developer", "assistant", "user"]
BlockType = Literal["text", "image_url", "file"]
ResponseFormat = Literal["text", "json_object", "json_schema"]

JSONPrimitive: TypeAlias = str | int | float | bool | None
JSONType: TypeAlias = JSONPrimitive | dict[str, "JSONType"] | list["JSONType"]


def _to_dict(response: object) -> dict[str, object]:
    """Convert a LiteLLM response object into a plain dictionary."""
    if hasattr(response, "model_dump"):
        return response.model_dump()
    return asdict(response)


def _format(
    text: str,
    data: dict[str, object] | None,
    *,
    tags: tuple[str, str] = ("<<", ">>"),
) -> str:
    """Render placeholders inside text using provided data."""
    if not data:
        return text

    open_tag, close_tag = tags

    for key, value in data.items():
        placeholder = f"{open_tag}{key}{close_tag}"
        text = text.replace(placeholder, str(value))

    return text


class TextBlock(BaseModel):
    """Represents a text content block."""

    type: Literal["text"] = "text"
    text: str

    def core(self) -> dict[str, object]:
        """Return a LiteLLM-compatible representation."""
        return self.model_dump(exclude_none=True)


class ImageBlock(BaseModel):
    """Represents an image content block."""

    type: Literal["image_url"] = "image_url"
    image_url: dict[str, object]

    def core(self) -> dict[str, object]:
        """Return a LiteLLM-compatible representation."""
        return self.model_dump(exclude_none=True)


class FileBlock(BaseModel):
    """Represents a file content block."""

    type: Literal["file"] = "file"
    file: dict[str, object]

    def core(self) -> dict[str, object]:
        """Return a LiteLLM-compatible representation."""
        return self.model_dump(exclude_none=True)


ContentBlock: TypeAlias = Annotated[
    TextBlock | ImageBlock | FileBlock,
    Field(discriminator="type"),
]


class Message(BaseModel):
    """Represents a chat message exchanged with a model."""

    role: MessageRole = "user"
    content: str | list[ContentBlock] = ""

    annotations: list[dict[str, object]] | None = None
    stats: dict[str, object] | None = None
    meta: dict[str, object] | None = None

    model_config = ConfigDict(
        extra="allow",
        from_attributes=True,
        validate_assignment=True,
    )

    @property
    def text(self) -> str:
        """Return the textual content of the message."""
        if isinstance(self.content, str):
            return self.content

        return "\n\n".join(
            block.text for block in self.content if isinstance(block, TextBlock)
        )

    @property
    def data(self) -> JSONType | None:
        """Parse message text as JSON on access."""
        raw = self.text.strip()
        if not raw:
            return None

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def _ensure_blocks(self) -> list[ContentBlock]:
        """Ensure content is represented as a list of content blocks."""
        if isinstance(self.content, list):
            return self.content

        if not self.content:
            self.content = []
        else:
            self.content = [TextBlock(text=self.content)]

        return self.content

    def add_text(self, text: str) -> None:
        """Append a text block to the message."""
        self._ensure_blocks().append(TextBlock(text=text))

    def add_image(
        self,
        url: str,
        detail: Literal["low", "high"] | None = None,
    ) -> None:
        """Append an image block to the message."""
        image_url: dict[str, object] = {"url": url}
        if detail is not None:
            image_url["detail"] = detail

        self._ensure_blocks().append(ImageBlock(image_url=image_url))

    def add_file(
        self,
        file_id: str,
        format: str = "application/pdf",
    ) -> None:
        """Append a file block to the message."""
        self._ensure_blocks().append(
            FileBlock(
                file={
                    "file_id": file_id,
                    "format": format,
                }
            )
        )

    def _render(
        self,
        *,
        data: dict[str, object] | None = None,
        text_mode: bool = False,
        tags: tuple[str, str] = ("<<", ">>"),
    ) -> str | list[dict[str, object]]:
        """Render content without mutating the original message."""

        if isinstance(self.content, str):
            return _format(self.content, data, tags=tags)

        rendered_blocks: list[ContentBlock] = []

        for block in self.content:
            if isinstance(block, TextBlock):
                rendered_blocks.append(
                    block.model_copy(
                        update={
                            "text": _format(block.text, data, tags=tags)
                        }
                    )
                )
            else:
                rendered_blocks.append(block)

        if text_mode:
            return "\n\n".join(
                b.text for b in rendered_blocks if isinstance(b, TextBlock)
            )

        return [b.core() for b in rendered_blocks]

    def core(
        self,
        *,
        text_mode: bool = False,
        data: dict[str, object] | None = None,
        tags: tuple[str, str] = ("<<", ">>"),
    ) -> dict[str, object]:
        """Return a LiteLLM-compatible message payload."""
        return {
            "role": self.role,
            "content": self._render(
                data=data,
                text_mode=text_mode,
                tags=tags,
            ),
        }

    @classmethod
    def from_completion(cls, response: object) -> Message:
        """Create a Message from a completion response."""
        data = _to_dict(response)

        choices = data.get("choices", [])
        if not choices:
            raise ValueError("Completion response contains no choices")

        choice = choices[0].copy()
        message = choice.get("message", {}).copy()

        content = message.get("content", "")
        role = message.get("role", "assistant")
        annotations = message.get("annotations")

        stats = {
            **{k: v for k, v in data.items() if k != "choices"},
            "choice": choice,
            "message": message,
        }

        return cls(
            role=role,
            content=content,
            annotations=annotations,
            stats=stats,
        )

    @staticmethod
    def merge(
        messages: list[Message],
        *,
        output_role: MessageRole = "user",
    ) -> Message:
        """Merge multiple messages into a single message."""
        parts = [f"{m.role}: {m.text}" for m in messages]

        return Message(
            role=output_role,
            content="\n\n---\n\n".join(parts),
        )

    @staticmethod
    def from_chunks(
        chunks: list[MessageChunk],
    ) -> Message:
        """Assemble a final Message from streamed chunks."""
        if not chunks:
            raise ValueError("Cannot assemble Message from empty chunks")

        content_parts: list[str] = []
        annotations: list[dict[str, object]] = []

        for chunk in chunks:
            content_parts.append(chunk.text)
            if chunk.annotations:
                annotations.extend(chunk.annotations)

        last_chunk = chunks[-1]

        return Message(
            role=last_chunk.role,
            content="".join(content_parts),
            annotations=annotations or None,
            stats=last_chunk.stats,
        )


class MessageChunk(Message):
    """Represents a streamed partial message chunk."""

    @classmethod
    def from_completion(cls, response: object) -> MessageChunk:
        """Create a MessageChunk from a streamed completion response."""
        data = _to_dict(response)

        choices = data.get("choices", [])
        if not choices:
            raise ValueError("Streamed response contains no choices")

        choice = choices[0].copy()
        delta = choice.get("delta", {}) or {}

        content = delta.get("content", "")
        annotations = delta.get("annotations")

        stats = {
            **{k: v for k, v in data.items() if k != "choices"},
            "choice": choice,
            "delta": delta,
        }

        return cls(
            role="assistant",
            content=content or "",
            annotations=annotations,
            stats=stats,
        )


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
        "googleMaps": {"enableWidget": "ENABLE_WIDGET"},
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

            tools.append({name: cls._TOOLS[name].copy()})

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
                return provider.make(names, model=model)

        raise ValueError(
            f"No tool provider supports model '{model}'",
        )


class Agent(BaseModel):
    """High-level wrapper around LiteLLM for chat-based agents."""

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
        """Construct LiteLLM completion parameters."""
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
        """Stream a synchronous completion response."""
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
        """Stream an asynchronous completion response."""
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
    def load(cls, path: str | Path) -> Agent:
        """Load an Agent instance from a config file."""
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


from print9 import print9

MODEL_NAME = "gemini/gemini-3-flash-preview"


INSTRUCTION = (
    "You are a helpful weather assistant. "
    "You always answer in <<language>>. "
    "The user name is <<name>>."
)

DATA = {
    "language": "Vietnamese",
    "name": "Vincent",
}




def test_work_sync():
    agent = Agent(
        instruction=INSTRUCTION,
        model=MODEL_NAME,
    )

    result = agent.work(
        messages=[
            Message(
                role="user",
                content="What is the weather today?",
            )
        ],
        data=DATA,
    )

    print9("\n[SYNC WORK OUTPUT]", color="blue")
    print9(result.text, color="white")

    assert isinstance(result.text, str)
    assert len(result.text) > 0

test_work_sync()