from __future__ import annotations

import json
from abc import ABC
from dataclasses import asdict
from typing import Literal

from pydantic import BaseModel, ConfigDict

MessageRole = Literal["system", "developer", "assistant", "user"]
BlockType = Literal["text", "image_url", "file"]


def _to_dict(response: object) -> dict[str, object]:
    """Convert a LiteLLM response object into a plain dictionary."""
    if hasattr(response, "model_dump"):
        return response.model_dump()
    return asdict(response)


class ContentBlock(BaseModel, ABC):
    """Base class for structured message content blocks."""

    type: BlockType

    def core(self) -> dict[str, object]:
        """Return a LiteLLM-compatible dictionary representation."""
        return self.model_dump(exclude_none=True)


class TextBlock(ContentBlock):
    """Text content block."""

    type: Literal["text"] = "text"
    text: str


class ImageBlock(ContentBlock):
    """Image URL content block."""

    type: Literal["image_url"] = "image_url"
    image_url: dict[str, object]


class FileBlock(ContentBlock):
    """File reference content block."""

    type: Literal["file"] = "file"
    file: dict[str, object]


class Message(BaseModel):
    """Represents a chat message exchanged with a model.

    Supports plain text or structured content blocks, along with
    metadata, annotations, and parsed JSON data.
    """

    role: MessageRole = "user"
    content: str | list[ContentBlock] = ""

    data: dict[str, object] | list[object] | None = None
    annotations: list[dict[str, object]] | None = None
    stats: dict[str, object] | None = None
    meta: dict[str, object] | None = None

    model_config = ConfigDict(
        extra="allow",
        from_attributes=True,
        validate_assignment=True,
    )

    @property
    def _blocks(self) -> list[ContentBlock]:
        """Ensure content is represented as a list of content blocks."""
        if isinstance(self.content, list):
            return self.content

        if not self.content:
            self.content = []
        else:
            self.content = [TextBlock(text=self.content)]

        return self.content

    @property
    def text(self) -> str:
        """Return the textual content of the message."""
        if isinstance(self.content, str):
            return self.content

        return "\n".join(
            block.text for block in self.content if isinstance(block, TextBlock)
        )

    def add_text_block(self, text: str) -> None:
        """Append a text block to the message."""
        self._blocks.append(TextBlock(text=text))

    def add_image_block(
        self,
        url: str,
        detail: Literal["low", "high"] | None = None,
    ) -> None:
        """Append an image block to the message."""
        image_url: dict[str, object] = {"url": url}
        if detail is not None:
            image_url["detail"] = detail

        self._blocks.append(ImageBlock(image_url=image_url))

    def add_file_block(
        self,
        file_id: str,
        format: str = "application/pdf",
    ) -> None:
        """Append a file reference block to the message."""
        self._blocks.append(
            FileBlock(
                file={
                    "file_id": file_id,
                    "format": format,
                }
            )
        )

    def data_from_content(self) -> dict[str, object] | list[object]:
        """Parse message content as JSON and store it in `data`."""
        if not isinstance(self.content, str):
            raise TypeError("Content must be raw JSON string")

        try:
            parsed = json.loads(self.content)
        except json.JSONDecodeError as exc:
            raise ValueError("Failed to parse message content as JSON") from exc

        self.data = parsed
        return parsed

    def format(self, data: dict[str, object]) -> None:
        """Replace <<key>> placeholders in text content with values."""
        if not data:
            return

        def replace(text: str) -> str:
            for key, value in data.items():
                text = text.replace(f"<<{key}>>", str(value))
            return text

        if isinstance(self.content, str):
            self.content = replace(self.content)
            return

        for block in self._blocks:
            if isinstance(block, TextBlock):
                block.text = replace(block.text)

    def core(
        self,
        *,
        text_mode: bool = False,
    ) -> dict[str, object]:
        """Return a LiteLLM-compatible message payload."""
        if text_mode:
            return {
                "role": self.role,
                "content": self.text,
            }

        if isinstance(self.content, list):
            return {
                "role": self.role,
                "content": [b.core() for b in self.content],
            }

        return {
            "role": self.role,
            "content": self.content,
        }

    @classmethod
    def from_completion(cls, response: object) -> Message:
        """Create a Message from a LiteLLM completion response."""
        data = _to_dict(response)

        choices = data.pop("choices")
        choice = choices[0]

        message = choice.pop("message")
        content = message.pop("content", "")
        role = message.pop("role", "assistant")
        annotations = message.pop("annotations", None)

        stats = {
            **data,
            "choice": choice,
            "message": message,
        }

        return cls(
            role=role,
            content=content,
            annotations=annotations,
            stats=stats or None,
        )

    @staticmethod
    def merge(
        messages: list[Message],
        *,
        output_role: MessageRole = "user",
    ) -> Message:
        """Merge multiple messages into a single formatted message."""
        parts = [f"{m.role}: {m.text}" for m in messages]

        return Message(
            role=output_role,
            content="\n\n---\n\n".join(parts),
        )


class MessageChunk(Message):
    """Represents a streamed partial message chunk."""

    @classmethod
    def from_completion(cls, response: object) -> MessageChunk:
        """Create a MessageChunk from a streamed completion response."""
        data = _to_dict(response)

        choices = data.pop("choices")
        choice = choices[0]
        delta = choice.pop("delta", {}) or {}

        content = delta.get("content", "")
        annotations = delta.pop("annotations", None)

        stats = {
            **data,
            "choice": choice,
            "delta": delta,
        }

        return cls(
            role="assistant",
            content=content or "",
            annotations=annotations,
            stats=stats or None,
        )