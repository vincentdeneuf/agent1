from __future__ import annotations

import json
from abc import ABC
from dataclasses import asdict
from typing import Literal

from pydantic import BaseModel, ConfigDict

MessageRole = Literal["system", "developer", "assistant", "user"]
BlockType = Literal["text", "image_url", "file"]


def _to_dict(response: object) -> dict[str, object]:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    return asdict(response)


# =========================
# Internal Block Models
# =========================


class ContentBlock(BaseModel, ABC):
    type: BlockType

    def core(self) -> dict[str, object]:
        return self.model_dump(exclude_none=True)


class TextBlock(ContentBlock):
    type: Literal["text"] = "text"
    text: str


class ImageBlock(ContentBlock):
    type: Literal["image_url"] = "image_url"
    image_url: dict[str, object]


class FileBlock(ContentBlock):
    type: Literal["file"] = "file"
    file: dict[str, object]


# =========================
# Message
# =========================


class Message(BaseModel):
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

    # -------------------------
    # Internal Block Access
    # -------------------------

    @property
    def _blocks(self) -> list[ContentBlock]:
        if isinstance(self.content, list):
            return self.content

        if not self.content:
            self.content = []
        else:
            self.content = [TextBlock(text=self.content)]

        return self.content

    # -------------------------
    # Public Views
    # -------------------------

    @property
    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content

        return "\n".join(
            block.text for block in self.content if isinstance(block, TextBlock)
        )

    # -------------------------
    # Block Builders
    # -------------------------

    def add_text_block(self, text: str) -> None:
        self._blocks.append(TextBlock(text=text))

    def add_image_block(
        self,
        url: str,
        detail: Literal["low", "high"] | None = None,
    ) -> None:
        image_url: dict[str, object] = {"url": url}
        if detail is not None:
            image_url["detail"] = detail

        self._blocks.append(ImageBlock(image_url=image_url))

    def add_file_block(
        self,
        file_id: str,
        format: str = "application/pdf",
    ) -> None:
        self._blocks.append(
            FileBlock(
                file={
                    "file_id": file_id,
                    "format": format,
                }
            )
        )

    # -------------------------
    # Utilities
    # -------------------------

    def data_from_content(self) -> dict[str, object] | list[object]:
        if not isinstance(self.content, str):
            raise TypeError("Content must be raw JSON string")

        self.data = json.loads(self.content)
        return self.data

    def format(self, data: dict[str, object]) -> None:
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

    # -------------------------
    # Serialization
    # -------------------------

    def core(
        self,
        *,
        text_mode: bool = False,
    ) -> dict[str, object]:
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

    # -------------------------
    # Factories
    # -------------------------

    @classmethod
    def from_completion(cls, response: object) -> Message:
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
        parts = [f"{m.role}: {m.text}" for m in messages]

        return Message(
            role=output_role,
            content="\n\n---\n\n".join(parts),
        )


# =========================
# Streaming Chunk
# =========================


class MessageChunk(Message):
    @classmethod
    def from_completion(cls, response: object) -> MessageChunk:
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
