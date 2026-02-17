from __future__ import annotations

import json
from dataclasses import asdict
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

MessageRole = Literal["system", "developer", "assistant", "user"]
BlockType = Literal["text", "image_url", "file"]

JSONType: TypeAlias = dict[str, object] | list[object]


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
    """Render placeholders inside text using provided data.

    Placeholders follow the pattern:
        <open_tag>key<close_tag>
    """
    if not data:
        return text

    open_tag, close_tag = tags

    for key, value in data.items():
        placeholder = f"{open_tag}{key}{close_tag}"
        text = text.replace(placeholder, str(value))

    return text


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str

    def core(self) -> dict[str, object]:
        return self.model_dump(exclude_none=True)


class ImageBlock(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: dict[str, object]

    def core(self) -> dict[str, object]:
        return self.model_dump(exclude_none=True)


class FileBlock(BaseModel):
    type: Literal["file"] = "file"
    file: dict[str, object]

    def core(self) -> dict[str, object]:
        return self.model_dump(exclude_none=True)


ContentBlock: TypeAlias = Annotated[
    TextBlock | ImageBlock | FileBlock,
    Field(discriminator="type"),
]


class Message(BaseModel):
    """Represents a chat message exchanged with a model.

    Supports plain text or structured content blocks,
    along with metadata and annotations.
    """

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
        self._ensure_blocks().append(TextBlock(text=text))

    def add_image(
        self,
        url: str,
        detail: Literal["low", "high"] | None = None,
    ) -> None:
        image_url: dict[str, object] = {"url": url}
        if detail is not None:
            image_url["detail"] = detail

        self._ensure_blocks().append(ImageBlock(image_url=image_url))

    def add_file(
        self,
        file_id: str,
        format: str = "application/pdf",
    ) -> None:
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

        # Plain string content
        if isinstance(self.content, str):
            return _format(
                self.content,
                data,
                tags=tags,
            )

        rendered_blocks: list[ContentBlock] = []

        for block in self.content:
            if isinstance(block, TextBlock):
                rendered_blocks.append(
                    block.model_copy(
                        update={
                            "text": _format(
                                block.text,
                                data,
                                tags=tags,
                            )
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
        parts = [f"{m.role}: {m.text}" for m in messages]

        return Message(
            role=output_role,
            content="\n\n---\n\n".join(parts),
        )

    @staticmethod
    def from_chunks(
        chunks: list[MessageChunk],
    ) -> Message:
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
