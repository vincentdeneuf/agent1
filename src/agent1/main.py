from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Literal

from pydantic import BaseModel, ConfigDict

MessageRole = Literal["system", "developer", "assistant", "user"]
APIType = Literal["completion", "responses"]


class BlockSchema:
    text: dict[APIType, str] = {
        "completion": "text",
        "responses": "input_text",
    }

    image: dict[APIType, str] = {
        "completion": "image_url",
        "responses": "input_image",
    }

    audio: dict[APIType, str] = {
        "responses": "input_audio",
    }

    file: dict[APIType, str] = {
        "completion": "file",
        "responses": "input_file",
    }


class ContentBlock(BaseModel, ABC):
    @abstractmethod
    def core(self, api_type: APIType) -> dict[str, object]:
        raise NotImplementedError

    @classmethod
    def from_responses_output_content(
        cls,
        output_content: list[dict[str, object]],
    ) -> list[ContentBlock]:
        blocks: list[ContentBlock] = []

        for item in output_content:
            block_type = item.get("type")

            if block_type == "output_text":
                text = item.get("text")
                if text:
                    blocks.append(TextBlock(text=text))

            elif block_type == "output_image":
                url = item.get("image_url")
                if url:
                    blocks.append(ImageBlock(url=url))

            elif block_type == "output_audio":
                audio = item.get("audio")
                if audio:
                    blocks.append(
                        AudioBlock(
                            data=audio.get("data"),
                            format=audio.get("format"),
                        )
                    )

        return blocks


class TextBlock(ContentBlock):
    text: str

    def core(self, api_type: APIType = "completion") -> dict[str, object]:
        return {
            "type": BlockSchema.text[api_type],
            "text": self.text,
        }


class ImageBlock(ContentBlock):
    url: str
    detail: Literal["low", "high"] | None = None

    def core(self, api_type: APIType = "completion") -> dict[str, object]:
        image_url: object

        if api_type == "responses":
            image_url = self.url
        else:
            image_url = {"url": self.url}
            if self.detail is not None:
                image_url["detail"] = self.detail

        return {
            "type": BlockSchema.image[api_type],
            "image_url": image_url,
        }


class FileBlock(ContentBlock):
    url: str

    def core(self, api_type: APIType = "completion") -> dict[str, object]:
        return {
            "type": BlockSchema.file[api_type],
            "file": {
                "file_id": self.url,
                "format": "application/pdf",
            },
        }


class AudioBlock(ContentBlock):
    data: str
    format: str

    def core(self, api_type: APIType = "responses") -> dict[str, object]:
        assert api_type == "responses"
        return {
            "type": BlockSchema.audio[api_type],
            "audio": {
                "data": self.data,
                "format": self.format,
            },
        }


class Message(BaseModel):
    content: str | list[ContentBlock] = ""
    role: MessageRole = "user"
    data: dict[str, object] | list[object] | None = None
    annotations: list[dict[str, object]] | None = None
    api_type: APIType | None = None
    stats: dict[str, object] | None = None
    meta: dict[str, object] | None = None

    model_config = ConfigDict(
        extra="allow",
        from_attributes=True,
        validate_assignment=True,
    )

    def data_from_content(self) -> dict[str, object] | list[object]:
        assert self.content
        assert isinstance(self.content, str)

        self.data = json.loads(self.content)
        return self.data

    def _content_as_list(self) -> None:
        if isinstance(self.content, list):
            return

        blocks: list[ContentBlock] = []
        if self.content:
            blocks.append(TextBlock(text=self.content))
        self.content = blocks

    def add_text(self, text: str) -> None:
        self._content_as_list()
        self.content.append(TextBlock(text=text))

    def add_image(
        self,
        url: str,
        detail: Literal["low", "high"] | None = None,
    ) -> None:
        self._content_as_list()
        self.content.append(ImageBlock(url=url, detail=detail))

    def add_file(self, url: str) -> None:
        self._content_as_list()
        self.content.append(FileBlock(url=url))

    def add_audio(self, data: str, format: str) -> None:
        self._content_as_list()
        self.content.append(AudioBlock(data=data, format=format))

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

        for block in self.content:
            if isinstance(block, TextBlock):
                block.text = replace(block.text)

    def core(
        self,
        api_type: APIType = "completion",
        *,
        text_mode: bool = False,
    ) -> dict[str, object]:
        if isinstance(self.content, list):
            if text_mode:
                texts: list[str] = []

                for block in self.content:
                    if isinstance(block, TextBlock):
                        texts.append(block.text)
                return {
                    "role": self.role,
                    "content": "\n".join(texts),
                }

            return {
                "role": self.role,
                "content": [block.core(api_type) for block in self.content],
            }

        return {
            "role": self.role,
            "content": self.content,
        }

    @classmethod
    def from_completion(cls, response: object) -> Message:
        data = (
            response.model_dump()
            if hasattr(response, "model_dump")
            else asdict(response)
        )

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
            api_type="completion",
            stats=stats or None,
        )

    @classmethod
    def from_responses(cls, response: object) -> Message:
        data = (
            response.model_dump()
            if hasattr(response, "model_dump")
            else asdict(response)
        )

        output = data.pop("output")
        assert output

        blocks: list[ContentBlock] = []

        for item in output:
            if item.get("type") != "message":
                continue

            output_content = item.get("content", [])
            blocks.extend(ContentBlock.from_responses_output_content(output_content))

        if len(blocks) == 1 and isinstance(blocks[0], TextBlock):
            content: str | list[ContentBlock] = blocks[0].text
        else:
            content = blocks

        return cls(
            role="assistant",
            content=content,
            api_type="responses",
            stats=data,
        )

    @classmethod
    def from_litellm(
        cls,
        response: object,
        *,
        api_type: APIType,
    ) -> Message:
        if api_type == "completion":
            return cls.from_completion(response)
        if api_type == "responses":
            return cls.from_responses(response)

    @classmethod
    def from_chunks(cls, chunks: list[MessageChunk]) -> Message:
        texts: list[str] = []
        role: MessageRole = "assistant"
        api_type: APIType | None = None
        annotations: list[dict[str, object]] | None = None
        stats: dict[str, object] = {}

        for chunk in chunks:
            if isinstance(chunk.content, str) and chunk.content:
                texts.append(chunk.content)

            if chunk.role:
                role = chunk.role

            if chunk.api_type:
                api_type = chunk.api_type

            if chunk.annotations:
                annotations = chunk.annotations

            if chunk.stats:
                stats.update(chunk.stats)

        return cls(
            role=role,
            content="".join(texts),
            api_type=api_type,
            annotations=annotations,
            stats=stats or None,
        )

    @staticmethod
    def merge(
        messages: list[Message],
        *,
        output_role: MessageRole = "user",
        api_type: APIType = "completion",
    ) -> Message:
        parts: list[str] = []

        for message in messages:
            core = message.core(
                api_type=api_type,
                text_mode=True,
            )

            role = core.get("role", message.role)
            content = core.get("content", "")

            parts.append(f"{role}: {content}")

        return Message(
            role=output_role,
            content="\n\n---\n\n".join(parts),
        )


class MessageChunk(Message):
    @classmethod
    def from_completion(cls, response: object) -> MessageChunk:
        data = (
            response.model_dump()
            if hasattr(response, "model_dump")
            else asdict(response)
        )

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
            api_type="completion",
            annotations=annotations,
            stats=stats or None,
        )
