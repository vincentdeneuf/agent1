"""
agent1 - A simple LLM abstraction layer

A modern, flexible abstraction layer for LLM applications built on Pydantic.
"""

__version__ = "0.1.0"
__author__ = "vincentdeneuf"
__email__ = "0189vn@gmail.com"

from .main import (
    APIType,
    AudioBlock,
    BlockSchema,
    ContentBlock,
    FileBlock,
    ImageBlock,
    Message,
    MessageChunk,
    MessageRole,
    TextBlock,
)

__all__ = [
    "Message",
    "MessageChunk",
    "ContentBlock",
    "TextBlock",
    "ImageBlock",
    "FileBlock",
    "AudioBlock",
    "BlockSchema",
    "MessageRole",
    "APIType",
]
