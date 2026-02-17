import json
import asyncio
import tempfile
from pathlib import Path

import pytest
import yaml

from agent1 import Message, Agent
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
                data=DATA,
            )
        ]
    )

    print9("\n[SYNC WORK OUTPUT]", color="blue")
    print9(result.text, color="white")

    assert isinstance(result.text, str)
    assert len(result.text) > 0




def test_stream_sync():
    agent = Agent(
        instruction=INSTRUCTION,
        model=MODEL_NAME,
    )

    final_message = None

    print9("\n[SYNC STREAM OUTPUT]", color="blue")

    for item in agent.stream(
        messages=[
            Message(
                role="user",
                content="What is the weather today?",
                data=DATA,
            )
        ]
    ):
        if hasattr(item, "text"):
            print9(item.text, color="white", end="", flush=True)
        final_message = item

    print9("", color="white")

    assert final_message is not None
    assert isinstance(final_message.text, str)
    assert len(final_message.text) > 0




@pytest.mark.asyncio
async def test_work_async():
    agent = Agent(
        instruction=INSTRUCTION,
        model=MODEL_NAME,
    )

    result = await agent.work_async(
        messages=[
            Message(
                role="user",
                content="What is the weather today?",
                data=DATA,
            )
        ]
    )

    print9("\n[ASYNC WORK OUTPUT]", color="cyan")
    print9(result.text, color="white")

    assert isinstance(result.text, str)
    assert len(result.text) > 0




@pytest.mark.asyncio
async def test_stream_async():
    agent = Agent(
        instruction=INSTRUCTION,
        model=MODEL_NAME,
    )

    final_message = None

    print9("\n[ASYNC STREAM OUTPUT]", color="cyan")

    async for item in agent.stream_async(
        messages=[
            Message(
                role="user",
                content="What is the weather today?",
                data=DATA,
            )
        ]
    ):
        if hasattr(item, "text"):
            print9(item.text, color="white", end="", flush=True)
        final_message = item

    print9("", color="white")

    assert final_message is not None
    assert isinstance(final_message.text, str)
    assert len(final_message.text) > 0


async def _run_async_tests():
    await test_work_async()
    await test_stream_async()


@pytest.mark.asyncio
async def test_load_configs(paths: list[str]):
    for path in paths:
        print9(f"\n[LOAD TEST] {path}", color="magenta")

        agent = Agent.load(path)

        # ---------- SYNC WORK ----------
        result = agent.work(
            messages=[
                Message(
                    role="user",
                    content="What is the weather today?",
                )
            ],
            data=DATA,
        )

        print9("[SYNC WORK]", color="blue")
        print9(result.text, color="white")

        assert isinstance(result.text, str)
        assert len(result.text) > 0

        # ---------- SYNC STREAM ----------
        final_message = None

        print9("[SYNC STREAM]", color="blue")

        for item in agent.stream(
            messages=[
                Message(
                    role="user",
                    content="What is the weather today?",
                )
            ],
            data=DATA,
        ):
            if hasattr(item, "text"):
                print9(item.text, color="white", end="", flush=True)
            final_message = item

        print9("", color="white")

        assert final_message is not None
        assert isinstance(final_message.text, str)
        assert len(final_message.text) > 0

        # ---------- ASYNC WORK ----------
        result_async = await agent.work_async(
            messages=[
                Message(
                    role="user",
                    content="What is the weather today?",
                )
            ],
            data=DATA,
        )

        print9("[ASYNC WORK]", color="cyan")
        print9(result_async.text, color="white")

        assert isinstance(result_async.text, str)
        assert len(result_async.text) > 0

        # ---------- ASYNC STREAM ----------
        final_async_message = None

        print9("[ASYNC STREAM]", color="cyan")

        async for item in agent.stream_async(
            messages=[
                Message(
                    role="user",
                    content="What is the weather today?",
                )
            ],
            data=DATA,
        ):
            if hasattr(item, "text"):
                print9(item.text, color="white", end="", flush=True)
            final_async_message = item

        print9("", color="white")

        assert final_async_message is not None
        assert isinstance(final_async_message.text, str)
        assert len(final_async_message.text) > 0

if __name__ == "__main__":
    print9("Running tests manually...\n", color="green")

    asyncio.run(
        test_load_configs(
            [
                # "tests/config/agent.toml",
                # "tests/config/agent.json",
                "tests/config/agent.yaml",
            ]
        )
    )