import asyncio
from pathlib import Path

import pytest

from agent1 import Message, Agent
from print9 import print9


MODEL_NAME = "gemini/gemini-3-flash-preview"

BASE_DIR = Path(__file__).parent
FILES_DIR = BASE_DIR / "files"

IMAGE_PATH = FILES_DIR / "img.jpg"
PDF_PATH = FILES_DIR / "test.pdf"

INSTRUCTION = (
    "You are a helpful assistant. "
    "You always answer in <<language>>. "
    "The user name is <<name>>."
)

DATA = {
    "language": "English",
    "name": "Vincent",
}


# =========================
# MESSAGE BUILDERS
# =========================
def _image_message() -> Message:
    msg = Message(role="user", content=[])
    msg.add_text("What do you see in this image?")
    msg.add_image(file=IMAGE_PATH, detail="low")
    return msg


def _pdf_message() -> Message:
    msg = Message(role="user", content=[])
    msg.add_text("What is this PDF about?")
    msg.add_file(file=PDF_PATH)
    return msg


# =========================
# SYNC WORK (IMAGE)
# =========================
def test_work_sync_image():
    agent = Agent(
        instruction=INSTRUCTION,
        model=MODEL_NAME,
    )

    result = agent.work(
        messages=[_image_message()],
        data=DATA,
    )

    print9("\n[SYNC WORK OUTPUT - IMAGE]", color="blue")
    print9(result.text, color="white")

    assert isinstance(result.text, str)
    assert len(result.text) > 0


# =========================
# SYNC WORK (PDF)
# =========================
def test_work_sync_pdf():
    agent = Agent(
        instruction=INSTRUCTION,
        model=MODEL_NAME,
    )

    result = agent.work(
        messages=[_pdf_message()],
        data=DATA,
    )

    print9("\n[SYNC WORK OUTPUT - PDF]", color="blue")
    print9(result.text, color="white")

    assert isinstance(result.text, str)
    assert len(result.text) > 0


# =========================
# ASYNC WORK (IMAGE)
# =========================
@pytest.mark.asyncio
async def test_work_async_image():
    agent = Agent(
        instruction=INSTRUCTION,
        model=MODEL_NAME,
    )

    result = await agent.work_async(
        messages=[_image_message()],
        data=DATA,
    )

    print9("\n[ASYNC WORK OUTPUT - IMAGE]", color="cyan")
    print9(result.text, color="white")

    assert isinstance(result.text, str)
    assert len(result.text) > 0


# =========================
# ASYNC WORK (PDF)
# =========================
@pytest.mark.asyncio
async def test_work_async_pdf():
    agent = Agent(
        instruction=INSTRUCTION,
        model=MODEL_NAME,
    )

    result = await agent.work_async(
        messages=[_pdf_message()],
        data=DATA,
    )

    print9("\n[ASYNC WORK OUTPUT - PDF]", color="cyan")
    print9(result.text, color="white")

    assert isinstance(result.text, str)
    assert len(result.text) > 0


# =========================
# LOAD CONFIG TEST
# =========================
@pytest.mark.asyncio
async def test_load_configs(paths: list[str]):
    for path in paths:
        print9(f"\n[LOAD TEST] {path}", color="magenta")

        agent = Agent.load(path)

        # ---------- IMAGE ----------
        result = agent.work(
            messages=[_image_message()],
            data=DATA,
        )

        assert isinstance(result.text, str)
        assert len(result.text) > 0

        # ---------- PDF ----------
        result_pdf = agent.work(
            messages=[_pdf_message()],
            data=DATA,
        )

        assert isinstance(result_pdf.text, str)
        assert len(result_pdf.text) > 0


# =========================
# SYNC STREAM (IMAGE)
# =========================
def test_stream_sync_image():
    agent = Agent(
        instruction=INSTRUCTION,
        model=MODEL_NAME,
    )

    final_message = None

    print9("\n[SYNC STREAM OUTPUT - IMAGE]", color="blue")

    for chunk in agent.stream(
        messages=[_image_message()],
        data=DATA,
    ):
        if hasattr(chunk, "text"):
            print9(chunk.text, color="white", end="", flush=True)
        final_message = chunk

    print9("", color="white")

    assert final_message is not None
    assert isinstance(final_message.text, str)
    assert len(final_message.text) > 0


# =========================
# SYNC STREAM (PDF)
# =========================
def test_stream_sync_pdf():
    agent = Agent(
        instruction=INSTRUCTION,
        model=MODEL_NAME,
    )

    final_message = None

    print9("\n[SYNC STREAM OUTPUT - PDF]", color="blue")

    for chunk in agent.stream(
        messages=[_pdf_message()],
        data=DATA,
    ):
        if hasattr(chunk, "text"):
            print9(chunk.text, color="white", end="", flush=True)
        final_message = chunk

    print9("", color="white")

    assert final_message is not None
    assert isinstance(final_message.text, str)
    assert len(final_message.text) > 0

# =========================
# ASYNC STREAM (IMAGE)
# =========================
@pytest.mark.asyncio
async def test_stream_async_image():
    agent = Agent(
        instruction=INSTRUCTION,
        model=MODEL_NAME,
    )

    final_message = None

    print9("\n[ASYNC STREAM OUTPUT - IMAGE]", color="cyan")

    async for chunk in agent.stream_async(
        messages=[_image_message()],
        data=DATA,
    ):
        if hasattr(chunk, "text"):
            print9(chunk.text, color="white", end="", flush=True)
        final_message = chunk

    print9("", color="white")

    assert final_message is not None
    assert isinstance(final_message.text, str)
    assert len(final_message.text) > 0


# =========================
# ASYNC STREAM (PDF)
# =========================
@pytest.mark.asyncio
async def test_stream_async_pdf():
    agent = Agent(
        instruction=INSTRUCTION,
        model=MODEL_NAME,
    )

    final_message = None

    print9("\n[ASYNC STREAM OUTPUT - PDF]", color="cyan")

    async for chunk in agent.stream_async(
        messages=[_pdf_message()],
        data=DATA,
    ):
        if hasattr(chunk, "text"):
            print9(chunk.text, color="white", end="", flush=True)
        final_message = chunk

    print9("", color="white")

    assert final_message is not None
    assert isinstance(final_message.text, str)
    assert len(final_message.text) > 0


if __name__ == "__main__":
    print9("Running tests manually...\n", color="green")

    async def run_all():
        # # ---------- SYNC WORK ----------
        # test_work_sync_image()
        # test_work_sync_pdf()

        # ---------- SYNC STREAM ----------
        test_stream_sync_image()
        test_stream_sync_pdf()

        # # ---------- ASYNC WORK ----------
        # await test_work_async_image()
        # await test_work_async_pdf()

        # # ---------- ASYNC STREAM ----------
        # await test_stream_async_image()
        # await test_stream_async_pdf()

        # # ---------- LOAD CONFIG ----------
        # await test_load_configs(
        #     [
        #         "tests/config/agent.yaml",
        #     ]
        # )

    asyncio.run(run_all())

    print9("\nAll manual tests completed ✅", color="green")