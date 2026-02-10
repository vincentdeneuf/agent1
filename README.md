# agent1

A simple LLM abstraction layer for Python applications.

## Features

- **Unified API**: Single interface for multiple LLM providers (OpenAI, Anthropic, etc.)
- **Flexible Content Types**: Support for text, images, audio, and file inputs
- **Message Abstraction**: Unified message handling across different API types
- **Pydantic Models**: Type-safe data structures with validation
- **Multiple API Types**: Support for both completion and responses API formats
- **Content Blocks**: Modular content representation for complex multimodal inputs
- **Python 3.11+**: Modern Python features and type hints

## Installation

```bash
pip install agent1
```

## Quick Start

### Basic Usage

```python
from agent1 import Message

# Create a simple text message
message = Message(
    content="Hello, how are you?",
    role="user"
)

# Convert to completion API format
completion_format = message.core(api_type="completion")
print(completion_format)
# Output: {'role': 'user', 'content': 'Hello, how are you?'}
```

### Advanced Usage with Content Blocks

```python
from agent1 import Message, TextBlock, ImageBlock

# Create a message with multiple content types
message = Message(role="user")
message.add_text("What do you see in this image?")
message.add_image("https://example.com/image.jpg", detail="high")

# Get formatted for different API types
completion_format = message.core(api_type="completion")
responses_format = message.core(api_type="responses")

print("Completion format:", completion_format)
print("Responses format:", responses_format)
```

### Message Formatting

```python
from agent1 import Message

# Create a template message
message = Message(
    content="Hello <<name>>, welcome to <<company>>!",
    role="system"
)

# Format with data
message.format({"name": "Alice", "company": "Acme Corp"})
print(message.content)
# Output: "Hello Alice, welcome to Acme Corp!"
```

## API Reference

### Message

The main message class for handling LLM interactions.

**Constructor Parameters:**
- `content`: str | list[ContentBlock] - Message content
- `role`: MessageRole - Message role ("system", "developer", "assistant", "user")
- `data`: dict | list | None - Additional data
- `annotations`: list[dict] | None - Message annotations
- `api_type`: APIType | None - API type ("completion", "responses")
- `stats`: dict | None - Response statistics
- `meta`: dict | None - Metadata

**Methods:**
- `add_text(text: str)`: Add text content block
- `add_image(url: str, detail: str | None)`: Add image content block
- `add_file(url: str)`: Add file content block
- `add_audio(data: str, format: str)`: Add audio content block
- `format(data: dict)`: Format message content with template variables
- `core(api_type: APIType, *, text_mode: bool)`: Convert to API format
- `data_from_content()`: Parse JSON data from content string

**Class Methods:**
- `from_completion(response)`: Create from completion API response
- `from_responses(response)`: Create from responses API response
- `from_litellm(response, api_type)`: Create from LiteLLM response
- `from_chunks(chunks)`: Create from message chunks
- `merge(messages, output_role, api_type)`: Merge multiple messages

### Content Blocks

**TextBlock**: Text content with `text` attribute
**ImageBlock**: Image content with `url` and optional `detail` attribute
**FileBlock**: File content with `url` attribute
**AudioBlock**: Audio content with `data` and `format` attributes

### MessageChunk

Extended Message class for handling streaming responses.

## Development

### Prerequisites

Install [uv](https://docs.astral.sh/uv/) for modern Python dependency management:

```bash
# On Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex

# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation for Development

```bash
# Clone the repository
git clone https://github.com/vincentdeneuf/agent1.git
cd agent1

# Install dependencies and create virtual environment
uv sync

# Activate the virtual environment (optional, uv commands work without it)
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
uv pip install -e .
```

### Running Tests

```bash
# Run tests with uv
uv run pytest

# Run tests with coverage
uv run pytest --cov=agent1 --cov-report=html
```

### Building the Package

```bash
# Build the package
uv build

# Build for distribution
uv build --release
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Changelog

### 0.1.0
- Initial release
- Unified message abstraction for multiple LLM providers
- Support for text, image, audio, and file content blocks
- Completion and responses API format support
- Pydantic-based type safety
- Message formatting and templating
- Streaming response support via MessageChunk