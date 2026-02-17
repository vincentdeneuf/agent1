# agent1

A simple LLM abstraction layer for Python applications.

## Features

- **Unified API**: Single interface for multiple LLM providers (OpenAI, Anthropic, etc.)
- **Flexible Content Types**: Support for text, images, audio, and file inputs
- **Message Abstraction**: Unified message handling across different API types

## Installation

```bash
pip install agent1
```

## Quick Start

```python
from agent1 import Message, Agent

# 1. Create a Message
message = Message(
    content="What is the capital of France?",
    role="user"
)

# 2. Create an Agent
agent = Agent(
    instruction="You are a helpful assistant.",
    model="gpt-4"
)

# 3. Load an Agent from config file
# agent = Agent.load("config/agent.toml")

# 4. Do work (synchronous)
response = agent.work([message])
print(response.content)

# 5. Stream responses
for chunk in agent.stream([message]):
    if hasattr(chunk, 'content'):
        print(chunk.content, end='', flush=True)
```

## Advanced

### Loading Agent from Config File

Create a config file (supports .toml, .json, .yaml, .yml):

```toml
# agent.toml
instruction = "You are a helpful assistant that specializes in Python programming."
model = "gpt-4"
temperature = 0.7
```

Load and use the agent:

```python
from agent1 import Agent, Message

# Load agent from config file
agent = Agent.load("agent.toml")

message = Message(
    content="How do I create a list comprehension in Python?",
    role="user"
)

response = agent.work([message])
print(response.content)
```

### Async Methods

For non-blocking operations, use async methods:

```python
import asyncio
from agent1 import Agent, Message

async def main():
    agent = Agent(
        instruction="You are a helpful assistant.",
        model="gpt-4"
    )
    
    message = Message(
        content="Explain quantum computing in simple terms.",
        role="user"
    )
    
    # Async work
    response = await agent.work_async([message])
    print(response.content)
    
    # Async streaming
    async for chunk in agent.stream_async([message]):
        if hasattr(chunk, 'content'):
            print(chunk.content, end='', flush=True)

# Run the async function
asyncio.run(main())
```
