# LM Studio Integration for OpenManus

This document describes how to use LM Studio as a local LLM provider with OpenManus, enabling you to run AI models completely locally without external API dependencies.

## Overview

LM Studio is a desktop application that allows you to run large language models locally on your machine. The OpenManus LM Studio integration provides:

- **Local LLM Hosting**: Run models completely locally without external API calls
- **Chat Interface**: Full conversational AI capabilities
- **Streaming Support**: Real-time response streaming
- **Model Management**: Easy model selection and configuration
- **No API Keys Required**: No external service dependencies

## Prerequisites

1. **LM Studio Application**: Download and install LM Studio from [https://lmstudio.ai/](https://lmstudio.ai/)
2. **Python Package**: Install the LM Studio Python SDK:
   ```bash
   pip install lmstudio
   ```
3. **Model**: Download and load a model in LM Studio (e.g., `deepseek/deepseek-r1-0528-qwen3-8b`)

## Setup Instructions

### 1. Start LM Studio Server

1. Open LM Studio application
2. Go to the "Local Server" tab
3. Load a model (e.g., `llama-3.2-1b-instruct`)
4. Click "Start Server" (default port: 1234)
5. Ensure the server is running and accessible

### 2. Configure OpenManus

Create or update your `config/config.toml` file with LM Studio settings:

```toml
[llm]
# Default LLM configuration using LM Studio
[llm.default]
model = "deepseek/deepseek-r1-0528-qwen3-8b"
base_url = "http://localhost:1234/v1"
api_key = "not-needed"
max_tokens = 2048
temperature = 0.7
api_type = "lmstudio"
api_version = "v1"
host = "localhost"
port = 1234

# Alternative LM Studio configuration
[llm.lmstudio_large]
model = "qwen2.5-7b-instruct"
base_url = "http://localhost:1234/v1"
api_key = "not-needed"
max_tokens = 4096
temperature = 0.8
api_type = "lmstudio"
api_version = "v1"
host = "localhost"
port = 1234
```

### 3. Test the Integration

Run the test script to verify everything is working:

```bash
python test_lmstudio_integration.py
```

## Usage Examples

### Basic Usage with OpenManus LLM Class

```python
from app.llm import LLM
from app.config import LLMSettings

# Create LM Studio configuration
lmstudio_settings = LLMSettings(
    model="deepseek/deepseek-r1-0528-qwen3-8b",
    base_url="http://localhost:1234/v1",
    api_key="not-needed",
    max_tokens=2048,
    temperature=0.7,
    api_type="lmstudio",
    api_version="v1",
    host="localhost",
    port=1234
)

# Create LLM instance
llm = LLM("lmstudio", {"lmstudio": lmstudio_settings})

# Send a message
messages = [{"role": "user", "content": "Hello! How are you?"}]
response = await llm.ask(messages)
print(response)
```

### Direct Provider Usage

```python
from app.providers.lmstudio_provider import create_lmstudio_provider

# Create provider
provider = create_lmstudio_provider(
    host="localhost",
    port=1234,
    default_model="deepseek/deepseek-r1-0528-qwen3-8b"
)

# Test availability
if provider.is_available:
    # Send chat completion
    messages = [{"role": "user", "content": "Hello!"}]
    response = await provider.chat_completion(messages)
    print(response.content)
```

### Streaming Responses

```python
# Enable streaming in LLM class
response = await llm.ask(messages, stream=True)

# Or with direct provider
streaming_response = await provider.stream_completion(messages)
async for chunk in streaming_response.content_generator:
    print(chunk, end="", flush=True)
```

## Configuration Options

### LM Studio Provider Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `host` | LM Studio server host | `localhost` |
| `port` | LM Studio server port | `1234` |
| `default_model` | Default model name | `deepseek/deepseek-r1-0528-qwen3-8b` |
| `timeout` | Request timeout (seconds) | `300` |
| `max_retries` | Maximum retry attempts | `3` |
| `retry_delay` | Delay between retries (seconds) | `1.0` |
| `temperature` | Sampling temperature | `0.7` |
| `max_tokens` | Maximum tokens to generate | `2048` |
| `top_p` | Top-p sampling parameter | `0.9` |

### Recommended Models

Popular models that work well with LM Studio:

- **Recommended (Default)**: `deepseek/deepseek-r1-0528-qwen3-8b` - High-performance reasoning model
- **Small/Fast**: `llama-3.2-1b-instruct`, `llama-3.2-3b-instruct`
- **Medium**: `qwen2.5-7b-instruct`, `mistral-7b-instruct`
- **Large**: `llama-3.1-8b-instruct`, `qwen2.5-14b-instruct`

## Troubleshooting

### Common Issues

1. **"LM Studio provider not available"**
   - Install the lmstudio package: `pip install lmstudio`
   - Restart your Python environment

2. **"Failed to configure LM Studio client"**
   - Ensure LM Studio server is running
   - Check that the host and port are correct
   - Verify no firewall is blocking the connection

3. **"Empty response from LM Studio"**
   - Check that a model is loaded in LM Studio
   - Verify the model name matches what's configured
   - Try a different model or adjust parameters

4. **Connection refused errors**
   - Ensure LM Studio local server is started
   - Check the port number (default: 1234)
   - Try restarting LM Studio

### Health Check

Use the built-in health check to diagnose issues:

```python
provider = create_lmstudio_provider()
health = await provider.health_check()
print(f"LM Studio healthy: {health}")
```

### Debug Mode

Enable debug logging to see detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Tips

1. **Model Selection**: Choose smaller models for faster responses
2. **GPU Acceleration**: Enable GPU acceleration in LM Studio for better performance
3. **Context Length**: Adjust context length based on your needs
4. **Temperature**: Lower temperature (0.1-0.3) for more focused responses
5. **Batch Size**: Adjust batch size in LM Studio for optimal performance

## Integration with OpenManus Features

The LM Studio provider integrates seamlessly with all OpenManus features:

- **Agent System**: Use local models for agent reasoning
- **MCP Integration**: Combine with MCP tools for enhanced capabilities
- **Flow System**: Use in multi-agent workflows
- **API Server**: Expose via REST API endpoints
- **WebSocket Support**: Real-time streaming via WebSocket

## Security Considerations

- LM Studio runs locally, so no data leaves your machine
- No API keys or external authentication required
- All processing happens on your local hardware
- Perfect for sensitive or confidential data processing

## Limitations

- Requires local computational resources
- Model performance depends on your hardware
- Limited by available system memory and GPU
- Some advanced features may not be available compared to cloud providers

## Support

For issues specific to:
- **LM Studio Integration**: Check this documentation and run the test script
- **LM Studio Application**: Visit [LM Studio documentation](https://lmstudio.ai/docs)
- **OpenManus**: Check the main OpenManus documentation
