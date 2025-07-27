# OpenRouter Research Findings

## Overview
- OpenRouter provides a unified API for 400+ AI models from 60+ providers
- OpenAI-compatible API (can use OpenAI SDK)
- Automatic fallbacks and cost optimization
- 8.4T monthly tokens processed
- 2.5M+ global users

## API Details
- Base URL: `https://openrouter.ai/api/v1`
- Authentication: Bearer token with `OPENROUTER_API_KEY`
- Compatible with OpenAI SDK
- Supports streaming responses
- Optional headers for app attribution:
  - `HTTP-Referer`: Site URL for rankings
  - `X-Title`: Site name for rankings

## Key Features
- Tool calling capabilities
- Structured outputs
- Image & PDF processing
- Web search integration
- Prompt caching
- Model routing and provider routing
- Uptime optimization
- Zero completion insurance

## Available Models
- GPT models (OpenAI)
- Claude models (Anthropic)
- Gemini models (Google)
- Llama models (Meta)
- Many other providers and models
- Models API provides comprehensive metadata

## Supported Parameters
- `tools` - Function calling
- `tool_choice` - Tool selection control
- `max_tokens` - Response length limiting
- `temperature` - Randomness control
- `top_p` - Nucleus sampling
- `reasoning` - Internal reasoning mode
- `structured_outputs` - JSON schema enforcement
- `response_format` - Output format specification
- `stop` - Custom stop sequences
- `frequency_penalty` - Repetition reduction
- `presence_penalty` - Topic diversity
- `seed` - Deterministic outputs

## Integration Strategy
1. Create OpenRouter client wrapper
2. Implement model management system
3. Add automatic fallback capabilities
4. Integrate with MCP system
5. Support streaming and tool calling
6. Add cost optimization features

