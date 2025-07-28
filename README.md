# OpenManus

**OpenManus** is a comprehensive AI agent framework that provides both a modern web interface and command-line tools for AI-powered task automation. Execute complex workflows, browse the web, analyze data, and interact with multiple AI models through an intuitive interface.

## 🌟 Key Features

- **Modern Web UI**: React-based interface with real-time progress updates
- **Multiple AI Agents**: Specialized agents for different task types
- **Multi-LLM Support**: OpenAI, Anthropic, Google, OpenRouter, and more
- **Web Browser Automation**: Intelligent web browsing and content extraction
- **Real-time Agent Thoughts**: See what the AI is thinking as it works
- **Flexible Deployment**: Local development or Docker containers

## 📋 Prerequisites

- **Python 3.12+** (Required)
- **Node.js 18+** (For UI)
- **Docker** (Optional)
- **API Keys** (At least one LLM provider)

## 🚀 Quick Start

### Local Setup

1. **Clone and Install**
   ```bash
   git clone https://github.com/yourusername/OpenManus.git
   cd OpenManus
   pip install -r requirements.txt
   playwright install
   ```

2. **Configure API Keys**
   ```bash
   cp config/config.example.toml config/config.toml
   # Edit config/config.toml with your API keys
   ```

3. **Start the Application**
   ```bash
   # Terminal 1: Start API server
   python api_server.py
   
   # Terminal 2: Start UI
   cd ui && npm install && npm run dev
   ```

4. **Access the Application**
   - Web UI: http://localhost:3000
   - API Server: http://localhost:8000

### Docker Setup

```bash
# Quick start with Docker Compose
docker-compose up -d

# Access at http://localhost:3000
```

## 🎯 Usage Modes

### Web UI (Recommended)

The web interface provides an intuitive way to interact with AI agents:

- **Task Types**: Choose from specialized interfaces
- **Model Selection**: Switch between different AI models
- **Real-time Updates**: See progress and agent thoughts live
- **Conversation History**: Persistent chat sessions
- **Dark/Light Mode**: Customizable themes

### Command Line

Direct agent execution for automation:

- **main.py**: Single-agent execution
- **run_mcp.py**: MCP-enabled agent
- **run_flow.py**: Multi-agent workflows

## 🔄 Task Types Comparison

| Task Type | Capabilities | Best For |
|-----------|-------------|----------|
| **General Chat** | LLM conversation | Q&A, general discussion |
| **Code Generation** | Programming focus | Writing, debugging code |
| **Data Analysis** | Data processing | CSV, JSON analysis |
| **Web Browsing** | Browser automation | Content extraction, scraping |
| **Manus Agent** | Full agent capabilities | Complex multi-step tasks |
| **Reasoning** | Logical analysis | Problem-solving, logic |
| **Creative Writing** | Creative content | Stories, poetry, content |

### When to Use Each Mode

- **Simple questions** → General Chat
- **Programming tasks** → Code Generation
- **Website content** → Web Browsing
- **Complex multi-step tasks** → Manus Agent
- **Data processing** → Data Analysis

## ⚙️ LLM Configuration

### Supported Providers

- **OpenAI**: GPT-4o, GPT-4o Mini, GPT-3.5 Turbo
- **Anthropic**: Claude 3.5 Sonnet, Claude 3 Haiku
- **Google**: Gemini 2.0 Flash, Gemini Pro
- **OpenRouter**: Unified access to 100+ models
- **Azure OpenAI**: Enterprise OpenAI models
- **AWS Bedrock**: Claude, Llama via AWS
- **Ollama**: Local model hosting

### OpenRouter Configuration

OpenRouter provides access to multiple AI models through a single API. It's often the most cost-effective option with access to the latest models.

**Benefits:**
- Access to 100+ models from different providers
- Competitive pricing and pay-per-use
- No need for multiple API keys
- Access to latest models like DeepSeek V3, Llama 3.1 405B

**Setup:**
1. Sign up at [openrouter.ai](https://openrouter.ai)
2. Get your API key from the dashboard
3. Add to your config:

```toml
[llm.openrouter]
api_key = "sk-or-v1-your-openrouter-api-key"
base_url = "https://openrouter.ai/api/v1"
model = "openai/gpt-4o"  # Default model
max_tokens = 4000
temperature = 0.7
```

### Provider-Specific Configurations

#### OpenAI
```toml
[llm.openai]
api_key = "sk-your-openai-api-key"
model = "gpt-4o"
max_tokens = 4000
temperature = 0.7
organization = "org-your-org-id"  # Optional
```

#### Anthropic
```toml
[llm.anthropic]
api_key = "sk-ant-your-anthropic-api-key"
model = "claude-3-5-sonnet-20241022"
max_tokens = 4000
temperature = 0.7
```

#### Google Gemini
```toml
[llm.google]
api_key = "your-google-api-key"
model = "gemini-2.0-flash-exp"
max_tokens = 4000
temperature = 0.7
```

#### Azure OpenAI
```toml
[llm.azure]
api_key = "your-azure-api-key"
endpoint = "https://your-resource.openai.azure.com/"
api_version = "2024-02-01"
deployment_name = "gpt-4o"
max_tokens = 4000
temperature = 0.7
```

#### AWS Bedrock
```toml
[llm.bedrock]
aws_access_key_id = "your-access-key"
aws_secret_access_key = "your-secret-key"
region = "us-east-1"
model = "anthropic.claude-3-5-sonnet-20241022-v2:0"
max_tokens = 4000
temperature = 0.7
```

#### Ollama (Local)
```toml
[llm.ollama]
base_url = "http://localhost:11434"
model = "llama3.1:8b"
max_tokens = 4000
temperature = 0.7
```

### Model Selection

**In the UI:**
- **Auto Select**: Uses the default configured provider
- **Direct Providers**: OpenAI, Anthropic, Google models
- **OpenRouter Models**: Access to 100+ models via OpenRouter
- **Real-time Switching**: Change models mid-conversation

**Configuration Priority:**
1. UI model selection (highest)
2. Task-specific model in config
3. Default provider model
4. Auto-select fallback

### Cost Optimization

**Recommended Setup for Cost Efficiency:**
1. **Primary**: OpenRouter for access to competitive pricing
2. **Backup**: Direct provider APIs for reliability
3. **Local**: Ollama for privacy-sensitive tasks

**Model Recommendations by Use Case:**
- **General Chat**: GPT-4o Mini, Claude 3 Haiku (cost-effective)
- **Complex Tasks**: GPT-4o, Claude 3.5 Sonnet (high capability)
- **Code Generation**: GPT-4o, DeepSeek V3 (specialized)
- **Creative Writing**: Claude 3.5 Sonnet, Llama 3.1 405B
- **Data Analysis**: GPT-4o, Gemini 2.0 Flash

## 🛠️ Setup Guide

### Local Development

**Requirements:**
- Python 3.12+ with pip
- Node.js 18+ with npm
- At least one LLM provider API key

**Steps:**
1. Clone repository and install dependencies
2. Configure API keys in `config/config.toml`
3. Install Playwright browsers
4. Start API server and UI

### Docker Deployment

**Requirements:**
- Docker and Docker Compose
- Valid configuration file

**Steps:**
1. Copy and edit configuration file
2. Run `docker-compose up -d`
3. Access at http://localhost:3000

**Docker Features:**
- Automated setup and dependencies
- Production-ready configuration
- Easy scaling and deployment
- Isolated environment

### Production Deployment

**Security:**
- Keep API keys secure
- Use environment variables
- Enable HTTPS in production
- Regular security updates

**Performance:**
- Choose appropriate models for your use case
- Monitor resource usage
- Consider load balancing for high traffic
- Enable logging and monitoring

## 🤖 Agent Capabilities

### Manus Agent (Full Capabilities)

- **Multi-tool coordination**: Browser + search + analysis
- **Complex reasoning**: Multi-step decision making
- **Autonomous operation**: Minimal human intervention
- **Real-time thoughts**: See the agent's thinking process

### Specialized Agents

- **Web Browsing**: Playwright-based automation
- **Code Generation**: Programming-focused responses
- **Data Analysis**: CSV/JSON processing with Python
- **Creative Writing**: Content generation and storytelling

### Agent Thoughts Feature

See what the AI is thinking in real-time:
- **Live Updates**: Thoughts stream to the UI as they happen
- **Transparency**: Understand the agent's decision process
- **Debugging**: Identify issues in agent reasoning
- **Learning**: Observe how agents approach problems

## 🔧 Advanced Features

### MCP Integration
- Connect to external tools and services
- Multi-server support with automatic failover
- Real-time health monitoring
- Extensible tool ecosystem

### Browser Automation
- Full Playwright integration
- Intelligent content extraction
- Error recovery and retries
- Multi-site workflow support

### Tool Ecosystem
- Python code execution
- File operations and editing
- Web search integration
- Data processing capabilities

## 📚 Getting Help

- **Configuration**: Check `config/config.example.toml` for all options
- **API Docs**: Visit `/docs` when the server is running
- **Issues**: Report bugs and request features on GitHub
- **Community**: Join discussions and get support

## 🤝 Contributing

We welcome contributions! Please check our contributing guidelines and feel free to submit issues and pull requests.

## 📄 License

MIT License - see LICENSE file for details.
