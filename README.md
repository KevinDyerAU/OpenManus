<p align="center">
  <img src="assets/logo.jpg" width="200"/>
</p>

English | [中文](README_zh.md) | [한국어](README_ko.md) | [日本語](README_ja.md)

[![GitHub stars](https://img.shields.io/github/stars/FoundationAgents/OpenManus?style=social)](https://github.com/FoundationAgents/OpenManus/stargazers)
&ensp;
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) &ensp;
[![Discord Follow](https://dcbadge.vercel.app/api/server/DYn29wFk9z?style=flat)](https://discord.gg/DYn29wFk9z)
[![Demo](https://img.shields.io/badge/Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/lyh-917/OpenManusDemo)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15186407.svg)](https://doi.org/10.5281/zenodo.15186407)

# 👋 OpenManus

Manus is incredible, but OpenManus can achieve any idea without an *Invite Code* 🛫!

OpenManus is a comprehensive AI agent framework that provides multiple execution modes and advanced capabilities for building intelligent automation systems. With support for browser automation, web scraping, data analysis, and multi-agent workflows, OpenManus empowers developers to create sophisticated AI-driven applications.

Our team members [@Xinbin Liang](https://github.com/mannaandpoem) and [@Jinyu Xiang](https://github.com/XiangJinyu) (core authors), along with [@Zhaoyang Yu](https://github.com/MoshiQAQ), [@Jiayi Zhang](https://github.com/didiforgithub), and [@Sirui Hong](https://github.com/stellaHSR), we are from [@MetaGPT](https://github.com/geekan/MetaGPT). The prototype is launched within 3 hours and we are keeping building!

It's a simple implementation, so we welcome any suggestions, contributions, and feedback!

Enjoy your own agent with OpenManus!

We're also excited to introduce [OpenManus-RL](https://github.com/OpenManus/OpenManus-RL), an open-source project dedicated to reinforcement learning (RL)- based (such as GRPO) tuning methods for LLM agents, developed collaboratively by researchers from UIUC and OpenManus.

## 🚀 Key Features

### Core Capabilities
- **Multiple Execution Modes**: Basic agent, MCP integration, multi-agent workflows, and REST API
- **Browser Automation**: Full Playwright integration with headless/headed modes
- **Web Scraping & Search**: Support for Google, Baidu, DuckDuckGo, and Bing search engines
- **Data Analysis**: Integrated data analysis agent with visualization capabilities
- **Sandbox Environment**: Secure Docker-based code execution
- **Multi-LLM Support**: OpenAI, Anthropic, Azure OpenAI, Ollama, and AWS Bedrock

### Advanced Features
- **Model Context Protocol (MCP)**: Server and client implementations
- **Proxy Support**: HTTP proxy configuration for browser and API requests
- **Async Processing**: Full asynchronous architecture for high performance
- **Extensible Architecture**: Plugin-based tool system
- **Comprehensive Logging**: Structured logging with Loguru
- **Configuration Management**: TOML-based configuration with environment support

## Project Demo

<video src="https://private-user-images.githubusercontent.com/61239030/420168772-6dcfd0d2-9142-45d9-b74e-d10aa75073c6.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDEzMTgwNTksIm5iZiI6MTc0MTMxNzc1OSwicGF0aCI6Ii82MTIzOTAzMC80MjAxNjg3NzItNmRjZmQwZDItOTE0Mi00NWQ5LWI3NGUtZDEwYWE3NTA3M2M2Lm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTAzMDclMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwMzA3VDAzMjIzOVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTdiZjFkNjlmYWNjMmEzOTliM2Y3M2VlYjgyNDRlZDJmOWE3NWZhZjE1MzhiZWY4YmQ3NjdkNTYwYTU5ZDA2MzYmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.UuHQCgWYkh0OQq9qsUWqGsUbhG3i9jcZDAMeHjLt5T4" data-canonical-src="https://private-user-images.githubusercontent.com/61239030/420168772-6dcfd0d2-9142-45d9-b74e-d10aa75073c6.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDEzMTgwNTksIm5iZiI6MTc0MTMxNzc1OSwicGF0aCI6Ii82MTIzOTAzMC80MjAxNjg3NzItNmRjZmQwZDItOTE0Mi00NWQ5LWI3NGUtZDEwYWE3NTA3M2M2Lm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTAzMDclMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwMzA3VDAzMjIzOVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTdiZjFkNjlmYWNjMmEzOTliM2Y3M2VlYjgyNDRlZDJmOWE3NWZhZjE1MzhiZWY4YmQ3NjdkNTYwYTU5ZDA2MzYmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.UuHQCgWYkh0OQq9qsUWqGsUbhG3i9jcZDAMeHjLt5T4" controls="controls" muted="muted" class="d-block rounded-bottom-2 border-top width-fit" style="max-height:640px; min-height: 200px"></video>

## 📦 Installation & Dependencies

### System Requirements
- **Python**: 3.12 or higher (required)
- **Node.js**: 16+ (for Playwright browser automation)
- **Docker**: Optional (for sandbox execution)
- **Operating System**: Windows, macOS, or Linux

### Core Dependencies Overview
OpenManus includes 40+ carefully selected Python packages:

**AI & LLM Integration:**
- `openai>=1.66.3` - OpenAI API client
- `pydantic>=2.10.6` - Data validation and settings
- `tiktoken>=0.9.0` - Token counting for OpenAI models
- `huggingface-hub>=0.29.2` - Hugging Face model integration

**Web & Browser Automation:**
- `playwright>=1.51.0` - Browser automation framework
- `browser-use>=0.1.40` - Browser interaction utilities
- `browsergym>=0.13.3` - Browser environment for RL
- `html2text>=2024.2.26` - HTML to text conversion
- `beautifulsoup4>=4.13.3` - HTML parsing
- `crawl4ai>=0.6.3` - Advanced web crawling

**Search & Data Processing:**
- `googlesearch-python>=1.3.0` - Google search integration
- `duckduckgo-search>=7.5.3` - DuckDuckGo search
- `baidusearch>=1.0.3` - Baidu search support
- `datasets>=3.4.1` - Dataset handling
- `numpy>=1.24.0` - Numerical computing

**API & Networking:**
- `fastapi>=0.115.11` - REST API framework
- `uvicorn>=0.34.0` - ASGI server
- `httpx>=0.27.0` - Async HTTP client
- `requests>=2.32.3` - HTTP library
- `aiofiles>=24.1.0` - Async file operations

**Configuration & Utilities:**
- `pyyaml>=6.0.2` - YAML configuration
- `tomli>=2.0.0` - TOML configuration
- `loguru>=0.7.3` - Advanced logging
- `tenacity>=9.0.0` - Retry mechanisms
- `colorama>=0.4.6` - Colored terminal output

### Installation Methods

We provide multiple installation methods. **Method 2 (using uv)** is recommended for faster installation and better dependency management.

#### Method 1: Using conda

1. **Create and activate conda environment:**
```bash
conda create -n open_manus python=3.12
conda activate open_manus
```

2. **Clone the repository:**
```bash
git clone https://github.com/FoundationAgents/OpenManus.git
cd OpenManus
```

3. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

#### Method 2: Using uv (Recommended)

1. **Install uv (Fast Python package installer):**
```bash
# On Unix/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (PowerShell):
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

2. **Clone the repository:**
```bash
git clone https://github.com/FoundationAgents/OpenManus.git
cd OpenManus
```

3. **Create and activate virtual environment:**
```bash
uv venv --python 3.12

# On Unix/macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

4. **Install dependencies:**
```bash
uv pip install -r requirements.txt
```

#### Method 3: Using Docker

1. **Build Docker image:**
```bash
docker build -t openmanus .
```

2. **Run container:**
```bash
docker run -it --rm -v $(pwd)/config:/app/config openmanus
```

### Additional Setup Steps

#### Browser Automation Setup
**Required for web scraping and browser automation features:**
```bash
playwright install
playwright install-deps  # Install system dependencies
```

#### Node.js Dependencies (for testing)
**Optional - only needed for E2E testing:**
```bash
npm install
```

#### Development Dependencies
**For contributors and developers:**
```bash
pip install pre-commit
pre-commit install
```

## ⚙️ Configuration

OpenManus uses a comprehensive TOML-based configuration system that supports multiple LLM providers, browser settings, search engines, and advanced features.

### Quick Setup

1. **Create configuration file:**
```bash
cp config/config.example.toml config/config.toml
```

2. **Edit configuration:**
```bash
# Open config/config.toml in your preferred editor
nano config/config.toml  # or vim, code, etc.
```

### LLM Provider Configuration

#### OpenAI Configuration
```toml
[llm]
model = "gpt-4o"
base_url = "https://api.openai.com/v1"
api_key = "sk-your-openai-api-key"
max_tokens = 4096
temperature = 0.0
```

#### Anthropic Claude Configuration
```toml
[llm]
model = "claude-3-7-sonnet-20250219"
base_url = "https://api.anthropic.com/v1/"
api_key = "your-anthropic-api-key"
max_tokens = 8192
temperature = 0.0
```

#### Azure OpenAI Configuration
```toml
[llm]
api_type = "azure"
model = "gpt-4o-mini"
base_url = "https://your-resource.openai.azure.com/openai/deployments/your-deployment"
api_key = "your-azure-api-key"
api_version = "2024-08-01-preview"
max_tokens = 8096
temperature = 0.0
```

#### Ollama (Local) Configuration
```toml
[llm]
api_type = "ollama"
model = "llama3.2"
base_url = "http://localhost:11434/v1"
api_key = "ollama"
max_tokens = 4096
temperature = 0.0
```

#### AWS Bedrock Configuration
```toml
[llm]
api_type = "aws"
model = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
base_url = "bedrock-runtime.us-west-2.amazonaws.com"
max_tokens = 8192
temperature = 1.0
api_key = "bear"  # Required but not used
```

#### OpenRouter Configuration (400+ Models)
**Access to 400+ AI models from 60+ providers through a single API:**
```toml
[llm]
api_type = "openrouter"
model = "openai/gpt-4o"                    # Default model
base_url = "https://openrouter.ai/api/v1"
api_key = "sk-or-your-openrouter-api-key"
max_tokens = 4096
temperature = 0.0

# OpenRouter-specific settings
[llm.openrouter]
site_url = "https://your-site.com"         # Optional: for model rankings
site_name = "Your App Name"               # Optional: for model rankings
enable_fallback = true                    # Enable automatic fallback
fallback_models = [                       # Fallback model chain
    "anthropic/claude-3.5-sonnet",
    "google/gemini-2.5-pro",
    "openai/gpt-4o-mini"
]
cost_limit = 1.0                          # Maximum cost per request ($)
rate_limit = 60                           # Requests per minute
timeout = 60                              # Request timeout (seconds)
max_retries = 3                           # Maximum retry attempts
retry_delay = 1.0                         # Retry delay (seconds)

# Model selection preferences
[llm.openrouter.preferences]
preferred_providers = ["openai", "anthropic", "google"]
budget_mode = false                       # Prioritize cost over performance
performance_mode = true                   # Prioritize performance over cost
auto_select = true                        # Enable automatic model selection
```

### Vision Model Configuration
```toml
[llm.vision]
model = "claude-3-7-sonnet-20250219"
base_url = "https://api.anthropic.com/v1/"
api_key = "your-anthropic-api-key"
max_tokens = 8192
temperature = 0.0
```

### Browser Configuration
```toml
[browser]
headless = false                    # Run browser in headless mode
disable_security = true             # Disable browser security features
extra_chromium_args = []            # Additional browser arguments
chrome_instance_path = ""           # Path to Chrome executable
wss_url = ""                       # WebSocket URL for remote browser
cdp_url = ""                       # Chrome DevTools Protocol URL

# Proxy configuration
[browser.proxy]
server = "http://proxy-server:port"
username = "proxy-username"
password = "proxy-password"
```

### Search Engine Configuration
```toml
[search]
engine = "Google"                           # Primary search engine
fallback_engines = ["DuckDuckGo", "Baidu", "Bing"]  # Fallback order
retry_delay = 60                            # Retry delay in seconds
max_retries = 3                             # Maximum retry attempts
lang = "en"                                 # Language code
country = "us"                              # Country code
```

### Sandbox Configuration
```toml
[sandbox]
use_sandbox = false                 # Enable Docker sandbox
image = "python:3.12-slim"         # Docker image
work_dir = "/workspace"             # Working directory
memory_limit = "1g"                # Memory limit
cpu_limit = 2.0                     # CPU limit
timeout = 300                       # Execution timeout
network_enabled = true              # Network access
```

### MCP Configuration
```toml
[mcp]
server_reference = "app.mcp.server"  # MCP server module reference
```

### Multi-Agent Configuration
```toml
[runflow]
use_data_analysis_agent = false     # Enable data analysis agent
```

### Callback Configuration
**Real-time event notifications and progress tracking:**
```toml
[callbacks]
# Enable callback system
enable_callbacks = true

# Delivery method: webhook, websocket, sse, polling
delivery_method = "webhook"
webhook_url = "https://your-app.com/api/callbacks"

# Event types to track
events = [
    "thinking",          # AI reasoning process
    "tool_use",          # Tool execution events
    "tool_result",       # Tool execution results
    "progress",          # Task progress updates
    "completion",        # Task completion
    "error",             # Error events
    "workflow_start",    # Workflow initiation
    "workflow_step",     # Individual workflow steps
    "workflow_complete", # Workflow completion
    "model_selection",   # AI model selection events
    "streaming_chunk"    # Real-time streaming data
]

# Delivery settings
include_intermediate_results = true   # Include partial results
timeout = 30                         # Callback timeout (seconds)
retry_attempts = 3                   # Retry failed callbacks
retry_delay = 1.0                    # Delay between retries (seconds)

# Custom headers for webhook callbacks
[callbacks.headers]
Authorization = "Bearer your-api-token"
Content-Type = "application/json"
X-API-Version = "v1"

# WebSocket configuration (if using websocket delivery)
[callbacks.websocket]
reconnect_attempts = 5
reconnect_delay = 2.0
heartbeat_interval = 30

# Server-Sent Events configuration (if using SSE delivery)
[callbacks.sse]
keep_alive_interval = 30
max_connections = 100
buffer_size = 1000
```

## 🚀 Quick Start

OpenManus provides multiple execution modes to suit different use cases. Choose the mode that best fits your needs:

### 1. Basic Agent Mode (Recommended for beginners)

**Single command execution:**
```bash
python main.py
```

**With command-line prompt:**
```bash
python main.py --prompt "Create a web scraper for news articles"
```

**Interactive mode:**
```bash
python main.py
# Then enter your prompt when asked
Enter your prompt: Analyze this website and extract key information
```

### 2. MCP (Model Context Protocol) Mode

**For advanced tool integration and server-client architecture:**

```bash
# Start MCP server
python run_mcp_server.py

# In another terminal, run MCP client
python run_mcp.py
```

**MCP with specific connection types:**
```bash
# STDIO connection (default)
python run_mcp.py --connection-type stdio

# SSE connection with custom server
python run_mcp.py --connection-type sse --server-url http://localhost:8000

# Single prompt execution
python run_mcp.py --prompt "Analyze data from this CSV file"
```

### 3. Multi-Agent Flow Mode

**For complex tasks requiring multiple specialized agents:**

```bash
python run_flow.py
```

**Enable data analysis capabilities:**
1. Update `config/config.toml`:
```toml
[runflow]
use_data_analysis_agent = true
```

2. Install additional dependencies:
```bash
pip install matplotlib seaborn pandas plotly
```

3. Run the flow:
```bash
python run_flow.py
# Enter complex prompts like:
# "Analyze sales data, create visualizations, and generate a report"
```

### 4. REST API Mode

**For web service integration:**

```bash
# Start the API server
python api_server.py

# Server will be available at http://localhost:8000
# API documentation at http://localhost:8000/docs
```

**Example API usage:**
```bash
# Using curl
curl -X POST "http://localhost:8000/agent/run" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Scrape product information from this e-commerce site"}'

# Using Python requests
import requests
response = requests.post(
    "http://localhost:8000/agent/run",
    json={"prompt": "Generate a data analysis report"}
)
print(response.json())
```

## 📖 Detailed Usage Guide

### Browser Automation Examples

**Web scraping:**
```bash
python main.py --prompt "Visit https://example.com and extract all product prices"
```

**Form filling:**
```bash
python main.py --prompt "Fill out the contact form on this website with test data"
```

**Screenshot and analysis:**
```bash
python main.py --prompt "Take a screenshot of this webpage and analyze its layout"
```

### Data Analysis Examples

**CSV analysis:**
```bash
python run_flow.py
# Prompt: "Load sales_data.csv, analyze trends, and create visualizations"
```

**Statistical analysis:**
```bash
python run_flow.py
# Prompt: "Perform correlation analysis on this dataset and generate insights"
```

### Search and Research Examples

**Web research:**
```bash
python main.py --prompt "Research the latest trends in AI and summarize findings"
```

**Competitive analysis:**
```bash
python main.py --prompt "Compare pricing strategies of top 5 competitors in the SaaS market"
```

### OpenRouter Usage Examples

**Access 400+ AI models with automatic selection:**
```bash
# Configure OpenRouter in config.toml
[llm]
api_type = "openrouter"
api_key = "sk-or-your-api-key"

# Run with automatic model selection
python main.py --prompt "Write a Python script for data analysis"
# OpenRouter will automatically select the best model for coding tasks
```

**Model-specific requests:**
```bash
# Use specific models for different tasks
python main.py --model "anthropic/claude-3.5-sonnet" --prompt "Analyze this complex document"
python main.py --model "openai/gpt-4o" --prompt "Generate creative content"
python main.py --model "google/gemini-2.5-pro" --prompt "Process this image and extract text"
```

**Cost-optimized usage:**
```toml
# Budget-friendly configuration
[llm.openrouter.preferences]
budget_mode = true
cost_limit = 0.01  # Maximum $0.01 per request
fallback_models = [
    "openai/gpt-4o-mini",
    "anthropic/claude-3-haiku",
    "google/gemini-1.5-flash"
]
```

**Performance-optimized usage:**
```toml
# High-performance configuration
[llm.openrouter.preferences]
performance_mode = true
preferred_providers = ["openai", "anthropic"]
fallback_models = [
    "openai/gpt-4o",
    "anthropic/claude-3.5-sonnet",
    "openai/gpt-4-turbo"
]
```

### Callback System Usage Examples

**Webhook callbacks for real-time monitoring:**
```python
# Your webhook endpoint (Flask example)
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/callbacks', methods=['POST'])
def handle_callback():
    event = request.json
    
    if event['event_type'] == 'thinking':
        print(f"AI is thinking: {event['data']['thought']}")
    elif event['event_type'] == 'tool_use':
        print(f"Using tool: {event['data']['tool_name']}")
    elif event['event_type'] == 'progress':
        print(f"Progress: {event['data']['progress']}% - {event['data']['message']}")
    elif event['event_type'] == 'completion':
        print(f"Task completed: {event['data']['result']}")
    
    return jsonify({"status": "received"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**WebSocket callbacks for real-time streaming:**
```python
# WebSocket client example
import asyncio
import websockets
import json

async def callback_handler():
    uri = "ws://localhost:8000/ws/callbacks/your-session-id"
    
    async with websockets.connect(uri) as websocket:
        async for message in websocket:
            event = json.loads(message)
            
            if event['event_type'] == 'streaming_chunk':
                print(event['data']['content'], end='', flush=True)
            elif event['event_type'] == 'tool_use':
                print(f"\n[TOOL] {event['data']['tool_name']}")
            elif event['event_type'] == 'completion':
                print(f"\n[DONE] {event['data']['result']}")
                break

# Run the callback handler
asyncio.run(callback_handler())
```

**Server-Sent Events (SSE) for web applications:**
```javascript
// JavaScript SSE client
const eventSource = new EventSource('/api/callbacks/sse/your-session-id');

eventSource.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    switch(data.event_type) {
        case 'thinking':
            updateThinkingDisplay(data.data.thought);
            break;
        case 'progress':
            updateProgressBar(data.data.progress, data.data.message);
            break;
        case 'tool_use':
            showToolExecution(data.data.tool_name, data.data.parameters);
            break;
        case 'completion':
            showResults(data.data.result);
            break;
    }
};

eventSource.onerror = function(event) {
    console.error('SSE connection error:', event);
};
```

**Polling-based callbacks:**
```python
# Polling client example
import requests
import time
from datetime import datetime

def poll_callbacks(session_id, base_url="http://localhost:8000"):
    last_check = datetime.now()
    
    while True:
        response = requests.get(
            f"{base_url}/api/callbacks/poll/{session_id}",
            params={"since": last_check.isoformat()}
        )
        
        if response.status_code == 200:
            events = response.json().get('events', [])
            
            for event in events:
                print(f"[{event['event_type']}] {event['data']}")
                
                if event['event_type'] == 'completion':
                    return event['data']['result']
            
            last_check = datetime.now()
        
        time.sleep(1)  # Poll every second

# Usage
result = poll_callbacks("your-session-id")
print(f"Final result: {result}")
```

### Advanced Configuration Examples

**Custom browser settings:**
```toml
[browser]
headless = true
extra_chromium_args = ["--disable-dev-shm-usage", "--no-sandbox"]

[browser.proxy]
server = "http://proxy.company.com:8080"
username = "user"
password = "pass"
```

**Multiple search engines:**
```toml
[search]
engine = "Google"
fallback_engines = ["DuckDuckGo", "Bing"]
max_retries = 5
retry_delay = 30
```

**Combined OpenRouter + Callbacks configuration:**
```toml
[llm]
api_type = "openrouter"
model = "openai/gpt-4o"
api_key = "sk-or-your-api-key"

[llm.openrouter]
enable_fallback = true
cost_limit = 0.50
rate_limit = 30

[callbacks]
enable_callbacks = true
delivery_method = "webhook"
webhook_url = "https://your-app.com/api/callbacks"
events = ["thinking", "tool_use", "progress", "completion", "model_selection"]
```

### Troubleshooting Common Issues

**Browser automation issues:**
```bash
# Install browser dependencies
playwright install-deps

# Run in headed mode for debugging
# Set headless = false in config.toml
```

**API rate limiting:**
```bash
# Configure retry settings in config.toml
[search]
retry_delay = 60
max_retries = 3
```

**Memory issues with large datasets:**
```bash
# Enable sandbox mode for isolation
[sandbox]
use_sandbox = true
memory_limit = "2g"
```

## How to contribute

We welcome any friendly suggestions and helpful contributions! Just create issues or submit pull requests.

Or contact @mannaandpoem via 📧email: mannaandpoem@gmail.com

**Note**: Before submitting a pull request, please use the pre-commit tool to check your changes. Run `pre-commit run --all-files` to execute the checks.

## Community Group
Join our networking group on Feishu and share your experience with other developers!

<div align="center" style="display: flex; gap: 20px;">
    <img src="assets/community_group.jpg" alt="OpenManus 交流群" width="300" />
</div>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=FoundationAgents/OpenManus&type=Date)](https://star-history.com/#FoundationAgents/OpenManus&Date)

## Sponsors
Thanks to [PPIO](https://ppinfra.com/user/register?invited_by=OCPKCN&utm_source=github_openmanus&utm_medium=github_readme&utm_campaign=link) for computing source support.
> PPIO: The most affordable and easily-integrated MaaS and GPU cloud solution.


## Acknowledgement

Thanks to [anthropic-computer-use](https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo)
and [browser-use](https://github.com/browser-use/browser-use) for providing basic support for this project!

Additionally, we are grateful to [AAAJ](https://github.com/metauto-ai/agent-as-a-judge), [MetaGPT](https://github.com/geekan/MetaGPT), [OpenHands](https://github.com/All-Hands-AI/OpenHands) and [SWE-agent](https://github.com/SWE-agent/SWE-agent).

We also thank stepfun(阶跃星辰) for supporting our Hugging Face demo space.

OpenManus is built by contributors from MetaGPT. Huge thanks to this agent community!

## Cite
```bibtex
@misc{openmanus2025,
  author = {Xinbin Liang and Jinyu Xiang and Zhaoyang Yu and Jiayi Zhang and Sirui Hong and Sheng Fan and Xiao Tang},
  title = {OpenManus: An open-source framework for building general AI agents},
  year = {2025},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.15186407},
  url = {https://doi.org/10.5281/zenodo.15186407},
}
```
