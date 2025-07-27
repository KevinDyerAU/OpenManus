# OpenRouter Integration and Headless Browser Setup Guide

## Overview

This guide covers the implementation of OpenRouter integration for unified LLM access and headless browser automation for OpenManus. The integration provides access to 400+ AI models through a single API and comprehensive web automation capabilities for AI agents.

## OpenRouter Integration

### Architecture Overview

The OpenRouter integration consists of two main components:

1. **OpenRouter Client** (`app/llm/openrouter_client.py`)
   - Unified API client for accessing 400+ models
   - OpenAI-compatible interface
   - Automatic fallback and retry mechanisms
   - Cost tracking and rate limiting
   - Streaming and tool calling support

2. **Model Manager** (`app/llm/model_manager.py`)
   - Intelligent model selection based on task requirements
   - Performance tracking and optimization
   - Cost analysis and budget management
   - Model capability matching
   - Recommendation system with reasoning

### Key Features

#### Unified Model Access
- Access to 400+ models from 60+ providers
- OpenAI-compatible API interface
- Automatic model discovery and metadata caching
- Provider-agnostic tool calling and structured outputs

#### Intelligent Model Selection
- Task-specific model recommendations
- Cost-performance optimization
- Capability-based filtering
- Automatic fallback chains

#### Cost Management
- Real-time cost estimation
- Budget limits and alerts
- Usage tracking and analytics
- Cost-performance analysis

#### Performance Optimization
- Automatic retry with exponential backoff
- Rate limiting and request queuing
- Response caching and optimization
- Performance metrics tracking

### Implementation Details

#### OpenRouter Client Configuration

```python
from app.llm import OpenRouterConfig, OpenRouterClient

# Basic configuration
config = OpenRouterConfig(
    api_key="your_openrouter_api_key",
    site_url="https://your-site.com",  # Optional for rankings
    site_name="Your App Name",         # Optional for rankings
    default_model="openai/gpt-4o",
    timeout=60,
    max_retries=3,
    enable_fallback=True,
    fallback_models=[
        "anthropic/claude-3.5-sonnet",
        "google/gemini-2.5-pro",
        "openai/gpt-4o-mini"
    ],
    cost_limit=1.0,  # $1 per request limit
    rate_limit=60    # 60 requests per minute
)

client = OpenRouterClient(config)
```

#### Model Management Setup

```python
from app.llm import ModelManager, TaskType, ModelSelectionCriteria

# Create model manager
manager = ModelManager(client)

# Define selection criteria
criteria = ModelSelectionCriteria(
    task_type=TaskType.CODE_GENERATION,
    max_cost=0.01,
    required_capabilities=[ModelCapability.TOOL_CALLING],
    preferred_providers=["openai", "anthropic"],
    min_context_length=8000
)

# Get recommendation
recommendation = await manager.get_model_recommendation(criteria)
```

#### Chat Completion with Automatic Model Selection

```python
from app.llm import ChatMessage, ChatCompletionRequest

# Define messages
messages = [
    ChatMessage(role="system", content="You are a helpful coding assistant."),
    ChatMessage(role="user", content="Write a Python function to calculate fibonacci numbers.")
]

# Execute with best model for the task
response = await manager.execute_with_best_model(
    messages=messages,
    criteria=criteria,
    max_tokens=1000,
    temperature=0.1
)

print(response.choices[0]["message"]["content"])
```

### Available Models and Capabilities

#### Model Providers
- **OpenAI**: GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-3.5-turbo
- **Anthropic**: Claude-3.5-sonnet, Claude-3-opus, Claude-3-haiku
- **Google**: Gemini-2.5-pro, Gemini-1.5-pro, Gemini-1.5-flash
- **Meta**: Llama-3.1-405b, Llama-3.1-70b, Llama-3.1-8b
- **Mistral**: Mistral-large, Mistral-medium, Mistral-small
- **Many others**: 400+ models total

#### Model Capabilities
- **Text Generation**: All models support basic text generation
- **Tool Calling**: Function calling and structured outputs
- **Vision**: Image understanding and analysis
- **Code Generation**: Specialized coding models
- **Reasoning**: Advanced reasoning capabilities
- **Long Context**: Models with 32K+ context windows
- **Structured Output**: JSON schema enforcement
- **Web Search**: Integrated web search capabilities

#### Task-Specific Model Selection

```python
# Code generation task
code_criteria = ModelSelectionCriteria(
    task_type=TaskType.CODE_GENERATION,
    required_capabilities=[ModelCapability.CODE_GENERATION],
    preferred_providers=["openai", "anthropic"]
)

# Vision analysis task
vision_criteria = ModelSelectionCriteria(
    task_type=TaskType.VISION_ANALYSIS,
    required_capabilities=[ModelCapability.VISION],
    min_context_length=16000
)

# Budget-optimized task
budget_criteria = ModelSelectionCriteria(
    task_type=TaskType.GENERAL_CHAT,
    max_cost=0.001,
    tier=ModelTier.BUDGET
)
```

### Cost Analysis and Optimization

#### Cost Estimation

```python
# Estimate cost before execution
request = ChatCompletionRequest(
    model="openai/gpt-4o",
    messages=messages,
    max_tokens=1000
)

cost_estimate = await client.estimate_cost(request)
print(f"Estimated cost: ${cost_estimate['estimated_cost']:.4f}")
```

#### Budget Management

```python
# Set cost limits
config.cost_limit = 0.10  # $0.10 per request
config.rate_limit = 30    # 30 requests per minute

# Track usage
stats = client.get_usage_stats()
print(f"Total cost: ${stats['total_cost']:.2f}")
print(f"Total requests: {stats['total_requests']}")
```

#### Performance Analytics

```python
# Get performance statistics
perf_stats = await manager.get_performance_stats(
    model_id="openai/gpt-4o",
    task_type=TaskType.CODE_GENERATION
)

print(f"Success rate: {perf_stats['success_rate']:.2%}")
print(f"Average latency: {perf_stats['average_latency']:.2f}s")
print(f"Average cost: ${perf_stats['average_cost']:.4f}")
```

## Headless Browser Automation

### Architecture Overview

The headless browser automation system provides comprehensive web automation capabilities:

1. **Headless Browser** (`app/browser/headless_browser.py`)
   - Playwright-based browser automation
   - Multi-browser support (Chromium, Firefox, WebKit)
   - Session management and state persistence
   - Request/response logging and monitoring

2. **MCP Browser Tools** (`app/browser/mcp_browser_tools.py`)
   - Browser automation as MCP tools
   - Integration with AI agents
   - Bulk operations and monitoring
   - Screenshot and content extraction

### Key Features

#### Comprehensive Browser Automation
- Page navigation and interaction
- Form filling and submission
- Element clicking and manipulation
- JavaScript execution
- File downloads and uploads

#### Content Extraction
- Text extraction from pages or elements
- Link and image extraction
- Structured data extraction
- Screenshot capture
- PDF and document processing

#### Session Management
- Cookie and local storage management
- Authentication state persistence
- Multi-tab and window support
- Request/response interception

#### Monitoring and Analytics
- Real-time page change monitoring
- Request/response logging
- Performance metrics tracking
- Error handling and recovery

### Implementation Details

#### Basic Browser Setup

```python
from app.browser import HeadlessBrowser, BrowserConfig, BrowserType

# Create browser configuration
config = BrowserConfig(
    browser_type=BrowserType.CHROMIUM,
    headless=True,
    viewport_width=1920,
    viewport_height=1080,
    timeout=30000,
    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
)

# Create and start browser
browser = HeadlessBrowser(config)
await browser.start()
```

#### Page Navigation and Interaction

```python
# Navigate to page
result = await browser.navigate("https://example.com")
print(f"Navigation successful: {result.success}")
print(f"Load time: {result.load_time:.2f}s")

# Take screenshot
screenshot_path = await browser.take_screenshot(full_page=True)

# Extract content
text = await browser.extract_text()
links = await browser.extract_links()
images = await browser.extract_images()

# Interact with elements
await browser.click_element("button#submit")
await browser.fill_form([
    FormData(selector="input[name='email']", value="user@example.com"),
    FormData(selector="input[name='password']", value="password123")
])
```

#### MCP Integration

```python
from app.browser import BrowserMCPTools

# Create MCP tools
browser_tools = BrowserMCPTools()

# Get tool definitions for MCP server
tool_definitions = await browser_tools.get_tool_definitions()

# Execute browser operations via MCP
result = await browser_tools.navigate({
    "url": "https://example.com",
    "wait_until": "load"
})

# Extract data via MCP
extraction_result = await browser_tools.search_and_extract({
    "url": "https://news.example.com",
    "search_selectors": {
        "text": "article h1",
        "links": "article a",
        "images": "article img"
    },
    "take_screenshot": True
})
```

#### Advanced Browser Operations

```python
# Monitor page changes
monitoring_result = await browser_tools.monitor_changes({
    "url": "https://live-updates.example.com",
    "selector": "#live-content",
    "interval": 5,
    "max_checks": 20
})

# Bulk data extraction
bulk_result = await browser_tools.bulk_extract({
    "urls": [
        "https://page1.example.com",
        "https://page2.example.com",
        "https://page3.example.com"
    ],
    "extraction_config": {
        "text_selector": "h1",
        "links_selector": "a",
        "custom_selectors": {
            "price": ".price",
            "rating": ".rating"
        }
    },
    "delay_between_pages": 2
})
```

### Browser Configurations

#### Scraping-Optimized Browser

```python
from app.browser import create_scraping_browser

# Optimized for web scraping
browser = create_scraping_browser()
# - Blocks images and stylesheets for speed
# - Stealth headers to avoid detection
# - Extended timeouts for slow sites
# - Request/response logging enabled
```

#### Mobile Browser

```python
config = BrowserConfig(
    viewport_width=375,
    viewport_height=667,
    user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15"
)
browser = HeadlessBrowser(config)
```

#### Testing Browser

```python
from app.browser import create_testing_browser

# Visible browser for debugging
browser = create_testing_browser()
# - Visible (non-headless) for debugging
# - Slower execution for visibility
# - Enhanced logging and error reporting
```

### MCP Tool Integration

#### Available Browser Tools

The browser automation system provides comprehensive MCP tools:

1. **Navigation Tools**
   - `browser_navigate`: Navigate to URLs
   - `browser_get_page_info`: Get page information
   - `browser_take_screenshot`: Capture screenshots

2. **Content Extraction Tools**
   - `browser_extract_text`: Extract text content
   - `browser_extract_links`: Extract all links
   - `browser_extract_images`: Extract all images

3. **Interaction Tools**
   - `browser_click_element`: Click page elements
   - `browser_fill_form`: Fill and submit forms
   - `browser_wait_for_element`: Wait for elements
   - `browser_scroll_page`: Scroll pages

4. **Advanced Tools**
   - `browser_execute_javascript`: Execute custom JavaScript
   - `browser_download_file`: Download files
   - `browser_search_and_extract`: Combined search and extraction
   - `browser_monitor_changes`: Monitor page changes
   - `browser_bulk_extract`: Bulk data extraction

5. **Session Management Tools**
   - `browser_set_cookies`: Manage cookies
   - `browser_get_cookies`: Retrieve cookies
   - `browser_set_local_storage`: Manage local storage

#### Tool Usage Examples

```python
# MCP tool definitions
tools = [
    {
        "name": "browser_navigate",
        "description": "Navigate to a URL in the browser",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "wait_until": {"type": "string", "enum": ["load", "domcontentloaded", "networkidle"]}
            },
            "required": ["url"]
        }
    },
    {
        "name": "browser_search_and_extract",
        "description": "Search for elements and extract data",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "search_selectors": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "links": {"type": "string"},
                        "images": {"type": "string"}
                    }
                }
            },
            "required": ["url", "search_selectors"]
        }
    }
]
```

## Integration with OpenManus

### MCP Server Integration

```python
from app.mcp import EnhancedMcpServer
from app.browser import BrowserMCPTools
from app.llm import OpenRouterClient, ModelManager

# Create enhanced MCP server
server = EnhancedMcpServer()

# Add browser tools
browser_tools = BrowserMCPTools()
browser_tool_definitions = await browser_tools.get_tool_definitions()

for tool_def in browser_tool_definitions:
    server.register_tool(
        name=tool_def["name"],
        description=tool_def["description"],
        handler=getattr(browser_tools, tool_def["name"].replace("browser_", "")),
        schema=tool_def["inputSchema"]
    )

# Add LLM integration
client = OpenRouterClient(config)
manager = ModelManager(client)

# Start server
await server.start()
```

### Flow Integration

```python
from app.flow import EnhancedFlow, create_flow_step

# Create flow with browser and LLM integration
flow = EnhancedFlow(mcp_client=mcp_client)

# Add browser automation steps
browser_step = create_flow_step(
    name="Extract Website Data",
    step_type="mcp_tool",
    parameters={
        "tool_name": "browser_search_and_extract",
        "arguments": {
            "url": "https://example.com",
            "search_selectors": {
                "text": "h1, h2, p",
                "links": "a[href]",
                "images": "img[src]"
            }
        }
    }
)

# Add LLM analysis step
analysis_step = create_flow_step(
    name="Analyze Extracted Data",
    step_type="llm_completion",
    parameters={
        "task_type": "data_analysis",
        "prompt": "Analyze the extracted website data and provide insights",
        "max_tokens": 1000
    },
    dependencies=[browser_step.id]
)

# Execute flow
plan = FlowPlan(steps=[browser_step, analysis_step])
result = await flow.execute(plan)
```

### Agent Integration

```python
from app.agents import BaseAgent

class WebAnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.browser_tools = BrowserMCPTools()
        self.llm_client, self.model_manager = create_production_llm_setup(api_key)
    
    async def analyze_website(self, url: str) -> dict:
        # Extract website content
        extraction_result = await self.browser_tools.search_and_extract({
            "url": url,
            "search_selectors": {
                "text": "h1, h2, h3, p",
                "links": "a[href]",
                "images": "img[src]"
            },
            "take_screenshot": True
        })
        
        # Analyze with LLM
        criteria = create_task_criteria(TaskType.DATA_ANALYSIS)
        messages = [
            ChatMessage(role="system", content="You are a web content analyst."),
            ChatMessage(role="user", content=f"Analyze this website data: {extraction_result}")
        ]
        
        analysis = await self.model_manager.execute_with_best_model(messages, criteria)
        
        return {
            "url": url,
            "extraction": extraction_result,
            "analysis": analysis.choices[0]["message"]["content"]
        }
```

## Performance Optimization

### Browser Performance

#### Resource Blocking
```python
# Block unnecessary resources for faster loading
await browser.block_resources(["image", "stylesheet", "font", "media"])
```

#### Request Interception
```python
# Intercept and modify requests
async def request_handler(route):
    if "analytics" in route.request.url:
        await route.abort()  # Block analytics
    else:
        await route.continue_()

await browser.intercept_requests("**/*", request_handler)
```

#### Parallel Processing
```python
# Process multiple pages in parallel
import asyncio

async def process_page(url):
    browser = create_scraping_browser()
    async with browser:
        result = await browser.navigate(url)
        return await browser.extract_text()

# Process multiple URLs concurrently
urls = ["https://page1.com", "https://page2.com", "https://page3.com"]
results = await asyncio.gather(*[process_page(url) for url in urls])
```

### LLM Performance

#### Model Caching
```python
# Cache model information
models = await client.get_available_models()
# Models are automatically cached for 1 hour

# Cache recommendations
recommendation = await manager.get_model_recommendation(criteria)
# Recommendations are cached for 30 minutes
```

#### Batch Processing
```python
# Process multiple requests efficiently
requests = [
    ChatCompletionRequest(model="openai/gpt-4o-mini", messages=messages1),
    ChatCompletionRequest(model="openai/gpt-4o-mini", messages=messages2),
    ChatCompletionRequest(model="openai/gpt-4o-mini", messages=messages3)
]

# Execute in parallel with rate limiting
results = []
for request in requests:
    result = await client.chat_completion(request)
    results.append(result)
    await asyncio.sleep(0.1)  # Rate limiting
```

#### Streaming for Real-time Updates
```python
# Use streaming for real-time responses
async for chunk in client.chat_completion_stream(request):
    if chunk.get("choices"):
        content = chunk["choices"][0].get("delta", {}).get("content", "")
        if content:
            print(content, end="", flush=True)
```

## Security Considerations

### Browser Security

#### Sandboxing
```python
# Run browser in sandboxed environment
config = BrowserConfig(
    extra_args=[
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--disable-web-security"
    ]
)
```

#### Request Filtering
```python
# Filter malicious requests
async def security_handler(route):
    url = route.request.url
    if any(domain in url for domain in BLOCKED_DOMAINS):
        await route.abort()
    else:
        await route.continue_()

await browser.intercept_requests("**/*", security_handler)
```

#### Data Sanitization
```python
# Sanitize extracted data
import html
import re

def sanitize_text(text: str) -> str:
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Decode HTML entities
    text = html.unescape(text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

extracted_text = sanitize_text(await browser.extract_text())
```

### LLM Security

#### Input Validation
```python
def validate_prompt(prompt: str) -> bool:
    # Check for injection attempts
    dangerous_patterns = [
        r"ignore previous instructions",
        r"system prompt",
        r"jailbreak",
        r"developer mode"
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, prompt, re.IGNORECASE):
            return False
    
    return True

# Validate before sending to LLM
if validate_prompt(user_input):
    response = await client.chat_completion(request)
```

#### Output Filtering
```python
def filter_response(response: str) -> str:
    # Remove sensitive information patterns
    sensitive_patterns = [
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
        r'\b\d{16}\b',             # Credit card
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email
    ]
    
    for pattern in sensitive_patterns:
        response = re.sub(pattern, '[REDACTED]', response)
    
    return response

filtered_response = filter_response(response.choices[0]["message"]["content"])
```

## Monitoring and Logging

### Browser Monitoring

```python
# Enable comprehensive logging
browser.logger.setLevel(logging.DEBUG)

# Monitor requests and responses
request_log = browser.get_request_log()
response_log = browser.get_response_log()

# Track performance metrics
page_info = await browser.get_page_info()
print(f"Page load time: {page_info.load_time}s")
print(f"Content size: {len(page_info.content)} bytes")
```

### LLM Monitoring

```python
# Track usage statistics
stats = client.get_usage_stats()
print(f"Total requests: {stats['total_requests']}")
print(f"Total cost: ${stats['total_cost']:.2f}")
print(f"Average latency: {stats['average_latency']:.2f}s")

# Monitor model performance
perf_stats = await manager.get_performance_stats()
for model_id, stats in perf_stats.items():
    print(f"{model_id}: {stats['success_rate']:.2%} success rate")
```

### Error Handling

```python
# Comprehensive error handling
try:
    result = await browser.navigate(url)
    if not result.success:
        logger.error(f"Navigation failed: {result.error}")
        # Implement retry logic
        
except Exception as e:
    logger.error(f"Browser error: {e}")
    # Implement recovery logic
    await browser.close()
    browser = create_scraping_browser()
    await browser.start()

# LLM error handling with fallback
try:
    response = await manager.execute_with_best_model(messages, criteria)
except Exception as e:
    logger.error(f"LLM error: {e}")
    # Try with budget model as fallback
    budget_criteria = create_budget_criteria(criteria.task_type)
    response = await manager.execute_with_best_model(messages, budget_criteria)
```

## Deployment Considerations

### Docker Integration

The browser and LLM components are fully integrated with the Docker setup:

```dockerfile
# Install Playwright browsers
RUN pip install playwright && playwright install --with-deps chromium

# Set environment variables
ENV OPENROUTER_API_KEY=""
ENV BROWSER_HEADLESS=true
ENV BROWSER_TIMEOUT=30000
```

### Environment Configuration

```bash
# .env file
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_SITE_URL=https://your-site.com
OPENROUTER_SITE_NAME=Your App Name

BROWSER_HEADLESS=true
BROWSER_TYPE=chromium
BROWSER_TIMEOUT=30000
BROWSER_VIEWPORT_WIDTH=1920
BROWSER_VIEWPORT_HEIGHT=1080

# Cost and rate limiting
LLM_COST_LIMIT=1.0
LLM_RATE_LIMIT=60
LLM_DEFAULT_MODEL=openai/gpt-4o
```

### Production Deployment

```python
# Production configuration
production_config = {
    "llm": {
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "cost_limit": float(os.getenv("LLM_COST_LIMIT", "1.0")),
        "rate_limit": int(os.getenv("LLM_RATE_LIMIT", "60")),
        "enable_fallback": True,
        "performance_tracking": True
    },
    "browser": {
        "headless": os.getenv("BROWSER_HEADLESS", "true").lower() == "true",
        "timeout": int(os.getenv("BROWSER_TIMEOUT", "30000")),
        "viewport_width": int(os.getenv("BROWSER_VIEWPORT_WIDTH", "1920")),
        "viewport_height": int(os.getenv("BROWSER_VIEWPORT_HEIGHT", "1080"))
    }
}
```

This comprehensive integration provides OpenManus with powerful LLM capabilities through OpenRouter and sophisticated web automation through headless browsers, enabling AI agents to interact with both language models and web content seamlessly.

