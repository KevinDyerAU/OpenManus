# Chat and API Interface Development Guide

## Overview

This guide covers the implementation of comprehensive chat and API interfaces for OpenManus, providing both web-based chat functionality and programmatic API access for AI agent interactions.

## Architecture Overview

The chat and API interface system consists of three main components:

1. **FastAPI REST API** (`app/api/main.py`)
   - RESTful endpoints for all OpenManus functionality
   - WebSocket support for real-time chat
   - Authentication and authorization
   - Rate limiting and security measures

2. **React Chat Interface** (`openmanus-chat/`)
   - Modern web-based chat interface
   - Real-time messaging with WebSocket support
   - Model selection and task type configuration
   - Conversation management and history

3. **Authentication System** (`app/api/auth.py`)
   - JWT-based authentication
   - Role-based access control
   - Rate limiting and security middleware

## FastAPI REST API

### Core Features

#### Chat Endpoints
- **POST /chat**: Standard chat completion
- **POST /chat/stream**: Streaming chat completion
- **WebSocket /ws/chat**: Real-time chat with WebSocket

#### Browser Automation
- **POST /browser**: Execute browser automation actions
- **GET /browser/sessions**: Manage browser sessions

#### Workflow Execution
- **POST /flows**: Execute workflows
- **GET /flows/{flow_id}**: Get workflow status

#### Conversation Management
- **GET /conversations**: List conversations
- **GET /conversations/{id}**: Get conversation details
- **DELETE /conversations/{id}**: Delete conversation

#### System Endpoints
- **GET /**: API information
- **GET /health**: Health check
- **GET /models**: Available models
- **GET /stats**: Usage statistics

### API Configuration

```python
from app.api import APIConfig

config = APIConfig()
# Configuration loaded from environment variables:
# - API_HOST: Server host (default: 0.0.0.0)
# - API_PORT: Server port (default: 8000)
# - API_DEBUG: Debug mode (default: false)
# - OPENROUTER_API_KEY: OpenRouter API key
# - JWT_SECRET: JWT secret key
# - CORS_ORIGINS: Allowed CORS origins
# - RATE_LIMIT: Requests per minute (default: 60)
```

### Starting the API Server

```bash
# Development
cd /path/to/openmanus
python -m app.api.main

# Production with uvicorn
uvicorn app.api.main:app --host 0.0.0.0 --port 8000

# With environment variables
OPENROUTER_API_KEY=your_key uvicorn app.api.main:app
```

### API Usage Examples

#### Chat Completion

```python
import requests

# Standard chat
response = requests.post("http://localhost:8000/chat", json={
    "message": "Hello, how can you help me?",
    "model": "openai/gpt-4o",
    "task_type": "general_chat",
    "max_tokens": 1000,
    "temperature": 0.7
})

data = response.json()
print(f"Response: {data['response']}")
print(f"Model: {data['model_used']}")
print(f"Cost: ${data['cost']:.4f}")
```

#### Streaming Chat

```python
import requests

response = requests.post("http://localhost:8000/chat/stream", json={
    "message": "Write a Python function to calculate fibonacci numbers",
    "task_type": "code_generation",
    "stream": True
}, stream=True)

for line in response.iter_lines():
    if line:
        data = json.loads(line.decode().replace("data: ", ""))
        if data.get("type") == "content":
            print(data["content"], end="", flush=True)
```

#### WebSocket Chat

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/chat');

ws.onopen = () => {
    ws.send(JSON.stringify({
        type: 'chat',
        message: 'Hello from WebSocket!',
        task_type: 'general_chat'
    }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'content') {
        console.log(data.content);
    }
};
```

#### Browser Automation

```python
# Navigate to a webpage
response = requests.post("http://localhost:8000/browser", json={
    "action": "navigate",
    "parameters": {
        "url": "https://example.com",
        "wait_until": "load"
    }
})

# Extract content
response = requests.post("http://localhost:8000/browser", json={
    "action": "search_and_extract",
    "parameters": {
        "url": "https://news.example.com",
        "search_selectors": {
            "text": "h1, h2, p",
            "links": "a[href]",
            "images": "img[src]"
        },
        "take_screenshot": True
    }
})
```

#### Workflow Execution

```python
# Start a workflow
response = requests.post("http://localhost:8000/flows", json={
    "flow_name": "web_analysis",
    "parameters": {
        "url": "https://example.com",
        "analysis_type": "content_summary"
    }
})

flow_id = response.json()["flow_id"]

# Check status
status_response = requests.get(f"http://localhost:8000/flows/{flow_id}")
print(status_response.json())
```

### Authentication

#### User Registration

```python
response = requests.post("http://localhost:8000/auth/register", json={
    "username": "newuser",
    "email": "user@example.com",
    "password": "securepassword123"
})
```

#### User Login

```python
response = requests.post("http://localhost:8000/auth/login", json={
    "username": "newuser",
    "password": "securepassword123"
})

tokens = response.json()
access_token = tokens["access_token"]

# Use token in subsequent requests
headers = {"Authorization": f"Bearer {access_token}"}
response = requests.get("http://localhost:8000/protected-endpoint", headers=headers)
```

### Rate Limiting and Security

The API includes comprehensive security measures:

- **Rate Limiting**: 60 requests per minute per IP
- **CORS Protection**: Configurable allowed origins
- **JWT Authentication**: Secure token-based auth
- **Input Validation**: Pydantic model validation
- **Security Headers**: XSS, CSRF, and other protections

## React Chat Interface

### Features

#### Modern Chat Interface
- Real-time messaging with WebSocket support
- Message history and conversation management
- Typing indicators and loading states
- Dark/light theme support

#### Model Selection
- Automatic model selection based on task type
- Manual model selection from 400+ available models
- Model performance and cost information
- Provider-specific model grouping

#### Task Type Configuration
- Predefined task types (chat, coding, analysis, etc.)
- Task-specific model recommendations
- Custom task type support

#### Advanced Features
- Streaming response support
- Conversation export/import
- Usage statistics and cost tracking
- Browser automation interface (coming soon)
- Workflow execution interface (coming soon)

### Setup and Configuration

```bash
# Navigate to chat interface
cd openmanus-chat

# Install dependencies (already done by manus-create-react-app)
pnpm install

# Create environment file
cp .env.example .env

# Configure API endpoints
echo "VITE_API_URL=http://localhost:8000" > .env
echo "VITE_WS_URL=ws://localhost:8000/ws/chat" >> .env

# Start development server
pnpm run dev --host
```

### Environment Configuration

```bash
# .env file
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000/ws/chat
VITE_DEBUG=true
VITE_ENABLE_BROWSER_TOOLS=true
VITE_ENABLE_FLOW_EXECUTION=true
VITE_DEFAULT_THEME=light
VITE_MAX_MESSAGE_LENGTH=4000
```

### Chat Interface Components

#### Main Chat Component (`App.jsx`)
- Message display and input
- Model and task type selection
- Real-time WebSocket communication
- Conversation management

#### Key Features:
- **Message Types**: User, assistant, and system messages
- **Streaming Support**: Real-time response streaming
- **Model Selection**: Dropdown with 400+ models
- **Task Types**: Predefined task categories
- **Dark Mode**: Toggle between light and dark themes
- **Export/Import**: JSON conversation export
- **Statistics**: Token usage and cost tracking

#### UI Components Used:
- **shadcn/ui**: Modern, accessible UI components
- **Tailwind CSS**: Utility-first CSS framework
- **Lucide Icons**: Beautiful icon library
- **Framer Motion**: Smooth animations (available)

### Customization

#### Adding New Task Types

```javascript
const TASK_TYPES = [
    { id: 'custom_task', name: 'Custom Task', icon: CustomIcon },
    // ... existing task types
];
```

#### Custom Model Providers

```javascript
const MODELS = [
    { id: 'custom/model', name: 'Custom Model', provider: 'Custom' },
    // ... existing models
];
```

#### Theme Customization

The interface uses CSS custom properties for theming:

```css
:root {
    --primary: oklch(0.205 0 0);
    --primary-foreground: oklch(0.985 0 0);
    /* ... other theme variables */
}

.dark {
    --primary: oklch(0.922 0 0);
    --primary-foreground: oklch(0.205 0 0);
    /* ... dark theme variables */
}
```

## Integration Examples

### Full-Stack Chat Application

```python
# Backend: Start API server
from app.api.main import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

```bash
# Frontend: Start React development server
cd openmanus-chat
pnpm run dev --host
```

### Programmatic API Usage

```python
import asyncio
import aiohttp
import json

class OpenManusClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def chat(self, message, model=None, task_type="general_chat"):
        async with self.session.post(
            f"{self.base_url}/chat",
            json={
                "message": message,
                "model": model,
                "task_type": task_type
            }
        ) as response:
            return await response.json()
    
    async def chat_stream(self, message, model=None, task_type="general_chat"):
        async with self.session.post(
            f"{self.base_url}/chat/stream",
            json={
                "message": message,
                "model": model,
                "task_type": task_type,
                "stream": True
            }
        ) as response:
            async for line in response.content:
                if line:
                    data = json.loads(line.decode().replace(b"data: ", b""))
                    yield data
    
    async def browser_action(self, action, parameters):
        async with self.session.post(
            f"{self.base_url}/browser",
            json={
                "action": action,
                "parameters": parameters
            }
        ) as response:
            return await response.json()

# Usage example
async def main():
    async with OpenManusClient() as client:
        # Simple chat
        response = await client.chat("Hello, how are you?")
        print(response["response"])
        
        # Streaming chat
        async for chunk in client.chat_stream("Write a Python function"):
            if chunk.get("type") == "content":
                print(chunk["content"], end="", flush=True)
        
        # Browser automation
        result = await client.browser_action("navigate", {
            "url": "https://example.com"
        })
        print(result)

asyncio.run(main())
```

### WebSocket Integration

```python
import asyncio
import websockets
import json

async def websocket_chat():
    uri = "ws://localhost:8000/ws/chat"
    
    async with websockets.connect(uri) as websocket:
        # Send message
        await websocket.send(json.dumps({
            "type": "chat",
            "message": "Hello from Python WebSocket!",
            "task_type": "general_chat"
        }))
        
        # Receive responses
        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)
                
                if data["type"] == "content":
                    print(data["content"], end="", flush=True)
                elif data["type"] == "done":
                    print(f"\nConversation ID: {data['conversation_id']}")
                    break
                elif data["type"] == "error":
                    print(f"Error: {data['error']}")
                    break
                    
            except websockets.exceptions.ConnectionClosed:
                break

asyncio.run(websocket_chat())
```

## Deployment

### Docker Deployment

The chat and API interfaces are fully integrated with the Docker setup:

```dockerfile
# API Server
EXPOSE 8000
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# React Chat Interface (built and served by API)
COPY openmanus-chat/dist /app/static
```

### Environment Variables

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false

# OpenRouter Integration
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_SITE_URL=https://your-site.com
OPENROUTER_SITE_NAME=Your App Name

# Authentication
JWT_SECRET=your-jwt-secret-key
ACCESS_TOKEN_EXPIRE_MINUTES=30
ENABLE_REGISTRATION=true

# Security
CORS_ORIGINS=https://your-frontend.com,http://localhost:3000
RATE_LIMIT=60
MAX_CONNECTIONS=100

# Features
ENABLE_BROWSER_TOOLS=true
ENABLE_FLOW_EXECUTION=true
```

### Production Deployment

```bash
# Build React interface
cd openmanus-chat
pnpm run build

# Copy built files to API static directory
cp -r dist/* ../app/static/

# Start production API server
cd ..
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Nginx Configuration

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    # API endpoints
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
    
    # WebSocket support
    location /ws/ {
        proxy_pass http://localhost:8000/ws/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
    
    # Static files (React app)
    location / {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
    }
}
```

## Testing

### API Testing

```python
import pytest
import asyncio
from httpx import AsyncClient
from app.api.main import app

@pytest.mark.asyncio
async def test_chat_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/chat", json={
            "message": "Hello, test!",
            "task_type": "general_chat"
        })
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "conversation_id" in data

@pytest.mark.asyncio
async def test_health_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

@pytest.mark.asyncio
async def test_browser_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/browser", json={
            "action": "navigate",
            "parameters": {"url": "https://example.com"}
        })
        assert response.status_code == 200
```

### Frontend Testing

```bash
# Run React tests
cd openmanus-chat
pnpm test

# Run E2E tests with Playwright
pnpm test:e2e
```

### Load Testing

```python
import asyncio
import aiohttp
import time

async def load_test():
    async with aiohttp.ClientSession() as session:
        tasks = []
        
        for i in range(100):  # 100 concurrent requests
            task = session.post("http://localhost:8000/chat", json={
                "message": f"Test message {i}",
                "task_type": "general_chat"
            })
            tasks.append(task)
        
        start_time = time.time()
        responses = await asyncio.gather(*tasks)
        end_time = time.time()
        
        print(f"Completed 100 requests in {end_time - start_time:.2f} seconds")
        print(f"Success rate: {sum(1 for r in responses if r.status == 200)/len(responses)*100:.1f}%")

asyncio.run(load_test())
```

## Monitoring and Analytics

### Usage Statistics

The API provides comprehensive usage statistics:

```python
# Get usage stats
response = requests.get("http://localhost:8000/stats")
stats = response.json()

print(f"Total conversations: {stats['conversations']['total']}")
print(f"Active flows: {stats['flows']['running']}")
print(f"WebSocket connections: {stats['websockets']['active_connections']}")
```

### Health Monitoring

```python
# Health check with component status
response = requests.get("http://localhost:8000/health")
health = response.json()

for component, status in health["components"].items():
    print(f"{component}: {status}")
```

### Error Handling

The API includes comprehensive error handling:

- **400 Bad Request**: Invalid input data
- **401 Unauthorized**: Authentication required
- **403 Forbidden**: Insufficient permissions
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Server error
- **503 Service Unavailable**: Service temporarily unavailable

## Security Considerations

### Authentication Security
- JWT tokens with configurable expiration
- Secure password hashing with bcrypt
- Role-based access control
- Rate limiting per user and IP

### API Security
- Input validation with Pydantic
- CORS protection
- Security headers (XSS, CSRF protection)
- Request size limits
- SQL injection prevention

### WebSocket Security
- Authentication required for WebSocket connections
- Message validation and sanitization
- Connection limits and timeouts
- Rate limiting for WebSocket messages

### Best Practices
- Use HTTPS in production
- Rotate JWT secrets regularly
- Monitor for suspicious activity
- Implement proper logging
- Use environment variables for secrets
- Regular security updates

This comprehensive chat and API interface system provides a robust foundation for human interactions with the OpenManus AI agent platform, supporting both web-based chat and programmatic API access with enterprise-grade security and scalability features.

