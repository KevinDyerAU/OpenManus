# OpenManus Callback Integration Guide

## Overview

The OpenManus platform now includes comprehensive callback support for real-time updates during AI processing and workflow execution. This enhancement provides developers and users with immediate feedback on AI operations, enabling better user experiences and more responsive applications.

## Table of Contents

1. [Introduction](#introduction)
2. [Callback Architecture](#callback-architecture)
3. [API Integration](#api-integration)
4. [Chat Interface Integration](#chat-interface-integration)
5. [Callback Event Types](#callback-event-types)
6. [Delivery Methods](#delivery-methods)
7. [Configuration Options](#configuration-options)
8. [Implementation Examples](#implementation-examples)
9. [Testing and Validation](#testing-and-validation)
10. [Troubleshooting](#troubleshooting)
11. [Best Practices](#best-practices)
12. [Performance Considerations](#performance-considerations)

## Introduction

Real-time callbacks are essential for modern AI applications where users expect immediate feedback during processing. The OpenManus callback system provides multiple delivery methods and comprehensive event types to support various use cases, from simple progress updates to detailed AI reasoning insights.

The callback system is designed with flexibility and reliability in mind, supporting webhook delivery, WebSocket connections, Server-Sent Events (SSE), and polling mechanisms. This ensures compatibility with different application architectures and deployment scenarios.

## Callback Architecture

### Core Components

The callback system consists of several key components working together to provide reliable real-time updates:

**Callback Manager**: The central orchestrator that handles session management, event routing, and delivery coordination. It maintains active sessions, manages delivery configurations, and provides comprehensive statistics and monitoring capabilities.

**Event System**: A type-safe event framework that defines standardized event types and data structures. Events are categorized by type and include rich metadata for context and debugging.

**Delivery Engine**: Multiple delivery mechanisms including HTTP webhooks, WebSocket connections, Server-Sent Events, and polling endpoints. Each delivery method is optimized for specific use cases and network conditions.

**Session Management**: Secure session handling with unique identifiers, configuration persistence, and automatic cleanup. Sessions can be created, configured, monitored, and destroyed through dedicated API endpoints.

### Event Flow

The typical event flow follows this pattern:

1. **Session Creation**: A callback session is created with specific configuration including delivery method, event types, and endpoint details
2. **Event Generation**: During AI processing, various components emit events at key points in the workflow
3. **Event Processing**: The callback manager receives events, validates them against session configuration, and queues them for delivery
4. **Event Delivery**: Events are delivered using the configured method with retry logic and error handling
5. **Monitoring**: Delivery statistics and session health are continuously monitored and made available through API endpoints

## API Integration

### Callback Configuration in Chat Requests

The enhanced chat API now accepts an optional `callback_config` parameter that enables real-time updates during processing:

```json
{
  "message": "Analyze this data and provide insights",
  "conversation_id": "conv_123",
  "model": "openai/gpt-4o",
  "task_type": "data_analysis",
  "callback_config": {
    "delivery_method": "webhook",
    "webhook_url": "https://your-app.com/callbacks",
    "events": ["thinking", "tool_use", "progress", "completion"],
    "include_intermediate_results": true,
    "timeout": 30,
    "retry_attempts": 3,
    "headers": {
      "Authorization": "Bearer your-token",
      "X-Source": "OpenManus-Chat"
    }
  }
}
```

### Session Management Endpoints

The callback system provides dedicated endpoints for session management:

**Create Session**: `POST /callbacks/sessions`
```json
{
  "delivery_method": "websocket",
  "events": ["thinking", "tool_use", "progress", "completion"],
  "include_intermediate_results": true,
  "timeout": 30
}
```

**Get Session Stats**: `GET /callbacks/sessions/{session_id}/stats`
```json
{
  "session_id": "sess_abc123",
  "events_sent": 45,
  "events_delivered": 43,
  "events_failed": 2,
  "retries": 3,
  "queued_events": 0,
  "has_websocket": true,
  "has_sse": false,
  "delivery_rate": 0.956
}
```

**Delete Session**: `DELETE /callbacks/sessions/{session_id}`

### WebSocket Integration

For real-time applications, WebSocket connections provide the lowest latency delivery method:

```javascript
const ws = new WebSocket('ws://localhost:8000/callbacks/sessions/sess_abc123/ws');

ws.onopen = () => {
  console.log('Callback WebSocket connected');
};

ws.onmessage = (event) => {
  const callback = JSON.parse(event.data);
  handleCallbackEvent(callback);
};

function handleCallbackEvent(callback) {
  switch (callback.event_type) {
    case 'thinking':
      updateThinkingIndicator(callback.data.thought);
      break;
    case 'tool_use':
      showToolExecution(callback.data.tool_name, callback.data.parameters);
      break;
    case 'progress':
      updateProgressBar(callback.data.progress, callback.data.message);
      break;
    case 'completion':
      handleCompletion(callback.data.result);
      break;
  }
}
```

## Chat Interface Integration

### Callback Configuration Panel

The React chat interface includes a comprehensive callback configuration panel accessible through the "Callbacks" tab. This panel provides:

**Delivery Method Selection**: Choose between WebSocket, Webhook, SSE, or Polling delivery methods with descriptions and use case guidance.

**Event Type Configuration**: Visual selection of callback events with descriptions and icons. Users can enable or disable specific event types based on their needs.

**Advanced Options**: Configuration of timeout values, intermediate result inclusion, and retry behavior.

**Session Management**: Create, test, and delete callback sessions with real-time status monitoring.

**Event Monitoring**: Live display of received callback events with filtering and search capabilities.

### Real-time Event Display

The callback panel includes a real-time event viewer that displays:

- Event type with appropriate icons
- Timestamp and session information
- Event data in formatted JSON
- Filtering by event type
- Event history with pagination

### Integration with Chat Flow

When callbacks are enabled, the chat interface automatically:

1. Creates a callback session before sending messages
2. Displays real-time updates during AI processing
3. Shows thinking indicators, tool usage, and progress updates
4. Provides visual feedback for completion and errors
5. Cleans up sessions after completion

## Callback Event Types

### Core Event Types

**THINKING**: Emitted when the AI is reasoning or making decisions
```json
{
  "event_type": "thinking",
  "data": {
    "thought": "I need to analyze the data structure first..."
  }
}
```

**TOOL_USE**: Emitted when tools are being executed
```json
{
  "event_type": "tool_use",
  "data": {
    "tool_name": "web_search",
    "parameters": {
      "query": "latest market trends",
      "limit": 10
    }
  }
}
```

**TOOL_RESULT**: Emitted when tool execution completes
```json
{
  "event_type": "tool_result",
  "data": {
    "tool_name": "web_search",
    "result": {...},
    "success": true,
    "execution_time": 1.23
  }
}
```

**PROGRESS**: Emitted for task progress updates
```json
{
  "event_type": "progress",
  "data": {
    "progress": 0.65,
    "message": "Processing data analysis...",
    "current_step": "feature_extraction",
    "total_steps": 5
  }
}
```

**COMPLETION**: Emitted when tasks complete
```json
{
  "event_type": "completion",
  "data": {
    "result": {...},
    "success": true,
    "total_time": 15.67,
    "tokens_used": 1250
  }
}
```

### Workflow Event Types

**WORKFLOW_START**: Emitted when workflows begin
**WORKFLOW_STEP**: Emitted for each workflow step
**WORKFLOW_COMPLETE**: Emitted when workflows finish

### Model Event Types

**MODEL_SELECTION**: Emitted during model selection process
**STREAMING_CHUNK**: Emitted for streaming response chunks

### Error Event Types

**ERROR**: Emitted when errors occur during processing
```json
{
  "event_type": "error",
  "data": {
    "error": "Connection timeout",
    "error_type": "network_error",
    "recoverable": true,
    "retry_count": 2
  }
}
```

## Delivery Methods

### Webhook Delivery

HTTP POST requests to configured webhook URLs provide reliable delivery with retry logic:

**Advantages**:
- Reliable delivery with retry mechanisms
- Works with any HTTP-capable application
- Supports authentication headers
- Detailed delivery statistics

**Configuration**:
```json
{
  "delivery_method": "webhook",
  "webhook_url": "https://your-app.com/callbacks",
  "timeout": 30,
  "retry_attempts": 3,
  "headers": {
    "Authorization": "Bearer token",
    "Content-Type": "application/json"
  }
}
```

### WebSocket Delivery

Real-time bidirectional communication for low-latency applications:

**Advantages**:
- Lowest latency delivery
- Bidirectional communication
- Connection status monitoring
- Automatic reconnection

**Usage**:
```javascript
const ws = new WebSocket('ws://localhost:8000/callbacks/sessions/{session_id}/ws');
ws.onmessage = (event) => {
  const callback = JSON.parse(event.data);
  // Handle callback event
};
```

### Server-Sent Events (SSE)

Unidirectional streaming for web applications:

**Advantages**:
- Native browser support
- Automatic reconnection
- Simple implementation
- HTTP-based protocol

**Usage**:
```javascript
const eventSource = new EventSource('/callbacks/sessions/{session_id}/sse');
eventSource.onmessage = (event) => {
  const callback = JSON.parse(event.data);
  // Handle callback event
};
```

### Polling Delivery

Request-based delivery for applications that cannot maintain persistent connections:

**Advantages**:
- Works with any HTTP client
- No persistent connections required
- Simple implementation
- Firewall-friendly

**Usage**:
```javascript
async function pollEvents() {
  const response = await fetch('/callbacks/sessions/{session_id}/events?since=' + lastTimestamp);
  const data = await response.json();
  data.events.forEach(handleCallbackEvent);
}

setInterval(pollEvents, 5000); // Poll every 5 seconds
```

## Configuration Options

### Session Configuration

**delivery_method**: The delivery mechanism (webhook, websocket, sse, polling)
**webhook_url**: Target URL for webhook delivery (required for webhook method)
**events**: Array of event types to receive
**include_intermediate_results**: Whether to include partial results
**timeout**: Request timeout in seconds
**retry_attempts**: Number of retry attempts for failed deliveries
**headers**: Custom headers for webhook requests

### Event Filtering

Events can be filtered by type to reduce noise and focus on relevant updates:

```json
{
  "events": [
    "thinking",
    "tool_use", 
    "progress",
    "completion"
  ]
}
```

### Advanced Options

**Circuit Breaker**: Automatic failure detection and recovery
**Rate Limiting**: Prevent callback flooding
**Batching**: Group multiple events for efficiency
**Compression**: Reduce bandwidth usage for large payloads

## Implementation Examples

### Basic Webhook Integration

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/callbacks', methods=['POST'])
def handle_callback():
    callback = request.json
    
    event_type = callback['event_type']
    data = callback['data']
    
    if event_type == 'thinking':
        print(f"AI is thinking: {data['thought']}")
    elif event_type == 'progress':
        print(f"Progress: {data['progress']*100:.1f}% - {data['message']}")
    elif event_type == 'completion':
        print(f"Task completed: {data['success']}")
    
    return jsonify({'status': 'received'})

if __name__ == '__main__':
    app.run(port=5000)
```

### React Component Integration

```jsx
import { useState, useEffect } from 'react';

function CallbackDisplay({ sessionId }) {
  const [events, setEvents] = useState([]);
  const [ws, setWs] = useState(null);

  useEffect(() => {
    const websocket = new WebSocket(`ws://localhost:8000/callbacks/sessions/${sessionId}/ws`);
    
    websocket.onmessage = (event) => {
      const callback = JSON.parse(event.data);
      setEvents(prev => [...prev, callback]);
    };
    
    setWs(websocket);
    
    return () => websocket.close();
  }, [sessionId]);

  return (
    <div className="callback-display">
      {events.map((event, index) => (
        <div key={index} className={`event event-${event.event_type}`}>
          <span className="event-type">{event.event_type}</span>
          <span className="event-time">{new Date(event.timestamp).toLocaleTimeString()}</span>
          <pre className="event-data">{JSON.stringify(event.data, null, 2)}</pre>
        </div>
      ))}
    </div>
  );
}
```

### Node.js Integration

```javascript
const express = require('express');
const WebSocket = require('ws');

const app = express();
app.use(express.json());

// Webhook endpoint
app.post('/callbacks', (req, res) => {
  const callback = req.body;
  console.log('Received callback:', callback.event_type);
  
  // Process callback based on event type
  switch (callback.event_type) {
    case 'thinking':
      handleThinking(callback.data);
      break;
    case 'progress':
      handleProgress(callback.data);
      break;
    case 'completion':
      handleCompletion(callback.data);
      break;
  }
  
  res.json({ status: 'received' });
});

// WebSocket client
function connectWebSocket(sessionId) {
  const ws = new WebSocket(`ws://localhost:8000/callbacks/sessions/${sessionId}/ws`);
  
  ws.on('message', (data) => {
    const callback = JSON.parse(data);
    console.log('WebSocket callback:', callback);
  });
  
  return ws;
}

app.listen(3000, () => {
  console.log('Callback server running on port 3000');
});
```

## Testing and Validation

### Test Endpoint

The callback system includes a test endpoint for validating delivery:

```bash
curl -X POST "http://localhost:8000/callbacks/sessions/{session_id}/test" \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "progress",
    "test_data": {
      "message": "Test callback",
      "progress": 0.5
    }
  }'
```

### Validation Checklist

**Session Creation**:
- [ ] Session created successfully with valid configuration
- [ ] Session ID returned and stored
- [ ] Configuration parameters validated

**Event Delivery**:
- [ ] Events delivered to correct endpoint
- [ ] Event format matches specification
- [ ] Timestamps are accurate
- [ ] Metadata is included

**Error Handling**:
- [ ] Failed deliveries trigger retries
- [ ] Circuit breaker activates on repeated failures
- [ ] Error events are generated for failures
- [ ] Session cleanup occurs on errors

**Performance**:
- [ ] Delivery latency is acceptable
- [ ] Memory usage is stable
- [ ] No event loss under load
- [ ] Statistics are accurate

### Load Testing

```python
import asyncio
import aiohttp
import time

async def test_callback_load():
    async with aiohttp.ClientSession() as session:
        # Create callback session
        async with session.post('http://localhost:8000/callbacks/sessions', json={
            'delivery_method': 'webhook',
            'webhook_url': 'http://localhost:5000/callbacks',
            'events': ['progress', 'completion']
        }) as resp:
            session_data = await resp.json()
            session_id = session_data['session_id']
        
        # Send multiple test events
        tasks = []
        for i in range(100):
            task = session.post(f'http://localhost:8000/callbacks/sessions/{session_id}/test', json={
                'event_type': 'progress',
                'test_data': {'progress': i/100, 'message': f'Test {i}'}
            })
            tasks.append(task)
        
        start_time = time.time()
        await asyncio.gather(*tasks)
        end_time = time.time()
        
        print(f"Sent 100 events in {end_time - start_time:.2f} seconds")

asyncio.run(test_callback_load())
```

## Troubleshooting

### Common Issues

**Webhook Delivery Failures**:
- Verify webhook URL is accessible
- Check authentication headers
- Validate SSL certificates
- Review firewall settings

**WebSocket Connection Issues**:
- Confirm WebSocket support in client
- Check for proxy interference
- Verify session ID is valid
- Monitor connection lifecycle

**Event Loss**:
- Check delivery method reliability
- Verify event filtering configuration
- Monitor queue sizes
- Review retry settings

**Performance Issues**:
- Optimize event filtering
- Reduce payload sizes
- Implement batching
- Monitor resource usage

### Debugging Tools

**Session Statistics**:
```bash
curl "http://localhost:8000/callbacks/sessions/{session_id}/stats"
```

**Overall Statistics**:
```bash
curl "http://localhost:8000/callbacks/stats"
```

**Event Polling**:
```bash
curl "http://localhost:8000/callbacks/sessions/{session_id}/events?limit=10"
```

### Logging Configuration

Enable detailed logging for troubleshooting:

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Enable callback manager logging
logging.getLogger('callback_manager').setLevel(logging.DEBUG)
```

## Best Practices

### Security Considerations

**Authentication**: Always use authentication headers for webhook delivery to prevent unauthorized access to callback endpoints.

**HTTPS**: Use HTTPS for webhook URLs to ensure data encryption in transit.

**Validation**: Validate callback payloads to prevent injection attacks and ensure data integrity.

**Rate Limiting**: Implement rate limiting on callback endpoints to prevent abuse and ensure system stability.

### Performance Optimization

**Event Filtering**: Only subscribe to necessary event types to reduce network traffic and processing overhead.

**Batching**: Consider batching multiple events for high-frequency scenarios to improve efficiency.

**Caching**: Cache session configurations and statistics to reduce database load.

**Connection Pooling**: Use connection pooling for webhook delivery to improve performance.

### Reliability Patterns

**Retry Logic**: Implement exponential backoff for failed deliveries to handle temporary network issues.

**Circuit Breaker**: Use circuit breaker patterns to prevent cascading failures.

**Dead Letter Queue**: Implement dead letter queues for events that cannot be delivered after all retries.

**Health Checks**: Monitor callback endpoint health and disable delivery to unhealthy endpoints.

### Monitoring and Alerting

**Delivery Metrics**: Monitor delivery success rates, latency, and error rates.

**Session Health**: Track active sessions, event volumes, and resource usage.

**Alerting**: Set up alerts for high error rates, delivery failures, and performance degradation.

**Dashboards**: Create monitoring dashboards for real-time visibility into callback system health.

## Performance Considerations

### Scalability Factors

**Concurrent Sessions**: The system supports thousands of concurrent callback sessions with proper resource allocation.

**Event Volume**: High-frequency events may require batching or filtering to maintain performance.

**Delivery Latency**: WebSocket delivery provides sub-100ms latency, while webhooks typically deliver within 1-2 seconds.

**Memory Usage**: Session state and event queues consume memory proportional to active sessions and queue depths.

### Optimization Strategies

**Event Aggregation**: Combine related events to reduce delivery frequency while maintaining information completeness.

**Compression**: Enable compression for large payloads to reduce bandwidth usage and improve delivery speed.

**Connection Reuse**: Reuse HTTP connections for webhook delivery to reduce connection overhead.

**Async Processing**: Use asynchronous processing for event generation and delivery to prevent blocking.

### Resource Management

**Queue Management**: Implement queue size limits and cleanup policies to prevent memory exhaustion.

**Connection Limits**: Set appropriate limits on concurrent WebSocket connections based on server capacity.

**Timeout Configuration**: Configure appropriate timeouts for different delivery methods based on network conditions.

**Cleanup Policies**: Implement automatic cleanup of inactive sessions and old events to maintain system health.

---

*This guide provides comprehensive coverage of the OpenManus callback integration system. For additional support or questions, please refer to the API documentation or contact the development team.*

