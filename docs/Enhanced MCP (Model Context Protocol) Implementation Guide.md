# Enhanced MCP (Model Context Protocol) Implementation Guide

## Overview

The Enhanced MCP implementation for OpenManus provides a comprehensive, production-ready Model Context Protocol system with real-time capabilities, dynamic tool loading, security features, and extensive monitoring. This implementation is inspired by the Eko framework but extends it with additional OpenManus-specific features.

## Architecture Overview

The enhanced MCP system consists of several key components that work together to provide a robust and scalable tool execution platform:

### Core Components

1. **Enhanced Interfaces** (`interfaces.py`)
   - Comprehensive interface definitions for all MCP components
   - Type-safe data classes with validation
   - Security levels and permission management
   - Event system interfaces

2. **SSE Client** (`sse_client.py`)
   - Server-Sent Events based client implementation
   - Real-time communication with automatic reconnection
   - Tool execution with streaming support
   - Usage statistics and monitoring

3. **Tool Registry** (`tool_registry.py`)
   - Dynamic tool loading and management
   - Plugin system for extensibility
   - Security validation and rate limiting
   - Tool versioning and metadata management

4. **Enhanced Server** (`enhanced_server.py`)
   - FastAPI-based server with WebSocket support
   - Real-time event broadcasting
   - RESTful API with streaming endpoints
   - Health monitoring and statistics

5. **Client Factory** (`client_factory.py`)
   - Factory patterns for easy client creation
   - Multi-client management utilities
   - Security and logging integrations
   - Production-ready configurations

## Key Features

### Real-Time Communication

The enhanced MCP system supports multiple communication protocols:

- **Server-Sent Events (SSE)**: For real-time server-to-client updates
- **WebSockets**: For bidirectional real-time communication
- **HTTP REST API**: For standard request-response interactions
- **Streaming**: For long-running tool executions

### Dynamic Tool Loading

Tools can be loaded dynamically from multiple sources:

```python
from app.mcp import mcp_tool, McpSecurityLevel

@mcp_tool(
    name="example_tool",
    description="An example tool demonstrating the enhanced MCP system",
    category="examples",
    security_level=McpSecurityLevel.PUBLIC,
    tags=["example", "demo"]
)
async def example_tool(message: str, count: int = 1, context=None):
    """Example tool implementation"""
    return {
        "message": message,
        "repeated": [message] * count,
        "timestamp": datetime.now().isoformat()
    }
```

### Security and Validation

The system includes comprehensive security features:

- **Security Levels**: PUBLIC, AUTHENTICATED, RESTRICTED, ADMIN
- **Tool Validation**: Parameter validation and security checks
- **Rate Limiting**: Configurable rate limits per tool
- **Access Control**: Fine-grained permission management
- **Input Sanitization**: Automatic input validation and sanitization

### Monitoring and Analytics

Built-in monitoring provides insights into tool usage:

- **Usage Statistics**: Call counts, success rates, execution times
- **Performance Metrics**: Response times, error rates, throughput
- **Health Monitoring**: System health checks and status reporting
- **Event Tracking**: Comprehensive event logging and analysis

## Implementation Details

### Interface Definitions

The enhanced interfaces provide type-safe definitions for all MCP components:

```python
@dataclass
class McpToolSchema:
    """Enhanced tool schema definition"""
    name: str
    description: str
    parameters: List[McpToolParameter]
    returns: Dict[str, Any]
    version: str = "1.0.0"
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    security_level: McpSecurityLevel = McpSecurityLevel.PUBLIC
    rate_limit: Optional[int] = None
    timeout: Optional[int] = None
    deprecated: bool = False
    experimental: bool = False
```

### SSE Client Implementation

The SimpleSseMcpClient provides robust client functionality:

```python
from app.mcp import create_production_client

# Create a production-ready client
client = await create_production_client("http://localhost:8081")

# Connect to server
await client.connect()

# List available tools
tools = await client.list_tools(McpListToolParam())

# Execute a tool
result = await client.call_tool(McpCallToolParam(
    name="system_info",
    arguments={}
))

# Stream tool execution
async for partial_result in client.call_tool_stream(McpCallToolParam(
    name="monitor_resources",
    arguments={"duration": 10}
)):
    print(f"Partial result: {partial_result}")
```

### Tool Registry Features

The enhanced tool registry supports dynamic loading:

```python
from app.mcp import McpToolRegistry

# Create registry
registry = McpToolRegistry(plugin_directories=["./plugins"])

# Load plugins from directory
await registry.load_plugins_from_directory("./custom_tools")

# Register tool programmatically
await registry.register_tool(tool_schema, tool_handler)

# Get usage statistics
stats = await registry.get_tool_usage_stats("system_info")
```

### Server Configuration

The enhanced server provides comprehensive configuration options:

```python
from app.mcp import create_mcp_server

# Create server with custom configuration
server = create_mcp_server(
    host="0.0.0.0",
    port=8081,
    plugin_directories=["./plugins", "./custom_tools"]
)

# Start server
await server.start()
```

## Usage Examples

### Basic Client Usage

```python
import asyncio
from app.mcp import create_simple_client, McpCallToolParam

async def main():
    # Create and connect client
    client = await create_simple_client("http://localhost:8081")
    await client.connect()
    
    try:
        # Get system information
        result = await client.call_tool(McpCallToolParam(
            name="system_info",
            arguments={}
        ))
        
        print(f"System info: {result.content}")
        
    finally:
        await client.disconnect()

asyncio.run(main())
```

### Advanced Client with Event Handling

```python
from app.mcp import (
    create_production_client, 
    DefaultEventListener,
    McpConnectionStatus
)

class CustomEventListener(DefaultEventListener):
    async def on_tool_result(self, tool_name: str, result):
        print(f"Tool {tool_name} completed in {result.execution_time:.3f}s")

async def main():
    client = await create_production_client("http://localhost:8081")
    
    # Add custom event listener
    listener = CustomEventListener()
    client.add_event_listener(listener)
    
    await client.connect()
    
    # Use client...
    
    await client.disconnect()
```

### Multi-Server Client Management

```python
from app.mcp import McpClientManager, create_production_client

async def main():
    manager = McpClientManager()
    
    # Add multiple servers
    servers = [
        "http://server1:8081",
        "http://server2:8081", 
        "http://server3:8081"
    ]
    
    for i, url in enumerate(servers):
        client = await create_production_client(url)
        await manager.add_client(f"server_{i+1}", client)
    
    # Connect all clients
    results = await manager.connect_all()
    print(f"Connection results: {results}")
    
    # Health check all servers
    health = await manager.health_check_all()
    print(f"Health status: {health}")
    
    # Use specific client
    client = manager.get_client("server_1")
    if client:
        result = await client.call_tool(McpCallToolParam(
            name="system_info",
            arguments={}
        ))
    
    # Cleanup
    await manager.disconnect_all()
```

### Custom Tool Development

```python
from app.mcp import mcp_tool, McpSecurityLevel, McpToolResult

@mcp_tool(
    name="custom_calculator",
    description="Perform mathematical calculations",
    category="math",
    security_level=McpSecurityLevel.PUBLIC,
    tags=["math", "calculator"],
    timeout=10
)
async def custom_calculator(operation: str, a: float, b: float, context=None):
    """Custom calculator tool"""
    try:
        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            if b == 0:
                raise ValueError("Division by zero")
            result = a / b
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        return McpToolResult(
            success=True,
            content=[{
                "type": "calculation",
                "operation": operation,
                "operands": [a, b],
                "result": result
            }],
            metadata={
                "precision": "float64",
                "operation_type": operation
            }
        )
        
    except Exception as e:
        return McpToolResult(
            success=False,
            content=[],
            error=str(e)
        )
```

### Plugin Development

Create a plugin file `my_plugin.py`:

```python
from app.mcp import mcp_tool, McpSecurityLevel

@mcp_tool(
    name="plugin_tool",
    description="A tool from a plugin",
    category="plugin",
    security_level=McpSecurityLevel.AUTHENTICATED
)
async def plugin_tool(data: str, context=None):
    """Plugin tool implementation"""
    return {
        "processed_data": data.upper(),
        "plugin_name": "my_plugin",
        "timestamp": datetime.now().isoformat()
    }

# Export tools for automatic registration
TOOLS = [
    (plugin_tool._mcp_tool_schema, plugin_tool)
]
```

## Security Considerations

### Security Levels

The system implements four security levels:

1. **PUBLIC**: Accessible to all clients without authentication
2. **AUTHENTICATED**: Requires valid authentication
3. **RESTRICTED**: Requires special permissions
4. **ADMIN**: Requires administrative privileges

### Validation and Sanitization

All tool inputs are validated according to their schema:

```python
@dataclass
class McpToolParameter:
    name: str
    type: str
    description: str
    required: bool = False
    default: Any = None
    enum: Optional[List[str]] = None
    pattern: Optional[str] = None
    minimum: Optional[Union[int, float]] = None
    maximum: Optional[Union[int, float]] = None
    security_level: McpSecurityLevel = McpSecurityLevel.PUBLIC
    validation_rules: List[str] = field(default_factory=list)
```

### Rate Limiting

Tools can have rate limits to prevent abuse:

```python
from app.mcp import SecurityToolValidator

validator = SecurityToolValidator()
validator.set_rate_limit("expensive_tool", max_calls=10, window_seconds=60)
```

## Performance Optimization

### Connection Pooling

The SSE client implements connection pooling and reuse:

- Automatic reconnection with exponential backoff
- Connection health monitoring
- Resource cleanup and management

### Caching

Tool results can be cached for improved performance:

- Schema caching for faster tool lookup
- Result caching for expensive operations
- Metadata caching for reduced overhead

### Streaming

Long-running tools support streaming results:

```python
async for partial_result in client.call_tool_stream(params):
    # Process partial results as they arrive
    process_partial_result(partial_result)
```

## Monitoring and Observability

### Health Checks

Comprehensive health monitoring:

```python
# Client health check
health = await client.health_check()
print(f"Client health: {health}")

# Server health check
response = requests.get("http://localhost:8081/health")
print(f"Server health: {response.json()}")
```

### Usage Statistics

Detailed usage analytics:

```python
# Get tool usage statistics
stats = await client.get_tool_usage_stats("system_info")
print(f"Tool stats: {stats}")

# Get registry statistics
registry_stats = await registry.get_registry_stats()
print(f"Registry stats: {registry_stats}")
```

### Event Monitoring

Real-time event monitoring:

```python
class MonitoringEventListener(IMcpEventListener):
    async def on_tool_called(self, tool_name: str, params):
        # Log to monitoring system
        monitor.log_tool_call(tool_name, params)
    
    async def on_tool_result(self, tool_name: str, result):
        # Track performance metrics
        monitor.track_performance(tool_name, result.execution_time)
```

## Deployment Considerations

### Production Configuration

For production deployment:

```python
# Production client configuration
client = await create_production_client(
    url="https://mcp.production.com",
    enable_logging=True,
    enable_security=True,
    allowed_security_levels=[
        McpSecurityLevel.PUBLIC,
        McpSecurityLevel.AUTHENTICATED
    ]
)
```

### Scaling

The enhanced MCP system supports horizontal scaling:

- Multiple server instances with load balancing
- Client-side load balancing and failover
- Distributed tool registry with shared storage
- Event broadcasting across server instances

### High Availability

Built-in high availability features:

- Automatic failover and recovery
- Health monitoring and alerting
- Graceful degradation under load
- Circuit breaker patterns for resilience

## Integration with OpenManus

The enhanced MCP system integrates seamlessly with OpenManus:

### Flow Integration

```python
from app.flow import BaseFlow
from app.mcp import create_production_client

class McpEnabledFlow(BaseFlow):
    async def initialize(self):
        self.mcp_client = await create_production_client(
            "http://localhost:8081"
        )
        await self.mcp_client.connect()
    
    async def execute_step(self, step_data):
        # Use MCP tools in flow execution
        result = await self.mcp_client.call_tool(McpCallToolParam(
            name="process_data",
            arguments=step_data
        ))
        return result.content
```

### Agent Integration

```python
from app.agents import BaseAgent

class McpAgent(BaseAgent):
    def __init__(self, mcp_client):
        super().__init__()
        self.mcp_client = mcp_client
    
    async def execute_action(self, action):
        # Use MCP tools for agent actions
        return await self.mcp_client.call_tool(action.to_mcp_params())
```

## Migration from Original MCP

### Compatibility

The enhanced MCP system maintains backward compatibility:

- Original MCP interfaces are supported
- Existing tools can be migrated with minimal changes
- Configuration migration utilities provided

### Migration Steps

1. **Update Dependencies**: Install enhanced MCP components
2. **Update Imports**: Change import statements to use enhanced modules
3. **Update Configuration**: Migrate to new configuration format
4. **Test Integration**: Verify all functionality works correctly
5. **Deploy Gradually**: Roll out changes incrementally

### Migration Example

Original code:
```python
from app.mcp.server import McpServer

server = McpServer()
await server.start()
```

Enhanced code:
```python
from app.mcp import create_mcp_server

server = create_mcp_server(
    host="0.0.0.0",
    port=8081
)
await server.start()
```

## Best Practices

### Tool Development

1. **Use Type Hints**: Provide clear type annotations for all parameters
2. **Implement Error Handling**: Handle exceptions gracefully
3. **Add Validation**: Validate inputs thoroughly
4. **Document Thoroughly**: Provide comprehensive documentation
5. **Test Extensively**: Write comprehensive tests for all tools

### Client Usage

1. **Handle Connections**: Always properly connect and disconnect clients
2. **Implement Retries**: Handle network failures gracefully
3. **Monitor Performance**: Track tool execution times and success rates
4. **Use Security**: Implement appropriate security measures
5. **Cache Results**: Cache expensive operations when appropriate

### Server Deployment

1. **Configure Security**: Set appropriate security levels and validation
2. **Monitor Health**: Implement comprehensive health monitoring
3. **Scale Appropriately**: Plan for expected load and growth
4. **Backup Data**: Implement backup and recovery procedures
5. **Update Regularly**: Keep the system updated with latest features

This enhanced MCP implementation provides a robust, scalable, and feature-rich foundation for tool execution in the OpenManus system, with comprehensive real-time capabilities, security features, and monitoring that exceeds the capabilities of the original implementation while maintaining compatibility and ease of use.

