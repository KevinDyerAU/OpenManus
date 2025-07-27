"""
SimpleSseMcpClient - Server-Sent Events based MCP Client
Enhanced implementation based on Eko framework with additional OpenManus features
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, AsyncGenerator, Set
from datetime import datetime, timedelta
import aiohttp
import sseclient
from urllib.parse import urljoin

from .interfaces import (
    IMcpClient, IMcpEventListener, IMcpToolInterceptor, IMcpToolValidator,
    McpConnectionStatus, McpToolSchema, McpToolResult, McpListToolParam,
    McpListToolResult, McpCallToolParam, McpConnectionConfig, McpToolUsageStats,
    McpSecurityLevel
)


class SimpleSseMcpClient(IMcpClient):
    """
    Server-Sent Events based MCP Client implementation
    Provides real-time communication with MCP servers using SSE protocol
    """
    
    def __init__(self, config: McpConnectionConfig):
        """Initialize SSE MCP client with configuration"""
        self.config = config
        self.status = McpConnectionStatus.DISCONNECTED
        self.session: Optional[aiohttp.ClientSession] = None
        self.sse_connection: Optional[sseclient.SSEClient] = None
        
        # Event management
        self.event_listeners: Set[IMcpEventListener] = set()
        self.tool_interceptors: List[IMcpToolInterceptor] = []
        self.tool_validator: Optional[IMcpToolValidator] = None
        
        # Tool management
        self.available_tools: Dict[str, McpToolSchema] = {}
        self.tool_usage_stats: Dict[str, McpToolUsageStats] = {}
        
        # Connection management
        self.last_heartbeat: Optional[datetime] = None
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = self.config.retry_attempts
        
        # Streaming support
        self.active_streams: Dict[str, asyncio.Queue] = {}
        
        # Logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Background tasks
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._sse_listener_task: Optional[asyncio.Task] = None
    
    async def connect(self) -> None:
        """Establish connection to MCP server"""
        if self.status == McpConnectionStatus.CONNECTED:
            self.logger.warning("Already connected to MCP server")
            return
        
        self.status = McpConnectionStatus.CONNECTING
        await self._notify_status_change()
        
        try:
            # Create HTTP session
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            connector = aiohttp.TCPConnector(ssl=self.config.ssl_verify)
            
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers=self.config.headers
            )
            
            # Establish SSE connection
            await self._establish_sse_connection()
            
            # Start background tasks
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self._sse_listener_task = asyncio.create_task(self._sse_listener_loop())
            
            # Load available tools
            await self._load_available_tools()
            
            self.status = McpConnectionStatus.CONNECTED
            self.reconnect_attempts = 0
            self.last_heartbeat = datetime.now()
            
            await self._notify_status_change()
            self.logger.info(f"Successfully connected to MCP server: {self.config.url}")
            
        except Exception as e:
            self.status = McpConnectionStatus.ERROR
            await self._notify_status_change()
            self.logger.error(f"Failed to connect to MCP server: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from MCP server"""
        if self.status == McpConnectionStatus.DISCONNECTED:
            return
        
        self.status = McpConnectionStatus.DISCONNECTED
        
        # Cancel background tasks
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        if self._sse_listener_task:
            self._sse_listener_task.cancel()
            try:
                await self._sse_listener_task
            except asyncio.CancelledError:
                pass
        
        # Close SSE connection
        if self.sse_connection:
            self.sse_connection.close()
            self.sse_connection = None
        
        # Close HTTP session
        if self.session:
            await self.session.close()
            self.session = None
        
        # Clear active streams
        for queue in self.active_streams.values():
            queue.put_nowait(None)  # Signal end of stream
        self.active_streams.clear()
        
        await self._notify_status_change()
        self.logger.info("Disconnected from MCP server")
    
    async def reconnect(self) -> None:
        """Reconnect to MCP server with exponential backoff"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            self.logger.error(f"Max reconnection attempts ({self.max_reconnect_attempts}) reached")
            self.status = McpConnectionStatus.ERROR
            await self._notify_status_change()
            return
        
        self.status = McpConnectionStatus.RECONNECTING
        await self._notify_status_change()
        
        # Exponential backoff
        delay = self.config.retry_delay * (2 ** self.reconnect_attempts)
        self.logger.info(f"Reconnecting in {delay} seconds (attempt {self.reconnect_attempts + 1})")
        await asyncio.sleep(delay)
        
        self.reconnect_attempts += 1
        
        try:
            await self.disconnect()
            await self.connect()
        except Exception as e:
            self.logger.error(f"Reconnection attempt failed: {e}")
            await self.reconnect()  # Retry
    
    def is_connected(self) -> bool:
        """Check if client is connected"""
        return self.status == McpConnectionStatus.CONNECTED
    
    async def list_tools(self, params: McpListToolParam) -> McpListToolResult:
        """List available tools with filtering"""
        if not self.is_connected():
            raise ConnectionError("Not connected to MCP server")
        
        try:
            # Apply filters
            filtered_tools = []
            for tool in self.available_tools.values():
                # Category filter
                if params.category and tool.category != params.category:
                    continue
                
                # Tags filter
                if params.tags and not any(tag in tool.tags for tag in params.tags):
                    continue
                
                # Security level filter
                if params.security_level and tool.security_level != params.security_level:
                    continue
                
                # Deprecated filter
                if not params.include_deprecated and tool.deprecated:
                    continue
                
                # Experimental filter
                if not params.include_experimental and tool.experimental:
                    continue
                
                filtered_tools.append(tool)
            
            # Collect metadata
            categories = list(set(tool.category for tool in self.available_tools.values()))
            available_tags = list(set(tag for tool in self.available_tools.values() for tag in tool.tags))
            
            return McpListToolResult(
                tools=filtered_tools,
                total_count=len(self.available_tools),
                filtered_count=len(filtered_tools),
                categories=categories,
                available_tags=available_tags
            )
            
        except Exception as e:
            self.logger.error(f"Failed to list tools: {e}")
            raise
    
    async def get_tool_schema(self, tool_name: str) -> Optional[McpToolSchema]:
        """Get detailed schema for a specific tool"""
        if not self.is_connected():
            raise ConnectionError("Not connected to MCP server")
        
        return self.available_tools.get(tool_name)
    
    async def call_tool(self, params: McpCallToolParam) -> McpToolResult:
        """Execute a tool with parameters"""
        if not self.is_connected():
            raise ConnectionError("Not connected to MCP server")
        
        tool_schema = self.available_tools.get(params.name)
        if not tool_schema:
            raise ValueError(f"Tool '{params.name}' not found")
        
        # Validate tool call
        if self.tool_validator:
            if not await self.tool_validator.validate_tool_call(params.name, params):
                raise PermissionError(f"Tool call validation failed for '{params.name}'")
        
        # Apply interceptors (before)
        for interceptor in self.tool_interceptors:
            params = await interceptor.before_tool_call(params.name, params)
        
        # Notify listeners
        for listener in self.event_listeners:
            await listener.on_tool_called(params.name, params)
        
        start_time = time.time()
        
        try:
            # Make HTTP request to execute tool
            url = urljoin(self.config.url, f"/tools/{params.name}/call")
            
            request_data = {
                "arguments": params.arguments,
                "context": params.context or {},
                "timeout": params.timeout or tool_schema.timeout,
                "dry_run": params.dry_run
            }
            
            async with self.session.post(url, json=request_data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Tool execution failed: {error_text}")
                
                result_data = await response.json()
                
                # Create result object
                execution_time = time.time() - start_time
                result = McpToolResult(
                    success=result_data.get("success", False),
                    content=result_data.get("content", []),
                    metadata=result_data.get("metadata", {}),
                    execution_time=execution_time,
                    tool_name=params.name,
                    tool_version=tool_schema.version,
                    error=result_data.get("error"),
                    warnings=result_data.get("warnings", []),
                    usage_stats=result_data.get("usage_stats", {})
                )
                
                # Update usage statistics
                await self._update_usage_stats(params.name, execution_time, result.success)
                
                # Validate result
                if self.tool_validator:
                    if not await self.tool_validator.validate_tool_result(params.name, result):
                        raise ValueError(f"Tool result validation failed for '{params.name}'")
                
                # Apply interceptors (after)
                for interceptor in self.tool_interceptors:
                    result = await interceptor.after_tool_call(params.name, result)
                
                # Notify listeners
                for listener in self.event_listeners:
                    await listener.on_tool_result(params.name, result)
                
                return result
                
        except Exception as e:
            execution_time = time.time() - start_time
            await self._update_usage_stats(params.name, execution_time, False)
            
            # Notify listeners of error
            for listener in self.event_listeners:
                await listener.on_tool_error(params.name, e)
            
            self.logger.error(f"Tool execution failed for '{params.name}': {e}")
            raise
    
    async def call_tool_stream(self, params: McpCallToolParam) -> AsyncGenerator[McpToolResult, None]:
        """Execute a tool with streaming results"""
        if not self.is_connected():
            raise ConnectionError("Not connected to MCP server")
        
        stream_id = f"{params.name}_{int(time.time() * 1000)}"
        self.active_streams[stream_id] = asyncio.Queue()
        
        try:
            # Start streaming request
            url = urljoin(self.config.url, f"/tools/{params.name}/stream")
            
            request_data = {
                "arguments": params.arguments,
                "context": params.context or {},
                "stream_id": stream_id
            }
            
            async with self.session.post(url, json=request_data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Streaming tool execution failed: {error_text}")
                
                # Process streaming results
                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            result = McpToolResult(**data)
                            yield result
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            self.logger.warning(f"Failed to parse streaming result: {e}")
                            continue
        
        finally:
            # Clean up stream
            if stream_id in self.active_streams:
                del self.active_streams[stream_id]
    
    async def register_tool(self, schema: McpToolSchema, handler) -> bool:
        """Register a new tool dynamically"""
        if not self.is_connected():
            raise ConnectionError("Not connected to MCP server")
        
        try:
            url = urljoin(self.config.url, "/tools/register")
            
            request_data = {
                "schema": {
                    "name": schema.name,
                    "description": schema.description,
                    "parameters": [
                        {
                            "name": param.name,
                            "type": param.type,
                            "description": param.description,
                            "required": param.required,
                            "default": param.default,
                            "enum": param.enum,
                            "pattern": param.pattern,
                            "minimum": param.minimum,
                            "maximum": param.maximum,
                            "security_level": param.security_level.value,
                            "validation_rules": param.validation_rules
                        }
                        for param in schema.parameters
                    ],
                    "returns": schema.returns,
                    "version": schema.version,
                    "category": schema.category,
                    "tags": schema.tags,
                    "security_level": schema.security_level.value,
                    "rate_limit": schema.rate_limit,
                    "timeout": schema.timeout,
                    "deprecated": schema.deprecated,
                    "experimental": schema.experimental
                }
            }
            
            async with self.session.post(url, json=request_data) as response:
                if response.status == 200:
                    self.available_tools[schema.name] = schema
                    self.logger.info(f"Successfully registered tool: {schema.name}")
                    return True
                else:
                    error_text = await response.text()
                    self.logger.error(f"Failed to register tool '{schema.name}': {error_text}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Failed to register tool '{schema.name}': {e}")
            return False
    
    async def unregister_tool(self, tool_name: str) -> bool:
        """Unregister a tool"""
        if not self.is_connected():
            raise ConnectionError("Not connected to MCP server")
        
        try:
            url = urljoin(self.config.url, f"/tools/{tool_name}/unregister")
            
            async with self.session.delete(url) as response:
                if response.status == 200:
                    if tool_name in self.available_tools:
                        del self.available_tools[tool_name]
                    if tool_name in self.tool_usage_stats:
                        del self.tool_usage_stats[tool_name]
                    self.logger.info(f"Successfully unregistered tool: {tool_name}")
                    return True
                else:
                    error_text = await response.text()
                    self.logger.error(f"Failed to unregister tool '{tool_name}': {error_text}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Failed to unregister tool '{tool_name}': {e}")
            return False
    
    async def get_tool_usage_stats(self, tool_name: Optional[str] = None) -> List[McpToolUsageStats]:
        """Get usage statistics for tools"""
        if tool_name:
            stats = self.tool_usage_stats.get(tool_name)
            return [stats] if stats else []
        else:
            return list(self.tool_usage_stats.values())
    
    def add_event_listener(self, listener: IMcpEventListener) -> None:
        """Add event listener for MCP events"""
        self.event_listeners.add(listener)
    
    def remove_event_listener(self, listener: IMcpEventListener) -> None:
        """Remove event listener"""
        self.event_listeners.discard(listener)
    
    def add_tool_interceptor(self, interceptor: IMcpToolInterceptor) -> None:
        """Add tool execution interceptor"""
        self.tool_interceptors.append(interceptor)
    
    def remove_tool_interceptor(self, interceptor: IMcpToolInterceptor) -> None:
        """Remove tool execution interceptor"""
        if interceptor in self.tool_interceptors:
            self.tool_interceptors.remove(interceptor)
    
    def set_tool_validator(self, validator: IMcpToolValidator) -> None:
        """Set tool validator for security"""
        self.tool_validator = validator
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on MCP connection"""
        if not self.is_connected():
            return {
                "status": "unhealthy",
                "connected": False,
                "error": "Not connected to MCP server"
            }
        
        try:
            url = urljoin(self.config.url, "/health")
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    health_data = await response.json()
                    return {
                        "status": "healthy",
                        "connected": True,
                        "server_health": health_data,
                        "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
                        "available_tools": len(self.available_tools),
                        "active_streams": len(self.active_streams)
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "connected": True,
                        "error": f"Server health check failed: {response.status}"
                    }
                    
        except Exception as e:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e)
            }
    
    async def get_server_info(self) -> Dict[str, Any]:
        """Get information about the MCP server"""
        if not self.is_connected():
            raise ConnectionError("Not connected to MCP server")
        
        try:
            url = urljoin(self.config.url, "/info")
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to get server info: {error_text}")
                    
        except Exception as e:
            self.logger.error(f"Failed to get server info: {e}")
            raise
    
    # Private methods
    
    async def _establish_sse_connection(self) -> None:
        """Establish Server-Sent Events connection"""
        sse_url = urljoin(self.config.url, "/events")
        
        # Note: In a real implementation, you would use an async SSE client
        # For now, this is a placeholder for the SSE connection logic
        self.logger.info(f"Establishing SSE connection to: {sse_url}")
        
        # TODO: Implement actual SSE connection using aiohttp or similar
        # This would involve creating a persistent connection to receive
        # server-sent events for real-time updates
    
    async def _sse_listener_loop(self) -> None:
        """Background task to listen for SSE events"""
        while self.is_connected():
            try:
                # TODO: Implement SSE event listening
                # This would process incoming events from the server
                await asyncio.sleep(1)  # Placeholder
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"SSE listener error: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _heartbeat_loop(self) -> None:
        """Background task for connection heartbeat"""
        while self.is_connected():
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                
                # Send heartbeat
                url = urljoin(self.config.url, "/heartbeat")
                async with self.session.get(url) as response:
                    if response.status == 200:
                        self.last_heartbeat = datetime.now()
                    else:
                        self.logger.warning(f"Heartbeat failed: {response.status}")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")
                # Trigger reconnection
                asyncio.create_task(self.reconnect())
                break
    
    async def _load_available_tools(self) -> None:
        """Load available tools from server"""
        try:
            url = urljoin(self.config.url, "/tools")
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    tools_data = await response.json()
                    
                    for tool_data in tools_data.get("tools", []):
                        schema = self._parse_tool_schema(tool_data)
                        self.available_tools[schema.name] = schema
                        
                        # Initialize usage stats
                        if schema.name not in self.tool_usage_stats:
                            self.tool_usage_stats[schema.name] = McpToolUsageStats(
                                tool_name=schema.name
                            )
                    
                    self.logger.info(f"Loaded {len(self.available_tools)} tools from server")
                else:
                    self.logger.warning(f"Failed to load tools: {response.status}")
                    
        except Exception as e:
            self.logger.error(f"Failed to load available tools: {e}")
    
    def _parse_tool_schema(self, tool_data: Dict[str, Any]) -> McpToolSchema:
        """Parse tool schema from server response"""
        # TODO: Implement proper schema parsing
        # This is a simplified version
        return McpToolSchema(
            name=tool_data["name"],
            description=tool_data["description"],
            parameters=[],  # TODO: Parse parameters
            returns=tool_data.get("returns", {}),
            version=tool_data.get("version", "1.0.0"),
            category=tool_data.get("category", "general"),
            tags=tool_data.get("tags", []),
            security_level=McpSecurityLevel(tool_data.get("security_level", "public")),
            rate_limit=tool_data.get("rate_limit"),
            timeout=tool_data.get("timeout"),
            deprecated=tool_data.get("deprecated", False),
            experimental=tool_data.get("experimental", False)
        )
    
    async def _update_usage_stats(self, tool_name: str, execution_time: float, success: bool) -> None:
        """Update tool usage statistics"""
        if tool_name not in self.tool_usage_stats:
            self.tool_usage_stats[tool_name] = McpToolUsageStats(tool_name=tool_name)
        
        stats = self.tool_usage_stats[tool_name]
        stats.call_count += 1
        stats.total_execution_time += execution_time
        stats.average_execution_time = stats.total_execution_time / stats.call_count
        stats.last_used = datetime.now()
        
        if success:
            stats.success_count += 1
        else:
            stats.error_count += 1
        
        # Update peak usage hour
        current_hour = datetime.now().hour
        if stats.peak_usage_hour is None:
            stats.peak_usage_hour = current_hour
    
    async def _notify_status_change(self) -> None:
        """Notify listeners of connection status change"""
        for listener in self.event_listeners:
            await listener.on_connection_status_changed(self.status)

