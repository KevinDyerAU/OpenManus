"""
Enhanced MCP Client for OpenManus
Provides advanced Model Context Protocol client capabilities with improved error handling,
connection management, tool discovery, and integration features.
"""

import asyncio
import json
import logging
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Callable
from urllib.parse import urlparse

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.types import (
    CallToolResult, 
    ListToolsResult, 
    TextContent, 
    Tool,
    GetPromptResult,
    ListPromptsResult,
    ListResourcesResult,
    ReadResourceResult
)
from pydantic import BaseModel, Field

from app.logger import logger
from app.tool.base import BaseTool, ToolResult


class ConnectionType(str, Enum):
    """MCP connection types"""
    STDIO = "stdio"
    SSE = "sse"
    WEBSOCKET = "websocket"


class ServerStatus(str, Enum):
    """MCP server status"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    RECONNECTING = "reconnecting"


@dataclass
class ServerConfig:
    """Configuration for an MCP server"""
    server_id: str
    name: str
    connection_type: ConnectionType
    url: Optional[str] = None
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    auto_reconnect: bool = True
    health_check_interval: int = 60
    capabilities: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServerInfo:
    """Information about an MCP server"""
    config: ServerConfig
    status: ServerStatus
    session: Optional[ClientSession] = None
    exit_stack: Optional[AsyncExitStack] = None
    last_connected: Optional[datetime] = None
    last_error: Optional[str] = None
    tools: Dict[str, Tool] = field(default_factory=dict)
    prompts: List[str] = field(default_factory=list)
    resources: List[str] = field(default_factory=list)
    connection_attempts: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0


class ToolProxy(BaseTool):
    """Enhanced proxy for MCP server tools with caching and error handling"""
    
    def __init__(self, tool: Tool, server_info: ServerInfo, client: 'EnhancedMCPClient'):
        self.tool = tool
        self.server_info = server_info
        self.client = client
        self.name = f"{server_info.config.server_id}_{tool.name}"
        self.original_name = tool.name
        self.description = tool.description or f"Tool {tool.name} from {server_info.config.name}"
        self.last_used = None
        self.usage_count = 0
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes

    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with enhanced error handling and caching"""
        self.usage_count += 1
        self.last_used = datetime.now()
        
        # Check cache if enabled
        cache_key = self._get_cache_key(kwargs)
        if cache_key in self.cache:
            cached_result, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < timedelta(seconds=self.cache_ttl):
                logger.debug(f"Using cached result for {self.name}")
                return cached_result

        # Execute tool
        try:
            if not self.server_info.session:
                return ToolResult(error=f"Not connected to server {self.server_info.config.name}")

            logger.info(f"Executing tool: {self.original_name} on server {self.server_info.config.name}")
            
            # Track request
            self.server_info.total_requests += 1
            
            # Execute with timeout
            result = await asyncio.wait_for(
                self.server_info.session.call_tool(self.original_name, kwargs),
                timeout=self.server_info.config.timeout
            )
            
            # Process result
            content_str = self._process_result_content(result)
            tool_result = ToolResult(output=content_str or "No output returned.")
            
            # Cache successful result
            if cache_key:
                self.cache[cache_key] = (tool_result, datetime.now())
            
            self.server_info.successful_requests += 1
            return tool_result
            
        except asyncio.TimeoutError:
            error_msg = f"Tool {self.original_name} timed out after {self.server_info.config.timeout}s"
            logger.error(error_msg)
            self.server_info.failed_requests += 1
            return ToolResult(error=error_msg)
            
        except Exception as e:
            error_msg = f"Error executing tool {self.original_name}: {str(e)}"
            logger.error(error_msg)
            self.server_info.failed_requests += 1
            
            # Attempt reconnection if connection error
            if "connection" in str(e).lower():
                await self.client.reconnect_server(self.server_info.config.server_id)
            
            return ToolResult(error=error_msg)

    def _get_cache_key(self, kwargs: Dict[str, Any]) -> Optional[str]:
        """Generate cache key for arguments"""
        try:
            return json.dumps(kwargs, sort_keys=True)
        except (TypeError, ValueError):
            return None

    def _process_result_content(self, result: CallToolResult) -> str:
        """Process and format tool result content"""
        content_parts = []
        for item in result.content:
            if isinstance(item, TextContent):
                content_parts.append(item.text)
            else:
                content_parts.append(str(item))
        return "\n".join(content_parts)


class EnhancedMCPClient:
    """Enhanced MCP client with advanced features and management capabilities"""
    
    def __init__(self, 
                 auto_discover: bool = True,
                 health_check_enabled: bool = True,
                 metrics_enabled: bool = True):
        self.servers: Dict[str, ServerInfo] = {}
        self.tools: Dict[str, ToolProxy] = {}
        self.auto_discover = auto_discover
        self.health_check_enabled = health_check_enabled
        self.metrics_enabled = metrics_enabled
        self._health_check_task: Optional[asyncio.Task] = None
        self._discovery_callbacks: List[Callable] = []
        self._status_callbacks: List[Callable] = []
        
    async def add_server(self, config: ServerConfig) -> bool:
        """Add and connect to an MCP server"""
        try:
            server_info = ServerInfo(config=config, status=ServerStatus.DISCONNECTED)
            self.servers[config.server_id] = server_info
            
            success = await self._connect_server(server_info)
            if success and self.auto_discover:
                await self._discover_server_capabilities(server_info)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to add server {config.server_id}: {e}")
            return False

    async def _connect_server(self, server_info: ServerInfo) -> bool:
        """Connect to an MCP server"""
        config = server_info.config
        server_info.status = ServerStatus.CONNECTING
        server_info.connection_attempts += 1
        
        try:
            exit_stack = AsyncExitStack()
            
            if config.connection_type == ConnectionType.STDIO:
                if not config.command:
                    raise ValueError("Command required for stdio connection")
                
                server_params = StdioServerParameters(
                    command=config.command,
                    args=config.args or [],
                    env=config.env
                )
                session = await exit_stack.enter_async_context(stdio_client(server_params))
                
            elif config.connection_type == ConnectionType.SSE:
                if not config.url:
                    raise ValueError("URL required for SSE connection")
                
                session = await exit_stack.enter_async_context(sse_client(config.url))
                
            else:
                raise ValueError(f"Unsupported connection type: {config.connection_type}")
            
            # Initialize session
            await session.initialize()
            
            server_info.session = session
            server_info.exit_stack = exit_stack
            server_info.status = ServerStatus.CONNECTED
            server_info.last_connected = datetime.now()
            server_info.last_error = None
            
            logger.info(f"Connected to MCP server: {config.name}")
            await self._notify_status_change(config.server_id, ServerStatus.CONNECTED)
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to connect to server {config.name}: {e}"
            logger.error(error_msg)
            server_info.status = ServerStatus.ERROR
            server_info.last_error = error_msg
            
            await self._notify_status_change(config.server_id, ServerStatus.ERROR)
            return False

    async def _discover_server_capabilities(self, server_info: ServerInfo) -> None:
        """Discover tools, prompts, and resources from server"""
        if not server_info.session:
            return
            
        try:
            # Discover tools
            tools_result = await server_info.session.list_tools()
            for tool in tools_result.tools:
                server_info.tools[tool.name] = tool
                
                # Create tool proxy
                proxy = ToolProxy(tool, server_info, self)
                self.tools[proxy.name] = proxy
                
            logger.info(f"Discovered {len(server_info.tools)} tools from {server_info.config.name}")
            
            # Discover prompts
            try:
                prompts_result = await server_info.session.list_prompts()
                server_info.prompts = [p.name for p in prompts_result.prompts]
                logger.info(f"Discovered {len(server_info.prompts)} prompts from {server_info.config.name}")
            except Exception as e:
                logger.debug(f"Server {server_info.config.name} doesn't support prompts: {e}")
            
            # Discover resources
            try:
                resources_result = await server_info.session.list_resources()
                server_info.resources = [r.uri for r in resources_result.resources]
                logger.info(f"Discovered {len(server_info.resources)} resources from {server_info.config.name}")
            except Exception as e:
                logger.debug(f"Server {server_info.config.name} doesn't support resources: {e}")
                
            await self._notify_discovery_complete(server_info.config.server_id)
            
        except Exception as e:
            logger.error(f"Failed to discover capabilities from {server_info.config.name}: {e}")

    async def reconnect_server(self, server_id: str) -> bool:
        """Reconnect to a server"""
        if server_id not in self.servers:
            return False
            
        server_info = self.servers[server_id]
        if server_info.status == ServerStatus.CONNECTING:
            return False
            
        logger.info(f"Reconnecting to server {server_info.config.name}")
        server_info.status = ServerStatus.RECONNECTING
        
        # Cleanup existing connection
        await self._cleanup_server(server_info)
        
        # Attempt reconnection with exponential backoff
        for attempt in range(server_info.config.max_retries):
            await asyncio.sleep(server_info.config.retry_delay * (2 ** attempt))
            
            if await self._connect_server(server_info):
                if self.auto_discover:
                    await self._discover_server_capabilities(server_info)
                return True
                
        server_info.status = ServerStatus.ERROR
        return False

    async def disconnect_server(self, server_id: str) -> None:
        """Disconnect from a server"""
        if server_id not in self.servers:
            return
            
        server_info = self.servers[server_id]
        await self._cleanup_server(server_info)
        server_info.status = ServerStatus.DISCONNECTED
        
        # Remove tools from this server
        tools_to_remove = [name for name, tool in self.tools.items() 
                          if tool.server_info.config.server_id == server_id]
        for tool_name in tools_to_remove:
            del self.tools[tool_name]
            
        logger.info(f"Disconnected from server {server_info.config.name}")

    async def _cleanup_server(self, server_info: ServerInfo) -> None:
        """Clean up server connection resources"""
        if server_info.exit_stack:
            try:
                await server_info.exit_stack.aclose()
            except Exception as e:
                logger.error(f"Error cleaning up server connection: {e}")
            finally:
                server_info.exit_stack = None
                server_info.session = None

    async def get_server_status(self, server_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status information for a server"""
        if server_id not in self.servers:
            return None
            
        server_info = self.servers[server_id]
        return {
            "server_id": server_id,
            "name": server_info.config.name,
            "status": server_info.status.value,
            "connection_type": server_info.config.connection_type.value,
            "last_connected": server_info.last_connected.isoformat() if server_info.last_connected else None,
            "last_error": server_info.last_error,
            "connection_attempts": server_info.connection_attempts,
            "tools_count": len(server_info.tools),
            "prompts_count": len(server_info.prompts),
            "resources_count": len(server_info.resources),
            "total_requests": server_info.total_requests,
            "successful_requests": server_info.successful_requests,
            "failed_requests": server_info.failed_requests,
            "success_rate": (server_info.successful_requests / max(server_info.total_requests, 1)) * 100
        }

    async def get_all_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available tools"""
        tools_info = {}
        for name, proxy in self.tools.items():
            tools_info[name] = {
                "name": proxy.original_name,
                "server": proxy.server_info.config.name,
                "server_id": proxy.server_info.config.server_id,
                "description": proxy.description,
                "usage_count": proxy.usage_count,
                "last_used": proxy.last_used.isoformat() if proxy.last_used else None,
                "schema": proxy.tool.inputSchema if hasattr(proxy.tool, 'inputSchema') else None
            }
        return tools_info

    async def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool by name"""
        if tool_name not in self.tools:
            return ToolResult(error=f"Tool {tool_name} not found")
            
        return await self.tools[tool_name].execute(**kwargs)

    async def start_health_checks(self) -> None:
        """Start periodic health checks for all servers"""
        if not self.health_check_enabled or self._health_check_task:
            return
            
        self._health_check_task = asyncio.create_task(self._health_check_loop())

    async def stop_health_checks(self) -> None:
        """Stop health check task"""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None

    async def _health_check_loop(self) -> None:
        """Periodic health check loop"""
        while True:
            try:
                for server_id, server_info in self.servers.items():
                    if server_info.status == ServerStatus.CONNECTED:
                        await self._check_server_health(server_info)
                        
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(60)

    async def _check_server_health(self, server_info: ServerInfo) -> None:
        """Check health of a specific server"""
        try:
            if server_info.session:
                # Try to list tools as a health check
                await asyncio.wait_for(
                    server_info.session.list_tools(),
                    timeout=10
                )
        except Exception as e:
            logger.warning(f"Health check failed for {server_info.config.name}: {e}")
            if server_info.config.auto_reconnect:
                await self.reconnect_server(server_info.config.server_id)

    def add_discovery_callback(self, callback: Callable[[str], None]) -> None:
        """Add callback for when server discovery completes"""
        self._discovery_callbacks.append(callback)

    def add_status_callback(self, callback: Callable[[str, ServerStatus], None]) -> None:
        """Add callback for server status changes"""
        self._status_callbacks.append(callback)

    async def _notify_discovery_complete(self, server_id: str) -> None:
        """Notify callbacks that discovery is complete"""
        for callback in self._discovery_callbacks:
            try:
                await callback(server_id)
            except Exception as e:
                logger.error(f"Error in discovery callback: {e}")

    async def _notify_status_change(self, server_id: str, status: ServerStatus) -> None:
        """Notify callbacks of status changes"""
        for callback in self._status_callbacks:
            try:
                await callback(server_id, status)
            except Exception as e:
                logger.error(f"Error in status callback: {e}")

    async def shutdown(self) -> None:
        """Shutdown the client and cleanup all connections"""
        await self.stop_health_checks()
        
        for server_id in list(self.servers.keys()):
            await self.disconnect_server(server_id)
            
        logger.info("Enhanced MCP client shutdown complete")

    async def __aenter__(self):
        """Async context manager entry"""
        await self.start_health_checks()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.shutdown()


# Factory functions for common configurations

def create_stdio_server_config(
    server_id: str,
    name: str,
    command: str,
    args: Optional[List[str]] = None,
    **kwargs
) -> ServerConfig:
    """Create configuration for stdio MCP server"""
    return ServerConfig(
        server_id=server_id,
        name=name,
        connection_type=ConnectionType.STDIO,
        command=command,
        args=args or [],
        **kwargs
    )


def create_sse_server_config(
    server_id: str,
    name: str,
    url: str,
    **kwargs
) -> ServerConfig:
    """Create configuration for SSE MCP server"""
    return ServerConfig(
        server_id=server_id,
        name=name,
        connection_type=ConnectionType.SSE,
        url=url,
        **kwargs
    )


async def create_enhanced_mcp_client(
    server_configs: List[ServerConfig],
    auto_start_health_checks: bool = True
) -> EnhancedMCPClient:
    """Create and initialize an enhanced MCP client with multiple servers"""
    client = EnhancedMCPClient()
    
    # Add all servers
    for config in server_configs:
        await client.add_server(config)
    
    # Start health checks
    if auto_start_health_checks:
        await client.start_health_checks()
    
    return client
