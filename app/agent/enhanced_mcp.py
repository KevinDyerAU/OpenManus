"""
Enhanced MCP Agent for OpenManus
Integrates with the enhanced MCP client to provide advanced multi-server capabilities,
improved error handling, and better tool management.
"""

import asyncio
from typing import Any, Dict, List, Optional, Set
from datetime import datetime

from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.logger import logger
from app.mcp.enhanced_client import (
    EnhancedMCPClient, 
    ServerConfig, 
    ServerStatus,
    create_stdio_server_config,
    create_sse_server_config
)
from app.prompt.mcp import MULTIMEDIA_RESPONSE_PROMPT, NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import AgentState, Message
from app.tool.base import ToolResult


class EnhancedMCPAgent(ToolCallAgent):
    """Enhanced agent for interacting with multiple MCP servers with advanced capabilities.
    
    This agent uses the enhanced MCP client to connect to multiple MCP servers,
    provides intelligent tool routing, automatic failover, and comprehensive monitoring.
    """

    name: str = "enhanced_mcp_agent"
    description: str = "An enhanced agent that connects to multiple MCP servers with advanced capabilities."

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    # Enhanced MCP client
    mcp_client: Optional[EnhancedMCPClient] = None
    server_configs: List[ServerConfig] = Field(default_factory=list)
    
    # Configuration
    max_steps: int = 20
    auto_reconnect: bool = True
    health_checks_enabled: bool = True
    tool_timeout: int = 30
    
    # Tool management
    preferred_servers: Dict[str, str] = Field(default_factory=dict)  # tool_type -> server_id
    server_priorities: List[str] = Field(default_factory=list)  # Ordered list of server IDs by priority
    
    # Monitoring
    execution_stats: Dict[str, Any] = Field(default_factory=dict)
    last_health_check: Optional[datetime] = None
    
    # Special tool names that should trigger termination
    special_tool_names: List[str] = Field(default_factory=lambda: ["terminate"])

    async def initialize(self, server_configs: Optional[List[ServerConfig]] = None) -> None:
        """Initialize the enhanced MCP agent with multiple server configurations.
        
        Args:
            server_configs: List of server configurations to connect to
        """
        try:
            if server_configs:
                self.server_configs = server_configs
            
            if not self.server_configs:
                logger.warning("No server configurations provided to Enhanced MCP Agent")
                return
            
            # Initialize enhanced MCP client
            self.mcp_client = EnhancedMCPClient(
                auto_discover=True,
                health_check_enabled=self.health_checks_enabled,
                metrics_enabled=True
            )
            
            # Add status change callback
            self.mcp_client.add_status_callback(self._on_server_status_change)
            self.mcp_client.add_discovery_callback(self._on_server_discovery_complete)
            
            # Connect to all servers
            connected_servers = []
            for config in self.server_configs:
                logger.info(f"Connecting to MCP server: {config.name}")
                success = await self.mcp_client.add_server(config)
                if success:
                    connected_servers.append(config.server_id)
                    logger.info(f"Successfully connected to {config.name}")
                else:
                    logger.error(f"Failed to connect to {config.name}")
            
            if not connected_servers:
                raise RuntimeError("Failed to connect to any MCP servers")
            
            # Set server priorities (first connected servers have higher priority)
            self.server_priorities = connected_servers
            
            # Start health checks
            if self.health_checks_enabled:
                await self.mcp_client.start_health_checks()
            
            # Update available tools
            await self._update_available_tools()
            
            # Initialize execution stats
            self.execution_stats = {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "servers_connected": len(connected_servers),
                "tools_available": len(self.mcp_client.tools),
                "last_updated": datetime.now()
            }
            
            logger.info(f"Enhanced MCP Agent initialized with {len(connected_servers)} servers and {len(self.mcp_client.tools)} tools")
            
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced MCP Agent: {e}")
            raise

    async def _update_available_tools(self) -> None:
        """Update the list of available tools from all connected servers"""
        if not self.mcp_client:
            return
            
        # Get all tools from the enhanced client
        all_tools = await self.mcp_client.get_all_tools()
        
        # Update tool collection
        self.available_tools = self.mcp_client
        
        logger.info(f"Updated available tools: {len(all_tools)} tools from {len(self.mcp_client.servers)} servers")

    async def _on_server_status_change(self, server_id: str, status: ServerStatus) -> None:
        """Handle server status changes"""
        logger.info(f"Server {server_id} status changed to: {status.value}")
        
        if status == ServerStatus.CONNECTED:
            await self._update_available_tools()
        elif status == ServerStatus.ERROR and self.auto_reconnect:
            logger.info(f"Attempting to reconnect to server {server_id}")
            # The enhanced client handles reconnection automatically

    async def _on_server_discovery_complete(self, server_id: str) -> None:
        """Handle completion of server capability discovery"""
        logger.info(f"Discovery complete for server {server_id}")
        await self._update_available_tools()

    async def get_server_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status of all servers"""
        if not self.mcp_client:
            return {"status": "not_initialized"}
        
        health_status = {
            "overall_status": "healthy",
            "servers": {},
            "total_servers": len(self.mcp_client.servers),
            "connected_servers": 0,
            "total_tools": len(self.mcp_client.tools),
            "last_check": datetime.now().isoformat()
        }
        
        for server_id in self.mcp_client.servers:
            server_status = await self.mcp_client.get_server_status(server_id)
            if server_status:
                health_status["servers"][server_id] = server_status
                if server_status["status"] == "connected":
                    health_status["connected_servers"] += 1
        
        # Determine overall status
        if health_status["connected_servers"] == 0:
            health_status["overall_status"] = "critical"
        elif health_status["connected_servers"] < health_status["total_servers"]:
            health_status["overall_status"] = "degraded"
        
        self.last_health_check = datetime.now()
        return health_status

    async def execute_tool_with_fallback(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool with intelligent server selection and fallback"""
        if not self.mcp_client:
            return ToolResult(error="MCP client not initialized")
        
        self.execution_stats["total_executions"] += 1
        
        # Try to find the tool directly first
        if tool_name in self.mcp_client.tools:
            try:
                result = await self.mcp_client.execute_tool(tool_name, **kwargs)
                if not result.error:
                    self.execution_stats["successful_executions"] += 1
                    return result
            except Exception as e:
                logger.warning(f"Direct tool execution failed: {e}")
        
        # Try to find similar tools across servers
        similar_tools = self._find_similar_tools(tool_name)
        
        for similar_tool_name in similar_tools:
            try:
                logger.info(f"Trying fallback tool: {similar_tool_name}")
                result = await self.mcp_client.execute_tool(similar_tool_name, **kwargs)
                if not result.error:
                    self.execution_stats["successful_executions"] += 1
                    return result
            except Exception as e:
                logger.warning(f"Fallback tool {similar_tool_name} failed: {e}")
                continue
        
        # No successful execution
        self.execution_stats["failed_executions"] += 1
        return ToolResult(error=f"Tool {tool_name} not found and no suitable fallback available")

    def _find_similar_tools(self, tool_name: str) -> List[str]:
        """Find similar tools across all servers"""
        if not self.mcp_client:
            return []
        
        similar_tools = []
        
        # Look for tools with similar names
        for available_tool_name in self.mcp_client.tools.keys():
            # Extract the original tool name (remove server prefix)
            original_name = available_tool_name.split('_', 1)[-1] if '_' in available_tool_name else available_tool_name
            
            if original_name == tool_name or tool_name in original_name or original_name in tool_name:
                similar_tools.append(available_tool_name)
        
        # Sort by server priority
        similar_tools.sort(key=lambda x: self._get_server_priority(x))
        
        return similar_tools

    def _get_server_priority(self, tool_name: str) -> int:
        """Get priority score for a tool based on its server"""
        if not self.mcp_client or tool_name not in self.mcp_client.tools:
            return 999
        
        server_id = self.mcp_client.tools[tool_name].server_info.config.server_id
        try:
            return self.server_priorities.index(server_id)
        except ValueError:
            return 999

    async def list_available_tools_by_server(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get organized list of tools grouped by server"""
        if not self.mcp_client:
            return {}
        
        tools_by_server = {}
        all_tools = await self.mcp_client.get_all_tools()
        
        for tool_name, tool_info in all_tools.items():
            server_name = tool_info["server"]
            if server_name not in tools_by_server:
                tools_by_server[server_name] = []
            
            tools_by_server[server_name].append({
                "name": tool_info["name"],
                "full_name": tool_name,
                "description": tool_info["description"],
                "usage_count": tool_info["usage_count"],
                "last_used": tool_info["last_used"]
            })
        
        return tools_by_server

    async def get_execution_statistics(self) -> Dict[str, Any]:
        """Get comprehensive execution statistics"""
        stats = self.execution_stats.copy()
        
        if self.mcp_client:
            # Add server-specific stats
            server_stats = {}
            for server_id in self.mcp_client.servers:
                server_status = await self.mcp_client.get_server_status(server_id)
                if server_status:
                    server_stats[server_id] = {
                        "total_requests": server_status["total_requests"],
                        "successful_requests": server_status["successful_requests"],
                        "failed_requests": server_status["failed_requests"],
                        "success_rate": server_status["success_rate"]
                    }
            
            stats["server_statistics"] = server_stats
            stats["total_tools"] = len(self.mcp_client.tools)
        
        # Calculate success rate
        if stats["total_executions"] > 0:
            stats["success_rate"] = (stats["successful_executions"] / stats["total_executions"]) * 100
        else:
            stats["success_rate"] = 0
        
        stats["last_updated"] = datetime.now().isoformat()
        return stats

    async def run(self, prompt: str, **kwargs) -> str:
        """Enhanced run method with better tool execution and monitoring"""
        if not self.mcp_client:
            return "Enhanced MCP Agent not properly initialized"
        
        # Check server health before execution
        health_status = await self.get_server_health_status()
        if health_status["connected_servers"] == 0:
            return "No MCP servers are currently connected. Please check server configurations."
        
        # Run the standard agent workflow with enhanced tool execution
        try:
            return await super().run(prompt, **kwargs)
        except Exception as e:
            logger.error(f"Enhanced MCP Agent execution failed: {e}")
            return f"Agent execution failed: {str(e)}"

    async def cleanup(self) -> None:
        """Clean up resources and connections"""
        if self.mcp_client:
            await self.mcp_client.shutdown()
            self.mcp_client = None
        
        logger.info("Enhanced MCP Agent cleanup completed")

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()


# Utility functions for creating enhanced MCP agents

async def create_enhanced_mcp_agent_with_default_servers() -> EnhancedMCPAgent:
    """Create an enhanced MCP agent with default server configurations"""
    from app.config import config
    
    server_configs = []
    
    # Add default stdio server if configured
    if hasattr(config, 'mcp_config') and config.mcp_config.server_reference:
        server_configs.append(create_stdio_server_config(
            server_id="default_stdio",
            name="Default MCP Server",
            command="python",
            args=["-m", config.mcp_config.server_reference]
        ))
    
    # Add any additional configured servers
    # This can be extended based on configuration
    
    agent = EnhancedMCPAgent()
    if server_configs:
        await agent.initialize(server_configs)
    
    return agent


async def create_enhanced_mcp_agent_with_custom_servers(
    server_configs: List[ServerConfig]
) -> EnhancedMCPAgent:
    """Create an enhanced MCP agent with custom server configurations"""
    agent = EnhancedMCPAgent()
    await agent.initialize(server_configs)
    return agent
