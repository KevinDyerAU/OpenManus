"""
Unit Tests for MCP Components
Tests for enhanced MCP interfaces, tool registry, and client functionality
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from app.mcp.interfaces import (
    MCPTool, MCPToolRegistry, MCPClient, MCPServer,
    SecurityLevel, ToolValidationError, MCPError
)
from app.mcp.tool_registry import EnhancedMCPToolRegistry
from app.mcp.sse_client import SimpleSseMcpClient
from app.mcp.enhanced_server import EnhancedMCPServer
from app.mcp.tools.system_tools import SystemTools


class TestMCPInterfaces:
    """Test MCP interface definitions"""
    
    def test_mcp_tool_creation(self):
        """Test MCPTool creation and validation"""
        tool = MCPTool(
            name="test_tool",
            description="A test tool",
            parameters={
                "type": "object",
                "properties": {
                    "input": {"type": "string"}
                }
            },
            security_level=SecurityLevel.PUBLIC
        )
        
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.security_level == SecurityLevel.PUBLIC
        assert "input" in tool.parameters["properties"]
    
    def test_mcp_tool_validation(self):
        """Test MCPTool parameter validation"""
        tool = MCPTool(
            name="validation_tool",
            description="Tool for testing validation",
            parameters={
                "type": "object",
                "properties": {
                    "required_param": {"type": "string"},
                    "optional_param": {"type": "integer", "default": 42}
                },
                "required": ["required_param"]
            }
        )
        
        # Valid parameters
        valid_params = {"required_param": "test"}
        assert tool.validate_parameters(valid_params) == valid_params
        
        # Invalid parameters - missing required
        with pytest.raises(ToolValidationError):
            tool.validate_parameters({})
        
        # Invalid parameters - wrong type
        with pytest.raises(ToolValidationError):
            tool.validate_parameters({"required_param": 123})
    
    def test_security_levels(self):
        """Test security level enumeration"""
        assert SecurityLevel.PUBLIC.value == "public"
        assert SecurityLevel.AUTHENTICATED.value == "authenticated"
        assert SecurityLevel.RESTRICTED.value == "restricted"
        assert SecurityLevel.ADMIN.value == "admin"
        
        # Test ordering
        assert SecurityLevel.PUBLIC < SecurityLevel.AUTHENTICATED
        assert SecurityLevel.AUTHENTICATED < SecurityLevel.RESTRICTED
        assert SecurityLevel.RESTRICTED < SecurityLevel.ADMIN


class TestEnhancedMCPToolRegistry:
    """Test enhanced MCP tool registry"""
    
    @pytest.fixture
    def registry(self):
        """Create a test registry"""
        return EnhancedMCPToolRegistry()
    
    @pytest.fixture
    def sample_tool(self):
        """Create a sample tool for testing"""
        async def sample_function(input_text: str) -> str:
            return f"Processed: {input_text}"
        
        return MCPTool(
            name="sample_tool",
            description="A sample tool for testing",
            parameters={
                "type": "object",
                "properties": {
                    "input_text": {"type": "string"}
                },
                "required": ["input_text"]
            },
            function=sample_function,
            security_level=SecurityLevel.PUBLIC
        )
    
    def test_tool_registration(self, registry, sample_tool):
        """Test tool registration"""
        registry.register_tool(sample_tool)
        
        assert "sample_tool" in registry.tools
        assert registry.get_tool("sample_tool") == sample_tool
    
    def test_duplicate_tool_registration(self, registry, sample_tool):
        """Test duplicate tool registration handling"""
        registry.register_tool(sample_tool)
        
        # Should raise error for duplicate registration
        with pytest.raises(ValueError, match="Tool .* already registered"):
            registry.register_tool(sample_tool)
    
    def test_tool_discovery(self, registry):
        """Test automatic tool discovery"""
        # Mock tool discovery
        with patch.object(registry, '_discover_tools') as mock_discover:
            mock_discover.return_value = ["discovered_tool"]
            
            discovered = registry.discover_tools("test_module")
            assert "discovered_tool" in discovered
            mock_discover.assert_called_once_with("test_module")
    
    @pytest.mark.asyncio
    async def test_tool_execution(self, registry, sample_tool):
        """Test tool execution"""
        registry.register_tool(sample_tool)
        
        result = await registry.execute_tool(
            "sample_tool",
            {"input_text": "test input"}
        )
        
        assert result == "Processed: test input"
    
    @pytest.mark.asyncio
    async def test_tool_execution_validation(self, registry, sample_tool):
        """Test tool execution with parameter validation"""
        registry.register_tool(sample_tool)
        
        # Invalid parameters should raise error
        with pytest.raises(ToolValidationError):
            await registry.execute_tool("sample_tool", {})
    
    @pytest.mark.asyncio
    async def test_tool_execution_nonexistent(self, registry):
        """Test execution of non-existent tool"""
        with pytest.raises(ValueError, match="Tool .* not found"):
            await registry.execute_tool("nonexistent_tool", {})
    
    def test_tool_filtering_by_security(self, registry):
        """Test tool filtering by security level"""
        # Register tools with different security levels
        public_tool = MCPTool(
            name="public_tool",
            description="Public tool",
            parameters={"type": "object"},
            security_level=SecurityLevel.PUBLIC
        )
        
        admin_tool = MCPTool(
            name="admin_tool",
            description="Admin tool",
            parameters={"type": "object"},
            security_level=SecurityLevel.ADMIN
        )
        
        registry.register_tool(public_tool)
        registry.register_tool(admin_tool)
        
        # Filter by security level
        public_tools = registry.get_tools_by_security(SecurityLevel.PUBLIC)
        assert len(public_tools) == 1
        assert public_tools[0].name == "public_tool"
        
        admin_tools = registry.get_tools_by_security(SecurityLevel.ADMIN)
        assert len(admin_tools) == 2  # Admin can access all tools
    
    def test_tool_usage_analytics(self, registry, sample_tool):
        """Test tool usage analytics"""
        registry.register_tool(sample_tool)
        
        # Initially no usage
        stats = registry.get_usage_stats("sample_tool")
        assert stats["call_count"] == 0
        
        # Record usage
        registry._record_usage("sample_tool", success=True, duration=0.5)
        
        stats = registry.get_usage_stats("sample_tool")
        assert stats["call_count"] == 1
        assert stats["success_count"] == 1
        assert stats["average_duration"] == 0.5


class TestSimpleSseMcpClient:
    """Test SSE MCP client"""
    
    @pytest.fixture
    def client(self):
        """Create a test SSE client"""
        return SimpleSseMcpClient("http://test-server")
    
    @pytest.mark.asyncio
    async def test_client_connection(self, client):
        """Test client connection"""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.content.readline = AsyncMock(
                side_effect=[
                    b'data: {"type": "connection", "status": "connected"}\n\n',
                    b''
                ]
            )
            mock_get.return_value.__aenter__.return_value = mock_response
            
            await client.connect()
            assert client.connected
    
    @pytest.mark.asyncio
    async def test_client_message_handling(self, client):
        """Test client message handling"""
        messages = []
        
        def message_handler(message):
            messages.append(message)
        
        client.on_message = message_handler
        
        # Simulate receiving messages
        test_message = {"type": "tool_result", "result": "test"}
        await client._handle_message(json.dumps(test_message))
        
        assert len(messages) == 1
        assert messages[0] == test_message
    
    @pytest.mark.asyncio
    async def test_client_tool_execution(self, client):
        """Test tool execution through client"""
        with patch.object(client, '_send_message') as mock_send:
            mock_send.return_value = {"result": "success"}
            
            result = await client.execute_tool("test_tool", {"param": "value"})
            
            assert result == {"result": "success"}
            mock_send.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_client_reconnection(self, client):
        """Test automatic reconnection"""
        client.auto_reconnect = True
        client.reconnect_interval = 0.1
        
        with patch.object(client, 'connect') as mock_connect:
            mock_connect.side_effect = [Exception("Connection failed"), None]
            
            await client._handle_disconnect()
            
            # Should attempt reconnection
            assert mock_connect.call_count >= 1


class TestEnhancedMCPServer:
    """Test enhanced MCP server"""
    
    @pytest.fixture
    def server(self):
        """Create a test MCP server"""
        return EnhancedMCPServer()
    
    @pytest.fixture
    def sample_tool(self):
        """Create a sample tool for server testing"""
        async def sample_function(data: str) -> str:
            return f"Server processed: {data}"
        
        return MCPTool(
            name="server_tool",
            description="Server test tool",
            parameters={
                "type": "object",
                "properties": {
                    "data": {"type": "string"}
                },
                "required": ["data"]
            },
            function=sample_function
        )
    
    def test_server_initialization(self, server):
        """Test server initialization"""
        assert server.tool_registry is not None
        assert server.event_handlers == {}
        assert not server.running
    
    def test_tool_registration_on_server(self, server, sample_tool):
        """Test tool registration on server"""
        server.register_tool(sample_tool)
        
        assert "server_tool" in server.tool_registry.tools
    
    @pytest.mark.asyncio
    async def test_server_tool_execution(self, server, sample_tool):
        """Test tool execution on server"""
        server.register_tool(sample_tool)
        
        result = await server.execute_tool("server_tool", {"data": "test"})
        
        assert result == "Server processed: test"
    
    def test_event_handler_registration(self, server):
        """Test event handler registration"""
        def test_handler(event_data):
            return f"Handled: {event_data}"
        
        server.register_event_handler("test_event", test_handler)
        
        assert "test_event" in server.event_handlers
        assert server.event_handlers["test_event"] == test_handler
    
    @pytest.mark.asyncio
    async def test_event_emission(self, server):
        """Test event emission"""
        handled_events = []
        
        def test_handler(event_data):
            handled_events.append(event_data)
        
        server.register_event_handler("test_event", test_handler)
        
        await server.emit_event("test_event", {"message": "test"})
        
        assert len(handled_events) == 1
        assert handled_events[0] == {"message": "test"}
    
    @pytest.mark.asyncio
    async def test_server_websocket_handling(self, server):
        """Test WebSocket message handling"""
        mock_websocket = AsyncMock()
        
        # Mock incoming message
        test_message = {
            "type": "tool_call",
            "tool": "server_tool",
            "parameters": {"data": "websocket test"}
        }
        
        with patch.object(server, 'execute_tool') as mock_execute:
            mock_execute.return_value = "WebSocket result"
            
            await server.handle_websocket_message(mock_websocket, test_message)
            
            mock_execute.assert_called_once_with("server_tool", {"data": "websocket test"})


class TestSystemTools:
    """Test system tools implementation"""
    
    @pytest.fixture
    def system_tools(self):
        """Create system tools instance"""
        return SystemTools()
    
    @pytest.mark.asyncio
    async def test_get_system_info(self, system_tools):
        """Test system information retrieval"""
        info = await system_tools.get_system_info()
        
        assert "platform" in info
        assert "python_version" in info
        assert "cpu_count" in info
        assert "memory_total" in info
    
    @pytest.mark.asyncio
    async def test_list_processes(self, system_tools):
        """Test process listing"""
        processes = await system_tools.list_processes()
        
        assert isinstance(processes, list)
        if processes:  # If any processes are running
            process = processes[0]
            assert "pid" in process
            assert "name" in process
    
    @pytest.mark.asyncio
    async def test_get_resource_usage(self, system_tools):
        """Test resource usage monitoring"""
        usage = await system_tools.get_resource_usage()
        
        assert "cpu_percent" in usage
        assert "memory_percent" in usage
        assert "disk_usage" in usage
        assert isinstance(usage["cpu_percent"], (int, float))
        assert isinstance(usage["memory_percent"], (int, float))
    
    @pytest.mark.asyncio
    async def test_execute_command_safe(self, system_tools):
        """Test safe command execution"""
        # Test safe command
        result = await system_tools.execute_command("echo 'test'")
        
        assert result["success"] is True
        assert "test" in result["output"]
    
    @pytest.mark.asyncio
    async def test_execute_command_unsafe(self, system_tools):
        """Test unsafe command rejection"""
        # Test unsafe command
        result = await system_tools.execute_command("rm -rf /")
        
        assert result["success"] is False
        assert "not allowed" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_health_check(self, system_tools):
        """Test system health check"""
        health = await system_tools.health_check()
        
        assert "status" in health
        assert "checks" in health
        assert health["status"] in ["healthy", "warning", "critical"]


class TestMCPIntegration:
    """Integration tests for MCP components"""
    
    @pytest.mark.asyncio
    async def test_full_mcp_workflow(self):
        """Test complete MCP workflow"""
        # Create registry and server
        registry = EnhancedMCPToolRegistry()
        server = EnhancedMCPServer()
        
        # Create and register a tool
        async def workflow_tool(message: str) -> str:
            return f"Workflow: {message}"
        
        tool = MCPTool(
            name="workflow_tool",
            description="Tool for workflow testing",
            parameters={
                "type": "object",
                "properties": {
                    "message": {"type": "string"}
                },
                "required": ["message"]
            },
            function=workflow_tool
        )
        
        registry.register_tool(tool)
        server.tool_registry = registry
        
        # Execute tool through server
        result = await server.execute_tool("workflow_tool", {"message": "test"})
        
        assert result == "Workflow: test"
    
    @pytest.mark.asyncio
    async def test_mcp_error_handling(self):
        """Test MCP error handling"""
        registry = EnhancedMCPToolRegistry()
        
        # Create a tool that raises an error
        async def error_tool() -> str:
            raise Exception("Tool error")
        
        tool = MCPTool(
            name="error_tool",
            description="Tool that raises errors",
            parameters={"type": "object"},
            function=error_tool
        )
        
        registry.register_tool(tool)
        
        # Should handle error gracefully
        with pytest.raises(Exception):
            await registry.execute_tool("error_tool", {})
    
    @pytest.mark.asyncio
    async def test_mcp_security_enforcement(self):
        """Test MCP security level enforcement"""
        registry = EnhancedMCPToolRegistry()
        
        # Create admin-only tool
        admin_tool = MCPTool(
            name="admin_tool",
            description="Admin only tool",
            parameters={"type": "object"},
            security_level=SecurityLevel.ADMIN
        )
        
        registry.register_tool(admin_tool)
        
        # Test security filtering
        public_tools = registry.get_tools_by_security(SecurityLevel.PUBLIC)
        admin_tools = registry.get_tools_by_security(SecurityLevel.ADMIN)
        
        assert len(public_tools) == 0
        assert len(admin_tools) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

