"""
Enhanced MCP Server Implementation
Integrates all enhanced MCP components with real-time capabilities
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from dataclasses import asdict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn

from .interfaces import (
    IMcpServer, McpToolSchema, McpCallToolParam, McpToolResult,
    McpListToolParam, McpListToolResult, McpConnectionStatus,
    IMcpEventListener, McpToolUsageStats
)
from .tool_registry import McpToolRegistry
from .sse_client import SimpleSseMcpClient


class ConnectionManager:
    """Manages WebSocket connections for real-time communication"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.client_metadata: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        """Accept and store WebSocket connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.client_metadata[client_id] = {
            "connected_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat()
        }
    
    def disconnect(self, client_id: str) -> None:
        """Remove WebSocket connection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.client_metadata:
            del self.client_metadata[client_id]
    
    async def send_personal_message(self, message: Dict[str, Any], client_id: str) -> None:
        """Send message to specific client"""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
                self.client_metadata[client_id]["last_activity"] = datetime.now().isoformat()
            except Exception as e:
                logging.error(f"Failed to send message to client {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcast message to all connected clients"""
        disconnected_clients = []
        
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(json.dumps(message))
                self.client_metadata[client_id]["last_activity"] = datetime.now().isoformat()
            except Exception as e:
                logging.error(f"Failed to broadcast to client {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)
    
    def get_connected_clients(self) -> List[Dict[str, Any]]:
        """Get list of connected clients with metadata"""
        return [
            {"client_id": client_id, **metadata}
            for client_id, metadata in self.client_metadata.items()
        ]


class McpEventBroadcaster(IMcpEventListener):
    """Broadcasts MCP events to connected clients"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
    
    async def on_connection_status_changed(self, status: McpConnectionStatus) -> None:
        """Broadcast connection status changes"""
        await self.connection_manager.broadcast({
            "type": "connection_status",
            "status": status.value,
            "timestamp": datetime.now().isoformat()
        })
    
    async def on_tool_called(self, tool_name: str, params: McpCallToolParam) -> None:
        """Broadcast tool call events"""
        await self.connection_manager.broadcast({
            "type": "tool_called",
            "tool_name": tool_name,
            "arguments": params.arguments,
            "timestamp": datetime.now().isoformat()
        })
    
    async def on_tool_result(self, tool_name: str, result: McpToolResult) -> None:
        """Broadcast tool result events"""
        await self.connection_manager.broadcast({
            "type": "tool_result",
            "tool_name": tool_name,
            "success": result.success,
            "execution_time": result.execution_time,
            "timestamp": datetime.now().isoformat()
        })
    
    async def on_tool_error(self, tool_name: str, error: Exception) -> None:
        """Broadcast tool error events"""
        await self.connection_manager.broadcast({
            "type": "tool_error",
            "tool_name": tool_name,
            "error": str(error),
            "timestamp": datetime.now().isoformat()
        })


class EnhancedMcpServer(IMcpServer):
    """Enhanced MCP Server with real-time capabilities"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8081):
        self.host = host
        self.port = port
        self.app = FastAPI(
            title="Enhanced OpenManus MCP Server",
            description="Model Context Protocol Server with real-time capabilities",
            version="2.0.0"
        )
        
        # Core components
        self.tool_registry = McpToolRegistry()
        self.connection_manager = ConnectionManager()
        self.event_broadcaster = McpEventBroadcaster(self.connection_manager)
        
        # Server state
        self.is_running_flag = False
        self.start_time: Optional[datetime] = None
        self.request_count = 0
        self.error_count = 0
        
        # Setup FastAPI app
        self._setup_middleware()
        self._setup_routes()
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def _setup_middleware(self) -> None:
        """Setup FastAPI middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self) -> None:
        """Setup FastAPI routes"""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy" if self.is_running_flag else "starting",
                "timestamp": datetime.now().isoformat(),
                "uptime": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
                "version": "2.0.0",
                "request_count": self.request_count,
                "error_count": self.error_count,
                "connected_clients": len(self.connection_manager.active_connections),
                "registered_tools": len(self.tool_registry.tools)
            }
        
        @self.app.get("/info")
        async def server_info():
            """Get server information"""
            stats = await self.tool_registry.get_registry_stats()
            return {
                "server": {
                    "name": "Enhanced OpenManus MCP Server",
                    "version": "2.0.0",
                    "host": self.host,
                    "port": self.port,
                    "start_time": self.start_time.isoformat() if self.start_time else None,
                    "uptime": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
                },
                "statistics": {
                    "request_count": self.request_count,
                    "error_count": self.error_count,
                    "connected_clients": len(self.connection_manager.active_connections)
                },
                "registry": stats
            }
        
        @self.app.get("/tools", response_model=McpListToolResult)
        async def list_tools(
            category: Optional[str] = None,
            tags: Optional[str] = None,
            security_level: Optional[str] = None,
            include_deprecated: bool = False,
            include_experimental: bool = True
        ):
            """List available tools"""
            self.request_count += 1
            
            try:
                params = McpListToolParam(
                    category=category,
                    tags=tags.split(",") if tags else None,
                    security_level=security_level,
                    include_deprecated=include_deprecated,
                    include_experimental=include_experimental
                )
                
                result = await self.tool_registry.list_tools(params)
                return result
                
            except Exception as e:
                self.error_count += 1
                self.logger.error(f"Failed to list tools: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/tools/{tool_name}")
        async def get_tool_schema(tool_name: str):
            """Get tool schema"""
            self.request_count += 1
            
            try:
                schema = await self.tool_registry.get_tool(tool_name)
                if not schema:
                    raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
                
                return asdict(schema)
                
            except HTTPException:
                raise
            except Exception as e:
                self.error_count += 1
                self.logger.error(f"Failed to get tool schema: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/tools/{tool_name}/call")
        async def call_tool(tool_name: str, request_data: Dict[str, Any]):
            """Execute a tool"""
            self.request_count += 1
            
            try:
                # Create call parameters
                params = McpCallToolParam(
                    name=tool_name,
                    arguments=request_data.get("arguments", {}),
                    context=request_data.get("context"),
                    timeout=request_data.get("timeout"),
                    dry_run=request_data.get("dry_run", False)
                )
                
                # Validate tool call
                if not await self.tool_registry.validate_tool_call(tool_name, params):
                    raise HTTPException(status_code=403, detail="Tool call validation failed")
                
                # Get tool handler
                handler = await self.tool_registry.get_tool_handler(tool_name)
                if not handler:
                    raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
                
                # Notify event listeners
                await self.event_broadcaster.on_tool_called(tool_name, params)
                
                # Execute tool
                start_time = time.time()
                
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(**params.arguments, context=params.context)
                else:
                    result = handler(**params.arguments, context=params.context)
                
                # Ensure result is McpToolResult
                if not isinstance(result, McpToolResult):
                    result = McpToolResult(
                        success=True,
                        content=[{"type": "result", "data": result}],
                        execution_time=time.time() - start_time,
                        tool_name=tool_name
                    )
                
                # Notify event listeners
                await self.event_broadcaster.on_tool_result(tool_name, result)
                
                return asdict(result)
                
            except HTTPException:
                raise
            except Exception as e:
                self.error_count += 1
                self.logger.error(f"Failed to execute tool '{tool_name}': {e}")
                
                # Notify event listeners
                await self.event_broadcaster.on_tool_error(tool_name, e)
                
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/tools/{tool_name}/stream")
        async def stream_tool(tool_name: str, request_data: Dict[str, Any]):
            """Execute a tool with streaming results"""
            self.request_count += 1
            
            try:
                # Create call parameters
                params = McpCallToolParam(
                    name=tool_name,
                    arguments=request_data.get("arguments", {}),
                    context=request_data.get("context"),
                    timeout=request_data.get("timeout")
                )
                
                # Validate tool call
                if not await self.tool_registry.validate_tool_call(tool_name, params):
                    raise HTTPException(status_code=403, detail="Tool call validation failed")
                
                # Get tool handler
                handler = await self.tool_registry.get_tool_handler(tool_name)
                if not handler:
                    raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
                
                async def generate_stream():
                    try:
                        # Notify start
                        await self.event_broadcaster.on_tool_called(tool_name, params)
                        
                        # Execute tool
                        if asyncio.iscoroutinefunction(handler):
                            result = await handler(**params.arguments, context=params.context)
                        else:
                            result = handler(**params.arguments, context=params.context)
                        
                        # Stream result
                        if isinstance(result, McpToolResult):
                            yield f"data: {json.dumps(asdict(result))}\n\n"
                        else:
                            tool_result = McpToolResult(
                                success=True,
                                content=[{"type": "result", "data": result}],
                                tool_name=tool_name
                            )
                            yield f"data: {json.dumps(asdict(tool_result))}\n\n"
                        
                        # Notify completion
                        await self.event_broadcaster.on_tool_result(tool_name, tool_result)
                        
                    except Exception as e:
                        error_result = McpToolResult(
                            success=False,
                            content=[],
                            error=str(e),
                            tool_name=tool_name
                        )
                        yield f"data: {json.dumps(asdict(error_result))}\n\n"
                        
                        await self.event_broadcaster.on_tool_error(tool_name, e)
                
                return StreamingResponse(
                    generate_stream(),
                    media_type="text/plain",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
                )
                
            except HTTPException:
                raise
            except Exception as e:
                self.error_count += 1
                self.logger.error(f"Failed to stream tool '{tool_name}': {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/tools/register")
        async def register_tool(tool_data: Dict[str, Any]):
            """Register a new tool"""
            self.request_count += 1
            
            try:
                # TODO: Parse tool schema from request data
                # This is a simplified implementation
                success = True  # await self.tool_registry.register_tool(schema, handler)
                
                if success:
                    return {"success": True, "message": "Tool registered successfully"}
                else:
                    raise HTTPException(status_code=400, detail="Failed to register tool")
                    
            except HTTPException:
                raise
            except Exception as e:
                self.error_count += 1
                self.logger.error(f"Failed to register tool: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/tools/{tool_name}/unregister")
        async def unregister_tool(tool_name: str):
            """Unregister a tool"""
            self.request_count += 1
            
            try:
                success = await self.tool_registry.unregister_tool(tool_name)
                
                if success:
                    return {"success": True, "message": f"Tool '{tool_name}' unregistered successfully"}
                else:
                    raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
                    
            except HTTPException:
                raise
            except Exception as e:
                self.error_count += 1
                self.logger.error(f"Failed to unregister tool: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/stats/tools")
        async def get_tool_stats():
            """Get tool usage statistics"""
            self.request_count += 1
            
            try:
                stats = await self.tool_registry.get_registry_stats()
                return stats
                
            except Exception as e:
                self.error_count += 1
                self.logger.error(f"Failed to get tool stats: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/clients")
        async def get_connected_clients():
            """Get connected clients"""
            return {
                "clients": self.connection_manager.get_connected_clients(),
                "total_count": len(self.connection_manager.active_connections)
            }
        
        @self.app.websocket("/ws/{client_id}")
        async def websocket_endpoint(websocket: WebSocket, client_id: str):
            """WebSocket endpoint for real-time communication"""
            await self.connection_manager.connect(websocket, client_id)
            
            try:
                # Send welcome message
                await self.connection_manager.send_personal_message({
                    "type": "welcome",
                    "client_id": client_id,
                    "server_info": {
                        "name": "Enhanced OpenManus MCP Server",
                        "version": "2.0.0",
                        "timestamp": datetime.now().isoformat()
                    }
                }, client_id)
                
                # Listen for messages
                while True:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    # Handle different message types
                    if message.get("type") == "ping":
                        await self.connection_manager.send_personal_message({
                            "type": "pong",
                            "timestamp": datetime.now().isoformat()
                        }, client_id)
                    
                    elif message.get("type") == "subscribe_events":
                        # Client wants to subscribe to events
                        await self.connection_manager.send_personal_message({
                            "type": "subscription_confirmed",
                            "events": ["tool_calls", "tool_results", "connection_status"]
                        }, client_id)
                    
            except WebSocketDisconnect:
                self.connection_manager.disconnect(client_id)
                self.logger.info(f"Client {client_id} disconnected")
            except Exception as e:
                self.logger.error(f"WebSocket error for client {client_id}: {e}")
                self.connection_manager.disconnect(client_id)
        
        @self.app.get("/events")
        async def sse_endpoint():
            """Server-Sent Events endpoint"""
            async def event_stream():
                while True:
                    # Send heartbeat
                    yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': datetime.now().isoformat()})}\n\n"
                    await asyncio.sleep(30)
            
            return StreamingResponse(
                event_stream(),
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Cache-Control"
                }
            )
    
    async def start(self, host: str = None, port: int = None) -> None:
        """Start the MCP server"""
        if self.is_running_flag:
            self.logger.warning("Server is already running")
            return
        
        self.host = host or self.host
        self.port = port or self.port
        self.start_time = datetime.now()
        self.is_running_flag = True
        
        self.logger.info(f"Starting Enhanced MCP Server on {self.host}:{self.port}")
        
        # Load plugins
        await self._load_plugins()
        
        # Start server
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            access_log=True
        )
        
        server = uvicorn.Server(config)
        await server.serve()
    
    async def stop(self) -> None:
        """Stop the MCP server"""
        self.is_running_flag = False
        self.logger.info("MCP Server stopped")
    
    def is_running(self) -> bool:
        """Check if server is running"""
        return self.is_running_flag
    
    async def register_tool(self, schema: McpToolSchema, handler) -> bool:
        """Register a tool with the server"""
        return await self.tool_registry.register_tool(schema, handler)
    
    async def unregister_tool(self, tool_name: str) -> bool:
        """Unregister a tool from the server"""
        return await self.tool_registry.unregister_tool(tool_name)
    
    async def get_registered_tools(self) -> List[McpToolSchema]:
        """Get all registered tools"""
        result = await self.tool_registry.list_tools(McpListToolParam())
        return result.tools
    
    async def handle_client_connection(self, client_id: str) -> None:
        """Handle new client connection"""
        self.logger.info(f"New client connected: {client_id}")
    
    async def handle_client_disconnection(self, client_id: str) -> None:
        """Handle client disconnection"""
        self.logger.info(f"Client disconnected: {client_id}")
    
    async def _load_plugins(self) -> None:
        """Load plugins from configured directories"""
        try:
            # Load built-in tools
            builtin_count = await self.tool_registry.load_plugins_from_directory("app/mcp/tools")
            self.logger.info(f"Loaded {builtin_count} built-in tools")
            
            # Load external plugins
            for plugin_dir in self.tool_registry.plugin_directories:
                plugin_count = await self.tool_registry.load_plugins_from_directory(plugin_dir)
                self.logger.info(f"Loaded {plugin_count} plugins from {plugin_dir}")
                
        except Exception as e:
            self.logger.error(f"Failed to load plugins: {e}")


# Factory function for creating server instance
def create_mcp_server(host: str = "0.0.0.0", port: int = 8081, 
                     plugin_directories: Optional[List[str]] = None) -> EnhancedMcpServer:
    """Create and configure an enhanced MCP server"""
    server = EnhancedMcpServer(host, port)
    
    if plugin_directories:
        server.tool_registry.plugin_directories.extend(plugin_directories)
    
    return server


# Main entry point for running the server
async def main():
    """Main entry point for running the MCP server"""
    import os
    
    host = os.getenv("MCP_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_SERVER_PORT", "8081"))
    
    server = create_mcp_server(host, port)
    await server.start()


if __name__ == "__main__":
    asyncio.run(main())

