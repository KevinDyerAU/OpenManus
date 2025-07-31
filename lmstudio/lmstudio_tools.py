"""
LM Studio MCP Tools for OpenManus

This module provides MCP (Model Context Protocol) tools specifically designed
for LM Studio integration, including model management and local inference capabilities.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
import json
from datetime import datetime

try:
    import lmstudio as lms
    LMSTUDIO_AVAILABLE = True
except ImportError:
    LMSTUDIO_AVAILABLE = False
    lms = None

from ..mcp.interfaces import MCPTool, ToolParameter, ToolResult
from .lmstudio_provider import LMStudioProvider, LMStudioConfig

logger = logging.getLogger(__name__)


class LMStudioModelManager(MCPTool):
    """MCP tool for managing LM Studio models"""
    
    def __init__(self, provider: LMStudioProvider):
        self.provider = provider
        super().__init__(
            name="lmstudio_model_manager",
            description="Manage LM Studio models - list, load, and configure local models",
            parameters=[
                ToolParameter(
                    name="action",
                    type="string",
                    description="Action to perform: 'list', 'load', 'unload', 'info'",
                    required=True,
                    enum=["list", "load", "unload", "info"]
                ),
                ToolParameter(
                    name="model_id",
                    type="string", 
                    description="Model identifier (required for load, unload, info actions)",
                    required=False
                ),
                ToolParameter(
                    name="config",
                    type="object",
                    description="Model configuration parameters (for load action)",
                    required=False
                )
            ]
        )
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute model management action"""
        if not LMSTUDIO_AVAILABLE:
            return ToolResult(
                success=False,
                error="LM Studio is not available. Install with: pip install lmstudio"
            )
        
        action = parameters.get("action")
        model_id = parameters.get("model_id")
        config = parameters.get("config", {})
        
        try:
            if action == "list":
                models = await self.provider.get_available_models()
                return ToolResult(
                    success=True,
                    data={
                        "action": "list",
                        "models": models,
                        "count": len(models)
                    },
                    message=f"Found {len(models)} available models"
                )
            
            elif action == "info":
                if not model_id:
                    return ToolResult(
                        success=False,
                        error="model_id is required for info action"
                    )
                
                model_info = await self.provider.get_model_info(model_id)
                return ToolResult(
                    success=True,
                    data={
                        "action": "info",
                        "model": model_info
                    },
                    message=f"Retrieved info for model: {model_id}"
                )
            
            elif action == "load":
                if not model_id:
                    return ToolResult(
                        success=False,
                        error="model_id is required for load action"
                    )
                
                # Try to load the model
                try:
                    model = lms.llm(model_id)
                    # Test with a simple response
                    test_response = model.respond("Hello")
                    
                    return ToolResult(
                        success=True,
                        data={
                            "action": "load",
                            "model_id": model_id,
                            "config": config,
                            "test_response": test_response[:50] + "..." if len(test_response) > 50 else test_response
                        },
                        message=f"Successfully loaded model: {model_id}"
                    )
                except Exception as e:
                    return ToolResult(
                        success=False,
                        error=f"Failed to load model {model_id}: {str(e)}"
                    )
            
            elif action == "unload":
                # Note: LM Studio SDK doesn't provide explicit unload functionality
                # This is a placeholder for future implementation
                return ToolResult(
                    success=True,
                    data={
                        "action": "unload",
                        "model_id": model_id,
                        "message": "Model unload requested (handled by LM Studio server)"
                    },
                    message=f"Unload requested for model: {model_id}"
                )
            
            else:
                return ToolResult(
                    success=False,
                    error=f"Unknown action: {action}"
                )
                
        except Exception as e:
            logger.error(f"LM Studio model management error: {e}")
            return ToolResult(
                success=False,
                error=f"Model management failed: {str(e)}"
            )


class LMStudioChat(MCPTool):
    """MCP tool for LM Studio chat interactions"""
    
    def __init__(self, provider: LMStudioProvider):
        self.provider = provider
        super().__init__(
            name="lmstudio_chat",
            description="Chat with LM Studio models using local inference",
            parameters=[
                ToolParameter(
                    name="message",
                    type="string",
                    description="User message to send to the model",
                    required=True
                ),
                ToolParameter(
                    name="model",
                    type="string",
                    description="Model to use for the chat (optional, uses default if not specified)",
                    required=False
                ),
                ToolParameter(
                    name="system_prompt",
                    type="string",
                    description="System prompt to set the model's behavior",
                    required=False
                ),
                ToolParameter(
                    name="conversation_history",
                    type="array",
                    description="Previous conversation messages",
                    required=False
                ),
                ToolParameter(
                    name="temperature",
                    type="number",
                    description="Sampling temperature (0.0-2.0)",
                    required=False
                ),
                ToolParameter(
                    name="max_tokens",
                    type="integer",
                    description="Maximum tokens to generate",
                    required=False
                ),
                ToolParameter(
                    name="stream",
                    type="boolean",
                    description="Whether to stream the response",
                    required=False,
                    default=False
                )
            ]
        )
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute chat interaction"""
        if not LMSTUDIO_AVAILABLE:
            return ToolResult(
                success=False,
                error="LM Studio is not available. Install with: pip install lmstudio"
            )
        
        message = parameters.get("message")
        model = parameters.get("model")
        system_prompt = parameters.get("system_prompt")
        conversation_history = parameters.get("conversation_history", [])
        temperature = parameters.get("temperature")
        max_tokens = parameters.get("max_tokens")
        stream = parameters.get("stream", False)
        
        try:
            # Build message list
            messages = list(conversation_history)
            messages.append({"role": "user", "content": message})
            
            if stream:
                # Handle streaming response
                response_chunks = []
                async for chunk in self.provider.generate_streaming_response(
                    messages=messages,
                    model=model,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                ):
                    if chunk.content:
                        response_chunks.append(chunk.content)
                
                full_response = "".join(response_chunks)
                
                return ToolResult(
                    success=True,
                    data={
                        "response": full_response,
                        "model": model or self.provider.config.default_model,
                        "provider": "lmstudio",
                        "stream": True,
                        "chunks": len(response_chunks),
                        "conversation_history": messages + [{"role": "assistant", "content": full_response}]
                    },
                    message="Chat response generated successfully (streamed)"
                )
            else:
                # Handle regular response
                response = await self.provider.generate_response(
                    messages=messages,
                    model=model,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                return ToolResult(
                    success=True,
                    data={
                        "response": response.content,
                        "model": response.model,
                        "provider": response.provider,
                        "response_time": response.response_time,
                        "usage": response.usage,
                        "conversation_history": messages + [{"role": "assistant", "content": response.content}]
                    },
                    message="Chat response generated successfully"
                )
                
        except Exception as e:
            logger.error(f"LM Studio chat error: {e}")
            return ToolResult(
                success=False,
                error=f"Chat failed: {str(e)}"
            )


class LMStudioServerStatus(MCPTool):
    """MCP tool for checking LM Studio server status"""
    
    def __init__(self, provider: LMStudioProvider):
        self.provider = provider
        super().__init__(
            name="lmstudio_server_status",
            description="Check LM Studio server status and connectivity",
            parameters=[
                ToolParameter(
                    name="detailed",
                    type="boolean",
                    description="Whether to include detailed server information",
                    required=False,
                    default=False
                )
            ]
        )
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Check server status"""
        detailed = parameters.get("detailed", False)
        
        try:
            # Test connection
            connection_test = await self.provider.test_connection()
            
            # Get provider info
            provider_info = self.provider.get_provider_info()
            
            # Get available models if detailed
            models = []
            if detailed:
                models = await self.provider.get_available_models()
            
            return ToolResult(
                success=connection_test["status"] == "success",
                data={
                    "connection": connection_test,
                    "provider": provider_info,
                    "models": models if detailed else [],
                    "timestamp": datetime.now().isoformat()
                },
                message=connection_test["message"]
            )
            
        except Exception as e:
            logger.error(f"LM Studio status check error: {e}")
            return ToolResult(
                success=False,
                error=f"Status check failed: {str(e)}"
            )


class LMStudioConfigManager(MCPTool):
    """MCP tool for managing LM Studio configuration"""
    
    def __init__(self, provider: LMStudioProvider):
        self.provider = provider
        super().__init__(
            name="lmstudio_config",
            description="Manage LM Studio provider configuration",
            parameters=[
                ToolParameter(
                    name="action",
                    type="string",
                    description="Action to perform: 'get', 'update', 'reset'",
                    required=True,
                    enum=["get", "update", "reset"]
                ),
                ToolParameter(
                    name="config",
                    type="object",
                    description="Configuration parameters to update",
                    required=False
                )
            ]
        )
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Manage configuration"""
        action = parameters.get("action")
        config_updates = parameters.get("config", {})
        
        try:
            if action == "get":
                return ToolResult(
                    success=True,
                    data={
                        "action": "get",
                        "config": {
                            "host": self.provider.config.host,
                            "port": self.provider.config.port,
                            "default_model": self.provider.config.default_model,
                            "temperature": self.provider.config.temperature,
                            "max_tokens": self.provider.config.max_tokens,
                            "top_p": self.provider.config.top_p,
                            "timeout": self.provider.config.timeout,
                            "max_retries": self.provider.config.max_retries
                        }
                    },
                    message="Retrieved current configuration"
                )
            
            elif action == "update":
                # Update configuration
                for key, value in config_updates.items():
                    if hasattr(self.provider.config, key):
                        setattr(self.provider.config, key, value)
                
                # Reconfigure client if host/port changed
                if "host" in config_updates or "port" in config_updates:
                    self.provider.server_host = f"{self.provider.config.host}:{self.provider.config.port}"
                    lms.configure_default_client(self.provider.server_host)
                
                return ToolResult(
                    success=True,
                    data={
                        "action": "update",
                        "updated": config_updates,
                        "current_config": {
                            "host": self.provider.config.host,
                            "port": self.provider.config.port,
                            "default_model": self.provider.config.default_model,
                            "temperature": self.provider.config.temperature,
                            "max_tokens": self.provider.config.max_tokens
                        }
                    },
                    message=f"Updated {len(config_updates)} configuration parameters"
                )
            
            elif action == "reset":
                # Reset to default configuration
                self.provider.config = LMStudioConfig()
                self.provider.server_host = f"{self.provider.config.host}:{self.provider.config.port}"
                lms.configure_default_client(self.provider.server_host)
                
                return ToolResult(
                    success=True,
                    data={
                        "action": "reset",
                        "config": {
                            "host": self.provider.config.host,
                            "port": self.provider.config.port,
                            "default_model": self.provider.config.default_model,
                            "temperature": self.provider.config.temperature,
                            "max_tokens": self.provider.config.max_tokens
                        }
                    },
                    message="Configuration reset to defaults"
                )
            
            else:
                return ToolResult(
                    success=False,
                    error=f"Unknown action: {action}"
                )
                
        except Exception as e:
            logger.error(f"LM Studio config management error: {e}")
            return ToolResult(
                success=False,
                error=f"Configuration management failed: {str(e)}"
            )


def create_lmstudio_tools(provider: LMStudioProvider) -> List[MCPTool]:
    """
    Create all LM Studio MCP tools.
    
    Args:
        provider: LM Studio provider instance
        
    Returns:
        List of configured MCP tools
    """
    return [
        LMStudioModelManager(provider),
        LMStudioChat(provider),
        LMStudioServerStatus(provider),
        LMStudioConfigManager(provider)
    ]

