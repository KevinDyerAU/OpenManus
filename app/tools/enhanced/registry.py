"""
Enhanced MCP Tool Registry
Provides dynamic tool loading, management, validation, and plugin system
"""

import asyncio
import importlib
import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Set, Type
from datetime import datetime
import json
import hashlib

from .interfaces import (
    IMcpToolRegistry, McpToolSchema, McpListToolParam, McpListToolResult,
    McpToolParameter, McpSecurityLevel, McpCallToolParam, McpToolResult
)


class ToolRegistryError(Exception):
    """Base exception for tool registry errors"""
    pass


class ToolValidationError(ToolRegistryError):
    """Exception raised when tool validation fails"""
    pass


class ToolLoadingError(ToolRegistryError):
    """Exception raised when tool loading fails"""
    pass


class McpToolRegistry(IMcpToolRegistry):
    """
    Enhanced MCP Tool Registry with dynamic loading and plugin support
    """
    
    def __init__(self, plugin_directories: Optional[List[str]] = None):
        """Initialize tool registry with optional plugin directories"""
        self.tools: Dict[str, McpToolSchema] = {}
        self.handlers: Dict[str, Callable] = {}
        self.plugin_directories = plugin_directories or []
        self.loaded_plugins: Set[str] = set()
        self.tool_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Security and validation
        self.allowed_security_levels: Set[McpSecurityLevel] = {
            McpSecurityLevel.PUBLIC,
            McpSecurityLevel.AUTHENTICATED
        }
        self.tool_validators: Dict[str, Callable] = {}
        
        # Monitoring
        self.tool_load_history: List[Dict[str, Any]] = []
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize with built-in tools
        asyncio.create_task(self._load_builtin_tools())
    
    async def register_tool(self, schema: McpToolSchema, handler: Callable) -> bool:
        """Register a tool in the registry"""
        try:
            # Validate tool schema
            await self._validate_tool_schema(schema)
            
            # Validate handler
            await self._validate_tool_handler(schema.name, handler)
            
            # Check for conflicts
            if schema.name in self.tools:
                existing_version = self.tools[schema.name].version
                if schema.version <= existing_version:
                    self.logger.warning(
                        f"Tool '{schema.name}' version {schema.version} is not newer than existing version {existing_version}"
                    )
                    return False
            
            # Register tool
            self.tools[schema.name] = schema
            self.handlers[schema.name] = handler
            
            # Store metadata
            self.tool_metadata[schema.name] = {
                "registered_at": datetime.now().isoformat(),
                "handler_module": handler.__module__ if hasattr(handler, "__module__") else None,
                "handler_name": handler.__name__ if hasattr(handler, "__name__") else None,
                "schema_hash": self._calculate_schema_hash(schema)
            }
            
            # Record in history
            self.tool_load_history.append({
                "action": "register",
                "tool_name": schema.name,
                "version": schema.version,
                "timestamp": datetime.now().isoformat(),
                "success": True
            })
            
            self.logger.info(f"Successfully registered tool: {schema.name} v{schema.version}")
            return True
            
        except Exception as e:
            # Record failure in history
            self.tool_load_history.append({
                "action": "register",
                "tool_name": schema.name,
                "version": schema.version,
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "error": str(e)
            })
            
            self.logger.error(f"Failed to register tool '{schema.name}': {e}")
            return False
    
    async def unregister_tool(self, tool_name: str) -> bool:
        """Unregister a tool from the registry"""
        try:
            if tool_name not in self.tools:
                self.logger.warning(f"Tool '{tool_name}' not found in registry")
                return False
            
            # Remove tool and handler
            del self.tools[tool_name]
            del self.handlers[tool_name]
            
            # Remove metadata
            if tool_name in self.tool_metadata:
                del self.tool_metadata[tool_name]
            
            # Remove validator if exists
            if tool_name in self.tool_validators:
                del self.tool_validators[tool_name]
            
            # Record in history
            self.tool_load_history.append({
                "action": "unregister",
                "tool_name": tool_name,
                "timestamp": datetime.now().isoformat(),
                "success": True
            })
            
            self.logger.info(f"Successfully unregistered tool: {tool_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unregister tool '{tool_name}': {e}")
            return False
    
    async def get_tool(self, tool_name: str) -> Optional[McpToolSchema]:
        """Get tool schema by name"""
        return self.tools.get(tool_name)
    
    async def list_tools(self, params: McpListToolParam) -> McpListToolResult:
        """List tools with filtering"""
        filtered_tools = []
        
        for tool in self.tools.values():
            # Apply filters
            if params.category and tool.category != params.category:
                continue
            
            if params.tags and not any(tag in tool.tags for tag in params.tags):
                continue
            
            if params.security_level and tool.security_level != params.security_level:
                continue
            
            if not params.include_deprecated and tool.deprecated:
                continue
            
            if not params.include_experimental and tool.experimental:
                continue
            
            filtered_tools.append(tool)
        
        # Collect metadata
        categories = list(set(tool.category for tool in self.tools.values()))
        available_tags = list(set(tag for tool in self.tools.values() for tag in tool.tags))
        
        return McpListToolResult(
            tools=filtered_tools,
            total_count=len(self.tools),
            filtered_count=len(filtered_tools),
            categories=categories,
            available_tags=available_tags
        )
    
    async def update_tool(self, tool_name: str, schema: McpToolSchema) -> bool:
        """Update existing tool schema"""
        if tool_name not in self.tools:
            self.logger.error(f"Tool '{tool_name}' not found for update")
            return False
        
        try:
            # Validate updated schema
            await self._validate_tool_schema(schema)
            
            # Keep existing handler
            handler = self.handlers[tool_name]
            
            # Update tool
            self.tools[tool_name] = schema
            
            # Update metadata
            self.tool_metadata[tool_name].update({
                "updated_at": datetime.now().isoformat(),
                "schema_hash": self._calculate_schema_hash(schema)
            })
            
            self.logger.info(f"Successfully updated tool: {tool_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update tool '{tool_name}': {e}")
            return False
    
    async def get_tool_handler(self, tool_name: str) -> Optional[Callable]:
        """Get tool execution handler"""
        return self.handlers.get(tool_name)
    
    async def load_plugins_from_directory(self, directory: str) -> int:
        """Load plugins from a directory"""
        if not os.path.exists(directory):
            self.logger.warning(f"Plugin directory does not exist: {directory}")
            return 0
        
        loaded_count = 0
        plugin_path = Path(directory)
        
        for plugin_file in plugin_path.glob("*.py"):
            if plugin_file.name.startswith("__"):
                continue
            
            try:
                await self._load_plugin_file(plugin_file)
                loaded_count += 1
            except Exception as e:
                self.logger.error(f"Failed to load plugin {plugin_file}: {e}")
        
        self.logger.info(f"Loaded {loaded_count} plugins from {directory}")
        return loaded_count
    
    async def reload_plugin(self, plugin_name: str) -> bool:
        """Reload a specific plugin"""
        try:
            # Find plugin file
            plugin_file = None
            for directory in self.plugin_directories:
                potential_file = Path(directory) / f"{plugin_name}.py"
                if potential_file.exists():
                    plugin_file = potential_file
                    break
            
            if not plugin_file:
                self.logger.error(f"Plugin file not found: {plugin_name}")
                return False
            
            # Unregister existing tools from this plugin
            tools_to_remove = []
            for tool_name, metadata in self.tool_metadata.items():
                if metadata.get("plugin_name") == plugin_name:
                    tools_to_remove.append(tool_name)
            
            for tool_name in tools_to_remove:
                await self.unregister_tool(tool_name)
            
            # Reload plugin
            await self._load_plugin_file(plugin_file)
            
            self.logger.info(f"Successfully reloaded plugin: {plugin_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to reload plugin '{plugin_name}': {e}")
            return False
    
    async def get_tool_metadata(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific tool"""
        return self.tool_metadata.get(tool_name)
    
    async def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        total_tools = len(self.tools)
        categories = {}
        security_levels = {}
        
        for tool in self.tools.values():
            # Count by category
            categories[tool.category] = categories.get(tool.category, 0) + 1
            
            # Count by security level
            level = tool.security_level.value
            security_levels[level] = security_levels.get(level, 0) + 1
        
        return {
            "total_tools": total_tools,
            "loaded_plugins": len(self.loaded_plugins),
            "categories": categories,
            "security_levels": security_levels,
            "plugin_directories": self.plugin_directories,
            "load_history_count": len(self.tool_load_history)
        }
    
    async def validate_tool_call(self, tool_name: str, params: McpCallToolParam) -> bool:
        """Validate a tool call before execution"""
        tool = self.tools.get(tool_name)
        if not tool:
            return False
        
        try:
            # Check security level
            if tool.security_level not in self.allowed_security_levels:
                self.logger.warning(f"Tool '{tool_name}' security level not allowed: {tool.security_level}")
                return False
            
            # Check if tool is deprecated
            if tool.deprecated:
                self.logger.warning(f"Tool '{tool_name}' is deprecated")
                return False
            
            # Validate parameters
            await self._validate_tool_parameters(tool, params.arguments)
            
            # Run custom validator if exists
            if tool_name in self.tool_validators:
                validator = self.tool_validators[tool_name]
                if not await validator(params):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Tool call validation failed for '{tool_name}': {e}")
            return False
    
    def set_allowed_security_levels(self, levels: Set[McpSecurityLevel]) -> None:
        """Set allowed security levels for tool execution"""
        self.allowed_security_levels = levels
        self.logger.info(f"Updated allowed security levels: {[level.value for level in levels]}")
    
    def add_tool_validator(self, tool_name: str, validator: Callable) -> None:
        """Add custom validator for a specific tool"""
        self.tool_validators[tool_name] = validator
        self.logger.info(f"Added custom validator for tool: {tool_name}")
    
    def remove_tool_validator(self, tool_name: str) -> None:
        """Remove custom validator for a specific tool"""
        if tool_name in self.tool_validators:
            del self.tool_validators[tool_name]
            self.logger.info(f"Removed custom validator for tool: {tool_name}")
    
    # Private methods
    
    async def _load_builtin_tools(self) -> None:
        """Load built-in tools"""
        try:
            # Load built-in tools from the tools directory
            builtin_tools_dir = Path(__file__).parent / "tools"
            if builtin_tools_dir.exists():
                await self.load_plugins_from_directory(str(builtin_tools_dir))
        except Exception as e:
            self.logger.error(f"Failed to load built-in tools: {e}")
    
    async def _load_plugin_file(self, plugin_file: Path) -> None:
        """Load a single plugin file"""
        plugin_name = plugin_file.stem
        
        if plugin_name in self.loaded_plugins:
            # Reload existing plugin
            module_name = f"mcp_plugin_{plugin_name}"
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
        
        try:
            # Add plugin directory to Python path
            plugin_dir = str(plugin_file.parent)
            if plugin_dir not in sys.path:
                sys.path.insert(0, plugin_dir)
            
            # Import plugin module
            spec = importlib.util.spec_from_file_location(f"mcp_plugin_{plugin_name}", plugin_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Look for tool definitions
            await self._extract_tools_from_module(module, plugin_name)
            
            self.loaded_plugins.add(plugin_name)
            
        except Exception as e:
            raise ToolLoadingError(f"Failed to load plugin {plugin_name}: {e}")
    
    async def _extract_tools_from_module(self, module, plugin_name: str) -> None:
        """Extract tool definitions from a plugin module"""
        # Look for TOOLS list or individual tool functions
        if hasattr(module, "TOOLS"):
            tools_list = getattr(module, "TOOLS")
            for tool_def in tools_list:
                if isinstance(tool_def, dict):
                    await self._register_tool_from_dict(tool_def, module, plugin_name)
                elif isinstance(tool_def, tuple) and len(tool_def) == 2:
                    schema, handler = tool_def
                    await self._register_tool_with_plugin_metadata(schema, handler, plugin_name)
        
        # Look for functions with tool decorators
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) and hasattr(obj, "_mcp_tool_schema"):
                schema = obj._mcp_tool_schema
                await self._register_tool_with_plugin_metadata(schema, obj, plugin_name)
    
    async def _register_tool_from_dict(self, tool_def: Dict[str, Any], module, plugin_name: str) -> None:
        """Register a tool from dictionary definition"""
        try:
            # Create schema from dictionary
            schema = self._create_schema_from_dict(tool_def)
            
            # Get handler function
            handler_name = tool_def.get("handler")
            if not handler_name or not hasattr(module, handler_name):
                raise ToolValidationError(f"Handler '{handler_name}' not found in plugin {plugin_name}")
            
            handler = getattr(module, handler_name)
            
            await self._register_tool_with_plugin_metadata(schema, handler, plugin_name)
            
        except Exception as e:
            raise ToolLoadingError(f"Failed to register tool from dict in plugin {plugin_name}: {e}")
    
    async def _register_tool_with_plugin_metadata(self, schema: McpToolSchema, handler: Callable, plugin_name: str) -> None:
        """Register a tool with plugin metadata"""
        success = await self.register_tool(schema, handler)
        if success and schema.name in self.tool_metadata:
            self.tool_metadata[schema.name]["plugin_name"] = plugin_name
    
    def _create_schema_from_dict(self, tool_def: Dict[str, Any]) -> McpToolSchema:
        """Create tool schema from dictionary definition"""
        parameters = []
        for param_def in tool_def.get("parameters", []):
            param = McpToolParameter(
                name=param_def["name"],
                type=param_def["type"],
                description=param_def["description"],
                required=param_def.get("required", False),
                default=param_def.get("default"),
                enum=param_def.get("enum"),
                pattern=param_def.get("pattern"),
                minimum=param_def.get("minimum"),
                maximum=param_def.get("maximum"),
                security_level=McpSecurityLevel(param_def.get("security_level", "public")),
                validation_rules=param_def.get("validation_rules", [])
            )
            parameters.append(param)
        
        return McpToolSchema(
            name=tool_def["name"],
            description=tool_def["description"],
            parameters=parameters,
            returns=tool_def.get("returns", {}),
            version=tool_def.get("version", "1.0.0"),
            category=tool_def.get("category", "general"),
            tags=tool_def.get("tags", []),
            security_level=McpSecurityLevel(tool_def.get("security_level", "public")),
            rate_limit=tool_def.get("rate_limit"),
            timeout=tool_def.get("timeout"),
            deprecated=tool_def.get("deprecated", False),
            experimental=tool_def.get("experimental", False)
        )
    
    async def _validate_tool_schema(self, schema: McpToolSchema) -> None:
        """Validate tool schema"""
        if not schema.name:
            raise ToolValidationError("Tool name is required")
        
        if not schema.description:
            raise ToolValidationError("Tool description is required")
        
        # Validate parameters
        param_names = set()
        for param in schema.parameters:
            if param.name in param_names:
                raise ToolValidationError(f"Duplicate parameter name: {param.name}")
            param_names.add(param.name)
            
            if not param.name or not param.type:
                raise ToolValidationError("Parameter name and type are required")
        
        # Validate security level
        if schema.security_level not in McpSecurityLevel:
            raise ToolValidationError(f"Invalid security level: {schema.security_level}")
    
    async def _validate_tool_handler(self, tool_name: str, handler: Callable) -> None:
        """Validate tool handler function"""
        if not callable(handler):
            raise ToolValidationError(f"Handler for tool '{tool_name}' is not callable")
        
        # Check function signature
        sig = inspect.signature(handler)
        if len(sig.parameters) == 0:
            raise ToolValidationError(f"Handler for tool '{tool_name}' must accept at least one parameter")
    
    async def _validate_tool_parameters(self, tool: McpToolSchema, arguments: Dict[str, Any]) -> None:
        """Validate tool call parameters"""
        # Check required parameters
        for param in tool.parameters:
            if param.required and param.name not in arguments:
                raise ToolValidationError(f"Required parameter '{param.name}' is missing")
        
        # Validate parameter values
        for param_name, value in arguments.items():
            param = next((p for p in tool.parameters if p.name == param_name), None)
            if not param:
                continue  # Allow extra parameters
            
            # Type validation (basic)
            if param.type == "string" and not isinstance(value, str):
                raise ToolValidationError(f"Parameter '{param_name}' must be a string")
            elif param.type == "integer" and not isinstance(value, int):
                raise ToolValidationError(f"Parameter '{param_name}' must be an integer")
            elif param.type == "number" and not isinstance(value, (int, float)):
                raise ToolValidationError(f"Parameter '{param_name}' must be a number")
            elif param.type == "boolean" and not isinstance(value, bool):
                raise ToolValidationError(f"Parameter '{param_name}' must be a boolean")
            
            # Enum validation
            if param.enum and value not in param.enum:
                raise ToolValidationError(f"Parameter '{param_name}' must be one of: {param.enum}")
            
            # Range validation
            if param.minimum is not None and isinstance(value, (int, float)) and value < param.minimum:
                raise ToolValidationError(f"Parameter '{param_name}' must be >= {param.minimum}")
            
            if param.maximum is not None and isinstance(value, (int, float)) and value > param.maximum:
                raise ToolValidationError(f"Parameter '{param_name}' must be <= {param.maximum}")
    
    def _calculate_schema_hash(self, schema: McpToolSchema) -> str:
        """Calculate hash of tool schema for change detection"""
        schema_dict = {
            "name": schema.name,
            "description": schema.description,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required
                }
                for p in schema.parameters
            ],
            "version": schema.version
        }
        
        schema_json = json.dumps(schema_dict, sort_keys=True)
        return hashlib.md5(schema_json.encode()).hexdigest()


# Decorator for easy tool registration
def mcp_tool(name: str, description: str, category: str = "general", 
             security_level: McpSecurityLevel = McpSecurityLevel.PUBLIC,
             **kwargs):
    """Decorator to mark a function as an MCP tool"""
    def decorator(func):
        # Extract parameters from function signature
        sig = inspect.signature(func)
        parameters = []
        
        for param_name, param in sig.parameters.items():
            if param_name == "context":  # Skip context parameter
                continue
            
            param_type = "string"  # Default type
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
            
            tool_param = McpToolParameter(
                name=param_name,
                type=param_type,
                description=f"Parameter {param_name}",
                required=param.default == inspect.Parameter.empty
            )
            parameters.append(tool_param)
        
        # Create schema
        schema = McpToolSchema(
            name=name,
            description=description,
            parameters=parameters,
            returns={"type": "object"},
            category=category,
            security_level=security_level,
            **kwargs
        )
        
        # Attach schema to function
        func._mcp_tool_schema = schema
        return func
    
    return decorator

