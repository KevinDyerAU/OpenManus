#!/usr/bin/env python
"""
Enhanced MCP Runner for OpenManus
Provides advanced MCP capabilities with multi-server support, health monitoring,
and intelligent tool routing.
"""

import argparse
import asyncio
import json
import sys
from typing import List, Optional

from app.agent.enhanced_mcp import EnhancedMCPAgent, create_enhanced_mcp_agent_with_default_servers
from app.config import config
from app.logger import logger
from app.mcp.enhanced_client import (
    ServerConfig,
    create_stdio_server_config,
    create_sse_server_config,
    ConnectionType
)


class EnhancedMCPRunner:
    """Enhanced runner for MCP Agent with multi-server support and monitoring."""

    def __init__(self):
        self.root_path = config.root_path
        self.agent: Optional[EnhancedMCPAgent] = None

    async def initialize_with_config_file(self, config_file: str) -> None:
        """Initialize with server configurations from a JSON file."""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            server_configs = []
            for server_data in config_data.get('servers', []):
                if server_data['connection_type'] == 'stdio':
                    server_config = create_stdio_server_config(
                        server_id=server_data['server_id'],
                        name=server_data['name'],
                        command=server_data['command'],
                        args=server_data.get('args', []),
                        env=server_data.get('env'),
                        timeout=server_data.get('timeout', 30),
                        max_retries=server_data.get('max_retries', 3),
                        auto_reconnect=server_data.get('auto_reconnect', True)
                    )
                elif server_data['connection_type'] == 'sse':
                    server_config = create_sse_server_config(
                        server_id=server_data['server_id'],
                        name=server_data['name'],
                        url=server_data['url'],
                        timeout=server_data.get('timeout', 30),
                        max_retries=server_data.get('max_retries', 3),
                        auto_reconnect=server_data.get('auto_reconnect', True)
                    )
                else:
                    logger.warning(f"Unsupported connection type: {server_data['connection_type']}")
                    continue
                
                server_configs.append(server_config)
            
            self.agent = EnhancedMCPAgent()
            await self.agent.initialize(server_configs)
            
            logger.info(f"Initialized Enhanced MCP Agent with {len(server_configs)} servers from config file")
            
        except Exception as e:
            logger.error(f"Failed to initialize from config file: {e}")
            raise

    async def initialize_with_single_server(
        self,
        connection_type: str,
        server_url: Optional[str] = None,
        command: Optional[str] = None,
        args: Optional[List[str]] = None
    ) -> None:
        """Initialize with a single server configuration."""
        try:
            if connection_type == "stdio":
                if not command:
                    # Use default server reference from config
                    command = sys.executable
                    args = ["-m", config.mcp_config.server_reference]
                
                server_config = create_stdio_server_config(
                    server_id="single_stdio",
                    name="Single MCP Server (stdio)",
                    command=command,
                    args=args or []
                )
            elif connection_type == "sse":
                if not server_url:
                    raise ValueError("Server URL required for SSE connection")
                
                server_config = create_sse_server_config(
                    server_id="single_sse",
                    name="Single MCP Server (SSE)",
                    url=server_url
                )
            else:
                raise ValueError(f"Unsupported connection type: {connection_type}")
            
            self.agent = EnhancedMCPAgent()
            await self.agent.initialize([server_config])
            
            logger.info(f"Initialized Enhanced MCP Agent with single {connection_type} server")
            
        except Exception as e:
            logger.error(f"Failed to initialize single server: {e}")
            raise

    async def initialize_with_defaults(self) -> None:
        """Initialize with default server configurations."""
        try:
            self.agent = await create_enhanced_mcp_agent_with_default_servers()
            logger.info("Initialized Enhanced MCP Agent with default servers")
        except Exception as e:
            logger.error(f"Failed to initialize with defaults: {e}")
            raise

    async def run_interactive(self) -> None:
        """Run the agent in interactive mode with enhanced features."""
        if not self.agent:
            print("Agent not initialized. Please initialize first.")
            return
        
        print("\n" + "="*60)
        print("Enhanced MCP Agent Interactive Mode")
        print("="*60)
        print("Commands:")
        print("  help     - Show this help message")
        print("  status   - Show server health status")
        print("  tools    - List available tools by server")
        print("  stats    - Show execution statistics")
        print("  exit     - Exit the interactive mode")
        print("="*60)
        
        while True:
            try:
                user_input = input("\nEnter your request: ").strip()
                
                if user_input.lower() in ["exit", "quit", "q"]:
                    break
                elif user_input.lower() == "help":
                    print("\nAvailable commands:")
                    print("  help     - Show this help message")
                    print("  status   - Show server health status")
                    print("  tools    - List available tools by server")
                    print("  stats    - Show execution statistics")
                    print("  exit     - Exit the interactive mode")
                    continue
                elif user_input.lower() == "status":
                    await self._show_status()
                    continue
                elif user_input.lower() == "tools":
                    await self._show_tools()
                    continue
                elif user_input.lower() == "stats":
                    await self._show_stats()
                    continue
                elif not user_input:
                    continue
                
                # Execute the user's request
                print("\nProcessing request...")
                response = await self.agent.run(user_input)
                print(f"\nAgent Response:\n{response}")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {e}")
                logger.error(f"Interactive mode error: {e}")

    async def _show_status(self) -> None:
        """Show server health status."""
        if not self.agent:
            print("Agent not initialized.")
            return
        
        print("\n" + "="*50)
        print("SERVER HEALTH STATUS")
        print("="*50)
        
        health_status = await self.agent.get_server_health_status()
        
        print(f"Overall Status: {health_status['overall_status'].upper()}")
        print(f"Connected Servers: {health_status['connected_servers']}/{health_status['total_servers']}")
        print(f"Total Tools Available: {health_status['total_tools']}")
        print(f"Last Check: {health_status['last_check']}")
        
        print("\nServer Details:")
        for server_id, server_info in health_status['servers'].items():
            status_indicator = "🟢" if server_info['status'] == 'connected' else "🔴"
            print(f"  {status_indicator} {server_info['name']} ({server_id})")
            print(f"    Status: {server_info['status']}")
            print(f"    Tools: {server_info['tools_count']}")
            print(f"    Success Rate: {server_info['success_rate']:.1f}%")
            if server_info['last_error']:
                print(f"    Last Error: {server_info['last_error']}")

    async def _show_tools(self) -> None:
        """Show available tools grouped by server."""
        if not self.agent:
            print("Agent not initialized.")
            return
        
        print("\n" + "="*50)
        print("AVAILABLE TOOLS BY SERVER")
        print("="*50)
        
        tools_by_server = await self.agent.list_available_tools_by_server()
        
        for server_name, tools in tools_by_server.items():
            print(f"\n📡 {server_name} ({len(tools)} tools):")
            for tool in tools:
                usage_info = f" (used {tool['usage_count']} times)" if tool['usage_count'] > 0 else ""
                print(f"  • {tool['name']}{usage_info}")
                if tool['description']:
                    print(f"    {tool['description']}")

    async def _show_stats(self) -> None:
        """Show execution statistics."""
        if not self.agent:
            print("Agent not initialized.")
            return
        
        print("\n" + "="*50)
        print("EXECUTION STATISTICS")
        print("="*50)
        
        stats = await self.agent.get_execution_statistics()
        
        print(f"Total Executions: {stats['total_executions']}")
        print(f"Successful: {stats['successful_executions']}")
        print(f"Failed: {stats['failed_executions']}")
        print(f"Success Rate: {stats['success_rate']:.1f}%")
        print(f"Servers Connected: {stats['servers_connected']}")
        print(f"Tools Available: {stats['total_tools']}")
        
        if 'server_statistics' in stats:
            print("\nServer Statistics:")
            for server_id, server_stats in stats['server_statistics'].items():
                print(f"  {server_id}:")
                print(f"    Requests: {server_stats['total_requests']}")
                print(f"    Success Rate: {server_stats['success_rate']:.1f}%")

    async def run_single_prompt(self, prompt: str) -> None:
        """Run the agent with a single prompt."""
        if not self.agent:
            print("Agent not initialized. Please initialize first.")
            return
        
        try:
            response = await self.agent.run(prompt)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error executing prompt: {e}")
            logger.error(f"Single prompt execution error: {e}")

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.agent:
            await self.agent.cleanup()
            self.agent = None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Enhanced MCP Runner with multi-server support"
    )
    
    parser.add_argument(
        "--mode",
        choices=["interactive", "single"],
        default="interactive",
        help="Run mode: interactive or single prompt"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        help="Single prompt to execute (required for single mode)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to server configuration JSON file"
    )
    
    parser.add_argument(
        "--connection-type",
        choices=["stdio", "sse"],
        default="stdio",
        help="Connection type for single server mode"
    )
    
    parser.add_argument(
        "--server-url",
        type=str,
        help="Server URL for SSE connection"
    )
    
    parser.add_argument(
        "--command",
        type=str,
        help="Command for stdio connection"
    )
    
    parser.add_argument(
        "--args",
        nargs="*",
        help="Arguments for stdio command"
    )
    
    parser.add_argument(
        "--use-defaults",
        action="store_true",
        help="Use default server configurations"
    )
    
    return parser.parse_args()


async def run_enhanced_mcp():
    """Main entry point for the enhanced MCP runner."""
    args = parse_args()
    
    if args.mode == "single" and not args.prompt:
        print("Error: --prompt is required for single mode")
        sys.exit(1)
    
    runner = EnhancedMCPRunner()
    
    try:
        # Initialize based on arguments
        if args.config:
            await runner.initialize_with_config_file(args.config)
        elif args.use_defaults:
            await runner.initialize_with_defaults()
        else:
            await runner.initialize_with_single_server(
                connection_type=args.connection_type,
                server_url=args.server_url,
                command=args.command,
                args=args.args
            )
        
        # Run based on mode
        if args.mode == "interactive":
            await runner.run_interactive()
        else:
            await runner.run_single_prompt(args.prompt)
    
    except Exception as e:
        logger.error(f"Enhanced MCP Runner failed: {e}")
        print(f"Error: {e}")
        sys.exit(1)
    
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(run_enhanced_mcp())
