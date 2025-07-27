"""
Built-in System Tools for OpenManus MCP
Demonstrates enhanced MCP capabilities with various tool types
"""

import asyncio
import os
import platform
import psutil
import subprocess
import time
from datetime import datetime
from typing import Dict, Any, List

from ..tool_registry import mcp_tool
from ..interfaces import McpSecurityLevel, McpCallToolParam, McpToolResult


@mcp_tool(
    name="system_info",
    description="Get comprehensive system information including OS, CPU, memory, and disk usage",
    category="system",
    security_level=McpSecurityLevel.PUBLIC,
    tags=["system", "monitoring", "info"]
)
async def get_system_info(context: Dict[str, Any] = None) -> McpToolResult:
    """Get system information"""
    try:
        # Gather system information
        system_info = {
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "architecture": platform.architecture(),
                "hostname": platform.node()
            },
            "cpu": {
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "current_frequency": psutil.cpu_freq().current if psutil.cpu_freq() else None,
                "usage_percent": psutil.cpu_percent(interval=1)
            },
            "memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "used": psutil.virtual_memory().used,
                "percentage": psutil.virtual_memory().percent
            },
            "disk": {
                "total": psutil.disk_usage('/').total,
                "used": psutil.disk_usage('/').used,
                "free": psutil.disk_usage('/').free,
                "percentage": psutil.disk_usage('/').percent
            },
            "network": {
                "interfaces": list(psutil.net_if_addrs().keys()),
                "stats": dict(psutil.net_io_counters()._asdict())
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return McpToolResult(
            success=True,
            content=[{
                "type": "system_info",
                "data": system_info
            }],
            metadata={
                "tool_version": "1.0.0",
                "data_freshness": "real-time"
            }
        )
        
    except Exception as e:
        return McpToolResult(
            success=False,
            content=[],
            error=f"Failed to get system info: {str(e)}"
        )


@mcp_tool(
    name="execute_command",
    description="Execute a shell command with safety restrictions",
    category="system",
    security_level=McpSecurityLevel.RESTRICTED,
    tags=["system", "command", "shell"],
    timeout=30
)
async def execute_command(command: str, working_dir: str = None, timeout: int = 30, 
                         context: Dict[str, Any] = None) -> McpToolResult:
    """Execute a shell command safely"""
    try:
        # Security checks
        dangerous_commands = ['rm -rf', 'sudo', 'su', 'chmod 777', 'mkfs', 'dd if=']
        if any(dangerous in command.lower() for dangerous in dangerous_commands):
            return McpToolResult(
                success=False,
                content=[],
                error="Command contains potentially dangerous operations",
                warnings=["Command blocked by security policy"]
            )
        
        # Set working directory
        cwd = working_dir if working_dir and os.path.exists(working_dir) else os.getcwd()
        
        # Execute command
        start_time = time.time()
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=timeout
            )
            execution_time = time.time() - start_time
            
            return McpToolResult(
                success=process.returncode == 0,
                content=[{
                    "type": "command_output",
                    "stdout": stdout.decode('utf-8', errors='replace'),
                    "stderr": stderr.decode('utf-8', errors='replace'),
                    "return_code": process.returncode,
                    "execution_time": execution_time,
                    "command": command,
                    "working_directory": cwd
                }],
                metadata={
                    "security_level": "restricted",
                    "timeout_used": timeout
                }
            )
            
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return McpToolResult(
                success=False,
                content=[],
                error=f"Command timed out after {timeout} seconds",
                warnings=["Process was terminated due to timeout"]
            )
            
    except Exception as e:
        return McpToolResult(
            success=False,
            content=[],
            error=f"Failed to execute command: {str(e)}"
        )


@mcp_tool(
    name="list_processes",
    description="List running processes with optional filtering",
    category="system",
    security_level=McpSecurityLevel.AUTHENTICATED,
    tags=["system", "processes", "monitoring"]
)
async def list_processes(name_filter: str = None, limit: int = 50, 
                        context: Dict[str, Any] = None) -> McpToolResult:
    """List running processes"""
    try:
        processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent', 'memory_percent', 'status']):
            try:
                proc_info = proc.info
                
                # Apply name filter
                if name_filter and name_filter.lower() not in proc_info['name'].lower():
                    continue
                
                processes.append({
                    "pid": proc_info['pid'],
                    "name": proc_info['name'],
                    "username": proc_info['username'],
                    "cpu_percent": proc_info['cpu_percent'],
                    "memory_percent": proc_info['memory_percent'],
                    "status": proc_info['status']
                })
                
                # Apply limit
                if len(processes) >= limit:
                    break
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Sort by CPU usage
        processes.sort(key=lambda x: x['cpu_percent'] or 0, reverse=True)
        
        return McpToolResult(
            success=True,
            content=[{
                "type": "process_list",
                "processes": processes,
                "total_found": len(processes),
                "filter_applied": name_filter,
                "limit_applied": limit
            }],
            metadata={
                "timestamp": datetime.now().isoformat(),
                "system_load": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            }
        )
        
    except Exception as e:
        return McpToolResult(
            success=False,
            content=[],
            error=f"Failed to list processes: {str(e)}"
        )


@mcp_tool(
    name="monitor_resources",
    description="Monitor system resources over a specified duration",
    category="system",
    security_level=McpSecurityLevel.PUBLIC,
    tags=["system", "monitoring", "performance"],
    timeout=120
)
async def monitor_resources(duration: int = 10, interval: int = 1, 
                           context: Dict[str, Any] = None) -> McpToolResult:
    """Monitor system resources over time"""
    try:
        if duration > 60:
            return McpToolResult(
                success=False,
                content=[],
                error="Duration cannot exceed 60 seconds for safety",
                warnings=["Use shorter monitoring periods for real-time monitoring"]
            )
        
        measurements = []
        start_time = time.time()
        
        for i in range(duration):
            measurement = {
                "timestamp": datetime.now().isoformat(),
                "elapsed": i,
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_io": dict(psutil.disk_io_counters()._asdict()) if psutil.disk_io_counters() else None,
                "network_io": dict(psutil.net_io_counters()._asdict()) if psutil.net_io_counters() else None
            }
            measurements.append(measurement)
            
            if i < duration - 1:  # Don't sleep on last iteration
                await asyncio.sleep(interval)
        
        total_time = time.time() - start_time
        
        # Calculate averages
        avg_cpu = sum(m['cpu_percent'] for m in measurements) / len(measurements)
        avg_memory = sum(m['memory_percent'] for m in measurements) / len(measurements)
        
        return McpToolResult(
            success=True,
            content=[{
                "type": "resource_monitoring",
                "measurements": measurements,
                "summary": {
                    "duration": total_time,
                    "sample_count": len(measurements),
                    "average_cpu_percent": avg_cpu,
                    "average_memory_percent": avg_memory,
                    "peak_cpu_percent": max(m['cpu_percent'] for m in measurements),
                    "peak_memory_percent": max(m['memory_percent'] for m in measurements)
                }
            }],
            metadata={
                "monitoring_interval": interval,
                "requested_duration": duration,
                "actual_duration": total_time
            }
        )
        
    except Exception as e:
        return McpToolResult(
            success=False,
            content=[],
            error=f"Failed to monitor resources: {str(e)}"
        )


@mcp_tool(
    name="check_service_health",
    description="Check health of system services and network endpoints",
    category="system",
    security_level=McpSecurityLevel.PUBLIC,
    tags=["system", "health", "monitoring", "network"]
)
async def check_service_health(services: List[str] = None, endpoints: List[str] = None,
                              context: Dict[str, Any] = None) -> McpToolResult:
    """Check health of services and endpoints"""
    try:
        health_results = {
            "services": {},
            "endpoints": {},
            "overall_status": "healthy"
        }
        
        # Check system services (if specified)
        if services:
            for service in services:
                try:
                    # Check if service is running (simplified check)
                    result = subprocess.run(
                        ['systemctl', 'is-active', service],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    
                    health_results["services"][service] = {
                        "status": result.stdout.strip(),
                        "healthy": result.returncode == 0
                    }
                    
                    if result.returncode != 0:
                        health_results["overall_status"] = "degraded"
                        
                except Exception as e:
                    health_results["services"][service] = {
                        "status": "error",
                        "healthy": False,
                        "error": str(e)
                    }
                    health_results["overall_status"] = "degraded"
        
        # Check network endpoints (if specified)
        if endpoints:
            import aiohttp
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                for endpoint in endpoints:
                    try:
                        start_time = time.time()
                        async with session.get(endpoint) as response:
                            response_time = time.time() - start_time
                            
                            health_results["endpoints"][endpoint] = {
                                "status_code": response.status,
                                "healthy": 200 <= response.status < 400,
                                "response_time": response_time,
                                "content_length": response.headers.get('content-length')
                            }
                            
                            if not (200 <= response.status < 400):
                                health_results["overall_status"] = "degraded"
                                
                    except Exception as e:
                        health_results["endpoints"][endpoint] = {
                            "status": "error",
                            "healthy": False,
                            "error": str(e)
                        }
                        health_results["overall_status"] = "degraded"
        
        return McpToolResult(
            success=True,
            content=[{
                "type": "health_check",
                "results": health_results
            }],
            metadata={
                "check_timestamp": datetime.now().isoformat(),
                "services_checked": len(services) if services else 0,
                "endpoints_checked": len(endpoints) if endpoints else 0
            }
        )
        
    except Exception as e:
        return McpToolResult(
            success=False,
            content=[],
            error=f"Failed to check service health: {str(e)}"
        )


# Tool list for automatic registration
TOOLS = [
    (get_system_info._mcp_tool_schema, get_system_info),
    (execute_command._mcp_tool_schema, execute_command),
    (list_processes._mcp_tool_schema, list_processes),
    (monitor_resources._mcp_tool_schema, monitor_resources),
    (check_service_health._mcp_tool_schema, check_service_health)
]

