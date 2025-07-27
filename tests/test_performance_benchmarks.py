"""
Load Testing and Performance Benchmarks for OpenManus
Comprehensive performance testing for all system components
"""

import asyncio
import time
import statistics
import json
import aiohttp
import websockets
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Callable
import pytest
from dataclasses import dataclass
from unittest.mock import patch, AsyncMock


@dataclass
class PerformanceMetrics:
    """Performance metrics container"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    median_response_time: float
    p95_response_time: float
    p99_response_time: float
    min_response_time: float
    max_response_time: float
    requests_per_second: float
    error_rate: float
    total_duration: float


class LoadTester:
    """Load testing utility class"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make a single HTTP request and measure performance"""
        start_time = time.time()
        success = False
        status_code = 0
        response_data = None
        error = None
        
        try:
            url = f"{self.base_url}{endpoint}"
            async with self.session.request(method, url, **kwargs) as response:
                status_code = response.status
                response_data = await response.json()
                success = 200 <= status_code < 300
        except Exception as e:
            error = str(e)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        return {
            "success": success,
            "status_code": status_code,
            "response_time": response_time,
            "response_data": response_data,
            "error": error,
            "timestamp": start_time
        }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

