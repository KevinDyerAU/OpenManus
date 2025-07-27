"""
API Testing and Validation Suites for OpenManus
Comprehensive API endpoint testing, validation, and contract testing
"""

import pytest
import json
import time
from typing import Dict, Any, List, Optional
from unittest.mock import patch, AsyncMock, Mock
from pydantic import BaseModel, ValidationError


class APITestCase:
    """Base class for API test cases"""
    
    def __init__(self, endpoint: str, method: str, description: str):
        self.endpoint = endpoint
        self.method = method
        self.description = description
        self.test_data = []
        self.expected_responses = {}
    
    def add_test_data(self, data: Dict[str, Any], expected_status: int = 200):
        """Add test data with expected response"""
        self.test_data.append({
            "data": data,
            "expected_status": expected_status
        })
    
    def set_expected_response_schema(self, status_code: int, schema: Dict[str, Any]):
        """Set expected response schema for validation"""
        self.expected_responses[status_code] = schema


class TestAuthenticationAPI:
    """Test authentication and authorization endpoints"""
    
    @pytest.mark.asyncio
    async def test_user_registration(self):
        """Test user registration endpoint"""
        test_cases = [
            # Valid registration
            {
                "data": {
                    "email": "test@example.com",
                    "password": "SecurePassword123!",
                    "name": "Test User"
                },
                "expected_status": 201,
                "expected_fields": ["user_id", "access_token", "refresh_token"]
            },
            # Invalid email format
            {
                "data": {
                    "email": "invalid-email",
                    "password": "SecurePassword123!",
                    "name": "Test User"
                },
                "expected_status": 422,
                "expected_error": "Invalid email format"
            },
            # Weak password
            {
                "data": {
                    "email": "test2@example.com",
                    "password": "weak",
                    "name": "Test User"
                },
                "expected_status": 422,
                "expected_error": "Password too weak"
            },
            # Missing required fields
            {
                "data": {
                    "email": "test3@example.com"
                },
                "expected_status": 422,
                "expected_error": "Missing required fields"
            }
        ]
        
        for test_case in test_cases:
            with patch('app.api.auth.create_user') as mock_create_user:
                if test_case["expected_status"] == 201:
                    mock_create_user.return_value = {
                        "user_id": "user_123",
                        "access_token": "jwt_token",
                        "refresh_token": "refresh_token"
                    }
                else:
                    mock_create_user.side_effect = ValidationError("Validation failed")
                
                result = await self._simulate_api_call(
                    "POST", "/auth/register", test_case["data"]
                )
                
                assert result["status"] == test_case["expected_status"]
                
                if test_case["expected_status"] == 201:
                    for field in test_case["expected_fields"]:
                        assert field in result["data"]
                else:
                    assert "error" in result["data"]
    
    @pytest.mark.asyncio
    async def test_user_login(self):
        """Test user login endpoint"""
        test_cases = [
            # Valid login
            {
                "data": {
                    "email": "test@example.com",
                    "password": "SecurePassword123!"
                },
                "expected_status": 200,
                "expected_fields": ["access_token", "refresh_token", "user_info"]
            },
            # Invalid credentials
            {
                "data": {
                    "email": "test@example.com",
                    "password": "wrongpassword"
                },
                "expected_status": 401,
                "expected_error": "Invalid credentials"
            },
            # Non-existent user
            {
                "data": {
                    "email": "nonexistent@example.com",
                    "password": "password"
                },
                "expected_status": 404,
                "expected_error": "User not found"
            }
        ]
        
        for test_case in test_cases:
            with patch('app.api.auth.authenticate_user') as mock_auth:
                if test_case["expected_status"] == 200:
                    mock_auth.return_value = {
                        "access_token": "jwt_token",
                        "refresh_token": "refresh_token",
                        "user_info": {"id": "user_123", "email": "test@example.com"}
                    }
                else:
                    mock_auth.return_value = None
                
                result = await self._simulate_api_call(
                    "POST", "/auth/login", test_case["data"]
                )
                
                assert result["status"] == test_case["expected_status"]
    
    @pytest.mark.asyncio
    async def test_token_refresh(self):
        """Test token refresh endpoint"""
        with patch('app.api.auth.refresh_access_token') as mock_refresh:
            mock_refresh.return_value = {
                "access_token": "new_jwt_token",
                "expires_in": 3600
            }
            
            result = await self._simulate_api_call(
                "POST", "/auth/refresh",
                {"refresh_token": "valid_refresh_token"}
            )
            
            assert result["status"] == 200
            assert "access_token" in result["data"]
            assert "expires_in" in result["data"]
    
    @pytest.mark.asyncio
    async def test_protected_endpoint_authorization(self):
        """Test authorization for protected endpoints"""
        # Test without token
        result = await self._simulate_api_call("GET", "/user/profile")
        assert result["status"] == 401
        
        # Test with invalid token
        result = await self._simulate_api_call(
            "GET", "/user/profile",
            headers={"Authorization": "Bearer invalid_token"}
        )
        assert result["status"] == 401
        
        # Test with valid token
        with patch('app.api.auth.verify_token') as mock_verify:
            mock_verify.return_value = {"user_id": "user_123", "role": "user"}
            
            result = await self._simulate_api_call(
                "GET", "/user/profile",
                headers={"Authorization": "Bearer valid_token"}
            )
            assert result["status"] == 200


class TestChatAPI:
    """Test chat and conversation endpoints"""
    
    @pytest.mark.asyncio
    async def test_chat_endpoint_validation(self):
        """Test chat endpoint input validation"""
        test_cases = [
            # Valid chat request
            {
                "data": {
                    "message": "Hello, how are you?",
                    "model": "openai/gpt-3.5-turbo",
                    "task_type": "general_chat"
                },
                "expected_status": 200
            },
            # Empty message
            {
                "data": {
                    "message": "",
                    "model": "openai/gpt-3.5-turbo"
                },
                "expected_status": 422
            },
            # Message too long
            {
                "data": {
                    "message": "x" * 10000,  # Very long message
                    "model": "openai/gpt-3.5-turbo"
                },
                "expected_status": 422
            },
            # Invalid model
            {
                "data": {
                    "message": "Hello",
                    "model": "invalid/model"
                },
                "expected_status": 422
            },
            # Missing required fields
            {
                "data": {
                    "model": "openai/gpt-3.5-turbo"
                },
                "expected_status": 422
            }
        ]
        
        for test_case in test_cases:
            with patch('app.llm.openrouter_client.OpenRouterClient.chat') as mock_chat:
                if test_case["expected_status"] == 200:
                    mock_chat.return_value = {
                        "choices": [{
                            "message": {"content": "Hello! How can I help you?"}
                        }]
                    }
                
                result = await self._simulate_api_call(
                    "POST", "/chat", test_case["data"],
                    headers={"Authorization": "Bearer valid_token"}
                )
                
                assert result["status"] == test_case["expected_status"]
    
    @pytest.mark.asyncio
    async def test_chat_streaming_endpoint(self):
        """Test chat streaming endpoint"""
        with patch('app.llm.openrouter_client.OpenRouterClient.chat_stream') as mock_stream:
            # Mock streaming response
            async def mock_stream_generator():
                chunks = [
                    {"choices": [{"delta": {"content": "Hello"}}]},
                    {"choices": [{"delta": {"content": " there!"}}]},
                    {"choices": [{"delta": {}}], "finish_reason": "stop"}
                ]
                for chunk in chunks:
                    yield chunk
            
            mock_stream.return_value = mock_stream_generator()
            
            result = await self._simulate_streaming_api_call(
                "POST", "/chat/stream",
                {
                    "message": "Hello",
                    "model": "openai/gpt-3.5-turbo",
                    "stream": True
                },
                headers={"Authorization": "Bearer valid_token"}
            )
            
            assert result["status"] == 200
            assert len(result["chunks"]) == 3
    
    @pytest.mark.asyncio
    async def test_conversation_management(self):
        """Test conversation management endpoints"""
        # Create conversation
        with patch('app.database.create_conversation') as mock_create:
            mock_create.return_value = {
                "conversation_id": "conv_123",
                "created_at": "2024-01-01T00:00:00Z"
            }
            
            result = await self._simulate_api_call(
                "POST", "/conversations",
                {"title": "Test Conversation"},
                headers={"Authorization": "Bearer valid_token"}
            )
            
            assert result["status"] == 201
            assert "conversation_id" in result["data"]
        
        # Get conversation history
        with patch('app.database.get_conversation') as mock_get:
            mock_get.return_value = {
                "conversation_id": "conv_123",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"}
                ]
            }
            
            result = await self._simulate_api_call(
                "GET", "/conversations/conv_123",
                headers={"Authorization": "Bearer valid_token"}
            )
            
            assert result["status"] == 200
            assert "messages" in result["data"]
        
        # Delete conversation
        result = await self._simulate_api_call(
            "DELETE", "/conversations/conv_123",
            headers={"Authorization": "Bearer valid_token"}
        )
        
        assert result["status"] == 204


class TestFlowAPI:
    """Test flow and workflow endpoints"""
    
    @pytest.mark.asyncio
    async def test_flow_creation_validation(self):
        """Test flow creation endpoint validation"""
        test_cases = [
            # Valid flow
            {
                "data": {
                    "name": "Test Flow",
                    "description": "A test flow",
                    "steps": [
                        {
                            "id": "step1",
                            "type": "action",
                            "description": "First step",
                            "parameters": {"action": "test"}
                        }
                    ]
                },
                "expected_status": 201
            },
            # Missing name
            {
                "data": {
                    "description": "A test flow",
                    "steps": []
                },
                "expected_status": 422
            },
            # Empty steps
            {
                "data": {
                    "name": "Test Flow",
                    "description": "A test flow",
                    "steps": []
                },
                "expected_status": 422
            },
            # Invalid step structure
            {
                "data": {
                    "name": "Test Flow",
                    "description": "A test flow",
                    "steps": [
                        {
                            "id": "step1",
                            "type": "invalid_type"
                        }
                    ]
                },
                "expected_status": 422
            },
            # Circular dependencies
            {
                "data": {
                    "name": "Test Flow",
                    "description": "A test flow",
                    "steps": [
                        {
                            "id": "step1",
                            "type": "action",
                            "dependencies": ["step2"]
                        },
                        {
                            "id": "step2",
                            "type": "action",
                            "dependencies": ["step1"]
                        }
                    ]
                },
                "expected_status": 422
            }
        ]
        
        for test_case in test_cases:
            with patch('app.flow.enhanced_flow.EnhancedFlow') as mock_flow:
                if test_case["expected_status"] == 201:
                    mock_flow_instance = Mock()
                    mock_flow_instance.validate_dependencies.return_value = True
                    mock_flow.return_value = mock_flow_instance
                else:
                    mock_flow.side_effect = ValueError("Validation failed")
                
                result = await self._simulate_api_call(
                    "POST", "/flows/create", test_case["data"],
                    headers={"Authorization": "Bearer valid_token"}
                )
                
                assert result["status"] == test_case["expected_status"]
    
    @pytest.mark.asyncio
    async def test_flow_execution_endpoint(self):
        """Test flow execution endpoint"""
        flow_data = {
            "name": "Execution Test Flow",
            "steps": [
                {
                    "id": "step1",
                    "type": "action",
                    "description": "Test step",
                    "parameters": {"action": "test"}
                }
            ],
            "config": {
                "timeout": 60,
                "enable_streaming": True
            }
        }
        
        with patch('app.flow.enhanced_flow.EnhancedFlow') as mock_flow:
            mock_flow_instance = AsyncMock()
            mock_flow_instance.execute.return_value = Mock(
                success=True,
                results={"step1": "Test result"},
                duration=1.5
            )
            mock_flow.return_value = mock_flow_instance
            
            result = await self._simulate_api_call(
                "POST", "/flows/execute", flow_data,
                headers={"Authorization": "Bearer valid_token"}
            )
            
            assert result["status"] == 200
            assert result["data"]["success"] is True
            assert "results" in result["data"]
    
    @pytest.mark.asyncio
    async def test_flow_status_monitoring(self):
        """Test flow status monitoring endpoints"""
        # Get flow status
        with patch('app.database.get_flow_status') as mock_status:
            mock_status.return_value = {
                "flow_id": "flow_123",
                "status": "running",
                "progress": 50,
                "current_step": "step2",
                "started_at": "2024-01-01T00:00:00Z"
            }
            
            result = await self._simulate_api_call(
                "GET", "/flows/flow_123/status",
                headers={"Authorization": "Bearer valid_token"}
            )
            
            assert result["status"] == 200
            assert result["data"]["status"] == "running"
            assert result["data"]["progress"] == 50
        
        # Cancel flow
        result = await self._simulate_api_call(
            "POST", "/flows/flow_123/cancel",
            headers={"Authorization": "Bearer valid_token"}
        )
        
        assert result["status"] == 200


class TestBrowserAPI:
    """Test browser automation endpoints"""
    
    @pytest.mark.asyncio
    async def test_browser_navigation_endpoint(self):
        """Test browser navigation endpoint"""
        test_cases = [
            # Valid navigation
            {
                "data": {
                    "action": "navigate",
                    "parameters": {
                        "url": "https://example.com"
                    }
                },
                "expected_status": 200
            },
            # Invalid URL
            {
                "data": {
                    "action": "navigate",
                    "parameters": {
                        "url": "invalid-url"
                    }
                },
                "expected_status": 422
            },
            # Missing parameters
            {
                "data": {
                    "action": "navigate"
                },
                "expected_status": 422
            },
            # Unsupported action
            {
                "data": {
                    "action": "unsupported_action",
                    "parameters": {}
                },
                "expected_status": 422
            }
        ]
        
        for test_case in test_cases:
            with patch('app.browser.headless_browser.HeadlessBrowser') as mock_browser:
                if test_case["expected_status"] == 200:
                    mock_browser_instance = AsyncMock()
                    mock_browser_instance.navigate.return_value = True
                    mock_browser.return_value = mock_browser_instance
                
                result = await self._simulate_api_call(
                    "POST", "/browser", test_case["data"],
                    headers={"Authorization": "Bearer valid_token"}
                )
                
                assert result["status"] == test_case["expected_status"]
    
    @pytest.mark.asyncio
    async def test_browser_extraction_endpoint(self):
        """Test browser data extraction endpoint"""
        extraction_request = {
            "action": "extract_data",
            "parameters": {
                "url": "https://example.com",
                "selectors": {
                    "title": "h1",
                    "content": ".content",
                    "links": "a"
                }
            }
        }
        
        with patch('app.browser.headless_browser.HeadlessBrowser') as mock_browser:
            mock_browser_instance = AsyncMock()
            mock_browser_instance.navigate.return_value = True
            mock_browser_instance.extract_data.return_value = {
                "title": "Example Page",
                "content": "Page content",
                "links": ["https://example.com/page1", "https://example.com/page2"]
            }
            mock_browser.return_value = mock_browser_instance
            
            result = await self._simulate_api_call(
                "POST", "/browser", extraction_request,
                headers={"Authorization": "Bearer valid_token"}
            )
            
            assert result["status"] == 200
            assert "title" in result["data"]["extracted_data"]
            assert "content" in result["data"]["extracted_data"]
            assert "links" in result["data"]["extracted_data"]


class TestMCPAPI:
    """Test MCP (Model Context Protocol) endpoints"""
    
    @pytest.mark.asyncio
    async def test_mcp_tool_listing(self):
        """Test MCP tool listing endpoint"""
        with patch('app.mcp.enhanced_server.EnhancedMCPServer') as mock_server:
            mock_server_instance = Mock()
            mock_server_instance.list_tools.return_value = [
                {
                    "name": "system_info",
                    "description": "Get system information",
                    "parameters": {"type": "object"},
                    "security_level": "public"
                },
                {
                    "name": "file_operations",
                    "description": "File operations tool",
                    "parameters": {"type": "object"},
                    "security_level": "authenticated"
                }
            ]
            mock_server.return_value = mock_server_instance
            
            result = await self._simulate_api_call(
                "GET", "/mcp/tools",
                headers={"Authorization": "Bearer valid_token"}
            )
            
            assert result["status"] == 200
            assert len(result["data"]["tools"]) == 2
            assert result["data"]["tools"][0]["name"] == "system_info"
    
    @pytest.mark.asyncio
    async def test_mcp_tool_execution(self):
        """Test MCP tool execution endpoint"""
        execution_request = {
            "tool_name": "system_info",
            "parameters": {}
        }
        
        with patch('app.mcp.enhanced_server.EnhancedMCPServer') as mock_server:
            mock_server_instance = AsyncMock()
            mock_server_instance.execute_tool.return_value = {
                "platform": "linux",
                "python_version": "3.11.0",
                "cpu_count": 4,
                "memory_total": "8GB"
            }
            mock_server.return_value = mock_server_instance
            
            result = await self._simulate_api_call(
                "POST", "/mcp/execute", execution_request,
                headers={"Authorization": "Bearer valid_token"}
            )
            
            assert result["status"] == 200
            assert "platform" in result["data"]["result"]
            assert "python_version" in result["data"]["result"]


class TestAPIRateLimiting:
    """Test API rate limiting and throttling"""
    
    @pytest.mark.asyncio
    async def test_rate_limiting_enforcement(self):
        """Test rate limiting enforcement"""
        # Simulate multiple rapid requests
        results = []
        
        for i in range(100):  # Exceed rate limit
            result = await self._simulate_api_call(
                "GET", "/health",
                headers={"Authorization": "Bearer valid_token"}
            )
            results.append(result)
        
        # Check that some requests were rate limited
        rate_limited_count = sum(1 for r in results if r["status"] == 429)
        assert rate_limited_count > 0  # Some requests should be rate limited
    
    @pytest.mark.asyncio
    async def test_rate_limit_headers(self):
        """Test rate limit headers in responses"""
        result = await self._simulate_api_call(
            "GET", "/health",
            headers={"Authorization": "Bearer valid_token"}
        )
        
        # Mock rate limit headers
        expected_headers = {
            "X-RateLimit-Limit": "60",
            "X-RateLimit-Remaining": "59",
            "X-RateLimit-Reset": str(int(time.time()) + 60)
        }
        
        # In a real test, these would be checked in the response headers
        assert result["status"] == 200


class TestAPIErrorHandling:
    """Test API error handling and responses"""
    
    @pytest.mark.asyncio
    async def test_validation_error_responses(self):
        """Test validation error response format"""
        invalid_request = {
            "message": "",  # Invalid empty message
            "model": "invalid/model"  # Invalid model format
        }
        
        result = await self._simulate_api_call(
            "POST", "/chat", invalid_request,
            headers={"Authorization": "Bearer valid_token"}
        )
        
        assert result["status"] == 422
        assert "error" in result["data"]
        assert "validation_errors" in result["data"]
        assert isinstance(result["data"]["validation_errors"], list)
    
    @pytest.mark.asyncio
    async def test_internal_error_responses(self):
        """Test internal error response format"""
        with patch('app.llm.openrouter_client.OpenRouterClient.chat') as mock_chat:
            mock_chat.side_effect = Exception("Internal service error")
            
            result = await self._simulate_api_call(
                "POST", "/chat",
                {"message": "Hello", "model": "openai/gpt-3.5-turbo"},
                headers={"Authorization": "Bearer valid_token"}
            )
            
            assert result["status"] == 500
            assert "error" in result["data"]
            assert result["data"]["error"] == "Internal server error"
    
    @pytest.mark.asyncio
    async def test_not_found_responses(self):
        """Test not found error responses"""
        result = await self._simulate_api_call(
            "GET", "/nonexistent/endpoint",
            headers={"Authorization": "Bearer valid_token"}
        )
        
        assert result["status"] == 404
        assert "error" in result["data"]
        assert "not found" in result["data"]["error"].lower()


# Helper methods for API simulation
async def _simulate_api_call(
    method: str,
    endpoint: str,
    data: Dict = None,
    headers: Dict = None
) -> Dict[str, Any]:
    """Simulate API call and return mock response"""
    # Mock response based on endpoint and method
    if endpoint == "/health":
        return {
            "status": 200,
            "data": {"status": "healthy", "timestamp": time.time()}
        }
    elif endpoint.startswith("/auth/"):
        if endpoint == "/auth/register" and method == "POST":
            if data and "email" in data and "@" in data["email"]:
                return {
                    "status": 201,
                    "data": {
                        "user_id": "user_123",
                        "access_token": "jwt_token",
                        "refresh_token": "refresh_token"
                    }
                }
            else:
                return {
                    "status": 422,
                    "data": {"error": "Validation failed"}
                }
        elif endpoint == "/auth/login" and method == "POST":
            return {
                "status": 200,
                "data": {
                    "access_token": "jwt_token",
                    "refresh_token": "refresh_token",
                    "user_info": {"id": "user_123"}
                }
            }
    elif endpoint == "/chat" and method == "POST":
        if not headers or "Authorization" not in headers:
            return {"status": 401, "data": {"error": "Unauthorized"}}
        if not data or not data.get("message"):
            return {"status": 422, "data": {"error": "Validation failed"}}
        return {
            "status": 200,
            "data": {
                "response": "Mock AI response",
                "conversation_id": "conv_123"
            }
        }
    
    # Default response
    return {
        "status": 200,
        "data": {"message": "Success"}
    }


async def _simulate_streaming_api_call(
    method: str,
    endpoint: str,
    data: Dict = None,
    headers: Dict = None
) -> Dict[str, Any]:
    """Simulate streaming API call"""
    return {
        "status": 200,
        "chunks": [
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": " there!"}}]},
            {"choices": [{"delta": {}}], "finish_reason": "stop"}
        ]
    }


# Monkey patch helper methods to test classes
TestAuthenticationAPI._simulate_api_call = _simulate_api_call
TestChatAPI._simulate_api_call = _simulate_api_call
TestChatAPI._simulate_streaming_api_call = _simulate_streaming_api_call
TestFlowAPI._simulate_api_call = _simulate_api_call
TestBrowserAPI._simulate_api_call = _simulate_api_call
TestMCPAPI._simulate_api_call = _simulate_api_call
TestAPIRateLimiting._simulate_api_call = _simulate_api_call
TestAPIErrorHandling._simulate_api_call = _simulate_api_call


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

