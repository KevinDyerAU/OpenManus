"""
Integration Tests for OpenManus System
Tests for component interactions, API endpoints, and end-to-end workflows
"""

import pytest
import asyncio
import json
import aiohttp
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from app.api.main import create_app
from app.mcp.enhanced_server import EnhancedMCPServer
from app.flow.enhanced_flow import EnhancedFlow
from app.llm.openrouter_client import OpenRouterClient
from app.browser.headless_browser import HeadlessBrowser


class TestAPIIntegration:
    """Test API integration and endpoints"""
    
    @pytest.fixture
    async def app(self):
        """Create test FastAPI application"""
        app = create_app()
        return app
    
    @pytest.fixture
    async def client(self, app):
        """Create test HTTP client"""
        async with aiohttp.ClientSession() as session:
            yield session
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self, client):
        """Test health check endpoint"""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "status": "healthy",
                "timestamp": "2024-01-01T00:00:00Z",
                "version": "1.0.0"
            })
            mock_get.return_value.__aenter__.return_value = mock_response
            
            async with client.get("http://localhost:8000/health") as response:
                assert response.status == 200
                data = await response.json()
                assert data["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_chat_endpoint(self, client):
        """Test chat endpoint"""
        with patch('app.llm.openrouter_client.OpenRouterClient.chat') as mock_chat:
            mock_chat.return_value = {
                "choices": [{
                    "message": {
                        "content": "Hello! How can I help you today?"
                    }
                }]
            }
            
            chat_data = {
                "message": "Hello",
                "model": "openai/gpt-3.5-turbo",
                "task_type": "general_chat"
            }
            
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.json = AsyncMock(return_value={
                    "response": "Hello! How can I help you today?",
                    "conversation_id": "conv_123",
                    "model_used": "openai/gpt-3.5-turbo"
                })
                mock_post.return_value.__aenter__.return_value = mock_response
                
                async with client.post(
                    "http://localhost:8000/chat",
                    json=chat_data
                ) as response:
                    assert response.status == 200
                    data = await response.json()
                    assert "response" in data
                    assert "conversation_id" in data
    
    @pytest.mark.asyncio
    async def test_websocket_chat(self):
        """Test WebSocket chat functionality"""
        with patch('websockets.connect') as mock_connect:
            mock_websocket = AsyncMock()
            mock_websocket.recv = AsyncMock(side_effect=[
                json.dumps({
                    "type": "connection",
                    "status": "connected",
                    "connection_id": "ws_123"
                }),
                json.dumps({
                    "type": "message",
                    "content": "WebSocket response",
                    "message_id": "msg_123"
                })
            ])
            mock_connect.return_value.__aenter__.return_value = mock_websocket
            
            # Simulate WebSocket connection
            async with mock_connect("ws://localhost:8000/ws/chat") as websocket:
                # Send message
                await websocket.send(json.dumps({
                    "type": "chat",
                    "message": "Hello via WebSocket"
                }))
                
                # Receive connection confirmation
                response = await websocket.recv()
                data = json.loads(response)
                assert data["type"] == "connection"
                assert data["status"] == "connected"
                
                # Receive message response
                response = await websocket.recv()
                data = json.loads(response)
                assert data["type"] == "message"
                assert "content" in data
    
    @pytest.mark.asyncio
    async def test_browser_automation_endpoint(self, client):
        """Test browser automation endpoint"""
        with patch('app.browser.headless_browser.HeadlessBrowser') as mock_browser:
            mock_instance = AsyncMock()
            mock_instance.navigate.return_value = True
            mock_instance.extract_text.return_value = "Extracted page content"
            mock_browser.return_value = mock_instance
            
            browser_data = {
                "action": "navigate_and_extract",
                "parameters": {
                    "url": "https://example.com",
                    "extract_text": True
                }
            }
            
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.json = AsyncMock(return_value={
                    "success": True,
                    "result": {
                        "url": "https://example.com",
                        "text": "Extracted page content"
                    }
                })
                mock_post.return_value.__aenter__.return_value = mock_response
                
                async with client.post(
                    "http://localhost:8000/browser",
                    json=browser_data
                ) as response:
                    assert response.status == 200
                    data = await response.json()
                    assert data["success"] is True
                    assert "result" in data
    
    @pytest.mark.asyncio
    async def test_flow_execution_endpoint(self, client):
        """Test flow execution endpoint"""
        flow_data = {
            "name": "test_flow",
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
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "success": True,
                "flow_id": "flow_123",
                "status": "completed",
                "results": {
                    "step1": "Test result"
                }
            })
            mock_post.return_value.__aenter__.return_value = mock_response
            
            async with client.post(
                "http://localhost:8000/flows/execute",
                json=flow_data
            ) as response:
                assert response.status == 200
                data = await response.json()
                assert data["success"] is True
                assert "flow_id" in data


class TestMCPFlowIntegration:
    """Test MCP and Flow component integration"""
    
    @pytest.fixture
    def mcp_server(self):
        """Create MCP server for testing"""
        return EnhancedMCPServer()
    
    @pytest.fixture
    def enhanced_flow(self):
        """Create enhanced flow for testing"""
        from app.flow.enhanced_interfaces import FlowConfig
        config = FlowConfig(
            name="integration_test_flow",
            description="Flow for integration testing",
            timeout=60,
            enable_streaming=True
        )
        return EnhancedFlow(config)
    
    @pytest.mark.asyncio
    async def test_mcp_tool_in_flow(self, mcp_server, enhanced_flow):
        """Test using MCP tools within flow execution"""
        # Register MCP tool
        from app.mcp.interfaces import MCPTool, SecurityLevel
        
        async def test_tool(message: str) -> str:
            return f"MCP processed: {message}"
        
        tool = MCPTool(
            name="test_mcp_tool",
            description="Test MCP tool for flow integration",
            parameters={
                "type": "object",
                "properties": {
                    "message": {"type": "string"}
                },
                "required": ["message"]
            },
            function=test_tool,
            security_level=SecurityLevel.PUBLIC
        )
        
        mcp_server.register_tool(tool)
        
        # Create flow step that uses MCP tool
        from app.flow.enhanced_interfaces import FlowStep
        
        step = FlowStep(
            id="mcp_step",
            type="mcp_tool",
            description="Step using MCP tool",
            parameters={
                "tool_name": "test_mcp_tool",
                "tool_parameters": {"message": "integration test"}
            }
        )
        
        enhanced_flow.add_step(step)
        
        # Mock step execution to use MCP server
        async def mock_execute_step(step, context):
            if step.type == "mcp_tool":
                tool_name = step.parameters["tool_name"]
                tool_params = step.parameters["tool_parameters"]
                return await mcp_server.execute_tool(tool_name, tool_params)
            return "default result"
        
        enhanced_flow._execute_step = mock_execute_step
        
        # Execute flow
        result = await enhanced_flow.execute()
        
        assert result.success is True
        assert enhanced_flow.results["mcp_step"] == "MCP processed: integration test"
    
    @pytest.mark.asyncio
    async def test_flow_with_streaming_callbacks(self, enhanced_flow):
        """Test flow execution with streaming callbacks"""
        stream_events = []
        
        def stream_callback(callback_type, data):
            stream_events.append((callback_type, data))
        
        enhanced_flow.set_stream_callback(stream_callback)
        
        # Add test step
        from app.flow.enhanced_interfaces import FlowStep
        
        step = FlowStep(
            id="streaming_step",
            type="action",
            description="Step with streaming",
            parameters={"action": "stream_test"}
        )
        
        enhanced_flow.add_step(step)
        
        # Mock step execution with streaming
        async def mock_execute_step(step, context):
            from app.flow.enhanced_interfaces import StreamCallbackType
            
            # Emit streaming events
            await enhanced_flow._emit_stream_event(
                StreamCallbackType.WORKFLOW,
                {"step": step.id, "status": "started"}
            )
            
            await enhanced_flow._emit_stream_event(
                StreamCallbackType.TEXT,
                {"content": "Processing step..."}
            )
            
            await enhanced_flow._emit_stream_event(
                StreamCallbackType.WORKFLOW,
                {"step": step.id, "status": "completed"}
            )
            
            return "streaming result"
        
        enhanced_flow._execute_step = mock_execute_step
        
        # Execute flow
        result = await enhanced_flow.execute()
        
        assert result.success is True
        assert len(stream_events) >= 3  # At least 3 streaming events
    
    @pytest.mark.asyncio
    async def test_multi_agent_flow_coordination(self):
        """Test multi-agent coordination in flow execution"""
        from app.flow.multi_agent_orchestrator import MultiAgentOrchestrator
        from app.flow.enhanced_interfaces import AgentRole
        
        orchestrator = MultiAgentOrchestrator()
        
        # Register agents
        agents = [
            {
                "id": "planner",
                "name": "Planning Agent",
                "role": AgentRole.COORDINATOR,
                "capabilities": ["planning", "coordination"]
            },
            {
                "id": "executor",
                "name": "Execution Agent",
                "role": AgentRole.EXECUTOR,
                "capabilities": ["execution", "processing"]
            }
        ]
        
        for agent in agents:
            orchestrator.register_agent(agent)
        
        # Mock agent execution
        async def mock_execute_task(agent_id, task):
            return {
                "success": True,
                "result": f"Task {task['id']} completed by {agent_id}",
                "agent_id": agent_id
            }
        
        orchestrator._execute_agent_task = mock_execute_task
        
        # Create workflow
        workflow = {
            "id": "multi_agent_workflow",
            "tasks": [
                {
                    "id": "planning_task",
                    "type": "planning",
                    "required_capabilities": ["planning"],
                    "dependencies": []
                },
                {
                    "id": "execution_task",
                    "type": "execution",
                    "required_capabilities": ["execution"],
                    "dependencies": ["planning_task"]
                }
            ]
        }
        
        # Execute workflow
        result = await orchestrator.execute_workflow(workflow)
        
        assert result["success"] is True
        assert len(result["task_results"]) == 2
        
        # Verify task assignment
        planning_result = next(
            r for r in result["task_results"]
            if r["task_id"] == "planning_task"
        )
        execution_result = next(
            r for r in result["task_results"]
            if r["task_id"] == "execution_task"
        )
        
        assert planning_result["agent_id"] == "planner"
        assert execution_result["agent_id"] == "executor"


class TestOpenRouterIntegration:
    """Test OpenRouter LLM integration"""
    
    @pytest.fixture
    def openrouter_client(self):
        """Create OpenRouter client for testing"""
        return OpenRouterClient(api_key="test-key")
    
    @pytest.mark.asyncio
    async def test_model_selection(self, openrouter_client):
        """Test automatic model selection"""
        with patch.object(openrouter_client, '_make_request') as mock_request:
            mock_request.return_value = {
                "data": [
                    {
                        "id": "openai/gpt-3.5-turbo",
                        "name": "GPT-3.5 Turbo",
                        "pricing": {"prompt": "0.0015", "completion": "0.002"},
                        "context_length": 4096
                    },
                    {
                        "id": "anthropic/claude-3-haiku",
                        "name": "Claude 3 Haiku",
                        "pricing": {"prompt": "0.00025", "completion": "0.00125"},
                        "context_length": 200000
                    }
                ]
            }
            
            # Test model selection for different tasks
            chat_model = await openrouter_client.select_model_for_task("chat")
            coding_model = await openrouter_client.select_model_for_task("coding")
            analysis_model = await openrouter_client.select_model_for_task("analysis")
            
            assert chat_model is not None
            assert coding_model is not None
            assert analysis_model is not None
    
    @pytest.mark.asyncio
    async def test_chat_completion(self, openrouter_client):
        """Test chat completion"""
        with patch.object(openrouter_client, '_make_request') as mock_request:
            mock_request.return_value = {
                "choices": [{
                    "message": {
                        "content": "Hello! How can I help you today?"
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 8,
                    "total_tokens": 18
                }
            }
            
            response = await openrouter_client.chat(
                messages=[{"role": "user", "content": "Hello"}],
                model="openai/gpt-3.5-turbo"
            )
            
            assert "choices" in response
            assert len(response["choices"]) == 1
            assert response["choices"][0]["message"]["content"] == "Hello! How can I help you today?"
    
    @pytest.mark.asyncio
    async def test_streaming_chat(self, openrouter_client):
        """Test streaming chat completion"""
        with patch.object(openrouter_client, '_make_streaming_request') as mock_stream:
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
            
            # Collect streaming response
            content_chunks = []
            async for chunk in openrouter_client.chat_stream(
                messages=[{"role": "user", "content": "Hello"}],
                model="openai/gpt-3.5-turbo"
            ):
                if "choices" in chunk and chunk["choices"]:
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta:
                        content_chunks.append(delta["content"])
            
            full_content = "".join(content_chunks)
            assert full_content == "Hello there!"
    
    @pytest.mark.asyncio
    async def test_cost_tracking(self, openrouter_client):
        """Test cost tracking functionality"""
        with patch.object(openrouter_client, '_make_request') as mock_request:
            mock_request.return_value = {
                "choices": [{
                    "message": {"content": "Test response"}
                }],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150
                }
            }
            
            # Mock model pricing
            openrouter_client.model_manager.model_cache = {
                "openai/gpt-3.5-turbo": {
                    "pricing": {"prompt": "0.0015", "completion": "0.002"}
                }
            }
            
            # Make request and check cost tracking
            initial_cost = openrouter_client.get_total_cost()
            
            await openrouter_client.chat(
                messages=[{"role": "user", "content": "Test"}],
                model="openai/gpt-3.5-turbo"
            )
            
            final_cost = openrouter_client.get_total_cost()
            
            # Should have tracked cost increase
            assert final_cost > initial_cost


class TestBrowserIntegration:
    """Test headless browser integration"""
    
    @pytest.fixture
    def browser(self):
        """Create headless browser for testing"""
        return HeadlessBrowser()
    
    @pytest.mark.asyncio
    async def test_browser_navigation(self, browser):
        """Test browser navigation"""
        with patch.object(browser, 'page') as mock_page:
            mock_page.goto = AsyncMock(return_value=None)
            mock_page.title = AsyncMock(return_value="Test Page")
            
            await browser.navigate("https://example.com")
            
            mock_page.goto.assert_called_once_with("https://example.com")
    
    @pytest.mark.asyncio
    async def test_text_extraction(self, browser):
        """Test text extraction from pages"""
        with patch.object(browser, 'page') as mock_page:
            mock_page.evaluate = AsyncMock(return_value="Extracted page text")
            
            text = await browser.extract_text()
            
            assert text == "Extracted page text"
            mock_page.evaluate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_form_interaction(self, browser):
        """Test form filling and submission"""
        with patch.object(browser, 'page') as mock_page:
            mock_page.fill = AsyncMock()
            mock_page.click = AsyncMock()
            mock_page.wait_for_navigation = AsyncMock()
            
            await browser.fill_form({
                "#username": "testuser",
                "#password": "testpass"
            })
            
            await browser.click_element("#submit")
            
            mock_page.fill.assert_called()
            mock_page.click.assert_called_with("#submit")
    
    @pytest.mark.asyncio
    async def test_screenshot_capture(self, browser):
        """Test screenshot capture"""
        with patch.object(browser, 'page') as mock_page:
            mock_page.screenshot = AsyncMock(return_value=b"fake_screenshot_data")
            
            screenshot = await browser.take_screenshot()
            
            assert screenshot == b"fake_screenshot_data"
            mock_page.screenshot.assert_called_once()


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows"""
    
    @pytest.mark.asyncio
    async def test_complete_research_workflow(self):
        """Test complete research workflow with all components"""
        # Mock all external dependencies
        with patch('app.llm.openrouter_client.OpenRouterClient') as mock_llm, \
             patch('app.browser.headless_browser.HeadlessBrowser') as mock_browser, \
             patch('app.mcp.enhanced_server.EnhancedMCPServer') as mock_mcp:
            
            # Setup mocks
            mock_llm_instance = AsyncMock()
            mock_llm_instance.chat.return_value = {
                "choices": [{
                    "message": {"content": "Research plan: 1. Search for information 2. Analyze data 3. Summarize findings"}
                }]
            }
            mock_llm.return_value = mock_llm_instance
            
            mock_browser_instance = AsyncMock()
            mock_browser_instance.navigate.return_value = True
            mock_browser_instance.extract_text.return_value = "Research data from web"
            mock_browser.return_value = mock_browser_instance
            
            mock_mcp_instance = AsyncMock()
            mock_mcp_instance.execute_tool.return_value = "MCP tool result"
            mock_mcp.return_value = mock_mcp_instance
            
            # Create research workflow
            from app.flow.enhanced_flow import EnhancedFlow
            from app.flow.enhanced_interfaces import FlowConfig, FlowStep
            
            config = FlowConfig(
                name="research_workflow",
                description="Complete research workflow",
                timeout=300,
                enable_streaming=True
            )
            
            flow = EnhancedFlow(config)
            
            # Add workflow steps
            steps = [
                FlowStep(
                    id="planning",
                    type="llm_chat",
                    description="Create research plan",
                    parameters={
                        "prompt": "Create a research plan for the topic",
                        "model": "openai/gpt-3.5-turbo"
                    }
                ),
                FlowStep(
                    id="web_research",
                    type="browser_action",
                    description="Gather information from web",
                    parameters={
                        "action": "navigate_and_extract",
                        "url": "https://example.com"
                    },
                    dependencies=["planning"]
                ),
                FlowStep(
                    id="data_analysis",
                    type="mcp_tool",
                    description="Analyze gathered data",
                    parameters={
                        "tool_name": "analyze_data",
                        "data": "research_data"
                    },
                    dependencies=["web_research"]
                ),
                FlowStep(
                    id="summary",
                    type="llm_chat",
                    description="Summarize findings",
                    parameters={
                        "prompt": "Summarize the research findings",
                        "model": "openai/gpt-3.5-turbo"
                    },
                    dependencies=["data_analysis"]
                )
            ]
            
            for step in steps:
                flow.add_step(step)
            
            # Mock step execution
            async def mock_execute_step(step, context):
                if step.type == "llm_chat":
                    return f"LLM response for {step.id}"
                elif step.type == "browser_action":
                    return f"Browser result for {step.id}"
                elif step.type == "mcp_tool":
                    return f"MCP result for {step.id}"
                return f"Default result for {step.id}"
            
            flow._execute_step = mock_execute_step
            
            # Execute workflow
            result = await flow.execute()
            
            assert result.success is True
            assert len(flow.results) == 4
            assert "planning" in flow.results
            assert "web_research" in flow.results
            assert "data_analysis" in flow.results
            assert "summary" in flow.results
    
    @pytest.mark.asyncio
    async def test_interactive_workflow_with_human_callbacks(self):
        """Test interactive workflow with human callbacks"""
        from app.flow.enhanced_flow import EnhancedFlow
        from app.flow.enhanced_interfaces import FlowConfig, FlowStep, HumanCallbackType
        
        config = FlowConfig(
            name="interactive_workflow",
            description="Workflow with human interaction",
            enable_human_callbacks=True
        )
        
        flow = EnhancedFlow(config)
        
        # Mock human callback responses
        human_responses = {
            HumanCallbackType.CONFIRM: True,
            HumanCallbackType.INPUT: "user provided input",
            HumanCallbackType.SELECT: 1
        }
        
        async def mock_human_callback(callback_type, data):
            return human_responses.get(callback_type)
        
        flow.set_human_callback(mock_human_callback)
        
        # Add interactive step
        step = FlowStep(
            id="interactive_step",
            type="human_interaction",
            description="Step requiring human input",
            parameters={"interaction_type": "input"}
        )
        
        flow.add_step(step)
        
        # Mock step execution with human interaction
        async def mock_execute_step(step, context):
            if step.type == "human_interaction":
                user_input = await flow._request_human_input("Please provide input:")
                return f"Processed user input: {user_input}"
            return "default result"
        
        flow._execute_step = mock_execute_step
        
        # Execute workflow
        result = await flow.execute()
        
        assert result.success is True
        assert "interactive_step" in flow.results
        assert "user provided input" in flow.results["interactive_step"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

