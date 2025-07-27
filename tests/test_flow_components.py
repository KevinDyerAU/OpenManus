"""
Unit Tests for Flow Components
Tests for enhanced flow interfaces, callback systems, and multi-agent orchestration
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List, Optional

from app.flow.enhanced_interfaces import (
    FlowStep, FlowConfig, FlowResult, FlowState,
    StreamCallbackType, HumanCallbackType, AgentRole,
    CommunicationType, ConflictType, ConflictResolution
)
from app.flow.enhanced_flow import EnhancedFlow
from app.flow.multi_agent_orchestrator import MultiAgentOrchestrator


class TestFlowInterfaces:
    """Test flow interface definitions"""
    
    def test_flow_step_creation(self):
        """Test FlowStep creation and validation"""
        step = FlowStep(
            id="test_step",
            type="action",
            description="A test step",
            parameters={"action": "test"},
            dependencies=["prev_step"]
        )
        
        assert step.id == "test_step"
        assert step.type == "action"
        assert step.description == "A test step"
        assert step.parameters == {"action": "test"}
        assert step.dependencies == ["prev_step"]
    
    def test_flow_config_creation(self):
        """Test FlowConfig creation"""
        config = FlowConfig(
            name="test_flow",
            description="A test flow",
            version="1.0.0",
            timeout=300,
            retry_attempts=3,
            enable_streaming=True,
            enable_human_callbacks=True
        )
        
        assert config.name == "test_flow"
        assert config.timeout == 300
        assert config.retry_attempts == 3
        assert config.enable_streaming is True
        assert config.enable_human_callbacks is True
    
    def test_flow_state_transitions(self):
        """Test FlowState enumeration and transitions"""
        assert FlowState.PENDING.value == "pending"
        assert FlowState.RUNNING.value == "running"
        assert FlowState.COMPLETED.value == "completed"
        assert FlowState.FAILED.value == "failed"
        assert FlowState.PAUSED.value == "paused"
        assert FlowState.CANCELLED.value == "cancelled"
    
    def test_callback_types(self):
        """Test callback type enumerations"""
        # Stream callback types
        assert StreamCallbackType.WORKFLOW.value == "workflow"
        assert StreamCallbackType.TEXT.value == "text"
        assert StreamCallbackType.THINKING.value == "thinking"
        assert StreamCallbackType.TOOL_USE.value == "tool_use"
        
        # Human callback types
        assert HumanCallbackType.CONFIRM.value == "confirm"
        assert HumanCallbackType.INPUT.value == "input"
        assert HumanCallbackType.SELECT.value == "select"
        assert HumanCallbackType.HELP.value == "help"
    
    def test_agent_roles(self):
        """Test agent role enumeration"""
        assert AgentRole.COORDINATOR.value == "coordinator"
        assert AgentRole.SPECIALIST.value == "specialist"
        assert AgentRole.VALIDATOR.value == "validator"
        assert AgentRole.EXECUTOR.value == "executor"
        assert AgentRole.MONITOR.value == "monitor"


class TestEnhancedFlow:
    """Test enhanced flow implementation"""
    
    @pytest.fixture
    def flow_config(self):
        """Create a test flow configuration"""
        return FlowConfig(
            name="test_flow",
            description="Test flow for unit testing",
            timeout=60,
            retry_attempts=2,
            enable_streaming=True,
            enable_human_callbacks=True
        )
    
    @pytest.fixture
    def enhanced_flow(self, flow_config):
        """Create an enhanced flow instance"""
        return EnhancedFlow(flow_config)
    
    @pytest.fixture
    def sample_steps(self):
        """Create sample flow steps"""
        return [
            FlowStep(
                id="step1",
                type="action",
                description="First step",
                parameters={"action": "initialize"}
            ),
            FlowStep(
                id="step2",
                type="action",
                description="Second step",
                parameters={"action": "process"},
                dependencies=["step1"]
            ),
            FlowStep(
                id="step3",
                type="action",
                description="Third step",
                parameters={"action": "finalize"},
                dependencies=["step2"]
            )
        ]
    
    def test_flow_initialization(self, enhanced_flow, flow_config):
        """Test flow initialization"""
        assert enhanced_flow.config == flow_config
        assert enhanced_flow.state == FlowState.PENDING
        assert enhanced_flow.steps == []
        assert enhanced_flow.results == {}
        assert enhanced_flow.context == {}
    
    def test_step_addition(self, enhanced_flow, sample_steps):
        """Test adding steps to flow"""
        for step in sample_steps:
            enhanced_flow.add_step(step)
        
        assert len(enhanced_flow.steps) == 3
        assert enhanced_flow.steps[0].id == "step1"
        assert enhanced_flow.steps[1].id == "step2"
        assert enhanced_flow.steps[2].id == "step3"
    
    def test_dependency_validation(self, enhanced_flow):
        """Test step dependency validation"""
        # Add step with non-existent dependency
        invalid_step = FlowStep(
            id="invalid_step",
            type="action",
            description="Invalid step",
            parameters={},
            dependencies=["nonexistent_step"]
        )
        
        enhanced_flow.add_step(invalid_step)
        
        # Should detect invalid dependency
        with pytest.raises(ValueError, match="Dependency .* not found"):
            enhanced_flow.validate_dependencies()
    
    def test_circular_dependency_detection(self, enhanced_flow):
        """Test circular dependency detection"""
        # Create circular dependency
        step1 = FlowStep(
            id="step1",
            type="action",
            description="Step 1",
            parameters={},
            dependencies=["step2"]
        )
        
        step2 = FlowStep(
            id="step2",
            type="action",
            description="Step 2",
            parameters={},
            dependencies=["step1"]
        )
        
        enhanced_flow.add_step(step1)
        enhanced_flow.add_step(step2)
        
        # Should detect circular dependency
        with pytest.raises(ValueError, match="Circular dependency detected"):
            enhanced_flow.validate_dependencies()
    
    @pytest.mark.asyncio
    async def test_flow_execution(self, enhanced_flow, sample_steps):
        """Test flow execution"""
        # Add steps
        for step in sample_steps:
            enhanced_flow.add_step(step)
        
        # Mock step execution
        async def mock_execute_step(step, context):
            return f"Result for {step.id}"
        
        enhanced_flow._execute_step = mock_execute_step
        
        # Execute flow
        result = await enhanced_flow.execute()
        
        assert result.success is True
        assert enhanced_flow.state == FlowState.COMPLETED
        assert len(enhanced_flow.results) == 3
        assert "step1" in enhanced_flow.results
        assert "step2" in enhanced_flow.results
        assert "step3" in enhanced_flow.results
    
    @pytest.mark.asyncio
    async def test_flow_execution_with_failure(self, enhanced_flow, sample_steps):
        """Test flow execution with step failure"""
        # Add steps
        for step in sample_steps:
            enhanced_flow.add_step(step)
        
        # Mock step execution with failure
        async def mock_execute_step(step, context):
            if step.id == "step2":
                raise Exception("Step 2 failed")
            return f"Result for {step.id}"
        
        enhanced_flow._execute_step = mock_execute_step
        
        # Execute flow
        result = await enhanced_flow.execute()
        
        assert result.success is False
        assert enhanced_flow.state == FlowState.FAILED
        assert "step1" in enhanced_flow.results
        assert "step2" not in enhanced_flow.results
    
    @pytest.mark.asyncio
    async def test_stream_callbacks(self, enhanced_flow):
        """Test stream callback functionality"""
        callback_events = []
        
        def stream_callback(callback_type, data):
            callback_events.append((callback_type, data))
        
        enhanced_flow.set_stream_callback(stream_callback)
        
        # Trigger stream events
        await enhanced_flow._emit_stream_event(StreamCallbackType.WORKFLOW, {"status": "started"})
        await enhanced_flow._emit_stream_event(StreamCallbackType.TEXT, {"content": "Processing..."})
        
        assert len(callback_events) == 2
        assert callback_events[0][0] == StreamCallbackType.WORKFLOW
        assert callback_events[1][0] == StreamCallbackType.TEXT
    
    @pytest.mark.asyncio
    async def test_human_callbacks(self, enhanced_flow):
        """Test human callback functionality"""
        # Mock human callback
        async def mock_human_callback(callback_type, data):
            if callback_type == HumanCallbackType.CONFIRM:
                return True
            elif callback_type == HumanCallbackType.INPUT:
                return "user input"
            elif callback_type == HumanCallbackType.SELECT:
                return 0
            return None
        
        enhanced_flow.set_human_callback(mock_human_callback)
        
        # Test confirmation
        result = await enhanced_flow._request_human_confirmation("Proceed?")
        assert result is True
        
        # Test input
        result = await enhanced_flow._request_human_input("Enter value:")
        assert result == "user input"
        
        # Test selection
        result = await enhanced_flow._request_human_selection("Choose:", ["option1", "option2"])
        assert result == 0
    
    @pytest.mark.asyncio
    async def test_flow_pause_resume(self, enhanced_flow, sample_steps):
        """Test flow pause and resume functionality"""
        # Add steps
        for step in sample_steps:
            enhanced_flow.add_step(step)
        
        # Mock step execution with pause
        async def mock_execute_step(step, context):
            if step.id == "step2":
                enhanced_flow.pause()
            return f"Result for {step.id}"
        
        enhanced_flow._execute_step = mock_execute_step
        
        # Start execution (should pause at step2)
        execution_task = asyncio.create_task(enhanced_flow.execute())
        await asyncio.sleep(0.1)  # Allow execution to start
        
        assert enhanced_flow.state == FlowState.PAUSED
        
        # Resume execution
        enhanced_flow.resume()
        result = await execution_task
        
        assert result.success is True
        assert enhanced_flow.state == FlowState.COMPLETED
    
    @pytest.mark.asyncio
    async def test_flow_cancellation(self, enhanced_flow, sample_steps):
        """Test flow cancellation"""
        # Add steps
        for step in sample_steps:
            enhanced_flow.add_step(step)
        
        # Mock long-running step execution
        async def mock_execute_step(step, context):
            await asyncio.sleep(1)  # Simulate long operation
            return f"Result for {step.id}"
        
        enhanced_flow._execute_step = mock_execute_step
        
        # Start execution and cancel
        execution_task = asyncio.create_task(enhanced_flow.execute())
        await asyncio.sleep(0.1)  # Allow execution to start
        
        enhanced_flow.cancel()
        result = await execution_task
        
        assert result.success is False
        assert enhanced_flow.state == FlowState.CANCELLED
    
    def test_context_management(self, enhanced_flow):
        """Test flow context management"""
        # Set context values
        enhanced_flow.set_context("key1", "value1")
        enhanced_flow.set_context("key2", {"nested": "value"})
        
        assert enhanced_flow.get_context("key1") == "value1"
        assert enhanced_flow.get_context("key2") == {"nested": "value"}
        assert enhanced_flow.get_context("nonexistent") is None
        
        # Update context
        enhanced_flow.update_context({"key3": "value3", "key1": "updated"})
        
        assert enhanced_flow.get_context("key1") == "updated"
        assert enhanced_flow.get_context("key3") == "value3"


class TestMultiAgentOrchestrator:
    """Test multi-agent orchestrator"""
    
    @pytest.fixture
    def orchestrator(self):
        """Create a multi-agent orchestrator"""
        return MultiAgentOrchestrator()
    
    @pytest.fixture
    def sample_agents(self):
        """Create sample agents for testing"""
        return [
            {
                "id": "agent1",
                "name": "Coordinator Agent",
                "role": AgentRole.COORDINATOR,
                "capabilities": ["planning", "coordination"],
                "max_concurrent_tasks": 5
            },
            {
                "id": "agent2",
                "name": "Specialist Agent",
                "role": AgentRole.SPECIALIST,
                "capabilities": ["analysis", "processing"],
                "max_concurrent_tasks": 3
            },
            {
                "id": "agent3",
                "name": "Validator Agent",
                "role": AgentRole.VALIDATOR,
                "capabilities": ["validation", "quality_check"],
                "max_concurrent_tasks": 2
            }
        ]
    
    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization"""
        assert orchestrator.agents == {}
        assert orchestrator.active_tasks == {}
        assert orchestrator.communication_log == []
        assert orchestrator.performance_metrics == {}
    
    def test_agent_registration(self, orchestrator, sample_agents):
        """Test agent registration"""
        for agent_config in sample_agents:
            orchestrator.register_agent(agent_config)
        
        assert len(orchestrator.agents) == 3
        assert "agent1" in orchestrator.agents
        assert orchestrator.agents["agent1"]["role"] == AgentRole.COORDINATOR
    
    def test_duplicate_agent_registration(self, orchestrator, sample_agents):
        """Test duplicate agent registration handling"""
        agent_config = sample_agents[0]
        orchestrator.register_agent(agent_config)
        
        # Should raise error for duplicate registration
        with pytest.raises(ValueError, match="Agent .* already registered"):
            orchestrator.register_agent(agent_config)
    
    def test_agent_selection_by_capability(self, orchestrator, sample_agents):
        """Test agent selection by capability"""
        for agent_config in sample_agents:
            orchestrator.register_agent(agent_config)
        
        # Find agents with specific capability
        planning_agents = orchestrator.find_agents_by_capability("planning")
        assert len(planning_agents) == 1
        assert planning_agents[0]["id"] == "agent1"
        
        analysis_agents = orchestrator.find_agents_by_capability("analysis")
        assert len(analysis_agents) == 1
        assert analysis_agents[0]["id"] == "agent2"
    
    def test_agent_selection_by_role(self, orchestrator, sample_agents):
        """Test agent selection by role"""
        for agent_config in sample_agents:
            orchestrator.register_agent(agent_config)
        
        # Find agents by role
        coordinators = orchestrator.find_agents_by_role(AgentRole.COORDINATOR)
        assert len(coordinators) == 1
        assert coordinators[0]["id"] == "agent1"
        
        specialists = orchestrator.find_agents_by_role(AgentRole.SPECIALIST)
        assert len(specialists) == 1
        assert specialists[0]["id"] == "agent2"
    
    @pytest.mark.asyncio
    async def test_task_assignment(self, orchestrator, sample_agents):
        """Test task assignment to agents"""
        for agent_config in sample_agents:
            orchestrator.register_agent(agent_config)
        
        # Mock agent execution
        async def mock_execute_task(agent_id, task):
            return f"Task {task['id']} completed by {agent_id}"
        
        orchestrator._execute_agent_task = mock_execute_task
        
        # Assign task
        task = {
            "id": "task1",
            "type": "analysis",
            "description": "Analyze data",
            "required_capabilities": ["analysis"]
        }
        
        result = await orchestrator.assign_task(task)
        
        assert result["success"] is True
        assert "agent2" in result["assigned_agent"]  # Should assign to specialist
    
    @pytest.mark.asyncio
    async def test_task_assignment_no_capable_agent(self, orchestrator, sample_agents):
        """Test task assignment when no agent has required capability"""
        for agent_config in sample_agents:
            orchestrator.register_agent(agent_config)
        
        # Task requiring non-existent capability
        task = {
            "id": "task1",
            "type": "unknown",
            "description": "Unknown task",
            "required_capabilities": ["nonexistent_capability"]
        }
        
        result = await orchestrator.assign_task(task)
        
        assert result["success"] is False
        assert "No capable agent found" in result["error"]
    
    @pytest.mark.asyncio
    async def test_inter_agent_communication(self, orchestrator, sample_agents):
        """Test inter-agent communication"""
        for agent_config in sample_agents:
            orchestrator.register_agent(agent_config)
        
        # Send message between agents
        message = {
            "type": CommunicationType.REQUEST,
            "content": "Need analysis results",
            "data": {"request_id": "req1"}
        }
        
        await orchestrator.send_message("agent1", "agent2", message)
        
        # Check communication log
        assert len(orchestrator.communication_log) == 1
        log_entry = orchestrator.communication_log[0]
        assert log_entry["from_agent"] == "agent1"
        assert log_entry["to_agent"] == "agent2"
        assert log_entry["message"]["type"] == CommunicationType.REQUEST
    
    @pytest.mark.asyncio
    async def test_broadcast_message(self, orchestrator, sample_agents):
        """Test broadcast message to all agents"""
        for agent_config in sample_agents:
            orchestrator.register_agent(agent_config)
        
        # Broadcast message
        message = {
            "type": CommunicationType.BROADCAST,
            "content": "System announcement",
            "data": {"priority": "high"}
        }
        
        await orchestrator.broadcast_message("agent1", message)
        
        # Should have 2 log entries (to agent2 and agent3)
        assert len(orchestrator.communication_log) == 2
    
    def test_conflict_detection(self, orchestrator, sample_agents):
        """Test conflict detection between agents"""
        for agent_config in sample_agents:
            orchestrator.register_agent(agent_config)
        
        # Create conflicting tasks
        task1 = {
            "id": "task1",
            "agent_id": "agent1",
            "resource": "database",
            "priority": 5
        }
        
        task2 = {
            "id": "task2",
            "agent_id": "agent2",
            "resource": "database",
            "priority": 3
        }
        
        conflicts = orchestrator.detect_conflicts([task1, task2])
        
        assert len(conflicts) == 1
        conflict = conflicts[0]
        assert conflict["type"] == ConflictType.RESOURCE
        assert conflict["tasks"] == ["task1", "task2"]
    
    def test_conflict_resolution(self, orchestrator):
        """Test conflict resolution strategies"""
        # Resource conflict
        conflict = {
            "type": ConflictType.RESOURCE,
            "tasks": ["task1", "task2"],
            "details": {
                "resource": "database",
                "task1_priority": 5,
                "task2_priority": 3
            }
        }
        
        resolution = orchestrator.resolve_conflict(conflict)
        
        assert resolution["strategy"] == ConflictResolution.PRIORITY
        assert resolution["winner"] == "task1"  # Higher priority
    
    def test_performance_tracking(self, orchestrator, sample_agents):
        """Test agent performance tracking"""
        for agent_config in sample_agents:
            orchestrator.register_agent(agent_config)
        
        # Record performance metrics
        orchestrator.record_performance("agent1", {
            "task_id": "task1",
            "duration": 5.0,
            "success": True,
            "quality_score": 0.95
        })
        
        orchestrator.record_performance("agent1", {
            "task_id": "task2",
            "duration": 3.0,
            "success": True,
            "quality_score": 0.88
        })
        
        # Get performance metrics
        metrics = orchestrator.get_performance_metrics("agent1")
        
        assert metrics["total_tasks"] == 2
        assert metrics["success_rate"] == 1.0
        assert metrics["average_duration"] == 4.0
        assert metrics["average_quality"] == 0.915
    
    @pytest.mark.asyncio
    async def test_load_balancing(self, orchestrator, sample_agents):
        """Test load balancing across agents"""
        for agent_config in sample_agents:
            orchestrator.register_agent(agent_config)
        
        # Simulate different load levels
        orchestrator.active_tasks["agent1"] = ["task1", "task2", "task3"]
        orchestrator.active_tasks["agent2"] = ["task4"]
        orchestrator.active_tasks["agent3"] = []
        
        # Find least loaded agent
        least_loaded = orchestrator.find_least_loaded_agent(["agent1", "agent2", "agent3"])
        
        assert least_loaded["id"] == "agent3"  # No active tasks
    
    @pytest.mark.asyncio
    async def test_orchestrator_workflow(self, orchestrator, sample_agents):
        """Test complete orchestrator workflow"""
        for agent_config in sample_agents:
            orchestrator.register_agent(agent_config)
        
        # Mock agent execution
        async def mock_execute_task(agent_id, task):
            await asyncio.sleep(0.1)  # Simulate work
            return {
                "success": True,
                "result": f"Task {task['id']} completed by {agent_id}",
                "duration": 0.1
            }
        
        orchestrator._execute_agent_task = mock_execute_task
        
        # Create workflow with multiple tasks
        workflow = {
            "id": "workflow1",
            "tasks": [
                {
                    "id": "task1",
                    "type": "planning",
                    "required_capabilities": ["planning"],
                    "dependencies": []
                },
                {
                    "id": "task2",
                    "type": "analysis",
                    "required_capabilities": ["analysis"],
                    "dependencies": ["task1"]
                },
                {
                    "id": "task3",
                    "type": "validation",
                    "required_capabilities": ["validation"],
                    "dependencies": ["task2"]
                }
            ]
        }
        
        # Execute workflow
        result = await orchestrator.execute_workflow(workflow)
        
        assert result["success"] is True
        assert len(result["task_results"]) == 3
        assert all(task_result["success"] for task_result in result["task_results"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

