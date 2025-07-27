# Enhanced Flow System Implementation Guide

## Overview

The Enhanced Flow System for OpenManus provides a comprehensive, production-ready workflow execution platform with real-time stream callbacks, human-in-the-loop interaction, multi-agent orchestration, and advanced state management. This implementation is inspired by the Eko framework's callback system and extends it with OpenManus-specific features.

## Architecture Overview

The enhanced flow system consists of several interconnected components that work together to provide robust, scalable, and interactive workflow execution:

### Core Components

1. **Enhanced Interfaces** (`enhanced_interfaces.py`)
   - Comprehensive interface definitions for all flow components
   - Type-safe data classes with validation
   - Stream and human callback interfaces
   - Multi-agent orchestration interfaces

2. **Enhanced Flow** (`enhanced_flow.py`)
   - Main flow execution engine with callback support
   - Stream callback management for real-time updates
   - Human callback system for interactive workflows
   - State persistence and memory management

3. **Multi-Agent Orchestrator** (`multi_agent_orchestrator.py`)
   - Coordinates multiple agents working together
   - Inter-agent communication system
   - Conflict resolution between agents
   - Resource allocation and task distribution

## Key Features

### Stream Callback System

The enhanced flow system supports comprehensive real-time streaming updates:

```python
from app.flow import StreamCallbackType, IStreamCallback, StreamCallbackData

class CustomStreamCallback(IStreamCallback):
    async def on_stream_event(self, data: StreamCallbackData) -> None:
        if data.type == StreamCallbackType.WORKFLOW:
            print(f"Workflow event: {data.content}")
        elif data.type == StreamCallbackType.TOOL_USE:
            print(f"Tool being used: {data.content}")
        elif data.type == StreamCallbackType.THINKING:
            print(f"AI thinking: {data.content}")
```

**Available Stream Callback Types:**
- `WORKFLOW`: High-level workflow events
- `TEXT`: Text generation and processing
- `THINKING`: AI reasoning and decision-making
- `TOOL_STREAMING`: Real-time tool execution updates
- `TOOL_USE`: Tool invocation events
- `TOOL_RUNNING`: Tool execution progress
- `TOOL_RESULT`: Tool completion results
- `FILE`: File operations and results
- `ERROR`: Error events and exceptions
- `FINISH`: Workflow completion events

### Human-in-the-Loop System

The system provides comprehensive human interaction capabilities:

```python
from app.flow import IHumanCallback, HumanCallbackData, HumanCallbackResponse

class CustomHumanCallback(IHumanCallback):
    async def on_human_confirm(self, data: HumanCallbackData) -> HumanCallbackResponse:
        # Request user confirmation
        user_response = await self.get_user_confirmation(data.message)
        return HumanCallbackResponse(
            callback_id=str(data.timestamp),
            response_type="confirm",
            value=user_response
        )
    
    async def on_human_input(self, data: HumanCallbackData) -> HumanCallbackResponse:
        # Request user input
        user_input = await self.get_user_input(data.message, data.default_value)
        return HumanCallbackResponse(
            callback_id=str(data.timestamp),
            response_type="input",
            value=user_input
        )
```

**Human Callback Types:**
- `onHumanConfirm`: Request user confirmation for actions
- `onHumanInput`: Request user input for parameters
- `onHumanSelect`: Request user selection from options
- `onHumanHelp`: Request user assistance or guidance

### Multi-Agent Orchestration

The system supports coordinating multiple agents for complex tasks:

```python
from app.flow import create_multi_agent_orchestrator, AgentInfo, AgentRole

# Create orchestrator
orchestrator = create_multi_agent_orchestrator()

# Register agents
research_agent = AgentInfo(
    id="research_agent",
    name="Research Specialist",
    role=AgentRole.SPECIALIST,
    capabilities=["web_search", "data_analysis", "summarization"],
    specializations=["academic_research", "market_analysis"]
)

writing_agent = AgentInfo(
    id="writing_agent", 
    name="Content Writer",
    role=AgentRole.EXECUTOR,
    capabilities=["content_generation", "editing", "formatting"],
    specializations=["technical_writing", "documentation"]
)

orchestrator.register_agent(research_agent)
orchestrator.register_agent(writing_agent)

# Orchestrate agents for a task
result = await orchestrator.orchestrate_agents(
    agents=[research_agent, writing_agent],
    task="Create a comprehensive research report on AI trends",
    context={"domain": "artificial_intelligence", "length": "10_pages"}
)
```

## Implementation Details

### Enhanced Flow Interfaces

The system provides comprehensive interfaces for all components:

```python
@dataclass
class FlowStep:
    """Individual step in a flow"""
    id: str
    name: str
    type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    timeout: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FlowPlan:
    """Flow execution plan"""
    id: str
    name: str
    description: str
    steps: List[FlowStep] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: int = 1
```

### Stream Callback Management

The stream callback system provides real-time updates throughout flow execution:

```python
class StreamCallbackManager:
    """Manages stream callbacks for flow execution"""
    
    def __init__(self):
        self.callbacks: Set[IStreamCallback] = set()
    
    async def emit(self, callback_type: StreamCallbackType, content: Any,
                  flow_id: str = None, step_id: str = None, 
                  metadata: Dict[str, Any] = None) -> None:
        """Emit stream callback to all registered callbacks"""
        data = StreamCallbackData(
            type=callback_type,
            content=content,
            metadata=metadata or {},
            flow_id=flow_id,
            step_id=step_id
        )
        
        for callback in self.callbacks:
            await callback.on_stream_event(data)
```

### Human Callback Management

The human callback system enables interactive workflows:

```python
class HumanCallbackManager:
    """Manages human callbacks for flow execution"""
    
    async def request_confirmation(self, message: str, flow_id: str = None,
                                 step_id: str = None, timeout: int = 3600) -> bool:
        """Request human confirmation"""
        data = HumanCallbackData(
            type=HumanCallbackType.CONFIRM,
            message=message,
            timeout=timeout,
            flow_id=flow_id,
            step_id=step_id
        )
        
        response = await self.callback_handler.on_human_confirm(data)
        return response.value
```

### Enhanced Flow Planning

The planning system creates optimized execution plans:

```python
class EnhancedFlowPlanner(IFlowPlanner):
    """Enhanced flow planner with AI-powered planning"""
    
    async def create_plan(self, goal: str, context: Dict[str, Any]) -> FlowPlan:
        """Create a flow plan for the given goal"""
        plan = FlowPlan(
            id=generate_flow_id(),
            name=f"Plan for: {goal}",
            description=f"Automatically generated plan to achieve: {goal}"
        )
        
        # Analyze goal and create appropriate steps
        steps = await self._analyze_goal_and_create_steps(goal, context)
        plan.steps = steps
        
        return plan
    
    async def replan(self, flow_id: str, current_state: FlowState,
                    new_context: Dict[str, Any]) -> FlowPlan:
        """Replan flow based on current state and new context"""
        remaining_goal = await self._analyze_remaining_work(current_state, new_context)
        new_plan = await self.create_plan(remaining_goal, new_context)
        new_plan.version = current_state.metadata.get("plan_version", 1) + 1
        return new_plan
```

## Usage Examples

### Basic Enhanced Flow

```python
import asyncio
from app.flow import create_enhanced_flow, create_default_config

async def main():
    # Create enhanced flow
    flow = create_enhanced_flow()
    
    # Initialize with configuration
    config = create_default_config()
    config.enable_stream_callbacks = True
    config.enable_human_callbacks = True
    await flow.initialize(config)
    
    # Plan the flow
    plan = await flow.plan(
        goal="Analyze market trends and create report",
        context={"industry": "technology", "timeframe": "2024"}
    )
    
    # Execute the flow
    state = await flow.execute(plan)
    
    print(f"Flow completed with status: {state.status}")
    print(f"Execution time: {state.execution_time:.2f} seconds")

asyncio.run(main())
```

### Interactive Flow with Human Callbacks

```python
from app.flow import create_interactive_flow, ConsoleHumanCallback

async def main():
    # Create interactive flow
    flow = create_interactive_flow()
    
    # Set up human callback handler
    human_callback = ConsoleHumanCallback()
    flow.set_human_callback(human_callback)
    
    # Plan and execute
    plan = await flow.plan("Create presentation with user input")
    state = await flow.execute(plan)

asyncio.run(main())
```

### Stream Callback Integration

```python
from app.flow import LoggingStreamCallback, StreamCallbackType

class CustomStreamCallback(LoggingStreamCallback):
    async def on_stream_event(self, data):
        if data.type == StreamCallbackType.THINKING:
            print(f"🤔 AI is thinking: {data.content}")
        elif data.type == StreamCallbackType.TOOL_USE:
            print(f"🔧 Using tool: {data.content}")
        elif data.type == StreamCallbackType.ERROR:
            print(f"❌ Error occurred: {data.content}")
        else:
            await super().on_stream_event(data)

async def main():
    flow = create_enhanced_flow()
    
    # Add custom stream callback
    callback = CustomStreamCallback()
    flow.add_stream_callback(callback)
    
    # Execute with real-time updates
    async for update in flow.stream_execute():
        print(f"Stream update: {update.type.value}")

asyncio.run(main())
```

### Multi-Agent Coordination

```python
from app.flow import create_multi_agent_flow, AgentInfo, AgentRole

async def main():
    # Define agents
    agents = [
        {
            "id": "researcher",
            "name": "Research Agent",
            "role": "specialist",
            "capabilities": ["web_search", "data_analysis"],
            "specializations": ["academic_research"]
        },
        {
            "id": "writer", 
            "name": "Writing Agent",
            "role": "executor",
            "capabilities": ["content_generation", "editing"],
            "specializations": ["technical_writing"]
        },
        {
            "id": "reviewer",
            "name": "Review Agent", 
            "role": "validator",
            "capabilities": ["quality_check", "fact_verification"],
            "specializations": ["content_review"]
        }
    ]
    
    # Create multi-agent flow
    flow, orchestrator = create_multi_agent_flow(agents)
    
    # Orchestrate agents for complex task
    result = await orchestrator.orchestrate_agents(
        agents=agents,
        task="Create comprehensive AI research report",
        context={
            "topic": "Large Language Models",
            "length": "15_pages",
            "audience": "technical",
            "deadline": "2024-12-31"
        }
    )
    
    print(f"Multi-agent task completed: {result}")

asyncio.run(main())
```

### Advanced Flow with State Persistence

```python
from app.flow import EnhancedFlow, InMemoryStateManager, InMemoryMemoryManager

async def main():
    # Create flow with custom state management
    state_manager = InMemoryStateManager()
    memory_manager = InMemoryMemoryManager()
    
    flow = EnhancedFlow()
    flow.executor.state_manager = state_manager
    flow.executor.memory_manager = memory_manager
    
    # Execute long-running flow
    plan = await flow.plan("Long-running data processing task")
    
    # Store important data in memory
    await memory_manager.store_memory(plan.id, "checkpoint_1", {"progress": 25})
    
    # Execute with state persistence
    state = await flow.execute(plan)
    
    # Retrieve stored memory
    checkpoint = await memory_manager.retrieve_memory(plan.id, "checkpoint_1")
    print(f"Retrieved checkpoint: {checkpoint}")

asyncio.run(main())
```

### Custom Step Types

```python
from app.flow import FlowStep, create_flow_step

# Create custom step types
def create_mcp_tool_step(tool_name: str, arguments: dict, dependencies: list = None):
    """Create MCP tool execution step"""
    return create_flow_step(
        name=f"Execute {tool_name}",
        step_type="mcp_tool",
        parameters={
            "tool_name": tool_name,
            "arguments": arguments
        },
        dependencies=dependencies or []
    )

def create_human_input_step(message: str, input_type: str = "input", 
                           options: list = None, dependencies: list = None):
    """Create human input step"""
    return create_flow_step(
        name="Human Input Required",
        step_type="human_input",
        parameters={
            "message": message,
            "input_type": input_type,
            "options": options
        },
        dependencies=dependencies or []
    )

def create_condition_step(condition: str, dependencies: list = None):
    """Create conditional step"""
    return create_flow_step(
        name="Conditional Check",
        step_type="condition",
        parameters={
            "condition": condition
        },
        dependencies=dependencies or []
    )

# Usage example
async def main():
    from app.flow import FlowPlan, generate_flow_id
    
    # Create custom plan with various step types
    plan = FlowPlan(
        id=generate_flow_id(),
        name="Custom Workflow",
        description="Workflow with custom step types"
    )
    
    # Add steps
    search_step = create_mcp_tool_step(
        tool_name="web_search",
        arguments={"query": "AI trends 2024"}
    )
    
    confirm_step = create_human_input_step(
        message="Do you want to proceed with analysis?",
        input_type="confirm",
        dependencies=[search_step.id]
    )
    
    condition_step = create_condition_step(
        condition="context['user_confirmed'] == True",
        dependencies=[confirm_step.id]
    )
    
    plan.steps = [search_step, confirm_step, condition_step]
    
    # Execute custom plan
    flow = create_enhanced_flow()
    state = await flow.execute(plan)

asyncio.run(main())
```

## Advanced Features

### Flow Event Listeners

Monitor flow execution with comprehensive event listeners:

```python
from app.flow import IFlowEventListener

class FlowMonitor(IFlowEventListener):
    async def on_flow_started(self, flow_id: str, plan: FlowPlan) -> None:
        print(f"📊 Flow started: {plan.name} ({flow_id})")
    
    async def on_flow_completed(self, flow_id: str, state: FlowState) -> None:
        print(f"✅ Flow completed: {flow_id} in {state.execution_time:.2f}s")
    
    async def on_flow_failed(self, flow_id: str, state: FlowState, error: Exception) -> None:
        print(f"❌ Flow failed: {flow_id} - {error}")
    
    async def on_step_started(self, flow_id: str, step: FlowStep) -> None:
        print(f"🔄 Step started: {step.name}")
    
    async def on_step_completed(self, flow_id: str, step: FlowStep) -> None:
        print(f"✓ Step completed: {step.name}")
    
    async def on_step_failed(self, flow_id: str, step: FlowStep, error: Exception) -> None:
        print(f"✗ Step failed: {step.name} - {error}")

# Usage
flow = create_enhanced_flow()
monitor = FlowMonitor()
flow.add_event_listener(monitor)
```

### Inter-Agent Communication

Enable agents to communicate and coordinate:

```python
from app.flow import AgentCommunicationHub, AgentMessage, CommunicationType

async def main():
    hub = AgentCommunicationHub()
    
    # Agent 1 sends message to Agent 2
    message = AgentMessage(
        id="msg_001",
        sender_id="agent_1",
        receiver_id="agent_2",
        message_type=CommunicationType.REQUEST,
        content={
            "action": "analyze_data",
            "data": {"dataset": "sales_2024.csv"},
            "priority": "high"
        },
        requires_response=True,
        response_timeout=300
    )
    
    # Send message and wait for response
    response = await hub.send_message(message)
    print(f"Response received: {response}")
    
    # Agent 2 receives messages
    messages = await hub.receive_messages("agent_2")
    for msg in messages:
        print(f"Received: {msg.content}")
        
        # Send response
        await hub.send_response(msg.id, {
            "status": "completed",
            "result": "Analysis complete"
        })

asyncio.run(main())
```

### Conflict Resolution

Handle conflicts between agents automatically:

```python
from app.flow import ConflictResolver, Conflict, ConflictType

async def main():
    resolver = ConflictResolver()
    
    # Define a resource conflict
    conflict = Conflict(
        id="conflict_001",
        conflict_type=ConflictType.RESOURCE_CONFLICT,
        description="Two agents need the same GPU resource",
        involved_agents=["agent_1", "agent_2"],
        involved_tasks=["task_1", "task_2"],
        severity=3
    )
    
    # Resolve conflict
    resolution = await resolver.resolve_conflict(conflict, agents, tasks)
    print(f"Conflict resolved: {resolution}")

asyncio.run(main())
```

## Integration with OpenManus

### MCP Integration

The enhanced flow system integrates seamlessly with the enhanced MCP system:

```python
from app.mcp import create_production_client
from app.flow import create_enhanced_flow

async def main():
    # Create MCP client
    mcp_client = await create_production_client("http://localhost:8081")
    await mcp_client.connect()
    
    # Create flow with MCP integration
    flow = create_enhanced_flow(mcp_client)
    
    # Plan flow that uses MCP tools
    plan = await flow.plan(
        goal="Process data using MCP tools",
        context={"data_source": "database", "output_format": "report"}
    )
    
    # Execute flow with MCP tool integration
    state = await flow.execute(plan)

asyncio.run(main())
```

### Agent Integration

Integrate with OpenManus agents:

```python
from app.agents import BaseAgent
from app.flow import create_enhanced_flow

class FlowEnabledAgent(BaseAgent):
    def __init__(self, mcp_client):
        super().__init__()
        self.flow = create_enhanced_flow(mcp_client)
    
    async def execute_complex_task(self, task_description: str):
        # Use enhanced flow for complex task execution
        plan = await self.flow.plan(task_description)
        state = await self.flow.execute(plan)
        return state.variables.get("final_result")
```

## Performance Optimization

### Parallel Execution

The system supports parallel step execution:

```python
# Steps without dependencies can execute in parallel
step1 = create_flow_step("Task A", "parallel_task")
step2 = create_flow_step("Task B", "parallel_task") 
step3 = create_flow_step("Task C", "parallel_task")

# Synchronization step waits for all parallel tasks
sync_step = create_flow_step(
    "Synchronize Results", 
    "synchronization",
    dependencies=[step1.id, step2.id, step3.id]
)
```

### Memory Management

Efficient memory usage with automatic cleanup:

```python
from app.flow import InMemoryMemoryManager

memory_manager = InMemoryMemoryManager()

# Store temporary data
await memory_manager.store_memory(flow_id, "temp_data", large_dataset)

# Retrieve when needed
data = await memory_manager.retrieve_memory(flow_id, "temp_data")

# Clean up when done
await memory_manager.delete_memory(flow_id, "temp_data")
```

### State Persistence

Robust state persistence for long-running flows:

```python
# Automatic state saving during execution
config = create_default_config()
config.auto_save_interval = 30  # Save every 30 seconds

flow = create_enhanced_flow()
await flow.initialize(config)

# State is automatically saved and can be recovered
state = await flow.get_state()
```

## Error Handling and Recovery

### Retry Mechanisms

Built-in retry logic for failed steps:

```python
step = create_flow_step(
    "Unreliable Operation",
    "network_request",
    parameters={
        "url": "https://api.example.com/data",
        "max_retries": 5,
        "retry_delay": 2,
        "exponential_backoff": True
    }
)
```

### Graceful Degradation

Handle failures gracefully:

```python
class RobustFlowExecutor(EnhancedFlowExecutor):
    async def execute_step(self, step: FlowStep, context: Dict[str, Any]) -> Any:
        try:
            return await super().execute_step(step, context)
        except Exception as e:
            # Log error but continue if step is not critical
            if not step.parameters.get("critical", False):
                self.logger.warning(f"Non-critical step failed: {step.name}")
                return {"error": str(e), "status": "failed_non_critical"}
            raise
```

### Recovery Strategies

Implement recovery strategies for different failure types:

```python
async def recover_from_failure(flow_id: str, error_type: str):
    state = await state_manager.load_state(flow_id)
    
    if error_type == "network_timeout":
        # Retry with longer timeout
        await flow.resume()
    elif error_type == "resource_unavailable":
        # Wait and retry
        await asyncio.sleep(60)
        await flow.resume()
    elif error_type == "data_corruption":
        # Replan from last checkpoint
        checkpoint = await memory_manager.retrieve_memory(flow_id, "last_checkpoint")
        new_plan = await planner.replan(flow_id, state, checkpoint)
        await flow.execute(new_plan)
```

## Best Practices

### Flow Design

1. **Modular Steps**: Design steps to be small, focused, and reusable
2. **Clear Dependencies**: Explicitly define step dependencies
3. **Error Handling**: Include error handling and recovery strategies
4. **Human Interaction**: Use human callbacks judiciously for critical decisions
5. **Resource Management**: Consider resource constraints and conflicts

### Performance

1. **Parallel Execution**: Identify steps that can run in parallel
2. **Memory Efficiency**: Clean up temporary data promptly
3. **State Management**: Use appropriate state persistence strategies
4. **Monitoring**: Implement comprehensive monitoring and logging
5. **Optimization**: Profile and optimize critical paths

### Security

1. **Input Validation**: Validate all inputs and parameters
2. **Access Control**: Implement proper access controls for sensitive operations
3. **Audit Logging**: Log all significant events and decisions
4. **Secure Communication**: Use secure channels for inter-agent communication
5. **Data Protection**: Protect sensitive data in memory and storage

This enhanced flow system provides a robust, scalable foundation for complex workflow execution in the OpenManus system, with comprehensive real-time capabilities, human interaction, and multi-agent coordination that significantly exceeds the capabilities of the original implementation while maintaining backward compatibility.

