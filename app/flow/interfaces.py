"""
Enhanced Flow Interfaces for OpenManus
Inspired by Eko framework with comprehensive callback systems and state management
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Callable, AsyncGenerator, Set
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from datetime import datetime
import uuid


class FlowStatus(Enum):
    """Flow execution status"""
    CREATED = "created"
    PLANNING = "planning"
    EXECUTING = "executing"
    PAUSED = "paused"
    WAITING_HUMAN = "waiting_human"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StreamCallbackType(Enum):
    """Types of stream callbacks"""
    WORKFLOW = "workflow"
    TEXT = "text"
    THINKING = "thinking"
    TOOL_STREAMING = "tool_streaming"
    TOOL_USE = "tool_use"
    TOOL_RUNNING = "tool_running"
    TOOL_RESULT = "tool_result"
    FILE = "file"
    ERROR = "error"
    FINISH = "finish"


class HumanCallbackType(Enum):
    """Types of human callbacks"""
    CONFIRM = "onHumanConfirm"
    INPUT = "onHumanInput"
    SELECT = "onHumanSelect"
    HELP = "onHumanHelp"


class FlowPriority(Enum):
    """Flow execution priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class StreamCallbackData:
    """Data structure for stream callbacks"""
    type: StreamCallbackType
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    flow_id: Optional[str] = None
    step_id: Optional[str] = None


@dataclass
class HumanCallbackData:
    """Data structure for human callbacks"""
    type: HumanCallbackType
    message: str
    options: Optional[List[str]] = None
    default_value: Optional[str] = None
    timeout: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    flow_id: Optional[str] = None
    step_id: Optional[str] = None


@dataclass
class HumanCallbackResponse:
    """Response from human callback"""
    callback_id: str
    response_type: str  # "confirm", "input", "select", "help"
    value: Any
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


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


@dataclass
class FlowState:
    """Flow execution state"""
    flow_id: str
    status: FlowStatus
    current_step: Optional[str] = None
    completed_steps: Set[str] = field(default_factory=set)
    failed_steps: Set[str] = field(default_factory=set)
    variables: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    last_activity: datetime = field(default_factory=datetime.now)
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FlowConfiguration:
    """Flow configuration settings"""
    max_execution_time: int = 3600  # 1 hour
    max_steps: int = 1000
    enable_replanning: bool = True
    enable_human_callbacks: bool = True
    enable_stream_callbacks: bool = True
    auto_save_interval: int = 30  # seconds
    retry_policy: Dict[str, Any] = field(default_factory=lambda: {
        "max_retries": 3,
        "retry_delay": 1,
        "exponential_backoff": True
    })
    timeout_policy: Dict[str, Any] = field(default_factory=lambda: {
        "default_step_timeout": 300,  # 5 minutes
        "human_callback_timeout": 3600  # 1 hour
    })


class IStreamCallback(ABC):
    """Interface for stream callbacks"""
    
    @abstractmethod
    async def on_stream_event(self, data: StreamCallbackData) -> None:
        """Handle stream event"""
        pass


class IHumanCallback(ABC):
    """Interface for human callbacks"""
    
    @abstractmethod
    async def on_human_confirm(self, data: HumanCallbackData) -> HumanCallbackResponse:
        """Handle human confirmation request"""
        pass
    
    @abstractmethod
    async def on_human_input(self, data: HumanCallbackData) -> HumanCallbackResponse:
        """Handle human input request"""
        pass
    
    @abstractmethod
    async def on_human_select(self, data: HumanCallbackData) -> HumanCallbackResponse:
        """Handle human selection request"""
        pass
    
    @abstractmethod
    async def on_human_help(self, data: HumanCallbackData) -> HumanCallbackResponse:
        """Handle human help request"""
        pass


class IFlowEventListener(ABC):
    """Interface for flow event listeners"""
    
    @abstractmethod
    async def on_flow_started(self, flow_id: str, plan: FlowPlan) -> None:
        """Called when flow starts"""
        pass
    
    @abstractmethod
    async def on_flow_completed(self, flow_id: str, state: FlowState) -> None:
        """Called when flow completes"""
        pass
    
    @abstractmethod
    async def on_flow_failed(self, flow_id: str, state: FlowState, error: Exception) -> None:
        """Called when flow fails"""
        pass
    
    @abstractmethod
    async def on_step_started(self, flow_id: str, step: FlowStep) -> None:
        """Called when step starts"""
        pass
    
    @abstractmethod
    async def on_step_completed(self, flow_id: str, step: FlowStep) -> None:
        """Called when step completes"""
        pass
    
    @abstractmethod
    async def on_step_failed(self, flow_id: str, step: FlowStep, error: Exception) -> None:
        """Called when step fails"""
        pass


class IFlowPlanner(ABC):
    """Interface for flow planning"""
    
    @abstractmethod
    async def create_plan(self, goal: str, context: Dict[str, Any]) -> FlowPlan:
        """Create a flow plan for the given goal"""
        pass
    
    @abstractmethod
    async def replan(self, flow_id: str, current_state: FlowState, 
                    new_context: Dict[str, Any]) -> FlowPlan:
        """Replan flow based on current state and new context"""
        pass
    
    @abstractmethod
    async def validate_plan(self, plan: FlowPlan) -> bool:
        """Validate a flow plan"""
        pass
    
    @abstractmethod
    async def optimize_plan(self, plan: FlowPlan) -> FlowPlan:
        """Optimize a flow plan for better performance"""
        pass


class IFlowExecutor(ABC):
    """Interface for flow execution"""
    
    @abstractmethod
    async def execute_flow(self, plan: FlowPlan, 
                          initial_context: Dict[str, Any] = None) -> FlowState:
        """Execute a flow plan"""
        pass
    
    @abstractmethod
    async def execute_step(self, step: FlowStep, context: Dict[str, Any]) -> Any:
        """Execute a single flow step"""
        pass
    
    @abstractmethod
    async def pause_flow(self, flow_id: str) -> bool:
        """Pause flow execution"""
        pass
    
    @abstractmethod
    async def resume_flow(self, flow_id: str) -> bool:
        """Resume flow execution"""
        pass
    
    @abstractmethod
    async def cancel_flow(self, flow_id: str) -> bool:
        """Cancel flow execution"""
        pass


class IFlowStateManager(ABC):
    """Interface for flow state management"""
    
    @abstractmethod
    async def save_state(self, state: FlowState) -> bool:
        """Save flow state"""
        pass
    
    @abstractmethod
    async def load_state(self, flow_id: str) -> Optional[FlowState]:
        """Load flow state"""
        pass
    
    @abstractmethod
    async def delete_state(self, flow_id: str) -> bool:
        """Delete flow state"""
        pass
    
    @abstractmethod
    async def list_active_flows(self) -> List[str]:
        """List active flow IDs"""
        pass
    
    @abstractmethod
    async def get_flow_history(self, flow_id: str) -> List[FlowState]:
        """Get flow execution history"""
        pass


class IFlowMemoryManager(ABC):
    """Interface for flow memory management"""
    
    @abstractmethod
    async def store_memory(self, flow_id: str, key: str, value: Any) -> bool:
        """Store memory for a flow"""
        pass
    
    @abstractmethod
    async def retrieve_memory(self, flow_id: str, key: str) -> Optional[Any]:
        """Retrieve memory for a flow"""
        pass
    
    @abstractmethod
    async def delete_memory(self, flow_id: str, key: str = None) -> bool:
        """Delete memory for a flow"""
        pass
    
    @abstractmethod
    async def list_memory_keys(self, flow_id: str) -> List[str]:
        """List memory keys for a flow"""
        pass


class IEnhancedFlow(ABC):
    """Enhanced flow interface with comprehensive capabilities"""
    
    @abstractmethod
    async def initialize(self, config: FlowConfiguration) -> None:
        """Initialize the flow with configuration"""
        pass
    
    @abstractmethod
    async def plan(self, goal: str, context: Dict[str, Any] = None) -> FlowPlan:
        """Plan the flow execution"""
        pass
    
    @abstractmethod
    async def execute(self, plan: FlowPlan = None, 
                     context: Dict[str, Any] = None) -> FlowState:
        """Execute the flow"""
        pass
    
    @abstractmethod
    async def stream_execute(self, plan: FlowPlan = None,
                           context: Dict[str, Any] = None) -> AsyncGenerator[StreamCallbackData, None]:
        """Execute flow with streaming updates"""
        pass
    
    @abstractmethod
    def add_stream_callback(self, callback: IStreamCallback) -> None:
        """Add stream callback"""
        pass
    
    @abstractmethod
    def remove_stream_callback(self, callback: IStreamCallback) -> None:
        """Remove stream callback"""
        pass
    
    @abstractmethod
    def set_human_callback(self, callback: IHumanCallback) -> None:
        """Set human callback handler"""
        pass
    
    @abstractmethod
    def add_event_listener(self, listener: IFlowEventListener) -> None:
        """Add event listener"""
        pass
    
    @abstractmethod
    def remove_event_listener(self, listener: IFlowEventListener) -> None:
        """Remove event listener"""
        pass
    
    @abstractmethod
    async def get_state(self) -> FlowState:
        """Get current flow state"""
        pass
    
    @abstractmethod
    async def pause(self) -> bool:
        """Pause flow execution"""
        pass
    
    @abstractmethod
    async def resume(self) -> bool:
        """Resume flow execution"""
        pass
    
    @abstractmethod
    async def cancel(self) -> bool:
        """Cancel flow execution"""
        pass


class IMultiAgentOrchestrator(ABC):
    """Interface for multi-agent orchestration"""
    
    @abstractmethod
    async def orchestrate_agents(self, agents: List[Any], 
                                task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate multiple agents for a task"""
        pass
    
    @abstractmethod
    async def coordinate_execution(self, agent_plans: Dict[str, FlowPlan]) -> FlowPlan:
        """Coordinate execution of multiple agent plans"""
        pass
    
    @abstractmethod
    async def handle_agent_communication(self, sender: str, receiver: str, 
                                       message: Dict[str, Any]) -> None:
        """Handle communication between agents"""
        pass
    
    @abstractmethod
    async def resolve_conflicts(self, conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve conflicts between agents"""
        pass


# Type aliases for convenience
StreamCallbackHandler = Callable[[StreamCallbackData], None]
HumanCallbackHandler = Callable[[HumanCallbackData], HumanCallbackResponse]
FlowEventHandler = Callable[[str, Dict[str, Any]], None]

# Utility functions

def generate_flow_id() -> str:
    """Generate unique flow ID"""
    return f"flow_{uuid.uuid4().hex[:8]}"

def generate_step_id() -> str:
    """Generate unique step ID"""
    return f"step_{uuid.uuid4().hex[:8]}"

def create_default_config() -> FlowConfiguration:
    """Create default flow configuration"""
    return FlowConfiguration()

def create_flow_step(name: str, step_type: str, parameters: Dict[str, Any] = None,
                    dependencies: List[str] = None) -> FlowStep:
    """Create a flow step"""
    return FlowStep(
        id=generate_step_id(),
        name=name,
        type=step_type,
        parameters=parameters or {},
        dependencies=dependencies or []
    )

def create_flow_plan(name: str, description: str, steps: List[FlowStep] = None) -> FlowPlan:
    """Create a flow plan"""
    return FlowPlan(
        id=generate_flow_id(),
        name=name,
        description=description,
        steps=steps or []
    )

def create_flow_state(flow_id: str) -> FlowState:
    """Create initial flow state"""
    return FlowState(
        flow_id=flow_id,
        status=FlowStatus.CREATED
    )

