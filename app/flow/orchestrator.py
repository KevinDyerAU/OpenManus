"""
Multi-Agent Orchestrator for OpenManus
Coordinates multiple agents working together on complex tasks
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid

from .enhanced_interfaces import (
    IMultiAgentOrchestrator, FlowPlan, FlowStep, FlowState, FlowStatus,
    IFlowEventListener, IStreamCallback, StreamCallbackData, StreamCallbackType,
    generate_flow_id, generate_step_id
)


class AgentRole(Enum):
    """Agent roles in multi-agent coordination"""
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"
    VALIDATOR = "validator"
    EXECUTOR = "executor"
    MONITOR = "monitor"


class CommunicationType(Enum):
    """Types of inter-agent communication"""
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    NOTIFICATION = "notification"
    COORDINATION = "coordination"
    CONFLICT_RESOLUTION = "conflict_resolution"


class ConflictType(Enum):
    """Types of conflicts between agents"""
    RESOURCE_CONFLICT = "resource_conflict"
    GOAL_CONFLICT = "goal_conflict"
    PRIORITY_CONFLICT = "priority_conflict"
    DEPENDENCY_CONFLICT = "dependency_conflict"
    DATA_CONFLICT = "data_conflict"


@dataclass
class AgentInfo:
    """Information about an agent in the orchestration"""
    id: str
    name: str
    role: AgentRole
    capabilities: List[str] = field(default_factory=list)
    specializations: List[str] = field(default_factory=list)
    current_task: Optional[str] = None
    status: str = "idle"
    load: float = 0.0
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentMessage:
    """Message between agents"""
    id: str
    sender_id: str
    receiver_id: Optional[str]  # None for broadcast
    message_type: CommunicationType
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 1
    requires_response: bool = False
    response_timeout: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTask:
    """Task assigned to an agent"""
    id: str
    agent_id: str
    task_type: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    priority: int = 1
    deadline: Optional[datetime] = None
    status: str = "assigned"
    result: Optional[Any] = None
    error: Optional[str] = None
    assigned_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class Conflict:
    """Conflict between agents or tasks"""
    id: str
    conflict_type: ConflictType
    description: str
    involved_agents: List[str]
    involved_tasks: List[str] = field(default_factory=list)
    severity: int = 1  # 1-5 scale
    resolution_strategy: Optional[str] = None
    resolved: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentCommunicationHub:
    """Manages communication between agents"""
    
    def __init__(self):
        self.message_queue: Dict[str, List[AgentMessage]] = {}
        self.broadcast_subscribers: Set[str] = set()
        self.message_history: List[AgentMessage] = []
        self.pending_responses: Dict[str, asyncio.Future] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def send_message(self, message: AgentMessage) -> Optional[Dict[str, Any]]:
        """Send message between agents"""
        try:
            # Add to history
            self.message_history.append(message)
            
            # Handle broadcast messages
            if message.receiver_id is None:
                await self._handle_broadcast(message)
                return None
            
            # Add to receiver's queue
            if message.receiver_id not in self.message_queue:
                self.message_queue[message.receiver_id] = []
            
            self.message_queue[message.receiver_id].append(message)
            
            # Handle response requirement
            if message.requires_response:
                future = asyncio.Future()
                self.pending_responses[message.id] = future
                
                try:
                    if message.response_timeout:
                        response = await asyncio.wait_for(future, timeout=message.response_timeout)
                    else:
                        response = await future
                    return response
                except asyncio.TimeoutError:
                    self.logger.warning(f"Message {message.id} timed out waiting for response")
                    del self.pending_responses[message.id]
                    return None
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to send message {message.id}: {e}")
            raise
    
    async def receive_messages(self, agent_id: str) -> List[AgentMessage]:
        """Receive messages for an agent"""
        messages = self.message_queue.get(agent_id, [])
        self.message_queue[agent_id] = []
        return messages
    
    async def send_response(self, original_message_id: str, response_content: Dict[str, Any]) -> None:
        """Send response to a message"""
        if original_message_id in self.pending_responses:
            future = self.pending_responses[original_message_id]
            if not future.done():
                future.set_result(response_content)
            del self.pending_responses[original_message_id]
    
    def subscribe_to_broadcasts(self, agent_id: str) -> None:
        """Subscribe agent to broadcast messages"""
        self.broadcast_subscribers.add(agent_id)
    
    def unsubscribe_from_broadcasts(self, agent_id: str) -> None:
        """Unsubscribe agent from broadcast messages"""
        self.broadcast_subscribers.discard(agent_id)
    
    async def _handle_broadcast(self, message: AgentMessage) -> None:
        """Handle broadcast message"""
        for agent_id in self.broadcast_subscribers:
            if agent_id != message.sender_id:  # Don't send to sender
                if agent_id not in self.message_queue:
                    self.message_queue[agent_id] = []
                self.message_queue[agent_id].append(message)
    
    def get_message_history(self, agent_id: str = None, 
                           message_type: CommunicationType = None) -> List[AgentMessage]:
        """Get message history with optional filtering"""
        messages = self.message_history
        
        if agent_id:
            messages = [m for m in messages if m.sender_id == agent_id or m.receiver_id == agent_id]
        
        if message_type:
            messages = [m for m in messages if m.message_type == message_type]
        
        return messages


class ConflictResolver:
    """Resolves conflicts between agents and tasks"""
    
    def __init__(self):
        self.resolution_strategies = {
            ConflictType.RESOURCE_CONFLICT: self._resolve_resource_conflict,
            ConflictType.GOAL_CONFLICT: self._resolve_goal_conflict,
            ConflictType.PRIORITY_CONFLICT: self._resolve_priority_conflict,
            ConflictType.DEPENDENCY_CONFLICT: self._resolve_dependency_conflict,
            ConflictType.DATA_CONFLICT: self._resolve_data_conflict
        }
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def resolve_conflict(self, conflict: Conflict, 
                             agents: Dict[str, AgentInfo],
                             tasks: Dict[str, AgentTask]) -> Dict[str, Any]:
        """Resolve a conflict"""
        try:
            strategy = self.resolution_strategies.get(conflict.conflict_type)
            if not strategy:
                raise ValueError(f"No resolution strategy for conflict type: {conflict.conflict_type}")
            
            resolution = await strategy(conflict, agents, tasks)
            
            conflict.resolved = True
            conflict.resolved_at = datetime.now()
            conflict.resolution_strategy = resolution.get("strategy", "unknown")
            
            return resolution
            
        except Exception as e:
            self.logger.error(f"Failed to resolve conflict {conflict.id}: {e}")
            raise
    
    async def _resolve_resource_conflict(self, conflict: Conflict,
                                       agents: Dict[str, AgentInfo],
                                       tasks: Dict[str, AgentTask]) -> Dict[str, Any]:
        """Resolve resource conflict"""
        # Simple strategy: assign resource to highest priority task
        involved_tasks = [tasks[task_id] for task_id in conflict.involved_tasks if task_id in tasks]
        
        if not involved_tasks:
            return {"strategy": "no_action", "reason": "no_tasks_involved"}
        
        # Sort by priority
        involved_tasks.sort(key=lambda t: t.priority, reverse=True)
        winner = involved_tasks[0]
        
        return {
            "strategy": "priority_based",
            "winner": winner.id,
            "action": "assign_resource_to_highest_priority",
            "affected_tasks": [t.id for t in involved_tasks[1:]]
        }
    
    async def _resolve_goal_conflict(self, conflict: Conflict,
                                   agents: Dict[str, AgentInfo],
                                   tasks: Dict[str, AgentTask]) -> Dict[str, Any]:
        """Resolve goal conflict"""
        # Strategy: negotiate compromise or escalate to human
        return {
            "strategy": "negotiation",
            "action": "request_human_intervention",
            "reason": "goal_conflicts_require_human_decision"
        }
    
    async def _resolve_priority_conflict(self, conflict: Conflict,
                                       agents: Dict[str, AgentInfo],
                                       tasks: Dict[str, AgentTask]) -> Dict[str, Any]:
        """Resolve priority conflict"""
        # Strategy: use agent performance metrics to decide
        involved_agents = [agents[agent_id] for agent_id in conflict.involved_agents if agent_id in agents]
        
        if not involved_agents:
            return {"strategy": "no_action", "reason": "no_agents_involved"}
        
        # Choose agent with best performance
        best_agent = max(involved_agents, 
                        key=lambda a: a.performance_metrics.get("success_rate", 0.5))
        
        return {
            "strategy": "performance_based",
            "winner": best_agent.id,
            "action": "assign_to_best_performer"
        }
    
    async def _resolve_dependency_conflict(self, conflict: Conflict,
                                         agents: Dict[str, AgentInfo],
                                         tasks: Dict[str, AgentTask]) -> Dict[str, Any]:
        """Resolve dependency conflict"""
        # Strategy: reorder tasks to resolve dependencies
        return {
            "strategy": "reordering",
            "action": "reorder_tasks_by_dependencies",
            "requires_replanning": True
        }
    
    async def _resolve_data_conflict(self, conflict: Conflict,
                                   agents: Dict[str, AgentInfo],
                                   tasks: Dict[str, AgentTask]) -> Dict[str, Any]:
        """Resolve data conflict"""
        # Strategy: use versioning or merge strategies
        return {
            "strategy": "data_versioning",
            "action": "create_data_versions",
            "requires_validation": True
        }


class MultiAgentOrchestrator(IMultiAgentOrchestrator):
    """Main multi-agent orchestrator implementation"""
    
    def __init__(self, mcp_client=None):
        self.mcp_client = mcp_client
        self.agents: Dict[str, AgentInfo] = {}
        self.tasks: Dict[str, AgentTask] = {}
        self.conflicts: Dict[str, Conflict] = {}
        self.communication_hub = AgentCommunicationHub()
        self.conflict_resolver = ConflictResolver()
        self.active_orchestrations: Dict[str, Dict[str, Any]] = {}
        self.event_listeners: Set[IFlowEventListener] = set()
        self.stream_callbacks: Set[IStreamCallback] = set()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def register_agent(self, agent_info: AgentInfo) -> None:
        """Register an agent with the orchestrator"""
        self.agents[agent_info.id] = agent_info
        self.communication_hub.subscribe_to_broadcasts(agent_info.id)
        self.logger.info(f"Registered agent: {agent_info.name} ({agent_info.id})")
    
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent"""
        if agent_id in self.agents:
            self.communication_hub.unsubscribe_from_broadcasts(agent_id)
            del self.agents[agent_id]
            self.logger.info(f"Unregistered agent: {agent_id}")
    
    async def orchestrate_agents(self, agents: List[Any], 
                                task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate multiple agents for a task"""
        orchestration_id = generate_flow_id()
        
        try:
            # Create orchestration context
            orchestration = {
                "id": orchestration_id,
                "task": task,
                "context": context,
                "agents": [agent.id if hasattr(agent, 'id') else str(agent) for agent in agents],
                "status": "planning",
                "started_at": datetime.now(),
                "steps": []
            }
            
            self.active_orchestrations[orchestration_id] = orchestration
            
            # Emit start event
            await self._emit_stream_event(
                StreamCallbackType.WORKFLOW,
                {"action": "orchestration_started", "task": task, "agents": len(agents)},
                orchestration_id
            )
            
            # Plan agent coordination
            agent_plans = await self._plan_agent_coordination(agents, task, context)
            
            # Coordinate execution
            coordinated_plan = await self.coordinate_execution(agent_plans)
            
            # Execute coordinated plan
            result = await self._execute_coordinated_plan(coordinated_plan, orchestration_id)
            
            # Update orchestration status
            orchestration["status"] = "completed"
            orchestration["completed_at"] = datetime.now()
            orchestration["result"] = result
            
            # Emit completion event
            await self._emit_stream_event(
                StreamCallbackType.FINISH,
                {"orchestration_id": orchestration_id, "result": result},
                orchestration_id
            )
            
            return result
            
        except Exception as e:
            # Handle orchestration failure
            if orchestration_id in self.active_orchestrations:
                self.active_orchestrations[orchestration_id]["status"] = "failed"
                self.active_orchestrations[orchestration_id]["error"] = str(e)
            
            await self._emit_stream_event(
                StreamCallbackType.ERROR,
                {"orchestration_id": orchestration_id, "error": str(e)},
                orchestration_id
            )
            
            raise
    
    async def coordinate_execution(self, agent_plans: Dict[str, FlowPlan]) -> FlowPlan:
        """Coordinate execution of multiple agent plans"""
        try:
            # Create master coordination plan
            master_plan = FlowPlan(
                id=generate_flow_id(),
                name="Multi-Agent Coordination Plan",
                description="Coordinated execution of multiple agent plans"
            )
            
            # Analyze dependencies between agent plans
            dependencies = await self._analyze_inter_plan_dependencies(agent_plans)
            
            # Create coordination steps
            coordination_steps = []
            
            # Add initialization step
            init_step = FlowStep(
                id=generate_step_id(),
                name="Initialize Multi-Agent Coordination",
                type="coordination_init",
                parameters={
                    "agent_plans": list(agent_plans.keys()),
                    "dependencies": dependencies
                }
            )
            coordination_steps.append(init_step)
            
            # Add agent execution steps
            for agent_id, plan in agent_plans.items():
                agent_step = FlowStep(
                    id=generate_step_id(),
                    name=f"Execute Agent Plan: {agent_id}",
                    type="agent_execution",
                    parameters={
                        "agent_id": agent_id,
                        "plan": plan,
                        "parallel_execution": True
                    },
                    dependencies=[init_step.id]
                )
                coordination_steps.append(agent_step)
            
            # Add synchronization points
            sync_step = FlowStep(
                id=generate_step_id(),
                name="Synchronize Agent Results",
                type="synchronization",
                parameters={
                    "agent_steps": [step.id for step in coordination_steps[1:]]
                },
                dependencies=[step.id for step in coordination_steps[1:]]
            )
            coordination_steps.append(sync_step)
            
            # Add finalization step
            final_step = FlowStep(
                id=generate_step_id(),
                name="Finalize Multi-Agent Coordination",
                type="coordination_finalize",
                parameters={
                    "consolidate_results": True
                },
                dependencies=[sync_step.id]
            )
            coordination_steps.append(final_step)
            
            master_plan.steps = coordination_steps
            
            return master_plan
            
        except Exception as e:
            self.logger.error(f"Failed to coordinate agent plans: {e}")
            raise
    
    async def handle_agent_communication(self, sender: str, receiver: str,
                                       message: Dict[str, Any]) -> None:
        """Handle communication between agents"""
        try:
            # Create agent message
            agent_message = AgentMessage(
                id=str(uuid.uuid4()),
                sender_id=sender,
                receiver_id=receiver,
                message_type=CommunicationType(message.get("type", "notification")),
                content=message.get("content", {}),
                priority=message.get("priority", 1),
                requires_response=message.get("requires_response", False),
                response_timeout=message.get("response_timeout")
            )
            
            # Send message through communication hub
            response = await self.communication_hub.send_message(agent_message)
            
            # Log communication
            self.logger.debug(f"Agent communication: {sender} -> {receiver}")
            
            # Emit stream event
            await self._emit_stream_event(
                StreamCallbackType.TOOL_USE,
                {
                    "action": "agent_communication",
                    "sender": sender,
                    "receiver": receiver,
                    "message_type": agent_message.message_type.value
                }
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to handle agent communication: {e}")
            raise
    
    async def resolve_conflicts(self, conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve conflicts between agents"""
        resolutions = {}
        
        try:
            for conflict_data in conflicts:
                # Create conflict object
                conflict = Conflict(
                    id=str(uuid.uuid4()),
                    conflict_type=ConflictType(conflict_data["type"]),
                    description=conflict_data["description"],
                    involved_agents=conflict_data.get("involved_agents", []),
                    involved_tasks=conflict_data.get("involved_tasks", []),
                    severity=conflict_data.get("severity", 1)
                )
                
                # Store conflict
                self.conflicts[conflict.id] = conflict
                
                # Resolve conflict
                resolution = await self.conflict_resolver.resolve_conflict(
                    conflict, self.agents, self.tasks
                )
                
                resolutions[conflict.id] = resolution
                
                # Emit resolution event
                await self._emit_stream_event(
                    StreamCallbackType.TOOL_RESULT,
                    {
                        "action": "conflict_resolved",
                        "conflict_id": conflict.id,
                        "resolution": resolution
                    }
                )
            
            return resolutions
            
        except Exception as e:
            self.logger.error(f"Failed to resolve conflicts: {e}")
            raise
    
    def add_stream_callback(self, callback: IStreamCallback) -> None:
        """Add stream callback"""
        self.stream_callbacks.add(callback)
    
    def remove_stream_callback(self, callback: IStreamCallback) -> None:
        """Remove stream callback"""
        self.stream_callbacks.discard(callback)
    
    def add_event_listener(self, listener: IFlowEventListener) -> None:
        """Add event listener"""
        self.event_listeners.add(listener)
    
    def remove_event_listener(self, listener: IFlowEventListener) -> None:
        """Remove event listener"""
        self.event_listeners.discard(listener)
    
    async def get_orchestration_status(self, orchestration_id: str) -> Dict[str, Any]:
        """Get status of an orchestration"""
        return self.active_orchestrations.get(orchestration_id, {})
    
    async def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get status of an agent"""
        agent = self.agents.get(agent_id)
        if not agent:
            return {"error": "Agent not found"}
        
        # Get recent messages
        recent_messages = self.communication_hub.get_message_history(agent_id)[-10:]
        
        # Get assigned tasks
        agent_tasks = [task for task in self.tasks.values() if task.agent_id == agent_id]
        
        return {
            "agent_info": agent,
            "recent_messages": len(recent_messages),
            "active_tasks": len([t for t in agent_tasks if t.status in ["assigned", "running"]]),
            "completed_tasks": len([t for t in agent_tasks if t.status == "completed"]),
            "failed_tasks": len([t for t in agent_tasks if t.status == "failed"])
        }
    
    async def _plan_agent_coordination(self, agents: List[Any], 
                                     task: str, context: Dict[str, Any]) -> Dict[str, FlowPlan]:
        """Plan coordination between agents"""
        agent_plans = {}
        
        # For each agent, create a specialized plan
        for i, agent in enumerate(agents):
            agent_id = agent.id if hasattr(agent, 'id') else f"agent_{i}"
            
            # Create agent-specific plan
            plan = FlowPlan(
                id=generate_flow_id(),
                name=f"Agent Plan: {agent_id}",
                description=f"Specialized plan for agent {agent_id}"
            )
            
            # Add agent-specific steps based on capabilities
            if agent_id in self.agents:
                agent_info = self.agents[agent_id]
                steps = await self._create_agent_steps(agent_info, task, context)
                plan.steps = steps
            
            agent_plans[agent_id] = plan
        
        return agent_plans
    
    async def _create_agent_steps(self, agent_info: AgentInfo, 
                                task: str, context: Dict[str, Any]) -> List[FlowStep]:
        """Create steps for a specific agent"""
        steps = []
        
        # Create steps based on agent capabilities
        for capability in agent_info.capabilities:
            if capability in task.lower():
                step = FlowStep(
                    id=generate_step_id(),
                    name=f"Execute {capability}",
                    type="agent_capability",
                    parameters={
                        "capability": capability,
                        "agent_id": agent_info.id,
                        "task_context": task
                    }
                )
                steps.append(step)
        
        # Add default step if no specific capabilities match
        if not steps:
            step = FlowStep(
                id=generate_step_id(),
                name="Execute General Task",
                type="general_execution",
                parameters={
                    "agent_id": agent_info.id,
                    "task": task
                }
            )
            steps.append(step)
        
        return steps
    
    async def _analyze_inter_plan_dependencies(self, agent_plans: Dict[str, FlowPlan]) -> Dict[str, List[str]]:
        """Analyze dependencies between agent plans"""
        dependencies = {}
        
        # Simple dependency analysis
        # In practice, this would be more sophisticated
        for agent_id, plan in agent_plans.items():
            dependencies[agent_id] = []
            
            # Check if this agent depends on others
            for step in plan.steps:
                if "depends_on" in step.parameters:
                    dep_agents = step.parameters["depends_on"]
                    if isinstance(dep_agents, list):
                        dependencies[agent_id].extend(dep_agents)
                    else:
                        dependencies[agent_id].append(dep_agents)
        
        return dependencies
    
    async def _execute_coordinated_plan(self, plan: FlowPlan, orchestration_id: str) -> Dict[str, Any]:
        """Execute the coordinated plan"""
        results = {}
        
        try:
            # Execute steps in dependency order
            for step in plan.steps:
                await self._emit_stream_event(
                    StreamCallbackType.TOOL_USE,
                    {"step": step.name, "type": step.type},
                    orchestration_id,
                    step.id
                )
                
                # Execute step based on type
                if step.type == "coordination_init":
                    result = await self._execute_coordination_init(step)
                elif step.type == "agent_execution":
                    result = await self._execute_agent_step(step)
                elif step.type == "synchronization":
                    result = await self._execute_synchronization(step)
                elif step.type == "coordination_finalize":
                    result = await self._execute_coordination_finalize(step)
                else:
                    result = {"step_type": step.type, "status": "completed"}
                
                results[step.id] = result
                
                await self._emit_stream_event(
                    StreamCallbackType.TOOL_RESULT,
                    {"step": step.name, "result": result},
                    orchestration_id,
                    step.id
                )
        
        except Exception as e:
            await self._emit_stream_event(
                StreamCallbackType.ERROR,
                {"error": str(e), "orchestration_id": orchestration_id},
                orchestration_id
            )
            raise
        
        return results
    
    async def _execute_coordination_init(self, step: FlowStep) -> Dict[str, Any]:
        """Execute coordination initialization"""
        return {
            "action": "coordination_initialized",
            "agent_plans": step.parameters.get("agent_plans", []),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _execute_agent_step(self, step: FlowStep) -> Dict[str, Any]:
        """Execute agent-specific step"""
        agent_id = step.parameters.get("agent_id")
        plan = step.parameters.get("plan")
        
        # In practice, this would delegate to the actual agent
        return {
            "action": "agent_execution_completed",
            "agent_id": agent_id,
            "plan_id": plan.id if plan else None,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _execute_synchronization(self, step: FlowStep) -> Dict[str, Any]:
        """Execute synchronization step"""
        agent_steps = step.parameters.get("agent_steps", [])
        
        return {
            "action": "synchronization_completed",
            "synchronized_steps": agent_steps,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _execute_coordination_finalize(self, step: FlowStep) -> Dict[str, Any]:
        """Execute coordination finalization"""
        return {
            "action": "coordination_finalized",
            "consolidate_results": step.parameters.get("consolidate_results", False),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _emit_stream_event(self, event_type: StreamCallbackType, content: Any,
                               flow_id: str = None, step_id: str = None) -> None:
        """Emit stream event to callbacks"""
        data = StreamCallbackData(
            type=event_type,
            content=content,
            flow_id=flow_id,
            step_id=step_id
        )
        
        for callback in self.stream_callbacks:
            try:
                await callback.on_stream_event(data)
            except Exception as e:
                self.logger.error(f"Stream callback error: {e}")


# Factory function for creating orchestrator
def create_multi_agent_orchestrator(mcp_client=None) -> MultiAgentOrchestrator:
    """Create a multi-agent orchestrator"""
    return MultiAgentOrchestrator(mcp_client)

