"""
Enhanced Flow Implementation for OpenManus
Comprehensive flow execution with stream callbacks, human interaction, and state management
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, AsyncGenerator, Set
from datetime import datetime, timedelta
import uuid

from .enhanced_interfaces import (
    IEnhancedFlow, IStreamCallback, IHumanCallback, IFlowEventListener,
    IFlowPlanner, IFlowExecutor, IFlowStateManager, IFlowMemoryManager,
    FlowStatus, StreamCallbackType, HumanCallbackType, FlowPriority,
    StreamCallbackData, HumanCallbackData, HumanCallbackResponse,
    FlowStep, FlowPlan, FlowState, FlowConfiguration,
    generate_flow_id, generate_step_id, create_flow_state
)


class StreamCallbackManager:
    """Manages stream callbacks for flow execution"""
    
    def __init__(self):
        self.callbacks: Set[IStreamCallback] = set()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def add_callback(self, callback: IStreamCallback) -> None:
        """Add stream callback"""
        self.callbacks.add(callback)
    
    def remove_callback(self, callback: IStreamCallback) -> None:
        """Remove stream callback"""
        self.callbacks.discard(callback)
    
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
        
        # Emit to all callbacks
        for callback in self.callbacks:
            try:
                await callback.on_stream_event(data)
            except Exception as e:
                self.logger.error(f"Stream callback error: {e}")


class HumanCallbackManager:
    """Manages human callbacks for flow execution"""
    
    def __init__(self):
        self.callback_handler: Optional[IHumanCallback] = None
        self.pending_callbacks: Dict[str, asyncio.Future] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def set_handler(self, handler: IHumanCallback) -> None:
        """Set human callback handler"""
        self.callback_handler = handler
    
    async def request_confirmation(self, message: str, flow_id: str = None,
                                 step_id: str = None, timeout: int = 3600) -> bool:
        """Request human confirmation"""
        if not self.callback_handler:
            raise RuntimeError("No human callback handler set")
        
        data = HumanCallbackData(
            type=HumanCallbackType.CONFIRM,
            message=message,
            timeout=timeout,
            flow_id=flow_id,
            step_id=step_id
        )
        
        try:
            response = await asyncio.wait_for(
                self.callback_handler.on_human_confirm(data),
                timeout=timeout
            )
            return response.value
        except asyncio.TimeoutError:
            self.logger.warning(f"Human confirmation timeout for flow {flow_id}")
            return False
    
    async def request_input(self, message: str, default_value: str = None,
                          flow_id: str = None, step_id: str = None,
                          timeout: int = 3600) -> str:
        """Request human input"""
        if not self.callback_handler:
            raise RuntimeError("No human callback handler set")
        
        data = HumanCallbackData(
            type=HumanCallbackType.INPUT,
            message=message,
            default_value=default_value,
            timeout=timeout,
            flow_id=flow_id,
            step_id=step_id
        )
        
        try:
            response = await asyncio.wait_for(
                self.callback_handler.on_human_input(data),
                timeout=timeout
            )
            return response.value
        except asyncio.TimeoutError:
            self.logger.warning(f"Human input timeout for flow {flow_id}")
            return default_value or ""
    
    async def request_selection(self, message: str, options: List[str],
                              flow_id: str = None, step_id: str = None,
                              timeout: int = 3600) -> str:
        """Request human selection"""
        if not self.callback_handler:
            raise RuntimeError("No human callback handler set")
        
        data = HumanCallbackData(
            type=HumanCallbackType.SELECT,
            message=message,
            options=options,
            timeout=timeout,
            flow_id=flow_id,
            step_id=step_id
        )
        
        try:
            response = await asyncio.wait_for(
                self.callback_handler.on_human_select(data),
                timeout=timeout
            )
            return response.value
        except asyncio.TimeoutError:
            self.logger.warning(f"Human selection timeout for flow {flow_id}")
            return options[0] if options else ""
    
    async def request_help(self, message: str, context: Dict[str, Any] = None,
                         flow_id: str = None, step_id: str = None,
                         timeout: int = 3600) -> Dict[str, Any]:
        """Request human help"""
        if not self.callback_handler:
            raise RuntimeError("No human callback handler set")
        
        data = HumanCallbackData(
            type=HumanCallbackType.HELP,
            message=message,
            metadata=context or {},
            timeout=timeout,
            flow_id=flow_id,
            step_id=step_id
        )
        
        try:
            response = await asyncio.wait_for(
                self.callback_handler.on_human_help(data),
                timeout=timeout
            )
            return response.value
        except asyncio.TimeoutError:
            self.logger.warning(f"Human help timeout for flow {flow_id}")
            return {}


class InMemoryStateManager(IFlowStateManager):
    """In-memory flow state manager"""
    
    def __init__(self):
        self.states: Dict[str, FlowState] = {}
        self.history: Dict[str, List[FlowState]] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def save_state(self, state: FlowState) -> bool:
        """Save flow state"""
        try:
            # Update timestamp
            state.last_activity = datetime.now()
            
            # Save current state
            self.states[state.flow_id] = state
            
            # Add to history
            if state.flow_id not in self.history:
                self.history[state.flow_id] = []
            
            # Create a copy for history
            import copy
            history_state = copy.deepcopy(state)
            self.history[state.flow_id].append(history_state)
            
            # Limit history size
            if len(self.history[state.flow_id]) > 100:
                self.history[state.flow_id] = self.history[state.flow_id][-100:]
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to save state for flow {state.flow_id}: {e}")
            return False
    
    async def load_state(self, flow_id: str) -> Optional[FlowState]:
        """Load flow state"""
        return self.states.get(flow_id)
    
    async def delete_state(self, flow_id: str) -> bool:
        """Delete flow state"""
        try:
            if flow_id in self.states:
                del self.states[flow_id]
            if flow_id in self.history:
                del self.history[flow_id]
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete state for flow {flow_id}: {e}")
            return False
    
    async def list_active_flows(self) -> List[str]:
        """List active flow IDs"""
        active_statuses = {FlowStatus.PLANNING, FlowStatus.EXECUTING, 
                          FlowStatus.PAUSED, FlowStatus.WAITING_HUMAN}
        return [
            flow_id for flow_id, state in self.states.items()
            if state.status in active_statuses
        ]
    
    async def get_flow_history(self, flow_id: str) -> List[FlowState]:
        """Get flow execution history"""
        return self.history.get(flow_id, [])


class InMemoryMemoryManager(IFlowMemoryManager):
    """In-memory flow memory manager"""
    
    def __init__(self):
        self.memory: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def store_memory(self, flow_id: str, key: str, value: Any) -> bool:
        """Store memory for a flow"""
        try:
            if flow_id not in self.memory:
                self.memory[flow_id] = {}
            self.memory[flow_id][key] = value
            return True
        except Exception as e:
            self.logger.error(f"Failed to store memory for flow {flow_id}: {e}")
            return False
    
    async def retrieve_memory(self, flow_id: str, key: str) -> Optional[Any]:
        """Retrieve memory for a flow"""
        return self.memory.get(flow_id, {}).get(key)
    
    async def delete_memory(self, flow_id: str, key: str = None) -> bool:
        """Delete memory for a flow"""
        try:
            if flow_id in self.memory:
                if key:
                    self.memory[flow_id].pop(key, None)
                else:
                    del self.memory[flow_id]
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete memory for flow {flow_id}: {e}")
            return False
    
    async def list_memory_keys(self, flow_id: str) -> List[str]:
        """List memory keys for a flow"""
        return list(self.memory.get(flow_id, {}).keys())


class EnhancedFlowPlanner(IFlowPlanner):
    """Enhanced flow planner with AI-powered planning"""
    
    def __init__(self, mcp_client=None):
        self.mcp_client = mcp_client
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def create_plan(self, goal: str, context: Dict[str, Any]) -> FlowPlan:
        """Create a flow plan for the given goal"""
        try:
            # For now, create a simple plan
            # In a real implementation, this would use AI planning
            plan = FlowPlan(
                id=generate_flow_id(),
                name=f"Plan for: {goal}",
                description=f"Automatically generated plan to achieve: {goal}"
            )
            
            # Add basic steps based on goal analysis
            steps = await self._analyze_goal_and_create_steps(goal, context)
            plan.steps = steps
            
            return plan
            
        except Exception as e:
            self.logger.error(f"Failed to create plan for goal '{goal}': {e}")
            raise
    
    async def replan(self, flow_id: str, current_state: FlowState,
                    new_context: Dict[str, Any]) -> FlowPlan:
        """Replan flow based on current state and new context"""
        try:
            # Analyze current progress and create new plan
            remaining_goal = await self._analyze_remaining_work(current_state, new_context)
            
            # Create new plan for remaining work
            new_plan = await self.create_plan(remaining_goal, new_context)
            new_plan.id = flow_id  # Keep same flow ID
            new_plan.version = current_state.metadata.get("plan_version", 1) + 1
            
            return new_plan
            
        except Exception as e:
            self.logger.error(f"Failed to replan flow {flow_id}: {e}")
            raise
    
    async def validate_plan(self, plan: FlowPlan) -> bool:
        """Validate a flow plan"""
        try:
            # Check for circular dependencies
            if self._has_circular_dependencies(plan.steps):
                return False
            
            # Check for missing dependencies
            step_ids = {step.id for step in plan.steps}
            for step in plan.steps:
                for dep in step.dependencies:
                    if dep not in step_ids:
                        return False
            
            # Check for valid step types
            for step in plan.steps:
                if not await self._is_valid_step_type(step.type):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to validate plan {plan.id}: {e}")
            return False
    
    async def optimize_plan(self, plan: FlowPlan) -> FlowPlan:
        """Optimize a flow plan for better performance"""
        try:
            # Create optimized copy
            import copy
            optimized_plan = copy.deepcopy(plan)
            
            # Optimize step order for parallel execution
            optimized_plan.steps = await self._optimize_step_order(optimized_plan.steps)
            
            # Merge compatible steps
            optimized_plan.steps = await self._merge_compatible_steps(optimized_plan.steps)
            
            return optimized_plan
            
        except Exception as e:
            self.logger.error(f"Failed to optimize plan {plan.id}: {e}")
            return plan
    
    async def _analyze_goal_and_create_steps(self, goal: str, context: Dict[str, Any]) -> List[FlowStep]:
        """Analyze goal and create appropriate steps"""
        steps = []
        
        # Simple goal analysis - in real implementation, use AI
        if "search" in goal.lower():
            steps.append(FlowStep(
                id=generate_step_id(),
                name="Search Information",
                type="search",
                parameters={"query": goal}
            ))
        
        if "analyze" in goal.lower():
            steps.append(FlowStep(
                id=generate_step_id(),
                name="Analyze Data",
                type="analyze",
                parameters={"data": context.get("data", {})}
            ))
        
        if "generate" in goal.lower() or "create" in goal.lower():
            steps.append(FlowStep(
                id=generate_step_id(),
                name="Generate Content",
                type="generate",
                parameters={"content_type": "text", "goal": goal}
            ))
        
        # Always add a completion step
        steps.append(FlowStep(
            id=generate_step_id(),
            name="Complete Task",
            type="complete",
            parameters={"goal": goal},
            dependencies=[step.id for step in steps]
        ))
        
        return steps
    
    async def _analyze_remaining_work(self, state: FlowState, context: Dict[str, Any]) -> str:
        """Analyze what work remains to be done"""
        completed_work = ", ".join(state.completed_steps)
        return f"Continue from completed steps: {completed_work}"
    
    def _has_circular_dependencies(self, steps: List[FlowStep]) -> bool:
        """Check for circular dependencies in steps"""
        # Simple cycle detection using DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle(step_id: str, step_map: Dict[str, FlowStep]) -> bool:
            if step_id in rec_stack:
                return True
            if step_id in visited:
                return False
            
            visited.add(step_id)
            rec_stack.add(step_id)
            
            step = step_map.get(step_id)
            if step:
                for dep in step.dependencies:
                    if has_cycle(dep, step_map):
                        return True
            
            rec_stack.remove(step_id)
            return False
        
        step_map = {step.id: step for step in steps}
        
        for step in steps:
            if step.id not in visited:
                if has_cycle(step.id, step_map):
                    return True
        
        return False
    
    async def _is_valid_step_type(self, step_type: str) -> bool:
        """Check if step type is valid"""
        valid_types = {
            "search", "analyze", "generate", "complete", "mcp_tool",
            "human_input", "condition", "loop", "parallel", "wait"
        }
        return step_type in valid_types
    
    async def _optimize_step_order(self, steps: List[FlowStep]) -> List[FlowStep]:
        """Optimize step order for parallel execution"""
        # Topological sort for dependency resolution
        # This is a simplified implementation
        return steps
    
    async def _merge_compatible_steps(self, steps: List[FlowStep]) -> List[FlowStep]:
        """Merge compatible steps for efficiency"""
        # Simple implementation - in practice, would analyze step compatibility
        return steps


class EnhancedFlowExecutor(IFlowExecutor):
    """Enhanced flow executor with comprehensive execution capabilities"""
    
    def __init__(self, mcp_client=None, state_manager: IFlowStateManager = None,
                 memory_manager: IFlowMemoryManager = None):
        self.mcp_client = mcp_client
        self.state_manager = state_manager or InMemoryStateManager()
        self.memory_manager = memory_manager or InMemoryMemoryManager()
        self.stream_manager = StreamCallbackManager()
        self.human_manager = HumanCallbackManager()
        self.event_listeners: Set[IFlowEventListener] = set()
        self.active_flows: Dict[str, asyncio.Task] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def add_event_listener(self, listener: IFlowEventListener) -> None:
        """Add event listener"""
        self.event_listeners.add(listener)
    
    def remove_event_listener(self, listener: IFlowEventListener) -> None:
        """Remove event listener"""
        self.event_listeners.discard(listener)
    
    async def execute_flow(self, plan: FlowPlan,
                          initial_context: Dict[str, Any] = None) -> FlowState:
        """Execute a flow plan"""
        flow_id = plan.id
        
        try:
            # Create initial state
            state = create_flow_state(flow_id)
            state.status = FlowStatus.EXECUTING
            state.started_at = datetime.now()
            state.context = initial_context or {}
            state.variables = plan.variables.copy()
            
            # Save initial state
            await self.state_manager.save_state(state)
            
            # Notify listeners
            for listener in self.event_listeners:
                await listener.on_flow_started(flow_id, plan)
            
            # Emit stream event
            await self.stream_manager.emit(
                StreamCallbackType.WORKFLOW,
                {"action": "started", "plan": plan.name},
                flow_id=flow_id
            )
            
            # Execute steps
            await self._execute_steps(plan, state)
            
            # Update final state
            state.status = FlowStatus.COMPLETED
            state.completed_at = datetime.now()
            state.execution_time = (state.completed_at - state.started_at).total_seconds()
            
            # Save final state
            await self.state_manager.save_state(state)
            
            # Notify listeners
            for listener in self.event_listeners:
                await listener.on_flow_completed(flow_id, state)
            
            # Emit completion event
            await self.stream_manager.emit(
                StreamCallbackType.FINISH,
                {"status": "completed", "execution_time": state.execution_time},
                flow_id=flow_id
            )
            
            return state
            
        except Exception as e:
            # Handle failure
            state = await self.state_manager.load_state(flow_id) or create_flow_state(flow_id)
            state.status = FlowStatus.FAILED
            state.error = str(e)
            state.completed_at = datetime.now()
            
            await self.state_manager.save_state(state)
            
            # Notify listeners
            for listener in self.event_listeners:
                await listener.on_flow_failed(flow_id, state, e)
            
            # Emit error event
            await self.stream_manager.emit(
                StreamCallbackType.ERROR,
                {"error": str(e)},
                flow_id=flow_id
            )
            
            raise
    
    async def execute_step(self, step: FlowStep, context: Dict[str, Any]) -> Any:
        """Execute a single flow step"""
        step.started_at = datetime.now()
        step.status = "running"
        
        try:
            # Emit step start event
            await self.stream_manager.emit(
                StreamCallbackType.TOOL_USE,
                {"step": step.name, "type": step.type, "parameters": step.parameters},
                step_id=step.id
            )
            
            # Execute based on step type
            if step.type == "mcp_tool":
                result = await self._execute_mcp_tool_step(step, context)
            elif step.type == "human_input":
                result = await self._execute_human_input_step(step, context)
            elif step.type == "condition":
                result = await self._execute_condition_step(step, context)
            elif step.type == "loop":
                result = await self._execute_loop_step(step, context)
            elif step.type == "wait":
                result = await self._execute_wait_step(step, context)
            else:
                result = await self._execute_generic_step(step, context)
            
            step.result = result
            step.status = "completed"
            step.completed_at = datetime.now()
            
            # Emit step completion event
            await self.stream_manager.emit(
                StreamCallbackType.TOOL_RESULT,
                {"step": step.name, "result": result, "success": True},
                step_id=step.id
            )
            
            return result
            
        except Exception as e:
            step.error = str(e)
            step.status = "failed"
            step.completed_at = datetime.now()
            
            # Emit step error event
            await self.stream_manager.emit(
                StreamCallbackType.ERROR,
                {"step": step.name, "error": str(e)},
                step_id=step.id
            )
            
            raise
    
    async def pause_flow(self, flow_id: str) -> bool:
        """Pause flow execution"""
        try:
            state = await self.state_manager.load_state(flow_id)
            if state and state.status == FlowStatus.EXECUTING:
                state.status = FlowStatus.PAUSED
                await self.state_manager.save_state(state)
                
                # Cancel active task if exists
                if flow_id in self.active_flows:
                    self.active_flows[flow_id].cancel()
                
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to pause flow {flow_id}: {e}")
            return False
    
    async def resume_flow(self, flow_id: str) -> bool:
        """Resume flow execution"""
        try:
            state = await self.state_manager.load_state(flow_id)
            if state and state.status == FlowStatus.PAUSED:
                state.status = FlowStatus.EXECUTING
                await self.state_manager.save_state(state)
                
                # Resume execution would require reloading the plan
                # This is a simplified implementation
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to resume flow {flow_id}: {e}")
            return False
    
    async def cancel_flow(self, flow_id: str) -> bool:
        """Cancel flow execution"""
        try:
            state = await self.state_manager.load_state(flow_id)
            if state and state.status in {FlowStatus.EXECUTING, FlowStatus.PAUSED}:
                state.status = FlowStatus.CANCELLED
                state.completed_at = datetime.now()
                await self.state_manager.save_state(state)
                
                # Cancel active task if exists
                if flow_id in self.active_flows:
                    self.active_flows[flow_id].cancel()
                    del self.active_flows[flow_id]
                
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to cancel flow {flow_id}: {e}")
            return False
    
    async def _execute_steps(self, plan: FlowPlan, state: FlowState) -> None:
        """Execute all steps in the plan"""
        # Build dependency graph
        step_map = {step.id: step for step in plan.steps}
        
        # Execute steps respecting dependencies
        while len(state.completed_steps) < len(plan.steps):
            # Find steps ready to execute
            ready_steps = []
            for step in plan.steps:
                if (step.id not in state.completed_steps and 
                    step.id not in state.failed_steps and
                    all(dep in state.completed_steps for dep in step.dependencies)):
                    ready_steps.append(step)
            
            if not ready_steps:
                # Check if we're stuck
                remaining_steps = [s for s in plan.steps 
                                 if s.id not in state.completed_steps and s.id not in state.failed_steps]
                if remaining_steps:
                    raise RuntimeError(f"Cannot proceed: no ready steps found. Remaining: {[s.name for s in remaining_steps]}")
                break
            
            # Execute ready steps (could be parallelized)
            for step in ready_steps:
                try:
                    # Notify listeners
                    for listener in self.event_listeners:
                        await listener.on_step_started(state.flow_id, step)
                    
                    # Execute step
                    result = await self.execute_step(step, state.context)
                    
                    # Update state
                    state.completed_steps.add(step.id)
                    state.variables[f"step_{step.id}_result"] = result
                    
                    # Save state
                    await self.state_manager.save_state(state)
                    
                    # Notify listeners
                    for listener in self.event_listeners:
                        await listener.on_step_completed(state.flow_id, step)
                    
                except Exception as e:
                    # Handle step failure
                    state.failed_steps.add(step.id)
                    
                    # Notify listeners
                    for listener in self.event_listeners:
                        await listener.on_step_failed(state.flow_id, step, e)
                    
                    # Check if this is a critical failure
                    if step.parameters.get("critical", False):
                        raise
                    
                    # Continue with other steps
                    self.logger.warning(f"Step {step.name} failed but continuing: {e}")
    
    async def _execute_mcp_tool_step(self, step: FlowStep, context: Dict[str, Any]) -> Any:
        """Execute MCP tool step"""
        if not self.mcp_client:
            raise RuntimeError("MCP client not available")
        
        from ..mcp.interfaces import McpCallToolParam
        
        tool_name = step.parameters.get("tool_name")
        tool_args = step.parameters.get("arguments", {})
        
        # Substitute variables in arguments
        tool_args = self._substitute_variables(tool_args, context)
        
        params = McpCallToolParam(
            name=tool_name,
            arguments=tool_args,
            context=context
        )
        
        result = await self.mcp_client.call_tool(params)
        return result.content
    
    async def _execute_human_input_step(self, step: FlowStep, context: Dict[str, Any]) -> Any:
        """Execute human input step"""
        input_type = step.parameters.get("input_type", "input")
        message = step.parameters.get("message", "Input required")
        
        if input_type == "confirm":
            return await self.human_manager.request_confirmation(message)
        elif input_type == "select":
            options = step.parameters.get("options", [])
            return await self.human_manager.request_selection(message, options)
        elif input_type == "help":
            return await self.human_manager.request_help(message, context)
        else:
            default = step.parameters.get("default")
            return await self.human_manager.request_input(message, default)
    
    async def _execute_condition_step(self, step: FlowStep, context: Dict[str, Any]) -> Any:
        """Execute conditional step"""
        condition = step.parameters.get("condition")
        # Simple condition evaluation - in practice, would use a proper expression evaluator
        return eval(condition, {"context": context})
    
    async def _execute_loop_step(self, step: FlowStep, context: Dict[str, Any]) -> Any:
        """Execute loop step"""
        iterations = step.parameters.get("iterations", 1)
        results = []
        
        for i in range(iterations):
            # Execute loop body (would need sub-steps in real implementation)
            result = f"Loop iteration {i+1}"
            results.append(result)
        
        return results
    
    async def _execute_wait_step(self, step: FlowStep, context: Dict[str, Any]) -> Any:
        """Execute wait step"""
        duration = step.parameters.get("duration", 1)
        await asyncio.sleep(duration)
        return f"Waited {duration} seconds"
    
    async def _execute_generic_step(self, step: FlowStep, context: Dict[str, Any]) -> Any:
        """Execute generic step"""
        # Default implementation for unknown step types
        return {"step_type": step.type, "parameters": step.parameters}
    
    def _substitute_variables(self, data: Any, context: Dict[str, Any]) -> Any:
        """Substitute variables in data structure"""
        if isinstance(data, str):
            # Simple variable substitution
            for key, value in context.items():
                data = data.replace(f"${{{key}}}", str(value))
            return data
        elif isinstance(data, dict):
            return {k: self._substitute_variables(v, context) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._substitute_variables(item, context) for item in data]
        else:
            return data


class EnhancedFlow(IEnhancedFlow):
    """Main enhanced flow implementation"""
    
    def __init__(self, mcp_client=None):
        self.config: Optional[FlowConfiguration] = None
        self.planner = EnhancedFlowPlanner(mcp_client)
        self.executor = EnhancedFlowExecutor(mcp_client)
        self.current_plan: Optional[FlowPlan] = None
        self.current_state: Optional[FlowState] = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def initialize(self, config: FlowConfiguration) -> None:
        """Initialize the flow with configuration"""
        self.config = config
        self.logger.info("Enhanced flow initialized")
    
    async def plan(self, goal: str, context: Dict[str, Any] = None) -> FlowPlan:
        """Plan the flow execution"""
        self.current_plan = await self.planner.create_plan(goal, context or {})
        return self.current_plan
    
    async def execute(self, plan: FlowPlan = None,
                     context: Dict[str, Any] = None) -> FlowState:
        """Execute the flow"""
        execution_plan = plan or self.current_plan
        if not execution_plan:
            raise ValueError("No plan available for execution")
        
        self.current_state = await self.executor.execute_flow(execution_plan, context)
        return self.current_state
    
    async def stream_execute(self, plan: FlowPlan = None,
                           context: Dict[str, Any] = None) -> AsyncGenerator[StreamCallbackData, None]:
        """Execute flow with streaming updates"""
        # This would need to be implemented with proper streaming
        # For now, just execute and yield final result
        state = await self.execute(plan, context)
        yield StreamCallbackData(
            type=StreamCallbackType.FINISH,
            content={"state": state},
            flow_id=state.flow_id
        )
    
    def add_stream_callback(self, callback: IStreamCallback) -> None:
        """Add stream callback"""
        self.executor.stream_manager.add_callback(callback)
    
    def remove_stream_callback(self, callback: IStreamCallback) -> None:
        """Remove stream callback"""
        self.executor.stream_manager.remove_callback(callback)
    
    def set_human_callback(self, callback: IHumanCallback) -> None:
        """Set human callback handler"""
        self.executor.human_manager.set_handler(callback)
    
    def add_event_listener(self, listener: IFlowEventListener) -> None:
        """Add event listener"""
        self.executor.add_event_listener(listener)
    
    def remove_event_listener(self, listener: IFlowEventListener) -> None:
        """Remove event listener"""
        self.executor.remove_event_listener(listener)
    
    async def get_state(self) -> FlowState:
        """Get current flow state"""
        if not self.current_state:
            raise ValueError("No active flow state")
        return self.current_state
    
    async def pause(self) -> bool:
        """Pause flow execution"""
        if self.current_state:
            return await self.executor.pause_flow(self.current_state.flow_id)
        return False
    
    async def resume(self) -> bool:
        """Resume flow execution"""
        if self.current_state:
            return await self.executor.resume_flow(self.current_state.flow_id)
        return False
    
    async def cancel(self) -> bool:
        """Cancel flow execution"""
        if self.current_state:
            return await self.executor.cancel_flow(self.current_state.flow_id)
        return False

