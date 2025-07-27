"""
Callback Manager for OpenManus API
Handles real-time callbacks and notifications during AI processing
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
import aiohttp
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class CallbackEventType(str, Enum):
    """Types of callback events"""
    THINKING = "thinking"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    PROGRESS = "progress"
    COMPLETION = "completion"
    ERROR = "error"
    WORKFLOW_START = "workflow_start"
    WORKFLOW_STEP = "workflow_step"
    WORKFLOW_COMPLETE = "workflow_complete"
    MODEL_SELECTION = "model_selection"
    STREAMING_CHUNK = "streaming_chunk"


class CallbackEvent(BaseModel):
    """Callback event data structure"""
    event_type: CallbackEventType
    timestamp: datetime
    session_id: str
    conversation_id: Optional[str] = None
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class CallbackDeliveryMethod(str, Enum):
    """Callback delivery methods"""
    WEBHOOK = "webhook"
    WEBSOCKET = "websocket"
    SSE = "sse"
    POLLING = "polling"


class CallbackConfig(BaseModel):
    """Callback configuration"""
    delivery_method: CallbackDeliveryMethod = CallbackDeliveryMethod.WEBHOOK
    webhook_url: Optional[str] = None
    events: List[CallbackEventType] = [
        CallbackEventType.THINKING,
        CallbackEventType.TOOL_USE,
        CallbackEventType.PROGRESS,
        CallbackEventType.COMPLETION
    ]
    include_intermediate_results: bool = True
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    headers: Optional[Dict[str, str]] = None


class CallbackManager:
    """Manages callback delivery and event handling"""
    
    def __init__(self):
        self.active_sessions: Dict[str, CallbackConfig] = {}
        self.websocket_connections: Dict[str, Any] = {}
        self.sse_connections: Dict[str, Any] = {}
        self.event_queue: Dict[str, List[CallbackEvent]] = {}
        self.delivery_stats: Dict[str, Dict[str, int]] = {}
        self._session = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self._session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._session:
            await self._session.close()
    
    def register_session(self, session_id: str, config: CallbackConfig) -> bool:
        """Register a session for callbacks"""
        try:
            self.active_sessions[session_id] = config
            self.event_queue[session_id] = []
            self.delivery_stats[session_id] = {
                "sent": 0,
                "delivered": 0,
                "failed": 0,
                "retries": 0
            }
            logger.info(f"Registered callback session: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to register session {session_id}: {e}")
            return False
    
    def unregister_session(self, session_id: str) -> bool:
        """Unregister a session"""
        try:
            self.active_sessions.pop(session_id, None)
            self.event_queue.pop(session_id, None)
            self.websocket_connections.pop(session_id, None)
            self.sse_connections.pop(session_id, None)
            self.delivery_stats.pop(session_id, None)
            logger.info(f"Unregistered callback session: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to unregister session {session_id}: {e}")
            return False
    
    def register_websocket(self, session_id: str, websocket: Any) -> bool:
        """Register WebSocket connection for a session"""
        try:
            self.websocket_connections[session_id] = websocket
            logger.info(f"Registered WebSocket for session: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to register WebSocket for {session_id}: {e}")
            return False
    
    def register_sse(self, session_id: str, sse_connection: Any) -> bool:
        """Register SSE connection for a session"""
        try:
            self.sse_connections[session_id] = sse_connection
            logger.info(f"Registered SSE for session: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to register SSE for {session_id}: {e}")
            return False
    
    async def emit_event(
        self,
        session_id: str,
        event_type: CallbackEventType,
        data: Dict[str, Any],
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Emit a callback event"""
        if session_id not in self.active_sessions:
            logger.warning(f"Session {session_id} not registered for callbacks")
            return False
        
        config = self.active_sessions[session_id]
        
        # Check if event type is enabled
        if event_type not in config.events:
            return True  # Not an error, just not configured
        
        # Create event
        event = CallbackEvent(
            event_type=event_type,
            timestamp=datetime.now(),
            session_id=session_id,
            conversation_id=conversation_id,
            data=data,
            metadata=metadata
        )
        
        # Add to queue
        self.event_queue[session_id].append(event)
        
        # Deliver event
        success = await self._deliver_event(session_id, event, config)
        
        # Update stats
        stats = self.delivery_stats[session_id]
        stats["sent"] += 1
        if success:
            stats["delivered"] += 1
        else:
            stats["failed"] += 1
        
        return success
    
    async def _deliver_event(
        self,
        session_id: str,
        event: CallbackEvent,
        config: CallbackConfig
    ) -> bool:
        """Deliver event based on configuration"""
        try:
            if config.delivery_method == CallbackDeliveryMethod.WEBHOOK:
                return await self._deliver_webhook(event, config)
            elif config.delivery_method == CallbackDeliveryMethod.WEBSOCKET:
                return await self._deliver_websocket(session_id, event)
            elif config.delivery_method == CallbackDeliveryMethod.SSE:
                return await self._deliver_sse(session_id, event)
            elif config.delivery_method == CallbackDeliveryMethod.POLLING:
                return True  # Events are queued for polling
            else:
                logger.error(f"Unknown delivery method: {config.delivery_method}")
                return False
        except Exception as e:
            logger.error(f"Failed to deliver event: {e}")
            return False
    
    async def _deliver_webhook(self, event: CallbackEvent, config: CallbackConfig) -> bool:
        """Deliver event via webhook"""
        if not config.webhook_url:
            logger.error("Webhook URL not configured")
            return False
        
        try:
            headers = config.headers or {}
            headers["Content-Type"] = "application/json"
            
            payload = {
                "event_type": event.event_type,
                "timestamp": event.timestamp.isoformat(),
                "session_id": event.session_id,
                "conversation_id": event.conversation_id,
                "data": event.data,
                "metadata": event.metadata
            }
            
            if not self._session:
                self._session = aiohttp.ClientSession()
            
            async with self._session.post(
                config.webhook_url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=config.timeout)
            ) as response:
                if response.status == 200:
                    logger.debug(f"Webhook delivered successfully: {event.event_type}")
                    return True
                else:
                    logger.error(f"Webhook delivery failed: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Webhook delivery error: {e}")
            return False
    
    async def _deliver_websocket(self, session_id: str, event: CallbackEvent) -> bool:
        """Deliver event via WebSocket"""
        if session_id not in self.websocket_connections:
            logger.warning(f"No WebSocket connection for session: {session_id}")
            return False
        
        try:
            websocket = self.websocket_connections[session_id]
            payload = {
                "event_type": event.event_type,
                "timestamp": event.timestamp.isoformat(),
                "session_id": event.session_id,
                "conversation_id": event.conversation_id,
                "data": event.data,
                "metadata": event.metadata
            }
            
            await websocket.send_text(json.dumps(payload))
            logger.debug(f"WebSocket delivered successfully: {event.event_type}")
            return True
            
        except Exception as e:
            logger.error(f"WebSocket delivery error: {e}")
            # Remove broken connection
            self.websocket_connections.pop(session_id, None)
            return False
    
    async def _deliver_sse(self, session_id: str, event: CallbackEvent) -> bool:
        """Deliver event via Server-Sent Events"""
        if session_id not in self.sse_connections:
            logger.warning(f"No SSE connection for session: {session_id}")
            return False
        
        try:
            sse_connection = self.sse_connections[session_id]
            payload = {
                "event_type": event.event_type,
                "timestamp": event.timestamp.isoformat(),
                "session_id": event.session_id,
                "conversation_id": event.conversation_id,
                "data": event.data,
                "metadata": event.metadata
            }
            
            # Format as SSE event
            sse_data = f"data: {json.dumps(payload)}\n\n"
            await sse_connection.send(sse_data)
            logger.debug(f"SSE delivered successfully: {event.event_type}")
            return True
            
        except Exception as e:
            logger.error(f"SSE delivery error: {e}")
            # Remove broken connection
            self.sse_connections.pop(session_id, None)
            return False
    
    def get_session_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a session"""
        if session_id not in self.active_sessions:
            return None
        
        stats = self.delivery_stats.get(session_id, {})
        return {
            "sent": stats.get("sent", 0),
            "delivered": stats.get("delivered", 0),
            "failed": stats.get("failed", 0),
            "retries": stats.get("retries", 0),
            "queued_events": len(self.event_queue.get(session_id, [])),
            "has_websocket": session_id in self.websocket_connections,
            "has_sse": session_id in self.sse_connections
        }
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get overall statistics"""
        total_sessions = len(self.active_sessions)
        total_sent = sum(stats.get("sent", 0) for stats in self.delivery_stats.values())
        total_delivered = sum(stats.get("delivered", 0) for stats in self.delivery_stats.values())
        total_failed = sum(stats.get("failed", 0) for stats in self.delivery_stats.values())
        
        return {
            "total_sessions": total_sessions,
            "total_sent": total_sent,
            "total_delivered": total_delivered,
            "total_failed": total_failed,
            "active_websockets": len(self.websocket_connections),
            "active_sse": len(self.sse_connections),
            "delivery_rate": total_delivered / total_sent if total_sent > 0 else 0
        }
    
    def get_queued_events(self, session_id: str, since: Optional[datetime] = None, limit: int = 100) -> List[CallbackEvent]:
        """Get queued events for polling"""
        if session_id not in self.event_queue:
            return []
        
        events = self.event_queue[session_id]
        
        if since:
            events = [e for e in events if e.timestamp > since]
        
        return events[-limit:] if limit else events
    
    def clear_queued_events(self, session_id: str, before: Optional[datetime] = None) -> int:
        """Clear queued events"""
        if session_id not in self.event_queue:
            return 0
        
        events = self.event_queue[session_id]
        
        if before:
            cleared_count = len([e for e in events if e.timestamp < before])
            self.event_queue[session_id] = [e for e in events if e.timestamp >= before]
        else:
            cleared_count = len(events)
            self.event_queue[session_id] = []
        
        return cleared_count


# Global callback manager instance
callback_manager = CallbackManager()


# Convenience functions for common events
async def emit_thinking_event(session_id: str, thought: str, conversation_id: Optional[str] = None):
    """Emit a thinking event"""
    await callback_manager.emit_event(
        session_id, CallbackEventType.THINKING,
        {"thought": thought}, conversation_id
    )


async def emit_tool_use_event(
    session_id: str,
    tool_name: str,
    parameters: Dict[str, Any],
    conversation_id: Optional[str] = None
):
    """Emit a tool use event"""
    await callback_manager.emit_event(
        session_id, CallbackEventType.TOOL_USE,
        {"tool_name": tool_name, "parameters": parameters}, conversation_id
    )


async def emit_tool_result_event(
    session_id: str,
    tool_name: str,
    result: Any,
    success: bool = True,
    conversation_id: Optional[str] = None
):
    """Emit a tool result event"""
    await callback_manager.emit_event(
        session_id, CallbackEventType.TOOL_RESULT,
        {"tool_name": tool_name, "result": result, "success": success}, conversation_id
    )


async def emit_progress_event(
    session_id: str,
    progress: float,
    message: str,
    conversation_id: Optional[str] = None
):
    """Emit a progress event"""
    await callback_manager.emit_event(
        session_id, CallbackEventType.PROGRESS,
        {"progress": progress, "message": message}, conversation_id
    )


async def emit_completion_event(
    session_id: str,
    result: Any,
    success: bool = True,
    conversation_id: Optional[str] = None
):
    """Emit a completion event"""
    await callback_manager.emit_event(
        session_id, CallbackEventType.COMPLETION,
        {"result": result, "success": success}, conversation_id
    )


async def emit_error_event(
    session_id: str,
    error: str,
    error_type: str = "general",
    conversation_id: Optional[str] = None
):
    """Emit an error event"""
    await callback_manager.emit_event(
        session_id, CallbackEventType.ERROR,
        {"error": error, "error_type": error_type}, conversation_id
    )


__all__ = [
    "CallbackEventType",
    "CallbackEvent",
    "CallbackDeliveryMethod",
    "CallbackConfig",
    "CallbackManager",
    "callback_manager",
    "emit_thinking_event",
    "emit_tool_use_event",
    "emit_tool_result_event",
    "emit_progress_event",
    "emit_completion_event",
    "emit_error_event",
]
