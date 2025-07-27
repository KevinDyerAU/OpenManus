"""
Callback-specific API endpoints for OpenManus
Provides endpoints for managing callbacks, webhooks, and real-time updates
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Query, Path
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .manager import (
    callback_manager, CallbackConfig, CallbackEventType, CallbackDeliveryMethod,
    CallbackEvent
)

logger = logging.getLogger(__name__)

# Create router for callback endpoints
router = APIRouter(prefix="/callbacks", tags=["callbacks"])


# Pydantic models for callback endpoints
class CallbackSessionRequest(BaseModel):
    """Request to create a callback session"""
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
    headers: Optional[Dict[str, str]] = None


class CallbackSessionResponse(BaseModel):
    """Response for callback session creation"""
    session_id: str
    config: CallbackConfig
    created_at: datetime
    status: str


class CallbackStatsResponse(BaseModel):
    """Callback statistics response"""
    session_id: str
    events_sent: int
    events_delivered: int
    events_failed: int
    retries: int
    queued_events: int
    has_websocket: bool
    has_sse: bool
    delivery_rate: float


class CallbackEventResponse(BaseModel):
    """Callback event response for polling"""
    events: List[CallbackEvent]
    total_count: int
    has_more: bool


# Callback session management endpoints
@router.post("/sessions", response_model=CallbackSessionResponse)
async def create_callback_session(request: CallbackSessionRequest):
    """Create a new callback session"""
    try:
        import uuid
        session_id = str(uuid.uuid4())
        
        # Create callback config
        config = CallbackConfig(
            delivery_method=request.delivery_method,
            webhook_url=request.webhook_url,
            events=request.events,
            include_intermediate_results=request.include_intermediate_results,
            timeout=request.timeout,
            retry_attempts=request.retry_attempts,
            headers=request.headers
        )
        
        # Register session
        success = callback_manager.register_session(session_id, config)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to create callback session")
        
        return CallbackSessionResponse(
            session_id=session_id,
            config=config,
            created_at=datetime.now(),
            status="active"
        )
        
    except Exception as e:
        logger.error(f"Failed to create callback session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}")
async def delete_callback_session(session_id: str = Path(..., description="Session ID")):
    """Delete a callback session"""
    try:
        success = callback_manager.unregister_session(session_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {"message": "Session deleted successfully", "session_id": session_id}
        
    except Exception as e:
        logger.error(f"Failed to delete callback session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/stats", response_model=CallbackStatsResponse)
async def get_callback_stats(session_id: str = Path(..., description="Session ID")):
    """Get callback statistics for a session"""
    try:
        stats = callback_manager.get_session_stats(session_id)
        
        if not stats:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return CallbackStatsResponse(
            session_id=session_id,
            events_sent=stats["sent"],
            events_delivered=stats["delivered"],
            events_failed=stats["failed"],
            retries=stats["retries"],
            queued_events=stats["queued_events"],
            has_websocket=stats["has_websocket"],
            has_sse=stats["has_sse"],
            delivery_rate=stats["delivered"] / stats["sent"] if stats["sent"] > 0 else 0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get stats for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_overall_callback_stats():
    """Get overall callback statistics"""
    try:
        stats = callback_manager.get_all_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get overall stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/events", response_model=CallbackEventResponse)
async def poll_callback_events(
    session_id: str = Path(..., description="Session ID"),
    since: Optional[datetime] = Query(None, description="Get events since this timestamp"),
    limit: int = Query(100, description="Maximum number of events to return"),
    offset: int = Query(0, description="Number of events to skip")
):
    """Poll for callback events (for polling-based delivery)"""
    try:
        events = callback_manager.get_queued_events(session_id, since, limit)
        
        # Apply offset
        if offset > 0:
            events = events[offset:]
        
        return CallbackEventResponse(
            events=events,
            total_count=len(events),
            has_more=len(events) == limit
        )
        
    except Exception as e:
        logger.error(f"Failed to poll events for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}/events")
async def clear_callback_events(
    session_id: str = Path(..., description="Session ID"),
    before: Optional[datetime] = Query(None, description="Clear events before this timestamp")
):
    """Clear queued callback events"""
    try:
        cleared_count = callback_manager.clear_queued_events(session_id, before)
        
        return {
            "message": f"Cleared {cleared_count} events",
            "session_id": session_id,
            "cleared_count": cleared_count
        }
        
    except Exception as e:
        logger.error(f"Failed to clear events for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket endpoint for real-time callbacks
@router.websocket("/sessions/{session_id}/ws")
async def callback_websocket(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time callback delivery"""
    await websocket.accept()
    
    # Register WebSocket connection
    callback_manager.register_websocket(session_id, websocket)
    
    try:
        while True:
            # Keep connection alive and handle incoming messages
            try:
                data = await websocket.receive_text()
                # Echo back for ping/pong
                await websocket.send_text(f"pong: {data}")
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error for session {session_id}: {e}")
                break
                
    except Exception as e:
        logger.error(f"WebSocket connection error for session {session_id}: {e}")
    finally:
        # Clean up connection
        if session_id in callback_manager.websocket_connections:
            del callback_manager.websocket_connections[session_id]


# Server-Sent Events endpoint for real-time callbacks
@router.get("/sessions/{session_id}/sse")
async def callback_sse(session_id: str = Path(..., description="Session ID")):
    """Server-Sent Events endpoint for real-time callback delivery"""
    
    async def event_stream():
        # Register SSE connection (simplified)
        callback_manager.register_sse(session_id, None)
        
        try:
            # Send initial connection event
            yield f"data: {{\"type\": \"connected\", \"session_id\": \"{session_id}\"}}\n\n"
            
            # Keep connection alive
            while True:
                # In a real implementation, you'd pull events from a queue
                # For now, send periodic heartbeat
                await asyncio.sleep(30)
                yield f"data: {{\"type\": \"heartbeat\", \"timestamp\": \"{datetime.now().isoformat()}\"}}\n\n"
                
        except Exception as e:
            logger.error(f"SSE error for session {session_id}: {e}")
        finally:
            # Clean up connection
            if session_id in callback_manager.sse_connections:
                del callback_manager.sse_connections[session_id]
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )


# Test endpoint for callback functionality
@router.post("/sessions/{session_id}/test")
async def test_callback(
    session_id: str = Path(..., description="Session ID"),
    event_type: CallbackEventType = Query(CallbackEventType.PROGRESS, description="Event type to test"),
    test_data: Optional[Dict[str, Any]] = None
):
    """Test callback delivery for a session"""
    try:
        if test_data is None:
            test_data = {
                "message": "Test callback event",
                "timestamp": datetime.now().isoformat()
            }
        
        success = await callback_manager.emit_event(
            session_id=session_id,
            event_type=event_type,
            data=test_data,
            metadata={"test": True}
        )
        
        return {
            "message": "Test callback sent",
            "session_id": session_id,
            "event_type": event_type,
            "success": success,
            "test_data": test_data
        }
        
    except Exception as e:
        logger.error(f"Failed to send test callback for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Utility function for cleanup
async def cleanup_callback_session(session_id: str):
    """Cleanup callback session after delay"""
    await asyncio.sleep(300)  # 5 minutes
    callback_manager.unregister_session(session_id)


__all__ = ["router"]
