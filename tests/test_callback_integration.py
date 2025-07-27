"""
Comprehensive test suite for OpenManus callback integration
Tests callback manager, API endpoints, and delivery mechanisms
"""

import asyncio
import json
import pytest
import time
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import WebSocket

# Import components to test
from app.callbacks.manager import (
    CallbackManager, CallbackConfig, CallbackEventType, CallbackDeliveryMethod,
    CallbackEvent, emit_thinking_event, emit_tool_use_event, emit_progress_event,
    emit_completion_event, emit_error_event
)
from app.api.main import app


class TestCallbackManager:
    """Test suite for CallbackManager core functionality"""
    
    @pytest.fixture
    def callback_manager(self):
        """Create a fresh callback manager for each test"""
        return CallbackManager()
    
    @pytest.fixture
    def sample_config(self):
        """Sample callback configuration for testing"""
        return CallbackConfig(
            delivery_method=CallbackDeliveryMethod.WEBHOOK,
            webhook_url="https://example.com/callback",
            events=[CallbackEventType.THINKING, CallbackEventType.PROGRESS],
            include_intermediate_results=True,
            timeout=30,
            retry_attempts=3
        )
    
    def test_session_registration(self, callback_manager, sample_config):
        """Test callback session registration"""
        session_id = str(uuid.uuid4())
        
        # Register session
        success = callback_manager.register_session(session_id, sample_config)
        assert success is True
        assert session_id in callback_manager.active_sessions
        assert session_id in callback_manager.event_queue
        assert session_id in callback_manager.delivery_stats
        
        # Verify configuration stored correctly
        stored_config = callback_manager.active_sessions[session_id]
        assert stored_config.delivery_method == sample_config.delivery_method
        assert stored_config.webhook_url == sample_config.webhook_url
        assert stored_config.events == sample_config.events
    
    def test_session_unregistration(self, callback_manager, sample_config):
        """Test callback session unregistration"""
        session_id = str(uuid.uuid4())
        
        # Register and then unregister
        callback_manager.register_session(session_id, sample_config)
        success = callback_manager.unregister_session(session_id)
        
        assert success is True
        assert session_id not in callback_manager.active_sessions
        assert session_id not in callback_manager.event_queue
        assert session_id not in callback_manager.delivery_stats
    
    def test_websocket_registration(self, callback_manager, sample_config):
        """Test WebSocket connection registration"""
        session_id = str(uuid.uuid4())
        mock_websocket = Mock()
        
        callback_manager.register_session(session_id, sample_config)
        success = callback_manager.register_websocket(session_id, mock_websocket)
        
        assert success is True
        assert session_id in callback_manager.websocket_connections
        assert callback_manager.websocket_connections[session_id] == mock_websocket
    
    @pytest.mark.asyncio
    async def test_event_emission(self, callback_manager, sample_config):
        """Test event emission and queuing"""
        session_id = str(uuid.uuid4())
        callback_manager.register_session(session_id, sample_config)
        
        # Emit a thinking event
        test_data = {"thought": "Testing event emission"}
        success = await callback_manager.emit_event(
            session_id=session_id,
            event_type=CallbackEventType.THINKING,
            data=test_data
        )
        
        assert success is True
        
        # Check event was queued
        events = callback_manager.event_queue[session_id]
        assert len(events) == 1
        
        event = events[0]
        assert event.event_type == CallbackEventType.THINKING
        assert event.session_id == session_id
        assert event.data == test_data
        assert isinstance(event.timestamp, datetime)
    
    @pytest.mark.asyncio
    async def test_event_filtering(self, callback_manager):
        """Test event filtering based on configuration"""
        session_id = str(uuid.uuid4())
        
        # Configure to only receive THINKING events
        config = CallbackConfig(
            delivery_method=CallbackDeliveryMethod.POLLING,
            events=[CallbackEventType.THINKING]
        )
        callback_manager.register_session(session_id, config)
        
        # Emit different event types
        await callback_manager.emit_event(session_id, CallbackEventType.THINKING, {"thought": "test"})
        await callback_manager.emit_event(session_id, CallbackEventType.PROGRESS, {"progress": 0.5})
        await callback_manager.emit_event(session_id, CallbackEventType.COMPLETION, {"result": "done"})
        
        # Only THINKING event should be queued
        events = callback_manager.event_queue[session_id]
        assert len(events) == 1
        assert events[0].event_type == CallbackEventType.THINKING
    
    @pytest.mark.asyncio
    async def test_webhook_delivery(self, callback_manager):
        """Test webhook delivery mechanism"""
        session_id = str(uuid.uuid4())
        
        config = CallbackConfig(
            delivery_method=CallbackDeliveryMethod.WEBHOOK,
            webhook_url="https://httpbin.org/post",
            events=[CallbackEventType.PROGRESS],
            timeout=10
        )
        
        # Mock the HTTP session
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value = mock_session
            
            callback_manager._session = mock_session
            callback_manager.register_session(session_id, config)
            
            # Emit event
            success = await callback_manager.emit_event(
                session_id=session_id,
                event_type=CallbackEventType.PROGRESS,
                data={"progress": 0.75, "message": "Almost done"}
            )
            
            assert success is True
            
            # Verify HTTP request was made
            mock_session.post.assert_called_once()
            call_args = mock_session.post.call_args
            assert call_args[0][0] == config.webhook_url
            assert call_args[1]['headers']['Content-Type'] == 'application/json'
    
    @pytest.mark.asyncio
    async def test_websocket_delivery(self, callback_manager, sample_config):
        """Test WebSocket delivery mechanism"""
        session_id = str(uuid.uuid4())
        mock_websocket = AsyncMock()
        
        # Configure for WebSocket delivery
        config = CallbackConfig(
            delivery_method=CallbackDeliveryMethod.WEBSOCKET,
            events=[CallbackEventType.THINKING]
        )
        
        callback_manager.register_session(session_id, config)
        callback_manager.register_websocket(session_id, mock_websocket)
        
        # Emit event
        test_data = {"thought": "WebSocket test"}
        success = await callback_manager.emit_event(
            session_id=session_id,
            event_type=CallbackEventType.THINKING,
            data=test_data
        )
        
        assert success is True
        
        # Verify WebSocket send was called
        mock_websocket.send_text.assert_called_once()
        sent_data = json.loads(mock_websocket.send_text.call_args[0][0])
        assert sent_data['type'] == 'callback'
        assert sent_data['event_type'] == CallbackEventType.THINKING
        assert sent_data['data'] == test_data
    
    def test_event_polling(self, callback_manager, sample_config):
        """Test event polling functionality"""
        session_id = str(uuid.uuid4())
        callback_manager.register_session(session_id, sample_config)
        
        # Add some events manually
        events = [
            CallbackEvent(
                event_type=CallbackEventType.THINKING,
                timestamp=datetime.now() - timedelta(minutes=5),
                session_id=session_id,
                data={"thought": "Old event"}
            ),
            CallbackEvent(
                event_type=CallbackEventType.PROGRESS,
                timestamp=datetime.now(),
                session_id=session_id,
                data={"progress": 0.5}
            )
        ]
        callback_manager.event_queue[session_id] = events
        
        # Poll all events
        all_events = callback_manager.get_queued_events(session_id)
        assert len(all_events) == 2
        
        # Poll events since specific time
        since_time = datetime.now() - timedelta(minutes=1)
        recent_events = callback_manager.get_queued_events(session_id, since_time)
        assert len(recent_events) == 1
        assert recent_events[0].data["progress"] == 0.5
    
    def test_event_cleanup(self, callback_manager, sample_config):
        """Test event queue cleanup"""
        session_id = str(uuid.uuid4())
        callback_manager.register_session(session_id, sample_config)
        
        # Add events
        events = [
            CallbackEvent(
                event_type=CallbackEventType.THINKING,
                timestamp=datetime.now() - timedelta(hours=2),
                session_id=session_id,
                data={"thought": "Old event"}
            ),
            CallbackEvent(
                event_type=CallbackEventType.PROGRESS,
                timestamp=datetime.now(),
                session_id=session_id,
                data={"progress": 0.5}
            )
        ]
        callback_manager.event_queue[session_id] = events
        
        # Clear old events
        cutoff_time = datetime.now() - timedelta(hours=1)
        cleared_count = callback_manager.clear_queued_events(session_id, cutoff_time)
        
        assert cleared_count == 1
        remaining_events = callback_manager.event_queue[session_id]
        assert len(remaining_events) == 1
        assert remaining_events[0].data["progress"] == 0.5
    
    def test_session_statistics(self, callback_manager, sample_config):
        """Test session statistics tracking"""
        session_id = str(uuid.uuid4())
        callback_manager.register_session(session_id, sample_config)
        
        # Simulate some delivery statistics
        stats = callback_manager.delivery_stats[session_id]
        stats["sent"] = 10
        stats["delivered"] = 8
        stats["failed"] = 2
        stats["retries"] = 3
        
        # Get session stats
        session_stats = callback_manager.get_session_stats(session_id)
        assert session_stats is not None
        assert session_stats["sent"] == 10
        assert session_stats["delivered"] == 8
        assert session_stats["failed"] == 2
        assert session_stats["retries"] == 3
        assert session_stats["queued_events"] == 0
    
    def test_overall_statistics(self, callback_manager, sample_config):
        """Test overall system statistics"""
        # Create multiple sessions with stats
        for i in range(3):
            session_id = str(uuid.uuid4())
            callback_manager.register_session(session_id, sample_config)
            stats = callback_manager.delivery_stats[session_id]
            stats["sent"] = 10 * (i + 1)
            stats["delivered"] = 8 * (i + 1)
            stats["failed"] = 2 * (i + 1)
            stats["retries"] = i
        
        # Get overall stats
        overall_stats = callback_manager.get_all_stats()
        assert overall_stats["total_sessions"] == 3
        assert overall_stats["total_events_sent"] == 60  # 10 + 20 + 30
        assert overall_stats["total_events_delivered"] == 48  # 8 + 16 + 24
        assert overall_stats["total_events_failed"] == 12  # 2 + 4 + 6
        assert overall_stats["total_retries"] == 3  # 0 + 1 + 2
        assert overall_stats["delivery_rate"] == 48/60  # 0.8


class TestCallbackAPI:
    """Test suite for callback API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client for API testing"""
        return TestClient(app)
    
    def test_create_callback_session(self, client):
        """Test callback session creation endpoint"""
        config_data = {
            "delivery_method": "webhook",
            "webhook_url": "https://example.com/callback",
            "events": ["thinking", "progress", "completion"],
            "include_intermediate_results": True,
            "timeout": 30,
            "retry_attempts": 3
        }
        
        response = client.post("/callbacks/sessions", json=config_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "session_id" in data
        assert "config" in data
        assert "created_at" in data
        assert data["status"] == "active"
        
        # Verify configuration
        config = data["config"]
        assert config["delivery_method"] == "webhook"
        assert config["webhook_url"] == "https://example.com/callback"
        assert set(config["events"]) == set(["thinking", "progress", "completion"])
    
    def test_delete_callback_session(self, client):
        """Test callback session deletion endpoint"""
        # First create a session
        config_data = {
            "delivery_method": "polling",
            "events": ["thinking"]
        }
        
        create_response = client.post("/callbacks/sessions", json=config_data)
        session_id = create_response.json()["session_id"]
        
        # Delete the session
        delete_response = client.delete(f"/callbacks/sessions/{session_id}")
        assert delete_response.status_code == 200
        
        data = delete_response.json()
        assert data["message"] == "Session deleted successfully"
        assert data["session_id"] == session_id
    
    def test_get_callback_stats(self, client):
        """Test callback statistics endpoint"""
        # Create a session
        config_data = {
            "delivery_method": "polling",
            "events": ["thinking", "progress"]
        }
        
        create_response = client.post("/callbacks/sessions", json=config_data)
        session_id = create_response.json()["session_id"]
        
        # Get stats
        stats_response = client.get(f"/callbacks/sessions/{session_id}/stats")
        assert stats_response.status_code == 200
        
        stats = stats_response.json()
        assert stats["session_id"] == session_id
        assert "events_sent" in stats
        assert "events_delivered" in stats
        assert "events_failed" in stats
        assert "delivery_rate" in stats
    
    def test_poll_callback_events(self, client):
        """Test callback event polling endpoint"""
        # Create a session
        config_data = {
            "delivery_method": "polling",
            "events": ["thinking", "progress"]
        }
        
        create_response = client.post("/callbacks/sessions", json=config_data)
        session_id = create_response.json()["session_id"]
        
        # Poll events (should be empty initially)
        poll_response = client.get(f"/callbacks/sessions/{session_id}/events")
        assert poll_response.status_code == 200
        
        data = poll_response.json()
        assert data["events"] == []
        assert data["total_count"] == 0
        assert data["has_more"] is False
    
    def test_test_callback_endpoint(self, client):
        """Test callback testing endpoint"""
        # Create a session
        config_data = {
            "delivery_method": "polling",
            "events": ["progress"]
        }
        
        create_response = client.post("/callbacks/sessions", json=config_data)
        session_id = create_response.json()["session_id"]
        
        # Send test callback
        test_response = client.post(
            f"/callbacks/sessions/{session_id}/test",
            params={"event_type": "progress"},
            json={"message": "Test callback", "progress": 0.5}
        )
        assert test_response.status_code == 200
        
        data = test_response.json()
        assert data["success"] is True
        assert data["session_id"] == session_id
        assert data["event_type"] == "progress"
    
    def test_clear_callback_events(self, client):
        """Test callback event clearing endpoint"""
        # Create a session
        config_data = {
            "delivery_method": "polling",
            "events": ["thinking"]
        }
        
        create_response = client.post("/callbacks/sessions", json=config_data)
        session_id = create_response.json()["session_id"]
        
        # Send some test events
        for i in range(5):
            client.post(f"/callbacks/sessions/{session_id}/test")
        
        # Clear events
        clear_response = client.delete(f"/callbacks/sessions/{session_id}/events")
        assert clear_response.status_code == 200
        
        data = clear_response.json()
        assert "cleared_count" in data
        assert data["session_id"] == session_id
    
    def test_overall_callback_stats(self, client):
        """Test overall callback statistics endpoint"""
        response = client.get("/callbacks/stats")
        assert response.status_code == 200
        
        stats = response.json()
        assert "total_sessions" in stats
        assert "active_websockets" in stats
        assert "active_sse" in stats
        assert "total_events_sent" in stats
        assert "total_events_delivered" in stats
        assert "delivery_rate" in stats


class TestChatIntegration:
    """Test suite for chat API callback integration"""
    
    @pytest.fixture
    def client(self):
        """Create test client for API testing"""
        return TestClient(app)
    
    def test_chat_with_callbacks(self, client):
        """Test chat endpoint with callback configuration"""
        # Mock the LLM client to avoid external dependencies
        with patch('app.api.main.api_state') as mock_state:
            mock_llm_client = AsyncMock()
            mock_response = Mock()
            mock_response.choices = [{"message": {"content": "Test response"}}]
            mock_response.usage = {"total_tokens": 100}
            mock_llm_client.chat_completion.return_value = mock_response
            
            mock_state.llm_client = mock_llm_client
            mock_state.model_manager = None
            mock_state.conversations = {}
            
            # Send chat request with callback configuration
            chat_data = {
                "message": "Hello, test message",
                "callback_config": {
                    "delivery_method": "polling",
                    "events": ["thinking", "completion"],
                    "include_intermediate_results": True,
                    "timeout": 30
                }
            }
            
            response = client.post("/chat", json=chat_data)
            assert response.status_code == 200
            
            data = response.json()
            assert "response" in data
            assert "session_id" in data
            assert data["session_id"] is not None
    
    def test_chat_without_callbacks(self, client):
        """Test chat endpoint without callback configuration"""
        with patch('app.api.main.api_state') as mock_state:
            mock_llm_client = AsyncMock()
            mock_response = Mock()
            mock_response.choices = [{"message": {"content": "Test response"}}]
            mock_response.usage = {"total_tokens": 100}
            mock_llm_client.chat_completion.return_value = mock_response
            
            mock_state.llm_client = mock_llm_client
            mock_state.model_manager = None
            mock_state.conversations = {}
            
            # Send chat request without callback configuration
            chat_data = {
                "message": "Hello, test message"
            }
            
            response = client.post("/chat", json=chat_data)
            assert response.status_code == 200
            
            data = response.json()
            assert "response" in data
            assert data.get("session_id") is None


class TestConvenienceFunctions:
    """Test suite for convenience callback functions"""
    
    @pytest.mark.asyncio
    async def test_emit_thinking_event(self):
        """Test thinking event emission convenience function"""
        with patch('app.api.callback_manager.callback_manager') as mock_manager:
            mock_manager.emit_event = AsyncMock(return_value=True)
            
            session_id = "test_session"
            thought = "I need to analyze this data"
            
            await emit_thinking_event(session_id, thought)
            
            mock_manager.emit_event.assert_called_once_with(
                session_id=session_id,
                event_type=CallbackEventType.THINKING,
                data={"thought": thought},
                conversation_id=None
            )
    
    @pytest.mark.asyncio
    async def test_emit_tool_use_event(self):
        """Test tool use event emission convenience function"""
        with patch('app.api.callback_manager.callback_manager') as mock_manager:
            mock_manager.emit_event = AsyncMock(return_value=True)
            
            session_id = "test_session"
            tool_name = "web_search"
            parameters = {"query": "test query", "limit": 10}
            
            await emit_tool_use_event(session_id, tool_name, parameters)
            
            mock_manager.emit_event.assert_called_once_with(
                session_id=session_id,
                event_type=CallbackEventType.TOOL_USE,
                data={"tool_name": tool_name, "parameters": parameters},
                conversation_id=None
            )
    
    @pytest.mark.asyncio
    async def test_emit_progress_event(self):
        """Test progress event emission convenience function"""
        with patch('app.api.callback_manager.callback_manager') as mock_manager:
            mock_manager.emit_event = AsyncMock(return_value=True)
            
            session_id = "test_session"
            progress = 0.75
            message = "Processing data..."
            
            await emit_progress_event(session_id, progress, message)
            
            mock_manager.emit_event.assert_called_once_with(
                session_id=session_id,
                event_type=CallbackEventType.PROGRESS,
                data={"progress": progress, "message": message},
                conversation_id=None
            )
    
    @pytest.mark.asyncio
    async def test_emit_completion_event(self):
        """Test completion event emission convenience function"""
        with patch('app.api.callback_manager.callback_manager') as mock_manager:
            mock_manager.emit_event = AsyncMock(return_value=True)
            
            session_id = "test_session"
            result = {"status": "success", "data": "processed"}
            
            await emit_completion_event(session_id, result)
            
            mock_manager.emit_event.assert_called_once_with(
                session_id=session_id,
                event_type=CallbackEventType.COMPLETION,
                data={"result": result, "success": True},
                conversation_id=None
            )
    
    @pytest.mark.asyncio
    async def test_emit_error_event(self):
        """Test error event emission convenience function"""
        with patch('app.api.callback_manager.callback_manager') as mock_manager:
            mock_manager.emit_event = AsyncMock(return_value=True)
            
            session_id = "test_session"
            error = "Connection timeout"
            error_type = "network_error"
            
            await emit_error_event(session_id, error, error_type)
            
            mock_manager.emit_event.assert_called_once_with(
                session_id=session_id,
                event_type=CallbackEventType.ERROR,
                data={"error": error, "error_type": error_type},
                conversation_id=None
            )


class TestErrorHandling:
    """Test suite for callback error handling"""
    
    @pytest.fixture
    def callback_manager(self):
        """Create a fresh callback manager for each test"""
        return CallbackManager()
    
    @pytest.mark.asyncio
    async def test_webhook_retry_logic(self, callback_manager):
        """Test webhook delivery retry logic"""
        session_id = str(uuid.uuid4())
        
        config = CallbackConfig(
            delivery_method=CallbackDeliveryMethod.WEBHOOK,
            webhook_url="https://httpbin.org/status/500",  # Returns 500 error
            events=[CallbackEventType.PROGRESS],
            retry_attempts=3,
            retry_delay=0.1  # Fast retry for testing
        )
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 500  # Simulate server error
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value = mock_session
            
            callback_manager._session = mock_session
            callback_manager.register_session(session_id, config)
            
            # Emit event (should trigger retries)
            success = await callback_manager.emit_event(
                session_id=session_id,
                event_type=CallbackEventType.PROGRESS,
                data={"progress": 0.5}
            )
            
            # Should fail after retries
            assert success is False
            
            # Verify retry attempts were made
            assert mock_session.post.call_count == config.retry_attempts
            
            # Check retry count in stats
            stats = callback_manager.delivery_stats[session_id]
            assert stats["retries"] == config.retry_attempts - 1
    
    @pytest.mark.asyncio
    async def test_websocket_connection_failure(self, callback_manager):
        """Test WebSocket delivery with connection failure"""
        session_id = str(uuid.uuid4())
        
        config = CallbackConfig(
            delivery_method=CallbackDeliveryMethod.WEBSOCKET,
            events=[CallbackEventType.THINKING]
        )
        
        # Mock WebSocket that raises exception
        mock_websocket = AsyncMock()
        mock_websocket.send_text.side_effect = Exception("Connection lost")
        
        callback_manager.register_session(session_id, config)
        callback_manager.register_websocket(session_id, mock_websocket)
        
        # Emit event (should handle exception)
        success = await callback_manager.emit_event(
            session_id=session_id,
            event_type=CallbackEventType.THINKING,
            data={"thought": "test"}
        )
        
        assert success is False
        
        # WebSocket should be removed from connections
        assert session_id not in callback_manager.websocket_connections
    
    def test_invalid_session_handling(self, callback_manager):
        """Test handling of invalid session IDs"""
        invalid_session_id = "invalid_session"
        
        # Try to get stats for non-existent session
        stats = callback_manager.get_session_stats(invalid_session_id)
        assert stats is None
        
        # Try to get events for non-existent session
        events = callback_manager.get_queued_events(invalid_session_id)
        assert events == []
        
        # Try to clear events for non-existent session
        cleared = callback_manager.clear_queued_events(invalid_session_id)
        assert cleared == 0


class TestPerformance:
    """Test suite for callback system performance"""
    
    @pytest.fixture
    def callback_manager(self):
        """Create a fresh callback manager for each test"""
        return CallbackManager()
    
    @pytest.mark.asyncio
    async def test_high_volume_events(self, callback_manager):
        """Test handling of high-volume event emission"""
        session_id = str(uuid.uuid4())
        
        config = CallbackConfig(
            delivery_method=CallbackDeliveryMethod.POLLING,
            events=[CallbackEventType.PROGRESS]
        )
        
        callback_manager.register_session(session_id, config)
        
        # Emit many events quickly
        start_time = time.time()
        event_count = 1000
        
        tasks = []
        for i in range(event_count):
            task = callback_manager.emit_event(
                session_id=session_id,
                event_type=CallbackEventType.PROGRESS,
                data={"progress": i / event_count, "step": i}
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # All events should succeed
        assert all(results)
        
        # Check performance
        duration = end_time - start_time
        events_per_second = event_count / duration
        print(f"Processed {event_count} events in {duration:.2f}s ({events_per_second:.0f} events/sec)")
        
        # Verify all events were queued
        queued_events = callback_manager.event_queue[session_id]
        assert len(queued_events) == event_count
    
    def test_memory_usage_with_many_sessions(self, callback_manager):
        """Test memory usage with many concurrent sessions"""
        session_count = 100
        events_per_session = 50
        
        config = CallbackConfig(
            delivery_method=CallbackDeliveryMethod.POLLING,
            events=[CallbackEventType.THINKING, CallbackEventType.PROGRESS]
        )
        
        # Create many sessions
        session_ids = []
        for i in range(session_count):
            session_id = f"session_{i}"
            callback_manager.register_session(session_id, config)
            session_ids.append(session_id)
            
            # Add events to each session
            for j in range(events_per_session):
                event = CallbackEvent(
                    event_type=CallbackEventType.PROGRESS,
                    timestamp=datetime.now(),
                    session_id=session_id,
                    data={"progress": j / events_per_session}
                )
                callback_manager.event_queue[session_id].append(event)
        
        # Verify all sessions and events exist
        assert len(callback_manager.active_sessions) == session_count
        
        total_events = sum(len(queue) for queue in callback_manager.event_queue.values())
        assert total_events == session_count * events_per_session
        
        # Test cleanup
        for session_id in session_ids:
            callback_manager.unregister_session(session_id)
        
        assert len(callback_manager.active_sessions) == 0
        assert len(callback_manager.event_queue) == 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])

