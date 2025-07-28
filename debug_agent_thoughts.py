#!/usr/bin/env python3
"""
Debug script to identify why Manus agent thoughts aren't reaching the UI
"""

import asyncio
import json
from app.logger import logger

class MockWebSocket:
    """Mock WebSocket to capture what would be sent to UI"""
    def __init__(self):
        self.sent_messages = []
    
    async def send_text(self, message):
        """Capture messages that would be sent to UI"""
        data = json.loads(message)
        self.sent_messages.append(data)
        print(f"MOCK WEBSOCKET SENT: {data}")

async def debug_agent_thoughts_flow():
    """Debug the complete agent thoughts flow"""
    print("=== Debugging Agent Thoughts Flow ===\n")
    
    # Create mock WebSocket
    mock_websocket = MockWebSocket()
    
    # Step 1: Test basic Loguru capture
    print("1. Testing basic Loguru capture...")
    captured_logs = []
    
    def websocket_sink(message):
        try:
            log_text = message.record["message"]
            level = message.record["level"].name
            
            # Only capture INFO and higher
            if message.record["level"].no < 20:
                return
            
            formatted_message = f"🤖 [{level}] {log_text}"
            captured_logs.append(formatted_message)
            
            # Create progress message for UI
            progress_msg = {
                "type": "progress",
                "status": "processing",
                "message": formatted_message,
                "progress": 50
            }
            
            # Send via mock WebSocket
            asyncio.create_task(mock_websocket.send_text(json.dumps(progress_msg)))
            print(f"SINK CAPTURED: {formatted_message}")
            
        except Exception as e:
            print(f"Error in websocket_sink: {e}")
    
    # Add sink
    sink_id = logger.add(websocket_sink, level="INFO", format="{message}", enqueue=True)
    
    try:
        # Step 2: Test logger messages
        print("\n2. Testing logger messages...")
        logger.info("🚀 Manus agent starting - thoughts will stream to UI")
        logger.info("🎯 Starting to process: Test message...")
        logger.info("✨ Agent thoughts: This is a test thought")
        logger.info("🔧 Activating tool: 'test_tool'...")
        logger.info("✅ Agent execution completed successfully")
        
        # Wait for async processing
        await asyncio.sleep(0.2)
        
        # Step 3: Test Manus agent creation and logging
        print("\n3. Testing Manus agent...")
        try:
            from app.agent.manus import Manus
            from app.llm import LLM
            
            # Create LLM and agent
            llm = LLM(config_name="default")
            agent = await Manus.create(llm=llm)
            
            # Test agent-specific logging
            logger.info("✨ Manus's thoughts: Analyzing the request...")
            logger.info("🔧 Tool arguments: {'url': 'https://example.com'}")
            logger.info("🏁 Special tool 'browser_tool' has completed the task!")
            
            await asyncio.sleep(0.2)
            await agent.cleanup()
            
        except Exception as e:
            print(f"Error with Manus agent: {e}")
        
        # Step 4: Analyze results
        print(f"\n4. Results Analysis:")
        print(f"   Captured logs: {len(captured_logs)}")
        print(f"   WebSocket messages sent: {len(mock_websocket.sent_messages)}")
        
        if captured_logs:
            print("   Captured messages:")
            for i, log in enumerate(captured_logs, 1):
                print(f"     {i}. {log}")
        
        if mock_websocket.sent_messages:
            print("   WebSocket messages:")
            for i, msg in enumerate(mock_websocket.sent_messages, 1):
                print(f"     {i}. {msg}")
                
                # Check if UI would recognize this
                if msg.get('message') and '🤖' in msg['message']:
                    print(f"        ✅ UI would recognize as agent thought")
                else:
                    print(f"        ❌ UI would NOT recognize as agent thought")
        
        # Step 5: Check for issues
        print(f"\n5. Issue Analysis:")
        if len(captured_logs) == 0:
            print("   ❌ ISSUE: No logs captured by sink")
        elif len(mock_websocket.sent_messages) == 0:
            print("   ❌ ISSUE: No WebSocket messages sent")
        elif not any('🤖' in msg.get('message', '') for msg in mock_websocket.sent_messages):
            print("   ❌ ISSUE: WebSocket messages missing 🤖 emoji")
        else:
            print("   ✅ All checks passed - should work in UI")
        
        return len(captured_logs) > 0 and len(mock_websocket.sent_messages) > 0
        
    finally:
        logger.remove(sink_id)

async def main():
    success = await debug_agent_thoughts_flow()
    
    print(f"\n=== FINAL DIAGNOSIS ===")
    if success:
        print("✅ Agent thoughts capture system is working correctly")
        print("   The issue may be:")
        print("   - WebSocket connection problems")
        print("   - API server not running properly")
        print("   - UI not connecting to WebSocket")
        print("   - Task type mismatch")
    else:
        print("❌ Agent thoughts capture system has issues")
        print("   Check:")
        print("   - Loguru sink configuration")
        print("   - Logger import and setup")
        print("   - WebSocket message formatting")

if __name__ == "__main__":
    asyncio.run(main())
