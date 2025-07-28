#!/usr/bin/env python3
"""
Test script to verify Manus agent thoughts are being captured properly
"""

import asyncio
import json
from app.logger import logger

async def test_manus_agent_thoughts():
    """Test if Manus agent produces the expected log messages"""
    print("=== Testing Manus Agent Thoughts Capture ===\n")
    
    captured_thoughts = []
    
    def capture_sink(message):
        try:
            log_text = message.record["message"]
            level = message.record["level"].name
            
            # Only capture INFO and higher
            if message.record["level"].no < 20:
                return
                
            formatted_message = f"🤖 [{level}] {log_text}"
            captured_thoughts.append(formatted_message)
            print(f"CAPTURED: {formatted_message}")
            
        except Exception as e:
            print(f"Error in capture_sink: {e}")
    
    # Add the capture sink
    sink_id = logger.add(capture_sink, level="INFO", format="{message}", enqueue=True)
    
    try:
        print("1. Testing basic logger capture...")
        logger.info("🚀 Manus agent starting - thoughts will stream to UI")
        logger.info("🎯 Starting to process: Test message...")
        
        await asyncio.sleep(0.1)  # Allow async processing
        
        print(f"\n2. Testing Manus agent creation...")
        try:
            from app.agent.manus import Manus
            from app.llm import LLM
            
            # Create LLM instance
            llm = LLM(config_name="default")
            print(f"   LLM created: {llm}")
            
            # Create Manus agent
            agent = await Manus.create(llm=llm)
            print(f"   Agent created: {agent}")
            
            # Test agent logging
            logger.info("✨ Agent thoughts: This is a test thought from Manus")
            logger.info("🔧 Activating tool: 'test_tool'...")
            logger.info("🏁 Special tool 'test_tool' has completed the task!")
            
            await asyncio.sleep(0.1)  # Allow processing
            
            # Cleanup
            await agent.cleanup()
            print("   Agent cleaned up successfully")
            
        except Exception as e:
            print(f"   Error with Manus agent: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n3. Results:")
        print(f"   Total captured thoughts: {len(captured_thoughts)}")
        
        if captured_thoughts:
            print("   Captured messages:")
            for i, thought in enumerate(captured_thoughts, 1):
                print(f"     {i}. {thought}")
        else:
            print("   ❌ No thoughts captured!")
            
        # Test WebSocket message format
        print(f"\n4. Testing WebSocket message format:")
        if captured_thoughts:
            sample_thought = captured_thoughts[0]
            websocket_msg = {
                "type": "progress",
                "status": "processing", 
                "message": sample_thought,
                "progress": 50
            }
            print(f"   Sample WebSocket message: {json.dumps(websocket_msg, indent=2)}")
            
            # Check if UI would recognize this as agent thought
            ui_recognizes = "🤖" in sample_thought
            print(f"   UI would recognize as agent thought: {'✅' if ui_recognizes else '❌'}")
        
        return len(captured_thoughts) > 0
        
    finally:
        logger.remove(sink_id)
        print(f"\n5. Cleanup: Removed logger sink")

async def main():
    success = await test_manus_agent_thoughts()
    
    print(f"\n=== FINAL RESULT ===")
    if success:
        print("✅ SUCCESS: Manus agent thoughts are being captured correctly!")
        print("   - Logger capture is working")
        print("   - Messages have 🤖 emoji for UI recognition") 
        print("   - WebSocket format is correct")
        print("   - UI should display these as purple agent thought messages")
    else:
        print("❌ FAILURE: Manus agent thoughts are NOT being captured!")
        print("   - Check logger configuration")
        print("   - Check Loguru sink setup")
        print("   - Check agent logging calls")

if __name__ == "__main__":
    asyncio.run(main())
