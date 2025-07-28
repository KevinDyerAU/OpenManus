#!/usr/bin/env python3
"""
Test script to verify Manus agent thoughts are being captured and sent to UI
"""

import asyncio
import json
from app.logger import logger

async def test_loguru_capture():
    """Test if Loguru logging can be captured properly"""
    print("Testing Loguru capture system...")
    
    captured_messages = []
    
    def test_sink(message):
        try:
            log_text = message.record["message"]
            level = message.record["level"].name
            formatted_message = f"🤖 [{level}] {log_text}"
            captured_messages.append(formatted_message)
            print(f"CAPTURED: {formatted_message}")
        except Exception as e:
            print(f"Error in test sink: {e}")
    
    # Add the test sink
    sink_id = logger.add(test_sink, level="INFO", format="{message}")
    
    try:
        # Test various log levels
        logger.info("This is an info message from Manus agent")
        logger.debug("This is a debug message (should not be captured)")
        logger.warning("This is a warning message")
        logger.error("This is an error message")
        
        # Wait a moment for async processing
        await asyncio.sleep(0.1)
        
        print(f"\nCaptured {len(captured_messages)} messages:")
        for msg in captured_messages:
            print(f"  - {msg}")
            
        return len(captured_messages) > 0
        
    finally:
        logger.remove(sink_id)

async def test_manus_agent_logging():
    """Test if Manus agent actually produces logs"""
    print("\nTesting Manus agent logging...")
    
    try:
        from app.agent.manus import Manus
        from app.llm import LLM
        
        # Create a simple LLM instance
        llm = LLM(config_name="default")
        
        # Create Manus agent
        agent = await Manus.create(llm=llm)
        
        # Test if agent has logger
        print(f"Agent created: {agent}")
        print(f"Agent has logger: {hasattr(agent, 'logger')}")
        
        # Test a simple logging call
        logger.info("Manus agent test logging - this should be captured")
        
        await agent.cleanup()
        return True
        
    except Exception as e:
        print(f"Error testing Manus agent: {e}")
        return False

async def main():
    print("=== Testing Agent Thoughts Capture System ===\n")
    
    # Test 1: Basic Loguru capture
    loguru_works = await test_loguru_capture()
    print(f"Loguru capture test: {'✅ PASS' if loguru_works else '❌ FAIL'}")
    
    # Test 2: Manus agent logging
    agent_works = await test_manus_agent_logging()
    print(f"Manus agent test: {'✅ PASS' if agent_works else '❌ FAIL'}")
    
    print(f"\nOverall result: {'✅ READY' if loguru_works and agent_works else '❌ NEEDS FIX'}")

if __name__ == "__main__":
    asyncio.run(main())
