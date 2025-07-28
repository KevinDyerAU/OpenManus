#!/usr/bin/env python3
"""
Simple test to verify agent thoughts capture
"""

import asyncio
from app.logger import logger

def test_loguru_capture():
    """Test basic Loguru capture functionality"""
    print("Testing Loguru capture...")
    
    captured = []
    
    def test_sink(message):
        log_text = message.record["message"]
        level = message.record["level"].name
        formatted = f"🤖 [{level}] {log_text}"
        captured.append(formatted)
        print(f"CAPTURED: {formatted}")
    
    # Add sink
    sink_id = logger.add(test_sink, level="INFO")
    
    try:
        # Test logging
        logger.info("🚀 Agent starting")
        logger.info("✨ Agent thoughts: Processing request")
        logger.info("🔧 Activating tool: test_tool")
        logger.info("✅ Task completed")
        
        print(f"\nResults: {len(captured)} messages captured")
        for msg in captured:
            print(f"  - {msg}")
            
        # Check UI compatibility
        ui_compatible = all("🤖" in msg for msg in captured)
        print(f"UI compatible: {'✅' if ui_compatible else '❌'}")
        
        return len(captured) > 0
        
    finally:
        logger.remove(sink_id)

if __name__ == "__main__":
    success = test_loguru_capture()
    print(f"\nOverall: {'✅ SUCCESS' if success else '❌ FAILED'}")
