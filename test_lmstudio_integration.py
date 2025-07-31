#!/usr/bin/env python3
"""
Test script for LM Studio integration with OpenManus

This script tests the LM Studio provider integration to ensure it works correctly
with the OpenManus system.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.llm import LLM
from app.config import LLMSettings
from app.providers.lmstudio_provider import LMStudioProvider, LMStudioConfig, create_lmstudio_provider


async def test_lmstudio_provider_direct():
    """Test LM Studio provider directly"""
    print("🧪 Testing LM Studio Provider directly...")
    
    try:
        # Create LM Studio provider
        provider = create_lmstudio_provider(
            host="localhost",
            port=1234,
            default_model="deepseek/deepseek-r1-0528-qwen3-8b"
        )
        
        print(f"✅ Provider created: {provider.display_name}")
        print(f"✅ Provider name: {provider.name}")
        
        # Test availability
        is_available = provider.is_available
        print(f"✅ Provider available: {is_available}")
        
        if not is_available:
            print("⚠️  LM Studio server not available. Make sure LM Studio is running on localhost:1234")
            return False
        
        # Test health check
        health = await provider.health_check()
        print(f"✅ Health check: {health}")
        
        # Test model listing
        models = await provider.list_models()
        print(f"✅ Available models: {models}")
        
        # Test chat completion
        messages = [
            {"role": "user", "content": "Hello! Please respond with just 'Hello from LM Studio!'"}
        ]
        
        print("🔄 Testing chat completion...")
        response = await provider.chat_completion(messages, stream=False)
        print(f"✅ Chat response: {response.content}")
        
        # Test streaming completion
        print("🔄 Testing streaming completion...")
        streaming_response = await provider.stream_completion(messages)
        
        collected_content = []
        async for chunk in streaming_response.content_generator:
            collected_content.append(chunk)
            print(chunk, end="", flush=True)
        
        print("\n✅ Streaming completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Direct provider test failed: {e}")
        return False


async def test_lmstudio_via_llm_class():
    """Test LM Studio integration via OpenManus LLM class"""
    print("\n🧪 Testing LM Studio via OpenManus LLM class...")
    
    try:
        # Create LLM settings for LM Studio
        lmstudio_settings = LLMSettings(
            model="deepseek/deepseek-r1-0528-qwen3-8b",
            base_url="http://localhost:1234/v1",
            api_key="not-needed",
            max_tokens=2048,
            temperature=0.7,
            api_type="lmstudio",
            api_version="v1",
            host="localhost",
            port=1234
        )
        
        # Create LLM instance with proper config structure
        llm_config = {
            "test_lmstudio": lmstudio_settings,
            "default": lmstudio_settings  # Add default fallback
        }
        llm = LLM("test_lmstudio", llm_config)
        
        print(f"✅ LLM instance created with model: {llm.model}")
        print(f"✅ API type: {llm.api_type}")
        
        # Test non-streaming ask
        print("🔄 Testing non-streaming ask...")
        messages = [
            {"role": "user", "content": "Say 'Hello from OpenManus + LM Studio integration!'"}
        ]
        
        response = await llm.ask(messages, stream=False)
        print(f"✅ Non-streaming response: {response}")
        
        # Test streaming ask
        print("🔄 Testing streaming ask...")
        response = await llm.ask(messages, stream=True)
        print(f"✅ Streaming response: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM class test failed: {e}")
        return False


def test_configuration_loading():
    """Test configuration loading for LM Studio"""
    print("\n🧪 Testing configuration loading...")
    
    try:
        from app.config import config
        
        # Test if LM Studio configuration fields are available
        print("✅ Configuration system loaded")
        
        # Check if we can access LLM config
        llm_config = config.llm
        print(f"✅ LLM configuration available: {type(llm_config)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


async def main():
    """Main test function"""
    print("🚀 Starting LM Studio Integration Tests for OpenManus\n")
    
    # Test configuration loading
    config_test = test_configuration_loading()
    
    # Test direct provider
    provider_test = await test_lmstudio_provider_direct()
    
    # Test via LLM class
    llm_test = await test_lmstudio_via_llm_class()
    
    # Summary
    print("\n📊 Test Summary:")
    print(f"Configuration Loading: {'✅ PASS' if config_test else '❌ FAIL'}")
    print(f"Direct Provider Test: {'✅ PASS' if provider_test else '❌ FAIL'}")
    print(f"LLM Class Integration: {'✅ PASS' if llm_test else '❌ FAIL'}")
    
    if all([config_test, provider_test, llm_test]):
        print("\n🎉 All tests passed! LM Studio integration is working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        print("\nTroubleshooting tips:")
        print("1. Make sure LM Studio is running on localhost:1234")
        print("2. Ensure you have a model loaded in LM Studio")
        print("3. Check that the lmstudio package is installed: pip install lmstudio")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
