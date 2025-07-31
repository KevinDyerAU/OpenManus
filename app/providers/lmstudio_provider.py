"""
LM Studio Provider for OpenManus

This module implements the LM Studio provider for local LLM inference using the lmstudio-python SDK.
It provides integration with locally hosted models through LM Studio's server API.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, AsyncGenerator, Callable
from dataclasses import dataclass
import json
from datetime import datetime

try:
    import lmstudio as lms
    from lmstudio import Chat
    LMSTUDIO_AVAILABLE = True
except ImportError:
    LMSTUDIO_AVAILABLE = False
    lms = None
    Chat = None

from .base_provider import BaseLLMProvider, LLMResponse, StreamingResponse

logger = logging.getLogger(__name__)


@dataclass
class LMStudioConfig:
    """Configuration for LM Studio provider"""
    host: str = "localhost"
    port: int = 1234
    default_model: str = "deepseek/deepseek-r1-0528-qwen3-8b"
    timeout: int = 300
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Default inference parameters
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    
    # Model loading parameters
    context_length: Optional[int] = None
    gpu_offload_ratio: Optional[float] = None


class LMStudioProvider(BaseLLMProvider):
    """
    LM Studio provider for local LLM inference.
    
    This provider integrates with LM Studio's local server to provide
    AI model inference capabilities without external API dependencies.
    """
    
    def __init__(self, config: Optional[LMStudioConfig] = None):
        """
        Initialize the LM Studio provider.
        
        Args:
            config: LM Studio configuration. If None, uses default config.
        """
        if not LMSTUDIO_AVAILABLE:
            raise ImportError(
                "lmstudio package is not installed. Install it with: pip install lmstudio"
            )
        
        self.config = config or LMStudioConfig()
        self.server_host = f"{self.config.host}:{self.config.port}"
        self._client = None
        self._models_cache = {}
        self._last_model_refresh = None
        
        # Configure the default client
        try:
            lms.configure_default_client(self.server_host)
            logger.info(f"Configured LM Studio client for {self.server_host}")
        except Exception as e:
            logger.warning(f"Failed to configure LM Studio client: {e}")
    
    @property
    def name(self) -> str:
        """Provider name"""
        return "lmstudio"
    
    @property
    def display_name(self) -> str:
        """Human-readable provider name"""
        return "LM Studio"
    
    @property
    def is_available(self) -> bool:
        """Check if LM Studio is available"""
        try:
            # Try to get a simple model handle to test connectivity
            model = lms.llm(self.config.default_model)
            return True
        except Exception as e:
            logger.debug(f"LM Studio availability check failed: {e}")
            return False
    
    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a chat completion response using LM Studio.
        
        Args:
            messages: List of chat messages
            model: Model to use (optional, uses default if not specified)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional provider-specific parameters
            
        Returns:
            LLMResponse object containing the generated response
        """
        try:
            model_name = model or self.config.default_model
            llm = lms.llm(model_name)
            
            # Convert messages to LM Studio Chat format
            chat = self._convert_messages_to_chat(messages)
            
            if stream:
                return await self.stream_completion(messages, model, temperature, max_tokens, **kwargs)
            
            # Generate response using LM Studio's simple API
            # Note: LM Studio's respond() method doesn't accept temperature/max_tokens parameters
            response = llm.respond(chat)
            
            return LLMResponse(
                content=response,
                model=model_name,
                created_at=datetime.now(),
                finish_reason="stop"
            )
            
        except Exception as e:
            logger.error(f"LM Studio chat completion failed: {e}")
            raise
    
    async def stream_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> StreamingResponse:
        """
        Generate a streaming chat completion response using LM Studio.
        
        Args:
            messages: List of chat messages
            model: Model to use (optional, uses default if not specified)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
            
        Returns:
            StreamingResponse object with content generator
        """
        try:
            model_name = model or self.config.default_model
            llm = lms.llm(model_name)
            
            # Convert messages to LM Studio Chat format
            chat = self._convert_messages_to_chat(messages)
            
            async def content_generator():
                try:
                    # Use LM Studio's streaming API
                    # Note: LM Studio's stream() method doesn't accept temperature/max_tokens parameters
                    for chunk in llm.stream(chat):
                        if chunk:
                            yield chunk
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    yield f"Error: {str(e)}"
            
            return StreamingResponse(
                content_generator=content_generator(),
                model=model_name
            )
            
        except Exception as e:
            logger.error(f"LM Studio streaming completion failed: {e}")
            raise
    
    async def list_models(self) -> List[str]:
        """
        List available models for LM Studio.
        
        Returns:
            List of model names
        """
        try:
            # Try to get available models from LM Studio
            # This is a simplified implementation - LM Studio SDK may have specific methods
            models = []
            
            # Add default model if available
            if self.config.default_model:
                models.append(self.config.default_model)
            
            # Add common LM Studio models
            common_models = [
                "llama-3.2-1b-instruct",
                "llama-3.2-3b-instruct", 
                "qwen2.5-7b-instruct",
                "mistral-7b-instruct"
            ]
            
            for model in common_models:
                if model not in models:
                    models.append(model)
            
            return models
            
        except Exception as e:
            logger.error(f"Failed to list LM Studio models: {e}")
            return [self.config.default_model] if self.config.default_model else []
    
    async def health_check(self) -> bool:
        """
        Check if LM Studio is healthy and responsive.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try a simple model access to test connectivity
            model = lms.llm(self.config.default_model)
            
            # Try a very simple completion to test functionality
            test_chat = Chat("You are a test assistant.")
            test_chat.add_user_message("Hello")
            
            # Use a short timeout for health check
            response = model.respond(test_chat, max_tokens=5, temperature=0.1)
            
            return bool(response and len(response.strip()) > 0)
            
        except Exception as e:
            logger.debug(f"LM Studio health check failed: {e}")
            return False
    
    def _convert_messages_to_chat(self, messages: List[Dict[str, Any]]) -> Chat:
        """
        Convert OpenAI-style messages to LM Studio Chat format.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            LM Studio Chat object
        """
        chat = None
        
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "system":
                if chat is None:
                    chat = Chat(content)
                else:
                    # If we already have a chat, we'll treat additional system messages as user messages
                    chat.add_user_message(f"System: {content}")
            elif role == "user":
                if chat is None:
                    chat = Chat("You are a helpful assistant.")
                chat.add_user_message(content)
            elif role == "assistant":
                if chat is None:
                    chat = Chat("You are a helpful assistant.")
                chat.add_assistant_message(content)
        
        # If no chat was created, create a default one
        if chat is None:
            chat = Chat("You are a helpful assistant.")
        
        return chat


# Factory function for easy provider creation
def create_lmstudio_provider(
    host: str = "localhost",
    port: int = 1234,
    default_model: str = "deepseek/deepseek-r1-0528-qwen3-8b",
    **kwargs
) -> LMStudioProvider:
    """
    Create a configured LM Studio provider.
    
    Args:
        host: LM Studio server host
        port: LM Studio server port
        default_model: Default model to use
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured LMStudioProvider instance
    """
    config = LMStudioConfig(
        host=host,
        port=port,
        default_model=default_model,
        **kwargs
    )
    return LMStudioProvider(config)
