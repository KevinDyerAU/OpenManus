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
from ..config import get_config

logger = logging.getLogger(__name__)


@dataclass
class LMStudioConfig:
    """Configuration for LM Studio provider"""
    host: str = "localhost"
    port: int = 1234
    default_model: str = "llama-3.2-1b-instruct"
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
            logger.debug(f"LM Studio not available: {e}")
            return False
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models from LM Studio.
        
        Returns:
            List of model information dictionaries
        """
        try:
            # Note: LM Studio SDK doesn't provide a direct model listing API
            # This is a placeholder implementation that returns common models
            # In practice, you would need to implement model discovery
            # through the LM Studio CLI or server API
            
            models = [
                {
                    "id": "llama-3.2-1b-instruct",
                    "name": "Llama 3.2 1B Instruct",
                    "provider": "lmstudio",
                    "type": "chat",
                    "context_length": 8192,
                    "description": "Fast and efficient instruction-following model"
                },
                {
                    "id": "llama-3.2-3b-instruct",
                    "name": "Llama 3.2 3B Instruct", 
                    "provider": "lmstudio",
                    "type": "chat",
                    "context_length": 8192,
                    "description": "Balanced performance and capability model"
                },
                {
                    "id": "qwen2.5-7b-instruct",
                    "name": "Qwen 2.5 7B Instruct",
                    "provider": "lmstudio",
                    "type": "chat",
                    "context_length": 32768,
                    "description": "High-capability multilingual model"
                }
            ]
            
            # Filter models that are actually available
            available_models = []
            for model_info in models:
                try:
                    model = lms.llm(model_info["id"])
                    available_models.append(model_info)
                except Exception as e:
                    logger.debug(f"Model {model_info['id']} not available: {e}")
            
            return available_models
            
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return []
    
    def _create_chat_context(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> Chat:
        """
        Create a Chat object from message history.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            system_prompt: Optional system prompt to prepend
            
        Returns:
            Chat object for LM Studio
        """
        # Start with system prompt if provided
        chat_messages = []
        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation messages
        chat_messages.extend(messages)
        
        # Create Chat object from history
        return Chat.from_history({"messages": chat_messages})
    
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response using LM Studio.
        
        Args:
            messages: Conversation messages
            model: Model identifier (optional, uses default if not provided)
            system_prompt: System prompt for the conversation
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with the generated content
        """
        try:
            # Use default model if not specified
            model_id = model or self.config.default_model
            
            # Get model handle
            llm_model = lms.llm(model_id)
            
            # Create chat context
            chat = self._create_chat_context(messages, system_prompt)
            
            # Prepare configuration
            config = {
                "temperature": temperature or self.config.temperature,
                "maxTokens": max_tokens or self.config.max_tokens,
                "topP": self.config.top_p,
            }
            
            # Add any additional config parameters
            for key, value in kwargs.items():
                if key in ["topP", "topK", "repeatPenalty", "structured"]:
                    config[key] = value
            
            # Generate response
            start_time = datetime.now()
            result = llm_model.respond(chat, config=config)
            end_time = datetime.now()
            
            # Calculate timing
            response_time = (end_time - start_time).total_seconds()
            
            return LLMResponse(
                content=result,
                model=model_id,
                provider="lmstudio",
                usage={
                    "prompt_tokens": 0,  # LM Studio doesn't provide token counts
                    "completion_tokens": 0,
                    "total_tokens": 0
                },
                response_time=response_time,
                metadata={
                    "server_host": self.server_host,
                    "config": config
                }
            )
            
        except Exception as e:
            logger.error(f"LM Studio generation failed: {e}")
            raise Exception(f"LM Studio generation failed: {str(e)}")
    
    async def generate_streaming_response(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        callback: Optional[Callable[[str], None]] = None,
        **kwargs
    ) -> AsyncGenerator[StreamingResponse, None]:
        """
        Generate a streaming response using LM Studio.
        
        Args:
            messages: Conversation messages
            model: Model identifier
            system_prompt: System prompt for the conversation
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            callback: Optional callback for each token
            **kwargs: Additional parameters
            
        Yields:
            StreamingResponse objects with incremental content
        """
        try:
            # Use default model if not specified
            model_id = model or self.config.default_model
            
            # Get model handle
            llm_model = lms.llm(model_id)
            
            # Create chat context
            chat = self._create_chat_context(messages, system_prompt)
            
            # Prepare configuration
            config = {
                "temperature": temperature or self.config.temperature,
                "maxTokens": max_tokens or self.config.max_tokens,
                "topP": self.config.top_p,
            }
            
            # Add any additional config parameters
            for key, value in kwargs.items():
                if key in ["topP", "topK", "repeatPenalty", "structured"]:
                    config[key] = value
            
            # Generate streaming response
            start_time = datetime.now()
            full_content = ""
            
            for fragment in llm_model.respond_stream(chat, config=config):
                content = fragment.content
                full_content += content
                
                # Call callback if provided
                if callback:
                    try:
                        callback(content)
                    except Exception as e:
                        logger.warning(f"Callback error: {e}")
                
                # Yield streaming response
                yield StreamingResponse(
                    content=content,
                    full_content=full_content,
                    model=model_id,
                    provider="lmstudio",
                    is_complete=False,
                    metadata={
                        "server_host": self.server_host,
                        "config": config
                    }
                )
            
            # Final response
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            yield StreamingResponse(
                content="",
                full_content=full_content,
                model=model_id,
                provider="lmstudio",
                is_complete=True,
                usage={
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                },
                response_time=response_time,
                metadata={
                    "server_host": self.server_host,
                    "config": config
                }
            )
            
        except Exception as e:
            logger.error(f"LM Studio streaming failed: {e}")
            raise Exception(f"LM Studio streaming failed: {str(e)}")
    
    async def test_connection(self) -> Dict[str, Any]:
        """
        Test connection to LM Studio server.
        
        Returns:
            Dictionary with connection test results
        """
        try:
            # Try to get a model handle
            model = lms.llm(self.config.default_model)
            
            # Try a simple response
            test_response = model.respond("Hello")
            
            return {
                "status": "success",
                "server_host": self.server_host,
                "default_model": self.config.default_model,
                "test_response": test_response[:100] + "..." if len(test_response) > 100 else test_response,
                "message": "LM Studio connection successful"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "server_host": self.server_host,
                "error": str(e),
                "message": "LM Studio connection failed"
            }
    
    async def get_model_info(self, model: str) -> Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            model: Model identifier
            
        Returns:
            Dictionary with model information
        """
        try:
            # Try to get model handle
            llm_model = lms.llm(model)
            
            return {
                "id": model,
                "name": model,
                "provider": "lmstudio",
                "type": "chat",
                "available": True,
                "server_host": self.server_host,
                "description": f"Local model hosted on {self.server_host}"
            }
            
        except Exception as e:
            return {
                "id": model,
                "name": model,
                "provider": "lmstudio",
                "available": False,
                "error": str(e),
                "server_host": self.server_host
            }
    
    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about the LM Studio provider.
        
        Returns:
            Dictionary with provider information
        """
        return {
            "name": self.name,
            "display_name": self.display_name,
            "type": "local",
            "description": "Local LLM inference using LM Studio",
            "features": [
                "Local inference",
                "No API costs",
                "Privacy-focused",
                "Streaming responses",
                "Custom model loading",
                "GPU acceleration"
            ],
            "requirements": [
                "LM Studio application installed",
                "lmstudio-python package",
                "Local models downloaded",
                "LM Studio server running"
            ],
            "server_host": self.server_host,
            "config": {
                "host": self.config.host,
                "port": self.config.port,
                "default_model": self.config.default_model,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens
            }
        }


# Factory function for easy provider creation
def create_lmstudio_provider(
    host: str = "localhost",
    port: int = 1234,
    default_model: str = "llama-3.2-1b-instruct",
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

