"""
Base Provider for OpenManus LLM Providers

This module defines the base interface that all LLM providers must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime


@dataclass
class LLMResponse:
    """Response from an LLM provider"""
    content: str
    model: str
    usage: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    created_at: Optional[datetime] = None


@dataclass
class StreamingResponse:
    """Streaming response from an LLM provider"""
    content_generator: AsyncGenerator[str, None]
    model: str
    usage: Optional[Dict[str, Any]] = None


class BaseLLMProvider(ABC):
    """
    Base class for all LLM providers.
    
    This abstract base class defines the interface that all LLM providers
    must implement to be compatible with OpenManus.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (used for identification)"""
        pass
    
    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable provider name"""
        pass
    
    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and ready to use"""
        pass
    
    @abstractmethod
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
        Generate a chat completion response.
        
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
        pass
    
    @abstractmethod
    async def stream_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> StreamingResponse:
        """
        Generate a streaming chat completion response.
        
        Args:
            messages: List of chat messages
            model: Model to use (optional, uses default if not specified)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
            
        Returns:
            StreamingResponse object with content generator
        """
        pass
    
    @abstractmethod
    async def list_models(self) -> List[str]:
        """
        List available models for this provider.
        
        Returns:
            List of model names
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the provider is healthy and responsive.
        
        Returns:
            True if healthy, False otherwise
        """
        pass
