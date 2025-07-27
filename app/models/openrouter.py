"""
OpenRouter Client for OpenManus
Provides unified access to 400+ AI models through OpenRouter API
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, AsyncGenerator, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import aiohttp
import openai
from openai import AsyncOpenAI


class ModelProvider(Enum):
    """Model providers available through OpenRouter"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    META = "meta"
    MISTRAL = "mistral"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    TOGETHER = "together"
    PERPLEXITY = "perplexity"
    AUTO = "auto"  # Let OpenRouter choose


class ModelCapability(Enum):
    """Model capabilities"""
    TEXT_GENERATION = "text_generation"
    TOOL_CALLING = "tool_calling"
    VISION = "vision"
    CODE_GENERATION = "code_generation"
    REASONING = "reasoning"
    STRUCTURED_OUTPUT = "structured_output"
    WEB_SEARCH = "web_search"


@dataclass
class ModelInfo:
    """Information about an available model"""
    id: str
    name: str
    provider: str
    description: str
    context_length: int
    pricing: Dict[str, float] = field(default_factory=dict)
    capabilities: List[ModelCapability] = field(default_factory=list)
    supported_parameters: List[str] = field(default_factory=list)
    architecture: Dict[str, Any] = field(default_factory=dict)
    top_provider: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatMessage:
    """Chat message structure"""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


@dataclass
class ChatCompletionRequest:
    """Chat completion request parameters"""
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    response_format: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None
    reasoning: Optional[bool] = None
    include_reasoning: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatCompletionResponse:
    """Chat completion response"""
    id: str
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]
    created: int
    provider: Optional[str] = None
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OpenRouterConfig:
    """OpenRouter client configuration"""
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    site_url: Optional[str] = None
    site_name: Optional[str] = None
    default_model: str = "openai/gpt-4o"
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_fallback: bool = True
    fallback_models: List[str] = field(default_factory=list)
    cost_limit: Optional[float] = None  # USD per request
    rate_limit: Optional[int] = None  # requests per minute


class OpenRouterClient:
    """OpenRouter client for unified LLM access"""
    
    def __init__(self, config: OpenRouterConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize OpenAI client for compatibility
        self.openai_client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
        
        # Model cache
        self.models_cache: Dict[str, ModelInfo] = {}
        self.models_last_updated: Optional[datetime] = None
        self.models_cache_ttl = timedelta(hours=1)
        
        # Usage tracking
        self.usage_stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "requests_by_model": {},
            "errors": 0
        }
        
        # Rate limiting
        self.request_times: List[datetime] = []
        
        # Session for direct API calls
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def get_available_models(self, refresh: bool = False) -> List[ModelInfo]:
        """Get list of available models"""
        try:
            # Check cache
            if (not refresh and self.models_cache and self.models_last_updated and
                datetime.now() - self.models_last_updated < self.models_cache_ttl):
                return list(self.models_cache.values())
            
            # Fetch models from API
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
            
            if self.config.site_url:
                headers["HTTP-Referer"] = self.config.site_url
            if self.config.site_name:
                headers["X-Title"] = self.config.site_name
            
            async with self.session.get(
                f"{self.config.base_url}/models",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            ) as response:
                response.raise_for_status()
                data = await response.json()
            
            # Parse models
            models = []
            for model_data in data.get("data", []):
                model_info = ModelInfo(
                    id=model_data["id"],
                    name=model_data.get("name", model_data["id"]),
                    provider=model_data.get("provider", "unknown"),
                    description=model_data.get("description", ""),
                    context_length=model_data.get("context_length", 0),
                    pricing=model_data.get("pricing", {}),
                    capabilities=self._parse_capabilities(model_data),
                    supported_parameters=model_data.get("supported_parameters", []),
                    architecture=model_data.get("architecture", {}),
                    top_provider=model_data.get("top_provider", {}),
                    metadata=model_data
                )
                models.append(model_info)
                self.models_cache[model_info.id] = model_info
            
            self.models_last_updated = datetime.now()
            self.logger.info(f"Loaded {len(models)} models from OpenRouter")
            
            return models
            
        except Exception as e:
            self.logger.error(f"Failed to get available models: {e}")
            return list(self.models_cache.values())  # Return cached models if available
    
    async def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get information about a specific model"""
        if model_id in self.models_cache:
            return self.models_cache[model_id]
        
        # Refresh models cache
        await self.get_available_models(refresh=True)
        return self.models_cache.get(model_id)
    
    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Create chat completion"""
        try:
            # Rate limiting check
            await self._check_rate_limit()
            
            # Cost check
            await self._check_cost_limit(request)
            
            # Prepare request
            openai_request = await self._prepare_openai_request(request)
            
            # Execute request with fallback
            response = await self._execute_with_fallback(openai_request, request.model)
            
            # Track usage
            self._track_usage(request.model, response)
            
            return self._parse_response(response)
            
        except Exception as e:
            self.usage_stats["errors"] += 1
            self.logger.error(f"Chat completion failed: {e}")
            raise
    
    async def chat_completion_stream(self, request: ChatCompletionRequest) -> AsyncGenerator[Dict[str, Any], None]:
        """Create streaming chat completion"""
        try:
            # Rate limiting check
            await self._check_rate_limit()
            
            # Cost check
            await self._check_cost_limit(request)
            
            # Prepare streaming request
            request.stream = True
            openai_request = await self._prepare_openai_request(request)
            
            # Execute streaming request
            stream = await self.openai_client.chat.completions.create(**openai_request)
            
            async for chunk in stream:
                yield chunk.model_dump()
                
        except Exception as e:
            self.usage_stats["errors"] += 1
            self.logger.error(f"Streaming chat completion failed: {e}")
            raise
    
    async def get_best_model_for_task(self, task_type: str, 
                                    capabilities: List[ModelCapability] = None,
                                    max_cost: float = None,
                                    min_context_length: int = None) -> Optional[str]:
        """Get the best model for a specific task"""
        try:
            models = await self.get_available_models()
            
            # Filter models by capabilities
            if capabilities:
                models = [m for m in models if all(cap in m.capabilities for cap in capabilities)]
            
            # Filter by cost
            if max_cost:
                models = [m for m in models 
                         if m.pricing.get("prompt", 0) <= max_cost]
            
            # Filter by context length
            if min_context_length:
                models = [m for m in models if m.context_length >= min_context_length]
            
            if not models:
                return None
            
            # Score models based on task type
            scored_models = []
            for model in models:
                score = self._score_model_for_task(model, task_type)
                scored_models.append((score, model))
            
            # Sort by score (higher is better)
            scored_models.sort(key=lambda x: x[0], reverse=True)
            
            return scored_models[0][1].id if scored_models else None
            
        except Exception as e:
            self.logger.error(f"Failed to get best model for task: {e}")
            return self.config.default_model
    
    async def estimate_cost(self, request: ChatCompletionRequest) -> Dict[str, float]:
        """Estimate cost for a request"""
        try:
            model_info = await self.get_model_info(request.model)
            if not model_info:
                return {"estimated_cost": 0.0, "error": "Model not found"}
            
            # Estimate token count (rough approximation)
            input_tokens = sum(len(msg.content.split()) * 1.3 for msg in request.messages)
            output_tokens = request.max_tokens or 1000
            
            # Calculate cost
            prompt_cost = model_info.pricing.get("prompt", 0) * input_tokens / 1000000
            completion_cost = model_info.pricing.get("completion", 0) * output_tokens / 1000000
            
            return {
                "estimated_cost": prompt_cost + completion_cost,
                "prompt_cost": prompt_cost,
                "completion_cost": completion_cost,
                "input_tokens": int(input_tokens),
                "output_tokens": output_tokens
            }
            
        except Exception as e:
            self.logger.error(f"Failed to estimate cost: {e}")
            return {"estimated_cost": 0.0, "error": str(e)}
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            **self.usage_stats,
            "models_cached": len(self.models_cache),
            "cache_last_updated": self.models_last_updated.isoformat() if self.models_last_updated else None
        }
    
    def reset_usage_stats(self) -> None:
        """Reset usage statistics"""
        self.usage_stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "requests_by_model": {},
            "errors": 0
        }
    
    async def _prepare_openai_request(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """Prepare request for OpenAI client"""
        openai_request = {
            "model": request.model,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    **({"name": msg.name} if msg.name else {}),
                    **({"tool_calls": msg.tool_calls} if msg.tool_calls else {}),
                    **({"tool_call_id": msg.tool_call_id} if msg.tool_call_id else {})
                }
                for msg in request.messages
            ]
        }
        
        # Add optional parameters
        if request.max_tokens is not None:
            openai_request["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            openai_request["temperature"] = request.temperature
        if request.top_p is not None:
            openai_request["top_p"] = request.top_p
        if request.frequency_penalty is not None:
            openai_request["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty is not None:
            openai_request["presence_penalty"] = request.presence_penalty
        if request.stop is not None:
            openai_request["stop"] = request.stop
        if request.stream is not None:
            openai_request["stream"] = request.stream
        if request.tools is not None:
            openai_request["tools"] = request.tools
        if request.tool_choice is not None:
            openai_request["tool_choice"] = request.tool_choice
        if request.response_format is not None:
            openai_request["response_format"] = request.response_format
        if request.seed is not None:
            openai_request["seed"] = request.seed
        
        # OpenRouter-specific parameters
        extra_headers = {}
        if self.config.site_url:
            extra_headers["HTTP-Referer"] = self.config.site_url
        if self.config.site_name:
            extra_headers["X-Title"] = self.config.site_name
        
        if extra_headers:
            openai_request["extra_headers"] = extra_headers
        
        return openai_request
    
    async def _execute_with_fallback(self, openai_request: Dict[str, Any], 
                                   primary_model: str) -> Any:
        """Execute request with fallback models"""
        models_to_try = [primary_model]
        
        if self.config.enable_fallback:
            models_to_try.extend(self.config.fallback_models)
            if self.config.default_model not in models_to_try:
                models_to_try.append(self.config.default_model)
        
        last_error = None
        
        for model in models_to_try:
            try:
                openai_request["model"] = model
                response = await self.openai_client.chat.completions.create(**openai_request)
                
                if model != primary_model:
                    self.logger.warning(f"Fallback to model {model} from {primary_model}")
                
                return response
                
            except Exception as e:
                last_error = e
                self.logger.warning(f"Model {model} failed: {e}")
                
                # Wait before trying next model
                if model != models_to_try[-1]:
                    await asyncio.sleep(self.config.retry_delay)
        
        raise last_error or Exception("All models failed")
    
    async def _check_rate_limit(self) -> None:
        """Check rate limiting"""
        if not self.config.rate_limit:
            return
        
        now = datetime.now()
        
        # Remove old requests (older than 1 minute)
        self.request_times = [t for t in self.request_times 
                             if now - t < timedelta(minutes=1)]
        
        # Check if we're within rate limit
        if len(self.request_times) >= self.config.rate_limit:
            sleep_time = 60 - (now - self.request_times[0]).total_seconds()
            if sleep_time > 0:
                self.logger.warning(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)
        
        self.request_times.append(now)
    
    async def _check_cost_limit(self, request: ChatCompletionRequest) -> None:
        """Check cost limits"""
        if not self.config.cost_limit:
            return
        
        cost_estimate = await self.estimate_cost(request)
        estimated_cost = cost_estimate.get("estimated_cost", 0)
        
        if estimated_cost > self.config.cost_limit:
            raise ValueError(f"Estimated cost ${estimated_cost:.4f} exceeds limit ${self.config.cost_limit:.4f}")
    
    def _parse_capabilities(self, model_data: Dict[str, Any]) -> List[ModelCapability]:
        """Parse model capabilities from model data"""
        capabilities = []
        
        # Check supported parameters for capabilities
        supported_params = model_data.get("supported_parameters", [])
        
        if "tools" in supported_params:
            capabilities.append(ModelCapability.TOOL_CALLING)
        if "structured_outputs" in supported_params:
            capabilities.append(ModelCapability.STRUCTURED_OUTPUT)
        if "reasoning" in supported_params:
            capabilities.append(ModelCapability.REASONING)
        
        # Check architecture for vision capability
        architecture = model_data.get("architecture", {})
        if architecture.get("modality") == "multimodal":
            capabilities.append(ModelCapability.VISION)
        
        # Default capability
        capabilities.append(ModelCapability.TEXT_GENERATION)
        
        return capabilities
    
    def _score_model_for_task(self, model: ModelInfo, task_type: str) -> float:
        """Score a model for a specific task type"""
        score = 0.0
        
        # Base score from pricing (lower cost = higher score)
        prompt_cost = model.pricing.get("prompt", 1.0)
        if prompt_cost > 0:
            score += 1.0 / prompt_cost
        
        # Task-specific scoring
        if task_type == "coding":
            if ModelCapability.CODE_GENERATION in model.capabilities:
                score += 10.0
            if "code" in model.name.lower():
                score += 5.0
        
        elif task_type == "reasoning":
            if ModelCapability.REASONING in model.capabilities:
                score += 10.0
            if "reasoning" in model.name.lower():
                score += 5.0
        
        elif task_type == "vision":
            if ModelCapability.VISION in model.capabilities:
                score += 10.0
        
        elif task_type == "tool_calling":
            if ModelCapability.TOOL_CALLING in model.capabilities:
                score += 10.0
        
        # Provider preferences
        if model.provider == "openai":
            score += 2.0
        elif model.provider == "anthropic":
            score += 1.5
        
        return score
    
    def _track_usage(self, model: str, response: Any) -> None:
        """Track usage statistics"""
        self.usage_stats["total_requests"] += 1
        
        if model not in self.usage_stats["requests_by_model"]:
            self.usage_stats["requests_by_model"][model] = 0
        self.usage_stats["requests_by_model"][model] += 1
        
        # Track tokens if available
        if hasattr(response, 'usage') and response.usage:
            total_tokens = response.usage.total_tokens
            self.usage_stats["total_tokens"] += total_tokens
    
    def _parse_response(self, response: Any) -> ChatCompletionResponse:
        """Parse OpenAI response to our format"""
        return ChatCompletionResponse(
            id=response.id,
            model=response.model,
            choices=[choice.model_dump() for choice in response.choices],
            usage=response.usage.model_dump() if response.usage else {},
            created=response.created,
            provider=getattr(response, 'provider', None),
            reasoning=getattr(response, 'reasoning', None)
        )


# Factory functions

def create_openrouter_client(api_key: str, 
                           site_url: str = None,
                           site_name: str = None,
                           default_model: str = "openai/gpt-4o") -> OpenRouterClient:
    """Create OpenRouter client with basic configuration"""
    config = OpenRouterConfig(
        api_key=api_key,
        site_url=site_url,
        site_name=site_name,
        default_model=default_model
    )
    return OpenRouterClient(config)

def create_production_openrouter_client(api_key: str,
                                      site_url: str = None,
                                      site_name: str = None) -> OpenRouterClient:
    """Create production OpenRouter client with optimizations"""
    config = OpenRouterConfig(
        api_key=api_key,
        site_url=site_url,
        site_name=site_name,
        default_model="openai/gpt-4o",
        enable_fallback=True,
        fallback_models=[
            "anthropic/claude-3.5-sonnet",
            "google/gemini-2.5-pro",
            "openai/gpt-4o-mini"
        ],
        cost_limit=1.0,  # $1 per request limit
        rate_limit=60,   # 60 requests per minute
        max_retries=3,
        retry_delay=2.0
    )
    return OpenRouterClient(config)

def create_budget_openrouter_client(api_key: str,
                                  site_url: str = None,
                                  site_name: str = None) -> OpenRouterClient:
    """Create budget-optimized OpenRouter client"""
    config = OpenRouterConfig(
        api_key=api_key,
        site_url=site_url,
        site_name=site_name,
        default_model="openai/gpt-4o-mini",
        enable_fallback=True,
        fallback_models=[
            "anthropic/claude-3-haiku",
            "google/gemini-1.5-flash",
            "meta/llama-3.1-8b-instruct"
        ],
        cost_limit=0.1,  # $0.10 per request limit
        rate_limit=30    # 30 requests per minute
    )
    return OpenRouterClient(config)

