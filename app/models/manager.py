"""
Model Management System for OpenManus
Intelligent model selection, caching, and optimization
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib

from .openrouter_client import (
    OpenRouterClient, ModelInfo, ModelCapability, ChatCompletionRequest,
    ChatCompletionResponse, ChatMessage
)


class TaskType(Enum):
    """Types of tasks for model selection"""
    GENERAL_CHAT = "general_chat"
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    REASONING = "reasoning"
    CREATIVE_WRITING = "creative_writing"
    TECHNICAL_WRITING = "technical_writing"
    DATA_ANALYSIS = "data_analysis"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    VISION_ANALYSIS = "vision_analysis"
    TOOL_CALLING = "tool_calling"
    STRUCTURED_OUTPUT = "structured_output"
    WEB_SEARCH = "web_search"
    LONG_CONTEXT = "long_context"


class ModelTier(Enum):
    """Model performance tiers"""
    PREMIUM = "premium"      # Highest quality, highest cost
    STANDARD = "standard"    # Good balance of quality and cost
    BUDGET = "budget"        # Lower cost, acceptable quality
    FREE = "free"           # Free models with limitations


@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_id: str
    task_type: TaskType
    success_rate: float = 0.0
    average_latency: float = 0.0
    average_cost: float = 0.0
    quality_score: float = 0.0
    total_requests: int = 0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class ModelSelectionCriteria:
    """Criteria for model selection"""
    task_type: TaskType
    max_cost: Optional[float] = None
    max_latency: Optional[float] = None
    min_quality: Optional[float] = None
    required_capabilities: List[ModelCapability] = field(default_factory=list)
    preferred_providers: List[str] = field(default_factory=list)
    min_context_length: Optional[int] = None
    tier: Optional[ModelTier] = None
    allow_fallback: bool = True


@dataclass
class ModelRecommendation:
    """Model recommendation with reasoning"""
    model_id: str
    model_info: ModelInfo
    confidence: float
    reasoning: str
    estimated_cost: float
    estimated_latency: float
    fallback_models: List[str] = field(default_factory=list)


class ModelManager:
    """Intelligent model management system"""
    
    def __init__(self, openrouter_client: OpenRouterClient):
        self.client = openrouter_client
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Performance tracking
        self.performance_cache: Dict[str, ModelPerformance] = {}
        
        # Model categorization
        self.model_tiers: Dict[str, ModelTier] = {}
        self.task_model_mapping: Dict[TaskType, List[str]] = {}
        
        # Caching
        self.recommendation_cache: Dict[str, Tuple[ModelRecommendation, datetime]] = {}
        self.cache_ttl = timedelta(minutes=30)
        
        # Configuration
        self.tier_cost_thresholds = {
            ModelTier.FREE: 0.0,
            ModelTier.BUDGET: 0.001,    # $0.001 per 1K tokens
            ModelTier.STANDARD: 0.01,   # $0.01 per 1K tokens
            ModelTier.PREMIUM: 0.1      # $0.1 per 1K tokens
        }
        
        # Initialize model categorization
        asyncio.create_task(self._initialize_model_categorization())
    
    async def get_model_recommendation(self, criteria: ModelSelectionCriteria) -> ModelRecommendation:
        """Get intelligent model recommendation based on criteria"""
        try:
            # Check cache
            cache_key = self._get_cache_key(criteria)
            if cache_key in self.recommendation_cache:
                recommendation, cached_time = self.recommendation_cache[cache_key]
                if datetime.now() - cached_time < self.cache_ttl:
                    return recommendation
            
            # Get available models
            models = await self.client.get_available_models()
            
            # Filter models by criteria
            candidate_models = self._filter_models(models, criteria)
            
            if not candidate_models:
                # Fallback to default model
                default_model = await self.client.get_model_info(self.client.config.default_model)
                if default_model:
                    return ModelRecommendation(
                        model_id=default_model.id,
                        model_info=default_model,
                        confidence=0.5,
                        reasoning="No models matched criteria, using default",
                        estimated_cost=0.01,
                        estimated_latency=2.0
                    )
                else:
                    raise ValueError("No suitable models found and no default available")
            
            # Score and rank models
            scored_models = await self._score_models(candidate_models, criteria)
            
            # Select best model
            best_model, score = scored_models[0]
            
            # Get fallback models
            fallback_models = [model.id for model, _ in scored_models[1:4]]  # Top 3 alternatives
            
            # Create recommendation
            recommendation = ModelRecommendation(
                model_id=best_model.id,
                model_info=best_model,
                confidence=min(score / 100.0, 1.0),
                reasoning=self._generate_reasoning(best_model, criteria, score),
                estimated_cost=await self._estimate_model_cost(best_model, criteria),
                estimated_latency=await self._estimate_model_latency(best_model, criteria),
                fallback_models=fallback_models
            )
            
            # Cache recommendation
            self.recommendation_cache[cache_key] = (recommendation, datetime.now())
            
            return recommendation
            
        except Exception as e:
            self.logger.error(f"Failed to get model recommendation: {e}")
            raise
    
    async def execute_with_best_model(self, 
                                    messages: List[ChatMessage],
                                    criteria: ModelSelectionCriteria,
                                    **kwargs) -> ChatCompletionResponse:
        """Execute request with automatically selected best model"""
        try:
            # Get recommendation
            recommendation = await self.get_model_recommendation(criteria)
            
            # Create request
            request = ChatCompletionRequest(
                model=recommendation.model_id,
                messages=messages,
                **kwargs
            )
            
            # Execute with fallback
            try:
                start_time = datetime.now()
                response = await self.client.chat_completion(request)
                end_time = datetime.now()
                
                # Track performance
                await self._track_performance(
                    recommendation.model_id,
                    criteria.task_type,
                    True,
                    (end_time - start_time).total_seconds(),
                    response.usage.get("total_tokens", 0)
                )
                
                return response
                
            except Exception as e:
                # Try fallback models
                for fallback_model in recommendation.fallback_models:
                    try:
                        request.model = fallback_model
                        start_time = datetime.now()
                        response = await self.client.chat_completion(request)
                        end_time = datetime.now()
                        
                        # Track performance
                        await self._track_performance(
                            fallback_model,
                            criteria.task_type,
                            True,
                            (end_time - start_time).total_seconds(),
                            response.usage.get("total_tokens", 0)
                        )
                        
                        self.logger.warning(f"Used fallback model {fallback_model} after {recommendation.model_id} failed")
                        return response
                        
                    except Exception as fallback_error:
                        self.logger.warning(f"Fallback model {fallback_model} also failed: {fallback_error}")
                        continue
                
                # Track failure
                await self._track_performance(recommendation.model_id, criteria.task_type, False, 0, 0)
                raise e
                
        except Exception as e:
            self.logger.error(f"Failed to execute with best model: {e}")
            raise
    
    async def get_models_by_task(self, task_type: TaskType, tier: ModelTier = None) -> List[ModelInfo]:
        """Get models suitable for a specific task type"""
        try:
            models = await self.client.get_available_models()
            
            # Filter by task suitability
            suitable_models = []
            for model in models:
                if self._is_model_suitable_for_task(model, task_type):
                    if tier is None or self._get_model_tier(model) == tier:
                        suitable_models.append(model)
            
            # Sort by performance if available
            suitable_models.sort(key=lambda m: self._get_model_performance_score(m.id, task_type), reverse=True)
            
            return suitable_models
            
        except Exception as e:
            self.logger.error(f"Failed to get models by task: {e}")
            return []
    
    async def get_cost_analysis(self, models: List[str], 
                              sample_request: ChatCompletionRequest) -> Dict[str, Dict[str, float]]:
        """Get cost analysis for multiple models"""
        try:
            analysis = {}
            
            for model_id in models:
                model_info = await self.client.get_model_info(model_id)
                if model_info:
                    sample_request.model = model_id
                    cost_estimate = await self.client.estimate_cost(sample_request)
                    
                    analysis[model_id] = {
                        "estimated_cost": cost_estimate.get("estimated_cost", 0),
                        "prompt_cost": cost_estimate.get("prompt_cost", 0),
                        "completion_cost": cost_estimate.get("completion_cost", 0),
                        "cost_per_1k_tokens": model_info.pricing.get("prompt", 0) * 1000,
                        "tier": self._get_model_tier(model_info).value
                    }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to get cost analysis: {e}")
            return {}
    
    async def get_performance_stats(self, model_id: str = None, 
                                  task_type: TaskType = None) -> Dict[str, Any]:
        """Get performance statistics"""
        try:
            if model_id and task_type:
                # Specific model and task
                key = f"{model_id}:{task_type.value}"
                if key in self.performance_cache:
                    perf = self.performance_cache[key]
                    return {
                        "model_id": perf.model_id,
                        "task_type": perf.task_type.value,
                        "success_rate": perf.success_rate,
                        "average_latency": perf.average_latency,
                        "average_cost": perf.average_cost,
                        "quality_score": perf.quality_score,
                        "total_requests": perf.total_requests,
                        "last_updated": perf.last_updated.isoformat()
                    }
                else:
                    return {"error": "No performance data available"}
            
            elif model_id:
                # All tasks for a model
                model_stats = {}
                for key, perf in self.performance_cache.items():
                    if perf.model_id == model_id:
                        model_stats[perf.task_type.value] = {
                            "success_rate": perf.success_rate,
                            "average_latency": perf.average_latency,
                            "average_cost": perf.average_cost,
                            "quality_score": perf.quality_score,
                            "total_requests": perf.total_requests
                        }
                return model_stats
            
            elif task_type:
                # All models for a task
                task_stats = {}
                for key, perf in self.performance_cache.items():
                    if perf.task_type == task_type:
                        task_stats[perf.model_id] = {
                            "success_rate": perf.success_rate,
                            "average_latency": perf.average_latency,
                            "average_cost": perf.average_cost,
                            "quality_score": perf.quality_score,
                            "total_requests": perf.total_requests
                        }
                return task_stats
            
            else:
                # Overall statistics
                return {
                    "total_models_tracked": len(set(perf.model_id for perf in self.performance_cache.values())),
                    "total_task_types": len(set(perf.task_type for perf in self.performance_cache.values())),
                    "total_requests": sum(perf.total_requests for perf in self.performance_cache.values()),
                    "average_success_rate": sum(perf.success_rate for perf in self.performance_cache.values()) / len(self.performance_cache) if self.performance_cache else 0
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get performance stats: {e}")
            return {"error": str(e)}
    
    def clear_performance_cache(self) -> None:
        """Clear performance cache"""
        self.performance_cache.clear()
        self.logger.info("Performance cache cleared")
    
    def clear_recommendation_cache(self) -> None:
        """Clear recommendation cache"""
        self.recommendation_cache.clear()
        self.logger.info("Recommendation cache cleared")
    
    async def _initialize_model_categorization(self) -> None:
        """Initialize model categorization"""
        try:
            models = await self.client.get_available_models()
            
            for model in models:
                # Categorize by tier
                self.model_tiers[model.id] = self._get_model_tier(model)
                
                # Map to tasks
                suitable_tasks = self._get_suitable_tasks(model)
                for task in suitable_tasks:
                    if task not in self.task_model_mapping:
                        self.task_model_mapping[task] = []
                    self.task_model_mapping[task].append(model.id)
            
            self.logger.info(f"Categorized {len(models)} models into {len(self.model_tiers)} tiers")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model categorization: {e}")
    
    def _filter_models(self, models: List[ModelInfo], 
                      criteria: ModelSelectionCriteria) -> List[ModelInfo]:
        """Filter models based on criteria"""
        filtered = []
        
        for model in models:
            # Check capabilities
            if criteria.required_capabilities:
                if not all(cap in model.capabilities for cap in criteria.required_capabilities):
                    continue
            
            # Check providers
            if criteria.preferred_providers:
                if model.provider not in criteria.preferred_providers:
                    continue
            
            # Check context length
            if criteria.min_context_length:
                if model.context_length < criteria.min_context_length:
                    continue
            
            # Check tier
            if criteria.tier:
                if self._get_model_tier(model) != criteria.tier:
                    continue
            
            # Check cost
            if criteria.max_cost:
                prompt_cost = model.pricing.get("prompt", 0)
                if prompt_cost > criteria.max_cost:
                    continue
            
            # Check task suitability
            if not self._is_model_suitable_for_task(model, criteria.task_type):
                continue
            
            filtered.append(model)
        
        return filtered
    
    async def _score_models(self, models: List[ModelInfo], 
                          criteria: ModelSelectionCriteria) -> List[Tuple[ModelInfo, float]]:
        """Score and rank models"""
        scored_models = []
        
        for model in models:
            score = 0.0
            
            # Performance score
            perf_score = self._get_model_performance_score(model.id, criteria.task_type)
            score += perf_score * 30  # 30% weight
            
            # Cost score (lower cost = higher score)
            cost_score = self._get_cost_score(model, criteria)
            score += cost_score * 25  # 25% weight
            
            # Capability score
            capability_score = self._get_capability_score(model, criteria)
            score += capability_score * 20  # 20% weight
            
            # Provider score
            provider_score = self._get_provider_score(model, criteria)
            score += provider_score * 15  # 15% weight
            
            # Context length score
            context_score = self._get_context_score(model, criteria)
            score += context_score * 10  # 10% weight
            
            scored_models.append((model, score))
        
        # Sort by score (highest first)
        scored_models.sort(key=lambda x: x[1], reverse=True)
        
        return scored_models
    
    def _get_model_tier(self, model: ModelInfo) -> ModelTier:
        """Determine model tier based on pricing"""
        prompt_cost = model.pricing.get("prompt", 0)
        
        if prompt_cost == 0:
            return ModelTier.FREE
        elif prompt_cost <= self.tier_cost_thresholds[ModelTier.BUDGET]:
            return ModelTier.BUDGET
        elif prompt_cost <= self.tier_cost_thresholds[ModelTier.STANDARD]:
            return ModelTier.STANDARD
        else:
            return ModelTier.PREMIUM
    
    def _is_model_suitable_for_task(self, model: ModelInfo, task_type: TaskType) -> bool:
        """Check if model is suitable for task type"""
        # Task-specific checks
        if task_type == TaskType.CODE_GENERATION:
            return (ModelCapability.CODE_GENERATION in model.capabilities or
                   "code" in model.name.lower() or
                   "codex" in model.name.lower())
        
        elif task_type == TaskType.VISION_ANALYSIS:
            return ModelCapability.VISION in model.capabilities
        
        elif task_type == TaskType.TOOL_CALLING:
            return ModelCapability.TOOL_CALLING in model.capabilities
        
        elif task_type == TaskType.STRUCTURED_OUTPUT:
            return ModelCapability.STRUCTURED_OUTPUT in model.capabilities
        
        elif task_type == TaskType.REASONING:
            return (ModelCapability.REASONING in model.capabilities or
                   "reasoning" in model.name.lower())
        
        elif task_type == TaskType.LONG_CONTEXT:
            return model.context_length >= 32000
        
        # Most models can handle general tasks
        return True
    
    def _get_suitable_tasks(self, model: ModelInfo) -> List[TaskType]:
        """Get tasks suitable for a model"""
        tasks = [TaskType.GENERAL_CHAT, TaskType.SUMMARIZATION]  # Basic tasks
        
        if ModelCapability.CODE_GENERATION in model.capabilities or "code" in model.name.lower():
            tasks.extend([TaskType.CODE_GENERATION, TaskType.CODE_REVIEW])
        
        if ModelCapability.VISION in model.capabilities:
            tasks.append(TaskType.VISION_ANALYSIS)
        
        if ModelCapability.TOOL_CALLING in model.capabilities:
            tasks.append(TaskType.TOOL_CALLING)
        
        if ModelCapability.STRUCTURED_OUTPUT in model.capabilities:
            tasks.append(TaskType.STRUCTURED_OUTPUT)
        
        if ModelCapability.REASONING in model.capabilities:
            tasks.append(TaskType.REASONING)
        
        if model.context_length >= 32000:
            tasks.append(TaskType.LONG_CONTEXT)
        
        # Add writing tasks for most models
        tasks.extend([TaskType.CREATIVE_WRITING, TaskType.TECHNICAL_WRITING, TaskType.TRANSLATION])
        
        return tasks
    
    def _get_model_performance_score(self, model_id: str, task_type: TaskType) -> float:
        """Get performance score for model and task"""
        key = f"{model_id}:{task_type.value}"
        if key in self.performance_cache:
            perf = self.performance_cache[key]
            # Combine success rate and quality score
            return (perf.success_rate * 0.7 + perf.quality_score * 0.3) * 100
        return 50.0  # Default score for unknown models
    
    def _get_cost_score(self, model: ModelInfo, criteria: ModelSelectionCriteria) -> float:
        """Get cost score (lower cost = higher score)"""
        prompt_cost = model.pricing.get("prompt", 0)
        
        if prompt_cost == 0:
            return 100.0  # Free models get max score
        
        # Normalize cost score (inverse relationship)
        max_cost = criteria.max_cost or 0.1  # Default max cost
        if prompt_cost >= max_cost:
            return 0.0
        
        return (1 - prompt_cost / max_cost) * 100
    
    def _get_capability_score(self, model: ModelInfo, criteria: ModelSelectionCriteria) -> float:
        """Get capability score"""
        if not criteria.required_capabilities:
            return 100.0
        
        # All required capabilities should be present (filtered earlier)
        # Score based on additional capabilities
        additional_caps = len(model.capabilities) - len(criteria.required_capabilities)
        return min(100.0, 50.0 + additional_caps * 10)
    
    def _get_provider_score(self, model: ModelInfo, criteria: ModelSelectionCriteria) -> float:
        """Get provider score"""
        if not criteria.preferred_providers:
            # Default provider preferences
            provider_scores = {
                "openai": 100.0,
                "anthropic": 90.0,
                "google": 80.0,
                "meta": 70.0,
                "mistral": 60.0
            }
            return provider_scores.get(model.provider, 50.0)
        
        if model.provider in criteria.preferred_providers:
            return 100.0
        return 0.0
    
    def _get_context_score(self, model: ModelInfo, criteria: ModelSelectionCriteria) -> float:
        """Get context length score"""
        if not criteria.min_context_length:
            return 100.0
        
        if model.context_length >= criteria.min_context_length:
            # Bonus for longer context
            bonus = min(50.0, (model.context_length - criteria.min_context_length) / 1000)
            return 50.0 + bonus
        
        return 0.0  # Should be filtered out earlier
    
    async def _estimate_model_cost(self, model: ModelInfo, criteria: ModelSelectionCriteria) -> float:
        """Estimate cost for model"""
        # Simple estimation based on average usage
        prompt_cost = model.pricing.get("prompt", 0)
        completion_cost = model.pricing.get("completion", 0)
        
        # Estimate tokens (rough)
        avg_input_tokens = 1000
        avg_output_tokens = 500
        
        return (prompt_cost * avg_input_tokens + completion_cost * avg_output_tokens) / 1000000
    
    async def _estimate_model_latency(self, model: ModelInfo, criteria: ModelSelectionCriteria) -> float:
        """Estimate latency for model"""
        key = f"{model.id}:{criteria.task_type.value}"
        if key in self.performance_cache:
            return self.performance_cache[key].average_latency
        
        # Default estimates based on model tier
        tier = self._get_model_tier(model)
        latency_estimates = {
            ModelTier.FREE: 5.0,
            ModelTier.BUDGET: 3.0,
            ModelTier.STANDARD: 2.0,
            ModelTier.PREMIUM: 1.5
        }
        return latency_estimates.get(tier, 3.0)
    
    async def _track_performance(self, model_id: str, task_type: TaskType, 
                               success: bool, latency: float, tokens: int) -> None:
        """Track model performance"""
        key = f"{model_id}:{task_type.value}"
        
        if key not in self.performance_cache:
            self.performance_cache[key] = ModelPerformance(
                model_id=model_id,
                task_type=task_type
            )
        
        perf = self.performance_cache[key]
        
        # Update metrics
        perf.total_requests += 1
        
        if success:
            # Update success rate
            perf.success_rate = ((perf.success_rate * (perf.total_requests - 1)) + 1.0) / perf.total_requests
            
            # Update average latency
            if perf.average_latency == 0:
                perf.average_latency = latency
            else:
                perf.average_latency = (perf.average_latency * 0.9) + (latency * 0.1)
        else:
            # Update success rate
            perf.success_rate = (perf.success_rate * (perf.total_requests - 1)) / perf.total_requests
        
        perf.last_updated = datetime.now()
    
    def _generate_reasoning(self, model: ModelInfo, criteria: ModelSelectionCriteria, score: float) -> str:
        """Generate reasoning for model selection"""
        reasons = []
        
        # Task suitability
        if self._is_model_suitable_for_task(model, criteria.task_type):
            reasons.append(f"Well-suited for {criteria.task_type.value}")
        
        # Cost efficiency
        tier = self._get_model_tier(model)
        reasons.append(f"{tier.value} tier model")
        
        # Capabilities
        if criteria.required_capabilities:
            matching_caps = [cap for cap in criteria.required_capabilities if cap in model.capabilities]
            if matching_caps:
                reasons.append(f"Supports required capabilities: {', '.join(cap.value for cap in matching_caps)}")
        
        # Performance
        if score > 80:
            reasons.append("High performance score")
        elif score > 60:
            reasons.append("Good performance score")
        
        # Provider
        if model.provider in ["openai", "anthropic", "google"]:
            reasons.append(f"Reliable {model.provider} provider")
        
        return "; ".join(reasons)
    
    def _get_cache_key(self, criteria: ModelSelectionCriteria) -> str:
        """Generate cache key for criteria"""
        key_data = {
            "task_type": criteria.task_type.value,
            "max_cost": criteria.max_cost,
            "max_latency": criteria.max_latency,
            "min_quality": criteria.min_quality,
            "required_capabilities": [cap.value for cap in criteria.required_capabilities],
            "preferred_providers": criteria.preferred_providers,
            "min_context_length": criteria.min_context_length,
            "tier": criteria.tier.value if criteria.tier else None
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()


# Factory functions

def create_model_manager(openrouter_client: OpenRouterClient) -> ModelManager:
    """Create model manager with OpenRouter client"""
    return ModelManager(openrouter_client)

def create_task_criteria(task_type: TaskType, **kwargs) -> ModelSelectionCriteria:
    """Create model selection criteria for a task"""
    return ModelSelectionCriteria(task_type=task_type, **kwargs)

def create_budget_criteria(task_type: TaskType, max_cost: float = 0.001) -> ModelSelectionCriteria:
    """Create budget-optimized criteria"""
    return ModelSelectionCriteria(
        task_type=task_type,
        max_cost=max_cost,
        tier=ModelTier.BUDGET,
        allow_fallback=True
    )

def create_premium_criteria(task_type: TaskType) -> ModelSelectionCriteria:
    """Create premium quality criteria"""
    return ModelSelectionCriteria(
        task_type=task_type,
        tier=ModelTier.PREMIUM,
        min_quality=0.8,
        preferred_providers=["openai", "anthropic", "google"]
    )

