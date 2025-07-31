"""
Configuration Management for OpenManus

This module handles configuration loading and management for the OpenManus platform,
including support for multiple LLM providers and deployment environments.
"""

import os
import toml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

# Configuration file paths
CONFIG_DIR = Path(__file__).parent.parent / "config"
DEFAULT_CONFIG_FILE = CONFIG_DIR / "config.toml"
LMSTUDIO_CONFIG_FILE = CONFIG_DIR / "lmstudio_config.toml"

@dataclass
class DatabaseConfig:
    """Database configuration"""
    url: str = "sqlite:///openmanus.db"
    echo: bool = False
    pool_size: int = 10
    max_overflow: int = 20

@dataclass
class RedisConfig:
    """Redis configuration"""
    url: str = "redis://localhost:6379/0"
    max_connections: int = 10
    socket_timeout: int = 5

@dataclass
class OpenRouterConfig:
    """OpenRouter API configuration"""
    api_key: str = ""
    base_url: str = "https://openrouter.ai/api/v1"
    default_model: str = "openai/gpt-3.5-turbo"
    timeout: int = 60
    max_retries: int = 3
    rate_limit: int = 60

@dataclass
class LMStudioConfig:
    """LM Studio configuration"""
    enabled: bool = True
    host: str = "localhost"
    port: int = 1234
    default_model: str = "llama-3.2-1b-instruct"
    timeout: int = 300
    max_retries: int = 3
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    priority: int = 5
    rate_limit: int = 60
    concurrent_requests: int = 4
    fallback_models: list = field(default_factory=lambda: [
        "llama-3.2-3b-instruct",
        "qwen2.5-7b-instruct"
    ])

@dataclass
class MCPConfig:
    """MCP (Model Context Protocol) configuration"""
    enabled: bool = True
    port: int = 8001
    max_connections: int = 100
    timeout: int = 30
    enable_logging: bool = True

@dataclass
class FlowConfig:
    """Flow system configuration"""
    enabled: bool = True
    max_concurrent_flows: int = 10
    default_timeout: int = 300
    enable_callbacks: bool = True
    callback_timeout: int = 30

@dataclass
class BrowserConfig:
    """Browser automation configuration"""
    enabled: bool = True
    headless: bool = True
    timeout: int = 30
    max_pages: int = 5
    user_agent: str = "OpenManus/1.0"

@dataclass
class APIConfig:
    """API server configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_enabled: bool = True
    rate_limit: int = 100
    auth_required: bool = False
    jwt_secret: str = "your-secret-key-change-in-production"

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5

@dataclass
class Config:
    """Main configuration class"""
    environment: str = "development"
    debug: bool = True
    
    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    openrouter: OpenRouterConfig = field(default_factory=OpenRouterConfig)
    lmstudio: LMStudioConfig = field(default_factory=LMStudioConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    browser: BrowserConfig = field(default_factory=BrowserConfig)
    api: APIConfig = field(default_factory=APIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

# Global configuration instance
_config: Optional[Config] = None

def load_config(config_file: Optional[str] = None) -> Config:
    """
    Load configuration from TOML file and environment variables.
    
    Args:
        config_file: Path to configuration file (optional)
        
    Returns:
        Loaded configuration object
    """
    global _config
    
    # Start with default configuration
    config = Config()
    
    # Load from TOML file if it exists
    if config_file:
        config_path = Path(config_file)
    else:
        config_path = DEFAULT_CONFIG_FILE
    
    if config_path.exists():
        try:
            toml_config = toml.load(config_path)
            _update_config_from_dict(config, toml_config)
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
    
    # Load LM Studio specific configuration
    if LMSTUDIO_CONFIG_FILE.exists():
        try:
            lmstudio_config = toml.load(LMSTUDIO_CONFIG_FILE)
            if "lmstudio" in lmstudio_config:
                _update_config_from_dict(config, {"lmstudio": lmstudio_config["lmstudio"]})
        except Exception as e:
            print(f"Warning: Failed to load LM Studio config: {e}")
    
    # Override with environment variables
    _load_env_overrides(config)
    
    _config = config
    return config

def _update_config_from_dict(config: Config, config_dict: Dict[str, Any]):
    """Update configuration object from dictionary"""
    for section_name, section_data in config_dict.items():
        if hasattr(config, section_name) and isinstance(section_data, dict):
            section_obj = getattr(config, section_name)
            for key, value in section_data.items():
                if hasattr(section_obj, key):
                    setattr(section_obj, key, value)

def _load_env_overrides(config: Config):
    """Load configuration overrides from environment variables"""
    
    # General settings
    config.environment = os.getenv("OPENMANUS_ENV", config.environment)
    config.debug = os.getenv("OPENMANUS_DEBUG", str(config.debug)).lower() == "true"
    
    # Database
    if os.getenv("DATABASE_URL"):
        config.database.url = os.getenv("DATABASE_URL")
    
    # Redis
    if os.getenv("REDIS_URL"):
        config.redis.url = os.getenv("REDIS_URL")
    
    # OpenRouter
    if os.getenv("OPENROUTER_API_KEY"):
        config.openrouter.api_key = os.getenv("OPENROUTER_API_KEY")
    if os.getenv("OPENROUTER_DEFAULT_MODEL"):
        config.openrouter.default_model = os.getenv("OPENROUTER_DEFAULT_MODEL")
    
    # LM Studio
    if os.getenv("LMSTUDIO_HOST"):
        config.lmstudio.host = os.getenv("LMSTUDIO_HOST")
    if os.getenv("LMSTUDIO_PORT"):
        config.lmstudio.port = int(os.getenv("LMSTUDIO_PORT"))
    if os.getenv("LMSTUDIO_DEFAULT_MODEL"):
        config.lmstudio.default_model = os.getenv("LMSTUDIO_DEFAULT_MODEL")
    if os.getenv("LMSTUDIO_ENABLED"):
        config.lmstudio.enabled = os.getenv("LMSTUDIO_ENABLED").lower() == "true"
    
    # API
    if os.getenv("API_HOST"):
        config.api.host = os.getenv("API_HOST")
    if os.getenv("API_PORT"):
        config.api.port = int(os.getenv("API_PORT"))
    if os.getenv("JWT_SECRET"):
        config.api.jwt_secret = os.getenv("JWT_SECRET")
    
    # Logging
    if os.getenv("LOG_LEVEL"):
        config.logging.level = os.getenv("LOG_LEVEL")
    if os.getenv("LOG_FILE"):
        config.logging.file = os.getenv("LOG_FILE")

def get_config() -> Config:
    """
    Get the current configuration.
    
    Returns:
        Current configuration object
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config

def reload_config(config_file: Optional[str] = None) -> Config:
    """
    Reload configuration from file.
    
    Args:
        config_file: Path to configuration file (optional)
        
    Returns:
        Reloaded configuration object
    """
    global _config
    _config = None
    return load_config(config_file)

def get_lmstudio_config() -> LMStudioConfig:
    """
    Get LM Studio specific configuration.
    
    Returns:
        LM Studio configuration object
    """
    return get_config().lmstudio

def is_lmstudio_enabled() -> bool:
    """
    Check if LM Studio provider is enabled.
    
    Returns:
        True if LM Studio is enabled, False otherwise
    """
    return get_config().lmstudio.enabled

def get_provider_configs() -> Dict[str, Any]:
    """
    Get all LLM provider configurations.
    
    Returns:
        Dictionary of provider configurations
    """
    config = get_config()
    return {
        "openrouter": config.openrouter,
        "lmstudio": config.lmstudio
    }

# Initialize configuration on module import
try:
    load_config()
except Exception as e:
    print(f"Warning: Failed to initialize configuration: {e}")
    _config = Config()  # Use defaults

