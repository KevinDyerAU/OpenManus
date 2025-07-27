"""Models module for OpenManus.

This module provides model management, OpenRouter integration, and client factory patterns.
Includes enhanced model management functionality migrated from the enhancements folder.
"""

try:
    from .manager import *
    from .openrouter import *
    MODEL_ENHANCEMENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Enhanced model components not available - {str(e)}")
    MODEL_ENHANCEMENTS_AVAILABLE = False

__all__ = []
if MODEL_ENHANCEMENTS_AVAILABLE:
    __all__.extend(["ModelManager", "OpenRouterClient"])

# This module will contain enhanced model management functionality
# migrated from the enhancements folder
