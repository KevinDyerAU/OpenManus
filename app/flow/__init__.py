"""Flow module for OpenManus.

This module provides workflow execution and management capabilities.
Includes enhanced flow system with callbacks, interfaces, and multi-agent orchestration.
"""

try:
    from .enhanced import *  # Enhanced flow system
    FLOW_ENHANCEMENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Enhanced flow components not available - {str(e)}")
    FLOW_ENHANCEMENTS_AVAILABLE = False

try:
    from .interfaces import *  # Enhanced interfaces
except ImportError:
    pass

try:
    from .orchestrator import *  # Multi-agent orchestrator
except ImportError:
    pass

# Re-export existing flow components
try:
    from .flow import *
    from .plan import *
except ImportError:
    pass  # Original flow components may not exist yet