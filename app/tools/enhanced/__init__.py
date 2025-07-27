"""Enhanced tools module for OpenManus.

This module contains enhanced tool implementations migrated from the enhancements folder.
"""

try:
    from .registry import *  # Enhanced tool registry
    ENHANCED_TOOLS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Enhanced tool registry not available - {str(e)}")
    ENHANCED_TOOLS_AVAILABLE = False

try:
    from .browser import *  # MCP browser tools
except ImportError:
    pass

try:
    from .system import *  # System tools
except ImportError:
    pass

try:
    from .headless_browser import *  # Headless browser functionality
except ImportError:
    pass

__all__ = []
if ENHANCED_TOOLS_AVAILABLE:
    # Export will be populated by the imported modules
    pass
