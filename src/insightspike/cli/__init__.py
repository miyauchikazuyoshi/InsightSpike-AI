"""CLI package initialization"""

try:
    from .spike import app as spike_app
except ImportError:
    spike_app = None

try:
    from .legacy import app as legacy_app
except ImportError:
    legacy_app = None
    
# from .commands.deps import app as deps_app  # Temporarily disabled
deps_app = None

__all__ = ["spike_app", "legacy_app", "deps_app"]