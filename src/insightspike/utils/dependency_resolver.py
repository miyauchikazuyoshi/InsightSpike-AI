"""Dependency resolution for platform-specific packages"""
from dataclasses import dataclass
from typing import Dict, List, Optional
from .platform_utils import PlatformInfo

@dataclass
class DependencySpec:
    """Specification for a dependency with version and platform constraints"""
    name: str
    version: str = ""
    extras: List[str] = None
    platform_tags: List[str] = None
    environment_markers: str = ""
    
    def __post_init__(self):
        if self.extras is None:
            self.extras = []
        if self.platform_tags is None:
            self.platform_tags = []

@dataclass
class ResolvedDependency:
    """Represents a resolved dependency with version and extras"""
    name: str
    version: str
    extras: List[str]
    platform_specific: bool = False

@dataclass
class ValidationResult:
    """Result of environment validation"""
    is_valid: bool
    issues: List[str]
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

class DependencyResolver:
    """Resolves platform-specific dependencies"""
    
    def __init__(self):
        self.platform_configs = {
            "macos": {
                "torch": {
                    "version": ">=2.0.0",
                    "description": "PyTorch for macOS with MPS support"
                },
                "torchvision": {
                    "version": ">=0.15.0", 
                    "description": "Computer vision library for PyTorch"
                },
                "faiss-cpu": {
                    "version": ">=1.7.0",
                    "description": "CPU-only FAISS for similarity search"
                }
            },
            "linux": {
                "torch": {
                    "version": ">=2.0.0",
                    "description": "PyTorch for Linux with CUDA support"
                },
                "torchvision": {
                    "version": ">=0.15.0",
                    "description": "Computer vision library for PyTorch"
                },
                "faiss-gpu": {
                    "version": ">=1.7.0", 
                    "description": "GPU-accelerated FAISS for similarity search"
                }
            },
            "windows": {
                "torch": {
                    "version": ">=2.0.0",
                    "description": "PyTorch for Windows"
                },
                "torchvision": {
                    "version": ">=0.15.0",
                    "description": "Computer vision library for PyTorch"  
                },
                "faiss-cpu": {
                    "version": ">=1.7.0",
                    "description": "CPU-only FAISS for similarity search"
                }
            }
        }
    
    def get_platform_dependencies(self, platform: str) -> Dict[str, Dict[str, str]]:
        """Get dependencies for a specific platform"""
        return self.platform_configs.get(platform, {})
    
    def resolve_dependencies(self, platform_info: PlatformInfo) -> List[ResolvedDependency]:
        """Resolve dependencies for given platform"""
        platform_deps = self.get_platform_dependencies(platform_info.platform)
        resolved = []
        
        for dep_name, dep_info in platform_deps.items():
            resolved_dep = ResolvedDependency(
                name=dep_name,
                version=dep_info.get("version", "latest"),
                extras=[],
                platform_specific=True
            )
            resolved.append(resolved_dep)
        
        return resolved
    
    def validate_environment(self, platform_info: PlatformInfo) -> ValidationResult:
        """Validate current environment against platform requirements"""
        issues = []
        warnings = []
        
        # Check Python version
        if platform_info.python_version < "3.8":
            issues.append("Python 3.8 or higher required")
        
        # Platform-specific checks
        if platform_info.platform == "macos" and platform_info.architecture == "arm64":
            warnings.append("ARM64 macOS detected - using CPU-only packages")
        elif platform_info.platform == "linux" and not platform_info.gpu_available:
            warnings.append("No GPU detected - consider CPU-only packages")
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            warnings=warnings
        )
