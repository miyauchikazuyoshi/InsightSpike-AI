"""Platform detection utilities for dependency management"""
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


def timestamp() -> str:
    """Generate timestamp in a standardized format."""
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


@dataclass
class PlatformInfo:
    """Platform information container"""

    platform: str
    architecture: str
    gpu_available: bool = False
    python_version: str = ""

    def __post_init__(self):
        if not self.python_version:
            self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"


class PlatformDetector:
    """Detects current platform and capabilities"""

    def detect_platform(self) -> PlatformInfo:
        """Detect current platform information"""
        system = platform.system().lower()
        machine = platform.machine().lower()

        # Normalize platform names
        if system == "darwin":
            platform_name = "macos"
        elif system == "linux":
            platform_name = "linux"
        elif system == "windows":
            platform_name = "windows"
        else:
            platform_name = system

        # Detect architecture
        if machine in ["x86_64", "amd64"]:
            arch = "x64"
        elif machine in ["arm64", "aarch64"]:
            arch = "arm64"
        else:
            arch = machine

        # Check GPU availability
        gpu_available = self._check_gpu_availability()

        return PlatformInfo(
            platform=platform_name, architecture=arch, gpu_available=gpu_available
        )

    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available"""
        try:
            # Try nvidia-smi for NVIDIA GPUs
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        try:
            # Try checking for Metal on macOS
            if platform.system().lower() == "darwin":
                import torch

                return torch.backends.mps.is_available()
        except ImportError:
            pass

        return False


class PlatformValidator:
    """Validates platform compatibility and requirements"""

    def __init__(self):
        self.detector = PlatformDetector()

    def validate_requirements(self, requirements: list) -> bool:
        """Validate platform against list of requirements"""
        platform_info = self.detector.detect_platform()

        for req in requirements:
            if req == "gpu" and not platform_info.gpu_available:
                return False
            if (
                req.startswith("python>=")
                and platform_info.python_version < req.split(">=")[1]
            ):
                return False

        return True

    def check_compatibility(self, target_platform: str) -> bool:
        """Check if current platform is compatible with target"""
        current = self.detector.detect_platform()
        return current.platform == target_platform


def get_current_platform_info() -> PlatformInfo:
    """Get current platform information"""
    detector = PlatformDetector()
    return detector.detect_platform()


def is_platform_compatible(platform_spec: str) -> bool:
    """Check if current platform matches specification"""
    current = get_current_platform_info()
    return current.platform == platform_spec


def get_optimal_dependency_config(platform_info: PlatformInfo) -> dict:
    """Get optimal dependency configuration for platform"""
    config = {
        "torch_index_url": None,
        "faiss_variant": "faiss-cpu",
        "cuda_support": False,
    }

    if platform_info.platform == "linux" and platform_info.gpu_available:
        config["torch_index_url"] = "https://download.pytorch.org/whl/cu118"
        config["faiss_variant"] = "faiss-gpu"
        config["cuda_support"] = True
    elif platform_info.platform == "macos":
        config["faiss_variant"] = "faiss-cpu"
        # macOS uses default PyPI for torch with Metal support

    return config


def is_macos() -> bool:
    """Check if current platform is macOS."""
    return platform.system().lower() == "darwin"


def is_linux() -> bool:
    """Check if current platform is Linux."""
    return platform.system().lower() == "linux"


def is_windows() -> bool:
    """Check if current platform is Windows."""
    return platform.system().lower() == "windows"
