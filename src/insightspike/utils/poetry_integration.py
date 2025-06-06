"""Poetry integration utilities"""
import json
import subprocess
from typing import Dict, List

from .dependency_resolver import DependencyResolver, ResolvedDependency
from .platform_utils import PlatformInfo


class PoetryIntegration:
    """Integrates with Poetry for dependency management"""

    def __init__(self):
        self.dependency_resolver = DependencyResolver()

    def export_requirements(self, platform_info: PlatformInfo) -> str:
        """Export platform-specific requirements.txt format"""
        resolved_deps = self.dependency_resolver.resolve_dependencies(platform_info)

        lines = [
            f"# Generated requirements for {platform_info.platform}",
            f"# Platform: {platform_info.platform}",
            f"# Architecture: {platform_info.architecture}",
            f"# GPU Available: {platform_info.gpu_available}",
            "",
        ]

        for dep in resolved_deps:
            if dep.version.startswith(">="):
                version_spec = dep.version
            else:
                version_spec = f"=={dep.version}" if dep.version != "latest" else ""

            line = f"{dep.name}{version_spec}"
            if dep.extras:
                extras_str = "[" + ",".join(dep.extras) + "]"
                line = f"{dep.name}{extras_str}{version_spec}"

            lines.append(line)

        return "\n".join(lines)

    def get_current_dependencies(self) -> Dict[str, str]:
        """Get currently installed dependencies from Poetry"""
        try:
            result = subprocess.run(
                ["poetry", "show", "--no-dev"],
                capture_output=True,
                text=True,
                check=True,
            )

            deps = {}
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        name, version = parts[0], parts[1]
                        deps[name] = version

            return deps
        except subprocess.CalledProcessError:
            return {}

    def install_platform_dependencies(self, platform_info: PlatformInfo) -> bool:
        """Install platform-specific dependencies using Poetry"""
        try:
            resolved_deps = self.dependency_resolver.resolve_dependencies(platform_info)

            for dep in resolved_deps:
                cmd = ["poetry", "add", f"{dep.name}{dep.version}"]

                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Failed to install {dep.name}: {result.stderr}")
                    return False

            return True
        except Exception as e:
            print(f"Installation failed: {e}")
            return False
