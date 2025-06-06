"""CLI dependency management commands"""
from pathlib import Path
from typing import Optional

import click

from ..utils.dependency_resolver import DependencyResolver
from ..utils.platform_utils import PlatformDetector
from ..utils.poetry_integration import PoetryIntegration


@click.command()
def platform_info():
    """Display platform information and capabilities."""
    detector = PlatformDetector()
    info = detector.detect_platform()

    click.echo(f"Platform: {info.platform}")
    click.echo(f"Architecture: {info.architecture}")
    click.echo(f"GPU Available: {info.gpu_available}")
    click.echo(f"Python Version: {info.python_version}")


@click.command()
@click.option("--platform", help="Platform to list dependencies for")
def list_deps(platform: Optional[str] = None):
    """List available dependencies for current or specified platform."""
    detector = PlatformDetector()
    current_platform = detector.detect_platform()

    target_platform = platform if platform else current_platform.platform

    resolver = DependencyResolver()
    deps = resolver.get_platform_dependencies(target_platform)

    click.echo(f"Dependencies for {target_platform}:")
    for pkg_name, pkg_info in deps.items():
        version = pkg_info.get("version", "latest")
        description = pkg_info.get("description", "No description")
        click.echo(f"  {pkg_name} ({version}): {description}")


@click.command()
@click.argument("package_name")
@click.option("--version", help="Specific version to add")
def add(package_name: str, version: Optional[str] = None):
    """Add a dependency to the project."""
    # This would integrate with Poetry to add dependencies
    version_spec = f"@{version}" if version else ""
    click.echo(f"Adding {package_name}{version_spec}")
    # Implementation would call Poetry here


@click.command()
@click.argument("preset_name")
def install_preset(preset_name: str):
    """Install a preset collection of dependencies."""
    detector = PlatformDetector()
    platform_info = detector.detect_platform()

    poetry_integration = PoetryIntegration()

    if preset_name == "platform":
        # Install platform-specific preset
        success = poetry_integration.install_platform_dependencies(platform_info)
        if success:
            click.echo("Platform dependencies installed successfully")
        else:
            click.echo("Failed to install platform dependencies")
    else:
        click.echo(f"Unknown preset: {preset_name}")


@click.command()
@click.option("--output", "-o", help="Output file path")
@click.option("--format", default="requirements", help="Export format")
def export_requirements(output: Optional[str] = None, format: str = "requirements"):
    """Export current dependencies to requirements format."""
    detector = PlatformDetector()
    platform_info = detector.detect_platform()

    poetry_integration = PoetryIntegration()

    if format == "requirements":
        content = poetry_integration.export_requirements(platform_info)

        if output:
            with open(output, "w") as f:
                f.write(content)
            click.echo(f"Requirements exported to {output}")
        else:
            click.echo(content)
    else:
        click.echo(f"Unsupported format: {format}")


# Create a click group for all dependency commands
@click.group()
def deps():
    """Dependency management commands."""
    pass


deps.add_command(platform_info)
deps.add_command(list_deps)
deps.add_command(add)
deps.add_command(install_preset)
deps.add_command(export_requirements)
