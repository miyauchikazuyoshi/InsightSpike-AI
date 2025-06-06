"""CLI dependency management commands using Typer"""
import typer
from rich import print
from rich.console import Console
from rich.table import Table

from ..utils.dependency_resolver import DependencyResolver
from ..utils.platform_utils import PlatformDetector
from ..utils.poetry_integration import PoetryIntegration

console = Console()
deps_app = typer.Typer(help="Dependency management commands")


@deps_app.command()
def list(platform: str = typer.Option(None, help="Platform to list dependencies for")):
    """List available dependencies for current or specified platform."""
    try:
        detector = PlatformDetector()
        current_platform = detector.detect_platform()

        target_platform = platform if platform else current_platform.platform

        print(f"[bold green]Dependencies for platform: {target_platform}[/bold green]")

        resolver = DependencyResolver()
        deps = resolver.get_platform_dependencies(target_platform)

        table = Table(title=f"Available Dependencies - {target_platform}")
        table.add_column("Package", style="cyan")
        table.add_column("Version", style="green")
        table.add_column("Description", style="yellow")

        for pkg_name, pkg_info in deps.items():
            description = pkg_info.get("description", "No description")
            version = pkg_info.get("version", "latest")
            table.add_row(pkg_name, version, description)

        console.print(table)

    except Exception as e:
        print(f"[red]Error listing dependencies: {e}[/red]")
        raise typer.Exit(1)


@deps_app.command()
def validate():
    """Validate current environment and dependencies."""
    try:
        detector = PlatformDetector()
        platform_info = detector.detect_platform()

        print(f"[bold blue]Platform Detection Results:[/bold blue]")
        print(f"Platform: {platform_info.platform}")
        print(f"Architecture: {platform_info.architecture}")
        print(f"GPU Available: {platform_info.gpu_available}")

        resolver = DependencyResolver()
        validation_result = resolver.validate_environment(platform_info)

        if validation_result.is_valid:
            print(f"[green]✓ Environment validation passed[/green]")
        else:
            print(f"[red]✗ Environment validation failed[/red]")
            for issue in validation_result.issues:
                print(f"  - {issue}")

    except Exception as e:
        print(f"[red]Error validating environment: {e}[/red]")
        raise typer.Exit(1)


@deps_app.command()
def export(
    format: str = typer.Option(
        "requirements", help="Export format: requirements, poetry, pipfile"
    ),
    output: str = typer.Option(None, help="Output file path"),
):
    """Export current dependencies to specified format."""
    try:
        detector = PlatformDetector()
        platform_info = detector.detect_platform()

        resolver = DependencyResolver()
        poetry_integration = PoetryIntegration()

        if format == "requirements":
            content = poetry_integration.export_requirements(platform_info)
            default_filename = f"requirements-{platform_info.platform}.txt"
        else:
            print(f"[red]Format '{format}' not yet supported[/red]")
            raise typer.Exit(1)

        output_file = output or default_filename

        with open(output_file, "w") as f:
            f.write(content)

        print(f"[green]Dependencies exported to: {output_file}[/green]")

    except Exception as e:
        print(f"[red]Error exporting dependencies: {e}[/red]")
        raise typer.Exit(1)
