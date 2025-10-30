#!/usr/bin/env python3
"""Simple CLI entry point without heavy imports"""

import os
os.environ["INSIGHTSPIKE_LITE_MODE"] = "1"

import typer

app = typer.Typer(
    name="spike",
    help="InsightSpike AI - Discover insights through knowledge synthesis",
    add_completion=True,
    rich_markup_mode="rich",
)

@app.command()
def version():
    """Show version information"""
    print("InsightSpike AI")
    print("Version: 0.8.0")
    print("Brain-inspired AI for insight detection")

@app.command()
def test():
    """Test command that works immediately"""
    print("CLI is working!")

if __name__ == "__main__":
    app()