#!/usr/bin/env python3
"""
InsightSpike-AI Setup Script
For pip-based installation with CLI command support
"""

from setuptools import setup, find_packages

# Read version from pyproject.toml (basic parsing)
def get_version():
    try:
        with open("pyproject.toml", "r") as f:
            for line in f:
                if line.startswith("version ="):
                    return line.split('"')[1]
    except:
        pass
    return "0.7.0"  # fallback

# Read README for description
def get_long_description():
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    except:
        return "InsightSpike prototype with PyG-based GNN layers"

setup(
    name="insightspike-ai",
    version=get_version(),
    description="InsightSpike prototype with PyG-based GNN layers",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="YOUR NAME",
    author_email="you@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    
    # CLI entry points - this is the key part!
    entry_points={
        "console_scripts": [
            "insightspike=insightspike.cli:app",
        ],
    },
    
    # Basic dependencies (minimal set)
    install_requires=[
        "typer>=0.7,<0.10",
        "click>=8.0,<8.2", 
        "rich>=13.6",
    ],
    
    # Optional dependencies for different environments
    extras_require={
        "full": [
            "torch>=2.2.2",
            "numpy>=1.24.0",
            "sentence-transformers>=2.5",
            "transformers>=4.30.0,<4.40.0",
            "accelerate>=0.29",
            "scipy>=1.12",
            "scikit-learn>=1.4",
            "networkx>=3.3",
            "datasets>=2.5",
            "matplotlib>=3.8",
            "nltk>=3.8",
            "pyyaml>=6.0",
        ],
        "colab": [
            "ipywidgets>=8.0",
        ],
    },
    
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
