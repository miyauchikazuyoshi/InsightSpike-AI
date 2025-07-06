#!/usr/bin/env python3
"""
Post-installation script for InsightSpike-AI
============================================

Automatically downloads required models after package installation.
"""

import os
import sys
import subprocess


def is_ci_environment():
    """Check if running in CI environment."""
    ci_vars = ['CI', 'GITHUB_ACTIONS', 'TRAVIS', 'CIRCLECI', 'GITLAB_CI']
    return any(os.environ.get(var) for var in ci_vars)


def should_download_models():
    """Check if we should download models."""
    # Skip in CI environments
    if is_ci_environment():
        print("CI environment detected, skipping model downloads")
        return False
    
    # Check environment variable
    if os.environ.get('INSIGHTSPIKE_SKIP_MODELS', '').lower() in ('1', 'true', 'yes'):
        print("Model download skipped (INSIGHTSPIKE_SKIP_MODELS set)")
        return False
    
    return True


def main():
    """Run post-installation tasks."""
    print("\n=== InsightSpike-AI Post-Installation ===")
    
    if not should_download_models():
        return
    
    # Check if this is a development install
    setup_models_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'setup_models.py'
    )
    
    if os.path.exists(setup_models_path):
        print("\nWould you like to download required models now?")
        print("This will download:")
        print("  - Sentence Transformer (~90MB)")
        print("  - TinyLlama (~1.1GB)")
        print("\nYou can always do this later with: python scripts/setup_models.py")
        
        try:
            response = input("\nDownload models now? (y/N): ")
            if response.lower() == 'y':
                subprocess.run([sys.executable, setup_models_path], check=True)
            else:
                print("\nSkipping model download.")
                print("Run 'python scripts/setup_models.py' when ready.")
        except (KeyboardInterrupt, EOFError):
            print("\nSkipping model download.")
    else:
        print("Note: Model setup script not found.")
        print("Models will be downloaded on first use.")


if __name__ == "__main__":
    main()