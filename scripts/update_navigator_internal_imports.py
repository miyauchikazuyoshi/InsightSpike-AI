#!/usr/bin/env python3
"""
Update internal imports in moved navigator files.
"""

import os
import re
from pathlib import Path

def update_imports_in_file(filepath):
    """Update imports in a single file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Update maze_config imports
    content = re.sub(
        r'from \.\.config\.maze_config',
        'from ..maze_config',
        content
    )
    
    # Update environments imports if needed
    content = re.sub(
        r'from \.\.environments\.maze',
        'from ...environments.maze',
        content
    )
    
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated: {filepath}")
        return True
    return False

def main():
    """Main function to update navigator files."""
    project_root = Path(__file__).parent.parent
    navigators_dir = project_root / "src" / "insightspike" / "maze_experimental" / "navigators"
    
    updated_files = []
    
    # Find all Python files in navigators directory
    for py_file in navigators_dir.glob("*.py"):
        if py_file.name != "__init__.py" and update_imports_in_file(py_file):
            updated_files.append(py_file)
    
    print(f"\nTotal files updated: {len(updated_files)}")

if __name__ == "__main__":
    main()