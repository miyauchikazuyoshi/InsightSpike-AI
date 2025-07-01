#!/usr/bin/env python3
"""
Fix Poetry CI configuration in GitHub Actions workflow
"""
import os
import re

def fix_poetry_config():
    """Fix Poetry installation configuration in CI workflow"""
    
    workflow_file = ".github/workflows/enhanced-ci.yml"
    
    # Check if GitHub workflows directory exists
    if not os.path.exists(".github/workflows"):
        print("⚠️ GitHub workflows directory not found")
        print("This script is intended for repositories with GitHub Actions CI")
        return
    
    if not os.path.exists(workflow_file):
        print(f"⚠️ Workflow file not found: {workflow_file}")
        print("Available workflow files:")
        for file in os.listdir(".github/workflows"):
            if file.endswith(('.yml', '.yaml')):
                print(f"  - {file}")
        return
    
    with open(workflow_file, 'r') as f:
        content = f.read()
    
    # Pattern to match Poetry installation blocks that need fixing
    pattern = r'(    - name: Install Poetry\n      uses: snok/install-poetry@v1\.3\.4\n      with:\n        version: \$\{\{ env\.POETRY_VERSION \}\}\n        virtualenvs-create: true\n        virtualenvs-in-project: true\n)\n(    - name: (?!Setup Poetry environment))'
    
    # Replacement with added configuration
    replacement = r'\1        installer-parallel: true\n\n    - name: Setup Poetry environment\n      run: |\n        poetry config virtualenvs.create true\n        poetry config virtualenvs.in-project true\n        echo "Poetry configuration completed"\n\n\2'
    
    # Apply the fix
    fixed_content = re.sub(pattern, replacement, content)
    
    with open(workflow_file, 'w') as f:
        f.write(fixed_content)
    
    print("✅ Fixed Poetry configuration in CI workflow")

if __name__ == "__main__":
    fix_poetry_config()
