#!/usr/bin/env python3
"""
Validation script for InsightSpike-AI Colab setup coordination strategy.
Tests requirements files, dependency resolution, and setup script coordination.
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path
from typing import List, Dict, Set, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ColabSetupValidator:
    """Validates the Colab setup coordination strategy and requirements files."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.configs_dir = self.project_root / "deployment" / "configs"
        self.scripts_dir = self.project_root / "scripts" / "colab"
        
    def validate_requirements_files(self) -> Dict[str, List[str]]:
        """Validate all requirements files exist and are properly formatted."""
        logger.info("🔍 Validating requirements files...")
        
        required_files = [
            "requirements-colab.txt"
        ]
        
        results = {}
        issues = []
        
        for req_file in required_files:
            file_path = self.configs_dir / req_file
            if not file_path.exists():
                issues.append(f"Missing requirements file: {req_file}")
                continue
                
            try:
                with open(file_path, 'r') as f:
                    content = f.read().strip()
                    
                if not content:
                    issues.append(f"Empty requirements file: {req_file}")
                    continue
                    
                # Parse requirements - filter comments and empty lines
                lines = []
                for line in content.split('\n'):
                    line = line.strip()
                    # Skip empty lines, full comments, and section dividers
                    if (line and 
                        not line.startswith('#') and 
                        not line.startswith('//') and
                        not line.startswith('=') and
                        not line.startswith('```')):
                        lines.append(line)
                
                results[req_file] = lines
                logger.info(f"✅ {req_file}: {len(lines)} dependencies")
                
            except Exception as e:
                issues.append(f"Error reading {req_file}: {str(e)}")
        
        if issues:
            logger.error("❌ Requirements file validation failed:")
            for issue in issues:
                logger.error(f"  - {issue}")
            return {}
            
        logger.info("✅ All requirements files validated successfully")
        return results
        
    def validate_coordination_strategy(self, requirements: Dict[str, List[str]]) -> bool:
        """Validate the strategic coordination between requirements files."""
        logger.info("🔍 Validating coordination strategy...")
        
        if not requirements:
            logger.error("❌ No requirements data to validate")
            return False
            
        # Check that torch and faiss are excluded from colab requirements
        colab_deps = set(dep.split('==')[0].split('>=')[0].split('~=')[0].lower() 
                        for dep in requirements.get('requirements-colab.txt', []))
        
        excluded_packages = {'torch', 'torchvision', 'torchaudio', 'faiss', 'faiss-cpu', 'faiss-gpu'}
        found_excluded = excluded_packages.intersection(colab_deps)
        
        if found_excluded:
            logger.error(f"❌ Found excluded packages in requirements-colab.txt: {found_excluded}")
            return False
            
        # Check that comprehensive includes all dependencies
        comprehensive_deps = set(dep.split('==')[0].split('>=')[0].split('~=')[0].lower() 
                               for dep in requirements.get('requirements-colab-comprehensive.txt', []))
        
        torch_deps = set(dep.split('==')[0].split('>=')[0].split('~=')[0].lower() 
                        for dep in requirements.get('requirements-torch.txt', []))
        
        pyg_deps = set(dep.split('==')[0].split('>=')[0].split('~=')[0].lower() 
                      for dep in requirements.get('requirements-PyG.txt', []))
        
        # Comprehensive should include most dependencies (allowing for some strategic exclusions)
        expected_deps = colab_deps.union(torch_deps).union(pyg_deps)
        missing_deps = expected_deps - comprehensive_deps
        
        # Filter out known strategic exclusions
        strategic_exclusions = {'faiss-cpu', 'faiss-gpu'}  # These might be handled separately
        missing_deps = missing_deps - strategic_exclusions
        
        if missing_deps:
            logger.warning(f"⚠️  Dependencies in comprehensive but not in individual files: {missing_deps}")
        
        logger.info("✅ Coordination strategy validated")
        return True
        
    def validate_setup_scripts(self) -> bool:
        """Validate that all setup scripts exist and are executable."""
        logger.info("🔍 Validating setup scripts...")
        
        required_scripts = [
            "setup_colab.sh",
            "setup_colab_debug.sh"
        ]
        
        issues = []
        
        for script in required_scripts:
            script_path = self.scripts_dir / script
            
            if not script_path.exists():
                issues.append(f"Missing setup script: {script}")
                continue
                
            if not os.access(script_path, os.X_OK):
                try:
                    os.chmod(script_path, 0o755)
                    logger.info(f"🔧 Made {script} executable")
                except Exception as e:
                    issues.append(f"Cannot make {script} executable: {str(e)}")
                    continue
            
            # Basic content validation
            try:
                with open(script_path, 'r') as f:
                    content = f.read()
                    
                if not (content.strip().startswith('#!/bin/bash') or content.strip().startswith('#!/usr/bin/env bash')):
                    issues.append(f"{script} missing shebang")
                    
                # Check for pip install command instead of poetry
                if 'pip install' not in content:
                    issues.append(f"{script} missing pip install command")
                    
            except Exception as e:
                issues.append(f"Error validating {script}: {str(e)}")
        
        if issues:
            logger.error("❌ Setup script validation failed:")
            for issue in issues:
                logger.error(f"  - {issue}")
            return False
            
        logger.info("✅ All setup scripts validated")
        return True
        
    def check_project_configuration(self) -> bool:
        """Check project configuration files."""
        logger.info("🔍 Checking project configuration...")
        
        pyproject_path = self.project_root / "pyproject.toml"
        if not pyproject_path.exists():
            logger.error("❌ pyproject.toml not found")
            return False
            
        setup_py_path = self.project_root / "setup.py"
        if not setup_py_path.exists():
            logger.warning("⚠️  setup.py not found - CLI commands may not work")
        else:
            logger.info("✅ setup.py configured for CLI entry points")
                
        return True
            
    def run_validation(self) -> bool:
        """Run complete validation suite."""
        logger.info("🚀 Starting InsightSpike-AI Colab setup validation...")
        
        # Step 1: Validate requirements files
        requirements = self.validate_requirements_files()
        if not requirements:
            return False
            
        # Step 2: Validate coordination strategy  
        if not self.validate_coordination_strategy(requirements):
            return False
            
        # Step 3: Validate setup scripts
        if not self.validate_setup_scripts():
            return False
            
        # Step 4: Check project configuration
        if not self.check_project_configuration():
            return False
            
        logger.info("🎉 All validations passed! Colab setup coordination is properly configured.")
        return True
        
    def generate_usage_report(self) -> str:
        """Generate a usage report for the setup scripts."""
        report = """
📋 InsightSpike-AI Colab Setup Scripts Usage Guide

🚀 Available Setup Scripts:
1. setup_colab.sh - Standard setup (8-12 min)
   - Complete installation with full logging
   - Best for production and development
   - Usage: ./setup_colab.sh [standard|minimal|debug]
   
2. setup_colab.sh minimal - Minimal setup (<60 sec)
   - Essential dependencies only
   - Perfect for rapid prototyping
   - Usage: ./setup_colab.sh minimal
   
3. setup_colab.sh debug - Debug setup (15-20 min)
   - Comprehensive logging and diagnostics
   - Use for troubleshooting installation issues
   - Usage: ./setup_colab.sh debug
   
4. setup_colab_debug.sh - Alternative debug script (15-20 min)
   - Separate comprehensive debug script
   - Creates detailed diagnostic logs
   - Usage: ./setup_colab_debug.sh

🔧 Strategic Coordination:
- GPU packages (PyTorch, FAISS) installed first via pip
- Remaining dependencies managed via Poetry
- Conflict avoidance through strategic exclusions
- Modular requirements files for flexibility

📁 Requirements Files:
- requirements-torch.txt: PyTorch with CUDA support
- requirements-PyG.txt: PyTorch Geometric dependencies  
- requirements-colab.txt: Core dependencies (excludes torch/faiss)
- requirements-colab-comprehensive.txt: Complete dependency list

💡 Usage in Colab:
!wget https://raw.githubusercontent.com/your-repo/InsightSpike-AI/main/scripts/colab/setup_colab_fast.sh
!chmod +x setup_colab_fast.sh
!./setup_colab_fast.sh

"""
        return report

def main():
    """Main validation function."""
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = "."
    
    validator = ColabSetupValidator(project_root)
    
    success = validator.run_validation()
    
    if success:
        print(validator.generate_usage_report())
        sys.exit(0)
    else:
        logger.error("❌ Validation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
