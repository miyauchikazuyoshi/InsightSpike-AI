#!/usr/bin/env python3
"""
Comprehensive validation script for Poetry CLI resolution
Tests all fallback methods and validates the complete solution
"""

import subprocess
import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

class PoetryResolutionValidator:
    """Validates the Poetry CLI resolution implementation"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.results = {}
        self.setup_environment()
    
    def setup_environment(self):
        """Setup environment for validation"""
        src_path = self.project_root / "src"
        if src_path.exists():
            sys.path.insert(0, str(src_path))
        
        current_pythonpath = os.environ.get('PYTHONPATH', '')
        new_pythonpath = f"{src_path}:{current_pythonpath}" if current_pythonpath else str(src_path)
        os.environ['PYTHONPATH'] = new_pythonpath
        
        print(f"✅ Validation environment setup: {src_path}")
    
    def test_poetry_cli_direct(self) -> Tuple[bool, str]:
        """Test direct Poetry CLI access"""
        try:
            result = subprocess.run(['poetry', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return True, f"Poetry CLI available: {result.stdout.strip()}"
            else:
                return False, f"Poetry CLI error: {result.stderr}"
        except Exception as e:
            return False, f"Poetry CLI not accessible: {e}"
    
    def test_alternative_runner(self) -> Tuple[bool, str]:
        """Test Poetry alternative runner"""
        try:
            sys.path.append('scripts/colab')
            from colab_experiment_runner import ColabExperimentRunner
            
            runner = ColabExperimentRunner()
            poetry_available = runner.test_poetry_availability()
            
            return True, f"Alternative runner loaded (Poetry: {poetry_available})"
        except Exception as e:
            return False, f"Alternative runner failed: {e}"
    
    def test_fix_script(self) -> Tuple[bool, str]:
        """Test Poetry CLI fix script"""
        try:
            fix_script = self.project_root / "scripts/colab/fix_poetry_cli.sh"
            if not fix_script.exists():
                return False, "Fix script not found"
            
            # Check if script is executable
            if not os.access(fix_script, os.X_OK):
                os.chmod(fix_script, 0o755)
            
            return True, "Poetry CLI fix script available and executable"
        except Exception as e:
            return False, f"Fix script validation failed: {e}"
    
    def test_colab_setup_scripts(self) -> Tuple[bool, str]:
        """Test Colab setup scripts"""
        setup_scripts = [
            "scripts/colab/setup_colab_minimal.sh",
            "scripts/colab/setup_colab_fast.sh", 
            "scripts/colab/setup_colab.sh",
            "scripts/colab/setup_colab_debug.sh"
        ]
        
        available_scripts = []
        for script in setup_scripts:
            script_path = self.project_root / script
            if script_path.exists():
                available_scripts.append(script)
                # Make executable
                os.chmod(script_path, 0o755)
        
        if available_scripts:
            return True, f"Setup scripts available: {len(available_scripts)}/4"
        else:
            return False, "No setup scripts found"
    
    def test_notebook_integration(self) -> Tuple[bool, str]:
        """Test Colab notebook integration"""
        try:
            notebook_path = self.project_root / "InsightSpike_Colab_Demo.ipynb"
            if not notebook_path.exists():
                return False, "Colab notebook not found"
            
            # Check for Poetry CLI fix cell
            with open(notebook_path, 'r') as f:
                notebook_content = f.read()
            
            required_features = [
                '"poetry_cli_fix"',  # Poetry CLI fix cell
                'colab_experiment_runner',  # Alternative runner
                'fix_poetry_cli.sh',  # Fix script reference
                'Poetry Alternative'  # Alternative method text
            ]
            
            missing_features = []
            for feature in required_features:
                if feature not in notebook_content:
                    missing_features.append(feature)
            
            if missing_features:
                return False, f"Missing notebook features: {missing_features}"
            else:
                return True, "Notebook integration complete with all features"
                
        except Exception as e:
            return False, f"Notebook validation failed: {e}"
    
    def test_fallback_methods(self) -> Tuple[bool, str]:
        """Test fallback execution methods"""
        try:
            # Test CLI command construction
            test_commands = [
                ['poetry', 'run', 'python', '-m', 'insightspike.cli', '--help'],
                ['python', '-m', 'insightspike.cli', '--help'],
                ['python', 'scripts/colab/colab_experiment_runner.py', '--help']
            ]
            
            working_methods = []
            
            for cmd in test_commands:
                try:
                    result = subprocess.run(cmd, capture_output=True, 
                                          text=True, timeout=10, cwd=self.project_root)
                    if result.returncode == 0 or 'usage:' in result.stdout:
                        working_methods.append(' '.join(cmd[:2]))
                except:
                    pass
            
            if working_methods:
                return True, f"Working methods: {', '.join(working_methods)}"
            else:
                return False, "No fallback methods working"
                
        except Exception as e:
            return False, f"Fallback method testing failed: {e}"
    
    def test_experiment_execution(self) -> Tuple[bool, str]:
        """Test experiment execution with alternatives"""
        try:
            # Create minimal test data
            test_data_dir = self.project_root / "data/raw"
            test_data_dir.mkdir(parents=True, exist_ok=True)
            
            test_file = test_data_dir / "validation_test.txt"
            test_content = "This is a test sentence for validation purposes."
            
            with open(test_file, 'w') as f:
                f.write(test_content)
            
            # Test alternative runner execution
            try:
                sys.path.append('scripts/colab')
                from colab_experiment_runner import ColabExperimentRunner
                
                runner = ColabExperimentRunner()
                runner.build_sample_data(str(test_file))
                
                return True, "Experiment execution validated successfully"
            except Exception as e:
                return False, f"Experiment execution failed: {e}"
                
        except Exception as e:
            return False, f"Experiment validation setup failed: {e}"
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation of Poetry CLI resolution"""
        print("🔬 Poetry CLI Resolution Comprehensive Validation")
        print("=" * 60)
        
        tests = [
            ("Poetry CLI Direct", self.test_poetry_cli_direct),
            ("Alternative Runner", self.test_alternative_runner),
            ("Fix Script", self.test_fix_script),
            ("Setup Scripts", self.test_colab_setup_scripts),
            ("Notebook Integration", self.test_notebook_integration),
            ("Fallback Methods", self.test_fallback_methods),
            ("Experiment Execution", self.test_experiment_execution)
        ]
        
        results = {}
        passed_tests = 0
        
        for test_name, test_func in tests:
            print(f"\n🧪 Testing: {test_name}")
            print("-" * 40)
            
            start_time = time.time()
            try:
                success, message = test_func()
                duration = time.time() - start_time
                
                status = "✅ PASS" if success else "❌ FAIL"
                print(f"{status} - {message}")
                print(f"⏱️ Duration: {duration:.2f}s")
                
                results[test_name] = {
                    'success': success,
                    'message': message,
                    'duration': duration
                }
                
                if success:
                    passed_tests += 1
                    
            except Exception as e:
                duration = time.time() - start_time
                print(f"❌ ERROR - {e}")
                print(f"⏱️ Duration: {duration:.2f}s")
                
                results[test_name] = {
                    'success': False,
                    'message': f"Test error: {e}",
                    'duration': duration
                }
        
        # Generate summary
        total_tests = len(tests)
        success_rate = (passed_tests / total_tests) * 100
        
        print("\n" + "=" * 60)
        print("📋 VALIDATION SUMMARY")
        print("=" * 60)
        print(f"🎯 Success Rate: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        print(f"⏱️ Total Duration: {sum(r['duration'] for r in results.values()):.2f}s")
        
        # Detailed results
        print("\n📊 Detailed Results:")
        for test_name, result in results.items():
            status = "✅" if result['success'] else "❌"
            print(f"   {status} {test_name}: {result['message']}")
        
        # Overall assessment
        if success_rate >= 85:
            print(f"\n🎉 EXCELLENT: Poetry CLI resolution is working well ({success_rate:.1f}%)")
        elif success_rate >= 70:
            print(f"\n✅ GOOD: Poetry CLI resolution is mostly working ({success_rate:.1f}%)")
        elif success_rate >= 50:
            print(f"\n⚠️ PARTIAL: Some Poetry CLI features working ({success_rate:.1f}%)")
        else:
            print(f"\n❌ POOR: Poetry CLI resolution needs improvement ({success_rate:.1f}%)")
        
        return {
            'success_rate': success_rate,
            'passed_tests': passed_tests,
            'total_tests': total_tests,
            'results': results
        }

def main():
    """Main validation execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Poetry CLI Resolution Validator")
    parser.add_argument("--project-root", type=str, 
                       help="Root directory of InsightSpike-AI project")
    
    args = parser.parse_args()
    
    validator = PoetryResolutionValidator(args.project_root)
    validation_results = validator.run_comprehensive_validation()
    
    # Exit with appropriate code
    if validation_results['success_rate'] >= 70:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure

if __name__ == "__main__":
    main()
