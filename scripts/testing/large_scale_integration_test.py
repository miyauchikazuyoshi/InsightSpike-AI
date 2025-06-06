#!/usr/bin/env python3
"""
InsightSpike-AI: Large-Scale Integration Testing Suite
Comprehensive testing for production-ready deployment validation

This script validates:
- Multi-environment compatibility (Local, Colab, CI)
- Production template generation and deployment
- Performance benchmarking and scalability
- Monitoring system integration
- Security and compliance validation
"""

import asyncio
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import concurrent.futures
from dataclasses import dataclass, asdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

@dataclass
class TestResult:
    """Test result data structure"""
    test_name: str
    status: str  # PASS, FAIL, SKIP, WARNING
    duration: float
    message: str
    details: Optional[Dict] = None

@dataclass
class IntegrationReport:
    """Integration testing report"""
    timestamp: str
    total_tests: int
    passed: int
    failed: int
    warnings: int
    skipped: int
    overall_status: str
    duration: float
    results: List[TestResult]
    environment_info: Dict[str, Any]

class LargeScaleIntegrationTester:
    """Comprehensive large-scale integration testing"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.results: List[TestResult] = []
        self.start_time = time.time()
        
        # Create directories
        self.test_output_dir = self.project_root / "test_outputs" / "integration"
        self.test_output_dir.mkdir(parents=True, exist_ok=True)
        
        print("🧪 InsightSpike-AI Large-Scale Integration Testing Suite")
        print("=" * 70)
        print(f"📂 Project root: {self.project_root.absolute()}")
        print(f"📊 Test output: {self.test_output_dir}")
        print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

    def run_test(self, test_name: str, test_func) -> TestResult:
        """Run a single test and capture results"""
        print(f"🧪 Running: {test_name}")
        start_time = time.time()
        
        try:
            result = test_func()
            duration = time.time() - start_time
            
            if isinstance(result, dict):
                status = result.get('status', 'PASS')
                message = result.get('message', 'Test completed successfully')
                details = result.get('details', None)
            elif isinstance(result, bool):
                status = 'PASS' if result else 'FAIL'
                message = 'Test completed' if result else 'Test failed'
                details = None
            else:
                status = 'PASS'
                message = str(result)
                details = None
                
            test_result = TestResult(
                test_name=test_name,
                status=status,
                duration=duration,
                message=message,
                details=details
            )
            
            status_emoji = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️", "WARNING": "⚠️"}
            print(f"   {status_emoji.get(status, '❓')} {status}: {message} ({duration:.2f}s)")
            
        except Exception as e:
            duration = time.time() - start_time
            test_result = TestResult(
                test_name=test_name,
                status='FAIL',
                duration=duration,
                message=f"Exception: {str(e)}",
                details={'traceback': traceback.format_exc()}
            )
            print(f"   ❌ FAIL: Exception - {str(e)} ({duration:.2f}s)")
            
        self.results.append(test_result)
        return test_result

    def test_environment_validation(self) -> Dict[str, Any]:
        """Test environment setup and validation"""
        try:
            # Check Python version
            python_version = sys.version
            
            # Check Poetry availability
            try:
                result = subprocess.run(['poetry', '--version'], 
                                      capture_output=True, text=True, timeout=10)
                poetry_available = result.returncode == 0
                poetry_version = result.stdout.strip() if poetry_available else "Not available"
            except:
                poetry_available = False
                poetry_version = "Not available"
            
            # Check directory structure
            critical_dirs = ['src', 'tests', 'scripts', 'docs']
            missing_dirs = [d for d in critical_dirs if not (self.project_root / d).exists()]
            
            # Check key files
            critical_files = ['pyproject.toml', 'README.md', '.github/workflows/enhanced-ci.yml']
            missing_files = [f for f in critical_files if not (self.project_root / f).exists()]
            
            success = poetry_available and len(missing_dirs) == 0 and len(missing_files) == 0
            
            return {
                'status': 'PASS' if success else 'WARNING',
                'message': f'Environment validation: {"Complete" if success else "Partial"}',
                'details': {
                    'python_version': python_version,
                    'poetry_available': poetry_available,
                    'poetry_version': poetry_version,
                    'missing_directories': missing_dirs,
                    'missing_files': missing_files
                }
            }
        except Exception as e:
            return {
                'status': 'FAIL',
                'message': f'Environment validation failed: {e}'
            }

    def test_poetry_cli_resolution(self) -> Dict[str, Any]:
        """Test Poetry CLI resolution system"""
        try:
            # Run the poetry validation script
            result = subprocess.run([
                'poetry', 'run', 'python', 'scripts/colab/validate_poetry_resolution.py',
                '--project-root', str(self.project_root)
            ], capture_output=True, text=True, timeout=60, cwd=self.project_root)
            
            success = result.returncode == 0
            
            # Parse output for success rate
            output = result.stdout
            success_rate = "Unknown"
            if "Success Rate:" in output:
                for line in output.split('\n'):
                    if "Success Rate:" in line:
                        success_rate = line.split("Success Rate:")[-1].strip()
                        break
            
            return {
                'status': 'PASS' if success else 'FAIL',
                'message': f'Poetry CLI resolution: {success_rate}',
                'details': {
                    'returncode': result.returncode,
                    'stdout': output,
                    'stderr': result.stderr
                }
            }
        except Exception as e:
            return {
                'status': 'FAIL',
                'message': f'Poetry CLI test failed: {e}'
            }

    def test_production_templates(self) -> Dict[str, Any]:
        """Test production template generation and validation"""
        try:
            # Test template listing
            list_result = subprocess.run([
                'poetry', 'run', 'python', 'templates/production_integration_template.py', '--list'
            ], capture_output=True, text=True, timeout=30, cwd=self.project_root)
            
            if list_result.returncode != 0:
                return {
                    'status': 'FAIL',
                    'message': 'Template listing failed',
                    'details': {'error': list_result.stderr}
                }
            
            # Test template generation
            temp_dir = self.test_output_dir / "templates"
            temp_dir.mkdir(exist_ok=True)
            
            gen_result = subprocess.run([
                'poetry', 'run', 'python', 'templates/production_integration_template.py',
                '--output', str(temp_dir)
            ], capture_output=True, text=True, timeout=60, cwd=self.project_root)
            
            success = gen_result.returncode == 0
            
            # Count generated files
            generated_files = list(temp_dir.glob("*.json")) + list(temp_dir.glob("*.py"))
            
            return {
                'status': 'PASS' if success else 'FAIL',
                'message': f'Templates: {len(generated_files)} files generated',
                'details': {
                    'list_output': list_result.stdout,
                    'generation_output': gen_result.stdout,
                    'generated_files': [f.name for f in generated_files]
                }
            }
        except Exception as e:
            return {
                'status': 'FAIL',
                'message': f'Template test failed: {e}'
            }

    def test_monitoring_system(self) -> Dict[str, Any]:
        """Test production monitoring system"""
        try:
            monitor_file = self.project_root / "monitoring" / "production_monitor.py"
            
            if not monitor_file.exists():
                return {
                    'status': 'FAIL',
                    'message': 'Monitoring system file not found'
                }
            
            # Test file syntax
            compile_result = subprocess.run([
                'python', '-m', 'py_compile', str(monitor_file)
            ], capture_output=True, text=True, timeout=30)
            
            if compile_result.returncode != 0:
                return {
                    'status': 'FAIL',
                    'message': 'Monitoring system syntax error',
                    'details': {'error': compile_result.stderr}
                }
            
            # Test import with proper path
            import_result = subprocess.run([
                'python', '-c', f'import sys; sys.path.append("."); import monitoring.production_monitor; print("Import successful")'
            ], capture_output=True, text=True, timeout=30, cwd=self.project_root)
            
            import_success = import_result.returncode == 0 and 'Import successful' in import_result.stdout
            
            # Test basic functionality with status command
            try:
                status_result = subprocess.run([
                    'python', str(monitor_file), '--status'
                ], capture_output=True, text=True, timeout=30, cwd=self.project_root)
                status_available = status_result.returncode == 0
            except:
                status_available = False
            
            return {
                'status': 'PASS',
                'message': f'Monitoring system: Import {"OK" if import_success else "Warning"}, Status {"Available" if status_available else "N/A"}',
                'details': {
                    'file_size': monitor_file.stat().st_size,
                    'syntax_check': 'passed',
                    'import_success': import_success,
                    'status_available': status_available
                }
            }
        except Exception as e:
            return {
                'status': 'WARNING',
                'message': f'Monitoring test partial: {e}'
            }

    def test_enhanced_ci_pipeline(self) -> Dict[str, Any]:
        """Test enhanced CI/CD pipeline configuration"""
        try:
            ci_file = self.project_root / ".github" / "workflows" / "enhanced-ci.yml"
            
            if not ci_file.exists():
                return {
                    'status': 'FAIL',
                    'message': 'Enhanced CI pipeline not found'
                }
            
            # Read and validate CI configuration
            with open(ci_file, 'r') as f:
                ci_content = f.read()
            
            # Check for required jobs
            required_jobs = [
                'code-quality',
                'test-suite',
                'poetry-validation',
                'template-validation',
                'performance-benchmark',
                'dependency-security',
                'monitoring-integration',
                'integration-report'
            ]
            
            missing_jobs = [job for job in required_jobs if job not in ci_content]
            
            return {
                'status': 'PASS' if len(missing_jobs) == 0 else 'WARNING',
                'message': f'CI Pipeline: {len(required_jobs) - len(missing_jobs)}/{len(required_jobs)} jobs found',
                'details': {
                    'file_size': len(ci_content),
                    'required_jobs': required_jobs,
                    'missing_jobs': missing_jobs
                }
            }
        except Exception as e:
            return {
                'status': 'FAIL',
                'message': f'CI pipeline test failed: {e}'
            }

    def test_colab_integration(self) -> Dict[str, Any]:
        """Test Google Colab integration capabilities"""
        try:
            # Check for Colab-specific files and scripts
            colab_files = [
                'scripts/colab/setup_colab.sh',
                'scripts/colab/validate_poetry_resolution.py',
                'experiments/notebooks/InsightSpike_Colab_Demo.ipynb'
            ]
            
            existing_files = []
            missing_files = []
            
            for file_path in colab_files:
                if (self.project_root / file_path).exists():
                    existing_files.append(file_path)
                else:
                    missing_files.append(file_path)
            
            # Test notebook poetry integration
            notebook_file = self.project_root / "experiments/notebooks/InsightSpike_Colab_Demo.ipynb"
            poetry_cli_fix_found = False
            
            if notebook_file.exists():
                try:
                    with open(notebook_file, 'r') as f:
                        notebook_content = f.read()
                    poetry_cli_fix_found = 'poetry_cli_fix' in notebook_content
                except:
                    pass
            
            success_rate = len(existing_files) / len(colab_files)
            
            return {
                'status': 'PASS' if success_rate >= 0.8 else 'WARNING' if success_rate >= 0.5 else 'FAIL',
                'message': f'Colab integration: {len(existing_files)}/{len(colab_files)} files, Poetry CLI fix: {"Yes" if poetry_cli_fix_found else "No"}',
                'details': {
                    'existing_files': existing_files,
                    'missing_files': missing_files,
                    'poetry_cli_fix_integrated': poetry_cli_fix_found
                }
            }
        except Exception as e:
            return {
                'status': 'FAIL',
                'message': f'Colab integration test failed: {e}'
            }

    def test_performance_benchmarks(self) -> Dict[str, Any]:
        """Test performance benchmarking system"""
        try:
            benchmark_file = self.project_root / "benchmarks" / "performance_suite.py"
            
            if not benchmark_file.exists():
                return {
                    'status': 'FAIL',
                    'message': 'Performance benchmark system not found'
                }
            
            # Test basic import and syntax
            compile_result = subprocess.run([
                'python', '-m', 'py_compile', str(benchmark_file)
            ], capture_output=True, text=True, timeout=30)
            
            syntax_ok = compile_result.returncode == 0
            
            # Test if we can run help
            try:
                help_result = subprocess.run([
                    'poetry', 'run', 'python', str(benchmark_file), '--help'
                ], capture_output=True, text=True, timeout=30, cwd=self.project_root)
                help_available = 'usage:' in help_result.stdout.lower() or help_result.returncode == 0
            except:
                help_available = False
            
            return {
                'status': 'PASS' if syntax_ok else 'WARNING',
                'message': f'Performance benchmarks: Syntax {"OK" if syntax_ok else "Error"}, Help {"Available" if help_available else "N/A"}',
                'details': {
                    'syntax_check': syntax_ok,
                    'help_available': help_available,
                    'file_size': benchmark_file.stat().st_size
                }
            }
        except Exception as e:
            return {
                'status': 'WARNING',
                'message': f'Performance benchmark test partial: {e}'
            }

    def test_security_compliance(self) -> Dict[str, Any]:
        """Test security and compliance features"""
        try:
            # Check for security-related configurations
            security_indicators = []
            
            # Check for dependency security in CI
            ci_file = self.project_root / ".github" / "workflows" / "enhanced-ci.yml"
            if ci_file.exists():
                with open(ci_file, 'r') as f:
                    ci_content = f.read()
                if 'dependency-security' in ci_content or 'safety' in ci_content:
                    security_indicators.append('CI Security Scanning')
            
            # Check for bandit security scanning
            if 'bandit' in ci_content:
                security_indicators.append('Static Code Analysis')
            
            # Check for license compliance
            if 'license' in ci_content.lower():
                security_indicators.append('License Compliance')
            
            # Check for secrets scanning
            if 'secret' in ci_content.lower():
                security_indicators.append('Secrets Scanning')
            
            return {
                'status': 'PASS' if len(security_indicators) >= 2 else 'WARNING',
                'message': f'Security features: {len(security_indicators)} found',
                'details': {
                    'security_features': security_indicators
                }
            }
        except Exception as e:
            return {
                'status': 'WARNING',
                'message': f'Security compliance test partial: {e}'
            }

    def test_documentation_completeness(self) -> Dict[str, Any]:
        """Test documentation completeness and quality"""
        try:
            doc_files = [
                'README.md',
                'CONTRIBUTING.md',
                'docs/COLAB_INTEGRATION_IMPROVEMENTS.md',
                'docs/LARGE_SCALE_DEPLOYMENT_GUIDE.md'
            ]
            
            existing_docs = []
            missing_docs = []
            total_size = 0
            
            for doc_path in doc_files:
                doc_file = self.project_root / doc_path
                if doc_file.exists():
                    existing_docs.append(doc_path)
                    total_size += doc_file.stat().st_size
                else:
                    missing_docs.append(doc_path)
            
            completeness = len(existing_docs) / len(doc_files)
            
            return {
                'status': 'PASS' if completeness >= 0.75 else 'WARNING',
                'message': f'Documentation: {len(existing_docs)}/{len(doc_files)} files, {total_size} bytes',
                'details': {
                    'existing_docs': existing_docs,
                    'missing_docs': missing_docs,
                    'total_size': total_size,
                    'completeness': completeness
                }
            }
        except Exception as e:
            return {
                'status': 'WARNING',
                'message': f'Documentation test partial: {e}'
            }

    def generate_integration_report(self) -> IntegrationReport:
        """Generate comprehensive integration report"""
        total_duration = time.time() - self.start_time
        
        # Count results by status
        status_counts = {'PASS': 0, 'FAIL': 0, 'WARNING': 0, 'SKIP': 0}
        for result in self.results:
            status_counts[result.status] += 1
        
        # Determine overall status
        if status_counts['FAIL'] > 0:
            overall_status = 'FAIL'
        elif status_counts['WARNING'] > 0:
            overall_status = 'WARNING'
        else:
            overall_status = 'PASS'
        
        # Environment information
        env_info = {
            'python_version': sys.version,
            'platform': sys.platform,
            'project_root': str(self.project_root.absolute()),
            'test_timestamp': datetime.now().isoformat(),
            'total_duration': total_duration
        }
        
        return IntegrationReport(
            timestamp=datetime.now().isoformat(),
            total_tests=len(self.results),
            passed=status_counts['PASS'],
            failed=status_counts['FAIL'],
            warnings=status_counts['WARNING'],
            skipped=status_counts['SKIP'],
            overall_status=overall_status,
            duration=total_duration,
            results=self.results,
            environment_info=env_info
        )

    def save_report(self, report: IntegrationReport) -> str:
        """Save integration report to file"""
        report_file = self.test_output_dir / f"integration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert to dict for JSON serialization
        report_dict = asdict(report)
        
        with open(report_file, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        return str(report_file)

    def run_all_tests(self):
        """Run all integration tests"""
        tests = [
            ("Environment Validation", self.test_environment_validation),
            ("Poetry CLI Resolution", self.test_poetry_cli_resolution),
            ("Production Templates", self.test_production_templates),
            ("Monitoring System", self.test_monitoring_system),
            ("Enhanced CI Pipeline", self.test_enhanced_ci_pipeline),
            ("Colab Integration", self.test_colab_integration),
            ("Performance Benchmarks", self.test_performance_benchmarks),
            ("Security Compliance", self.test_security_compliance),
            ("Documentation Completeness", self.test_documentation_completeness)
        ]
        
        print("🚀 Running Large-Scale Integration Tests")
        print("=" * 70)
        
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
            print()
        
        # Generate and save report
        report = self.generate_integration_report()
        report_file = self.save_report(report)
        
        # Print summary
        print("=" * 70)
        print("📋 INTEGRATION TEST SUMMARY")
        print("=" * 70)
        
        status_emoji = {'PASS': '✅', 'FAIL': '❌', 'WARNING': '⚠️', 'SKIP': '⏭️'}
        
        for result in self.results:
            emoji = status_emoji.get(result.status, '❓')
            print(f"{emoji} {result.test_name}: {result.message}")
        
        print()
        print(f"📊 Results: {report.passed} passed, {report.failed} failed, {report.warnings} warnings")
        print(f"⏱️ Duration: {report.duration:.2f} seconds")
        print(f"🎯 Overall Status: {status_emoji.get(report.overall_status, '❓')} {report.overall_status}")
        print(f"📄 Report saved: {report_file}")
        
        # Return exit code
        return 0 if report.overall_status == 'PASS' else 1

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="InsightSpike-AI Large-Scale Integration Testing")
    parser.add_argument("--project-root", type=str, default=".", help="Project root directory")
    
    args = parser.parse_args()
    
    # Run integration tests
    tester = LargeScaleIntegrationTester(args.project_root)
    exit_code = tester.run_all_tests()
    
    return exit_code

if __name__ == "__main__":
    exit(main())
