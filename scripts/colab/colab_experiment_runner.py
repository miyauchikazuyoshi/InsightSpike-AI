#!/usr/bin/env python3
"""
Poetry Alternative Experiment Runner for Google Colab
Provides experiment execution when Poetry CLI is not accessible
"""

import subprocess
import sys
import os
import importlib.util
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import time

class ColabExperimentRunner:
    """
    Alternative experiment runner for Colab environment
    Handles Poetry CLI issues and provides direct execution methods
    """
    
    def __init__(self, base_path: Optional[str] = None):
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.setup_environment()
    
    def setup_environment(self):
        """Setup Python environment for experiment execution"""
        # Add src to Python path for imports
        src_path = self.base_path / "src"
        if src_path.exists():
            sys.path.insert(0, str(src_path))
        
        # Set PYTHONPATH environment variable
        current_pythonpath = os.environ.get('PYTHONPATH', '')
        new_pythonpath = f"{src_path}:{current_pythonpath}" if current_pythonpath else str(src_path)
        os.environ['PYTHONPATH'] = new_pythonpath
        
        print(f"✅ Environment setup: PYTHONPATH={new_pythonpath}")
    
    def test_poetry_availability(self) -> bool:
        """Test if Poetry CLI is available"""
        try:
            result = subprocess.run(['poetry', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✅ Poetry available: {result.stdout.strip()}")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
            pass
        
        print("❌ Poetry CLI not available")
        return False
    
    def run_poetry_command(self, args: List[str]) -> bool:
        """
        Run Poetry command with fallback methods
        
        Args:
            args: Poetry command arguments (e.g., ['run', 'python', '-m', 'insightspike.cli', 'embed'])
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Method 1: Try Poetry directly
        if self.test_poetry_availability():
            try:
                cmd = ['poetry'] + args
                print(f"🚀 Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, check=True)
                return result.returncode == 0
            except subprocess.CalledProcessError as e:
                print(f"⚠️ Poetry command failed: {e}")
        
        # Method 2: Direct Python execution (fallback)
        if args and args[0] == 'run' and len(args) > 1:
            # Remove 'run' and execute directly
            direct_args = args[1:]
            try:
                print(f"🔄 Fallback: Running {' '.join(direct_args)}")
                result = subprocess.run(direct_args, check=True)
                return result.returncode == 0
            except subprocess.CalledProcessError as e:
                print(f"❌ Direct execution failed: {e}")
                return False
        
        print(f"❌ No fallback available for Poetry command: {' '.join(args)}")
        return False
    
    def run_insightspike_cli(self, command: str, *args) -> bool:
        """
        Run InsightSpike CLI commands with fallback methods
        
        Args:
            command: CLI command (embed, graph, loop, etc.)
            *args: Additional arguments
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Construct command arguments
        cli_args = ['run', 'python', '-m', 'insightspike.cli', command] + list(args)
        
        print(f"🧠 Running InsightSpike CLI: {command} {' '.join(args)}")
        return self.run_poetry_command(cli_args)
    
    def build_sample_data(self, data_path: Optional[str] = None) -> bool:
        """Build sample data for experiments"""
        if data_path is None:
            data_path = self.base_path / "data" / "raw" / "test_sentences.txt"
        else:
            data_path = Path(data_path)
        
        # Create directories
        data_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create sample content if not exists
        if not data_path.exists():
            sample_content = """The aurora borealis is caused by charged particles from the sun interacting with Earth's magnetic field.
Quantum entanglement is a phenomenon where particles become correlated in ways that defy classical physics.
Artificial intelligence uses machine learning algorithms to process data and make predictions.
The human brain contains billions of neurons that communicate through synapses.
Photosynthesis converts sunlight into chemical energy in plants using chlorophyll.
DNA contains the genetic instructions for all living organisms in a double helix structure.
Gravity is a fundamental force that attracts objects with mass toward each other.
Evolution explains how species change over time through natural selection and adaptation.
Neurons communicate through electrical and chemical signals across synaptic connections.
Machine learning models can identify complex patterns in large datasets automatically.
Complex systems exhibit emergent properties that arise from simple interactions.
Consciousness emerges from neural networks processing information across brain regions."""
            
            with open(data_path, 'w', encoding='utf-8') as f:
                f.write(sample_content)
            
            print(f"✅ Sample data created: {data_path}")
        else:
            print(f"✅ Sample data exists: {data_path}")
        
        return True
    
    def build_episodic_memory(self, data_path: Optional[str] = None) -> bool:
        """Build episodic memory from data"""
        if data_path is None:
            data_path = "data/raw/test_sentences.txt"
        
        print("🧠 Building episodic memory...")
        return self.run_insightspike_cli('embed', '--path', data_path)
    
    def build_similarity_graph(self) -> bool:
        """Build similarity graph"""
        print("🕸️ Building similarity graph...")
        return self.run_insightspike_cli('graph')
    
    def run_insight_query(self, query: str) -> bool:
        """Run insight detection query"""
        print(f"🔍 Running insight query: {query}")
        return self.run_insightspike_cli('loop', f'"{query}"')
    
    def run_large_scale_experiment(self, experiment_type: str = "quick") -> bool:
        """
        Run large-scale experiments
        
        Args:
            experiment_type: Type of experiment (quick, full, custom)
        
        Returns:
            bool: True if successful, False otherwise
        """
        print(f"🧪 Running large-scale experiment: {experiment_type}")
        
        experiment_script = self.base_path / "scripts" / "colab" / "colab_large_scale_experiment.py"
        
        if experiment_script.exists():
            # Method 1: Try with Poetry
            poetry_args = ['run', 'python', str(experiment_script), '--mode', experiment_type]
            if self.run_poetry_command(poetry_args):
                return True
            
            # Method 2: Direct execution
            try:
                cmd = [sys.executable, str(experiment_script), '--mode', experiment_type]
                print(f"🔄 Fallback: Running {' '.join(cmd)}")
                result = subprocess.run(cmd, check=True)
                return result.returncode == 0
            except subprocess.CalledProcessError as e:
                print(f"❌ Direct experiment execution failed: {e}")
                return False
        else:
            print(f"❌ Experiment script not found: {experiment_script}")
            return False
    
    def run_complete_demo(self) -> Dict[str, bool]:
        """
        Run complete InsightSpike-AI demo with all components
        
        Returns:
            Dict[str, bool]: Results of each step
        """
        results = {}
        
        print("🚀 Running Complete InsightSpike-AI Demo")
        print("=" * 50)
        
        # Step 1: Build sample data
        print("\n📊 Step 1: Building sample data...")
        results['sample_data'] = self.build_sample_data()
        
        # Step 2: Build episodic memory
        print("\n🧠 Step 2: Building episodic memory...")
        results['episodic_memory'] = self.build_episodic_memory()
        
        # Step 3: Build similarity graph
        print("\n🕸️ Step 3: Building similarity graph...")
        results['similarity_graph'] = self.build_similarity_graph()
        
        # Step 4: Run test queries
        print("\n🔍 Step 4: Running test queries...")
        test_queries = [
            "What is quantum entanglement?",
            "How do neurons communicate?",
            "What connects photosynthesis and consciousness?"
        ]
        
        query_results = []
        for query in test_queries:
            result = self.run_insight_query(query)
            query_results.append(result)
            time.sleep(2)  # Brief pause between queries
        
        results['test_queries'] = all(query_results)
        
        # Step 5: Run large-scale experiment
        print("\n🧪 Step 5: Running large-scale experiment...")
        results['large_scale_experiment'] = self.run_large_scale_experiment('quick')
        
        # Summary
        print("\n" + "=" * 50)
        print("📋 Demo Results Summary:")
        for step, success in results.items():
            status = "✅" if success else "❌"
            print(f"   {status} {step.replace('_', ' ').title()}")
        
        overall_success = all(results.values())
        print(f"\n🎉 Overall Demo Status: {'✅ SUCCESS' if overall_success else '⚠️ PARTIAL SUCCESS'}")
        
        return results

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Colab Experiment Runner for InsightSpike-AI")
    parser.add_argument("--command", choices=['demo', 'embed', 'graph', 'query', 'experiment'], 
                       default='demo', help="Command to run")
    parser.add_argument("--query", type=str, help="Query for insight detection")
    parser.add_argument("--mode", choices=['quick', 'full', 'custom'], 
                       default='quick', help="Experiment mode")
    parser.add_argument("--data-path", type=str, help="Path to data file")
    
    args = parser.parse_args()
    
    runner = ColabExperimentRunner()
    
    if args.command == 'demo':
        runner.run_complete_demo()
    elif args.command == 'embed':
        runner.build_sample_data(args.data_path)
        runner.build_episodic_memory(args.data_path)
    elif args.command == 'graph':
        runner.build_similarity_graph()
    elif args.command == 'query':
        if args.query:
            runner.run_insight_query(args.query)
        else:
            print("❌ Query required for query command")
    elif args.command == 'experiment':
        runner.run_large_scale_experiment(args.mode)

if __name__ == "__main__":
    main()
