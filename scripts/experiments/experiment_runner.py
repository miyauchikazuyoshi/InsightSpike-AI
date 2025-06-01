#!/usr/bin/env python3
"""
InsightSpike-AI Experiment Runner
Large-scale experiment execution and management for Colab environment
"""

import json
import time
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

class ExperimentRunner:
    """Manages execution of InsightSpike-AI experiments"""
    
    def __init__(self, base_dir="experiments"):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data"
        self.results_dir = self.base_dir / "results"
        
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def run_experiment_1(self):
        """Experiment 1: Paradox Resolution Task"""
        print("🧩 Starting Experiment 1: Paradox Resolution Task")
        print("=" * 60)
        
        # Paradox dataset
        paradox_dataset = [
            {
                "name": "Banach-Tarski Paradox",
                "setup": "A solid ball can be decomposed into finite pieces and reassembled into two identical balls of the same size as the original.",
                "resolution": "This uses the axiom of choice to create non-measurable sets. The pieces don't have well-defined volumes in the usual sense, so doubling volume isn't actually happening.",
                "cognitive_shift": "discrete_to_continuous",
                "expected_spike_timing": [0.3, 0.7]
            },
            {
                "name": "Zeno's Paradox",
                "setup": "Achilles can never overtake a tortoise if the tortoise has a head start, because he must always first reach where the tortoise was.",
                "resolution": "The infinite series of times converges to a finite value. Mathematics shows that ∑(1/2)ⁿ = 1, so infinite steps can occur in finite time.",
                "cognitive_shift": "infinite_to_finite",
                "expected_spike_timing": [0.4, 0.8]
            },
            {
                "name": "Monty Hall Problem",
                "setup": "You choose 1 of 3 doors. The host opens a losing door and offers to let you switch. Should you switch?",
                "resolution": "Yes! Your original choice has 1/3 probability, but the remaining door has 2/3 probability due to conditional probability.",
                "cognitive_shift": "intuition_to_logic",
                "expected_spike_timing": [0.5, 0.9]
            },
            {
                "name": "Ship of Theseus",
                "setup": "If all parts of a ship are gradually replaced, is it still the same ship? What if the old parts are reassembled?",
                "resolution": "This reveals the difference between physical and conceptual identity. Identity depends on continuity of function and pattern, not material substance.",
                "cognitive_shift": "material_to_pattern",
                "expected_spike_timing": [0.6, 0.85]
            }
        ]
        
        # Save dataset
        with open(self.data_dir / "paradox_dataset.json", 'w') as f:
            json.dump(paradox_dataset, f, indent=2)
        
        print(f"✅ Created paradox dataset with {len(paradox_dataset)} paradoxes")
        
        # Execute experiments
        results = []
        for i, paradox in enumerate(paradox_dataset, 1):
            print(f"\n🔍 Testing Paradox {i}: {paradox['name']}")
            
            full_query = f"Paradox: {paradox['setup']} Please explain why this seems impossible and then resolve it."
            
            start_time = time.time()
            try:
                # Simulate InsightSpike execution (replace with actual CLI call)
                print("🧠 Running InsightSpike analysis...")
                time.sleep(2)  # Simulate processing
                
                execution_time = time.time() - start_time
                result = {
                    "paradox_name": paradox['name'],
                    "execution_time": execution_time,
                    "cognitive_shift_type": paradox['cognitive_shift'],
                    "expected_spikes": paradox['expected_spike_timing'],
                    "status": "completed",
                    "timestamp": datetime.now().isoformat()
                }
                results.append(result)
                print(f"✅ Completed in {execution_time:.1f}s")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                result = {
                    "paradox_name": paradox['name'],
                    "status": "failed",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                results.append(result)
        
        # Save results
        with open(self.results_dir / "experiment1_paradox_resolution.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📁 Results saved to: {self.results_dir / 'experiment1_paradox_resolution.json'}")
        return results
    
    def run_experiment_2(self):
        """Experiment 2: Scaffolded Learning Task"""
        print("📚 Starting Experiment 2: Scaffolded Learning Task")
        print("=" * 60)
        
        # Concept hierarchies
        concept_hierarchies = {
            "mathematics": [
                {
                    "level": 1,
                    "concept": "Basic Arithmetic",
                    "example": "1 + 1 = 2. Addition combines quantities.",
                    "prerequisite": None,
                    "abstraction_level": "concrete"
                },
                {
                    "level": 2,
                    "concept": "Algebraic Equations", 
                    "example": "x + 1 = 2, therefore x = 1. Variables represent unknown quantities.",
                    "prerequisite": "Basic Arithmetic",
                    "abstraction_level": "symbolic"
                },
                {
                    "level": 3,
                    "concept": "Differential Equations",
                    "example": "dx/dt = -x describes exponential decay. Derivatives show rate of change.",
                    "prerequisite": "Algebraic Equations",
                    "abstraction_level": "dynamic"
                },
                {
                    "level": 4,
                    "concept": "Partial Differential Equations",
                    "example": "∂u/∂t = ∇²u is the heat equation. Multiple variables change simultaneously.",
                    "prerequisite": "Differential Equations",
                    "abstraction_level": "multidimensional"
                }
            ],
            "physics": [
                {
                    "level": 1,
                    "concept": "Newton's Laws",
                    "example": "F = ma. Force equals mass times acceleration in classical mechanics.",
                    "prerequisite": None,
                    "abstraction_level": "classical"
                },
                {
                    "level": 2,
                    "concept": "Special Relativity",
                    "example": "E = mc². Energy and mass are equivalent at high speeds.",
                    "prerequisite": "Newton's Laws",
                    "abstraction_level": "relativistic"
                },
                {
                    "level": 3,
                    "concept": "Quantum Mechanics",
                    "example": "HΨ = EΨ. The Schrödinger equation describes quantum states.",
                    "prerequisite": "Special Relativity",
                    "abstraction_level": "quantum"
                },
                {
                    "level": 4,
                    "concept": "Quantum Field Theory",
                    "example": "Lagrangian formalism unifies quantum mechanics and relativity.",
                    "prerequisite": "Quantum Mechanics",
                    "abstraction_level": "field_theoretic"
                }
            ]
        }
        
        # Save hierarchies
        for domain, hierarchy in concept_hierarchies.items():
            with open(self.data_dir / f"concept_hierarchy_{domain}.json", 'w') as f:
                json.dump(hierarchy, f, indent=2)
        
        print(f"✅ Created concept hierarchies for {len(concept_hierarchies)} domains")
        
        # Execute experiments
        results = []
        for domain, hierarchy in concept_hierarchies.items():
            print(f"\n🔬 Testing Domain: {domain.upper()}")
            
            for concept in hierarchy:
                level = concept['level']
                name = concept['concept']
                
                print(f"\n📊 Level {level}: {name}")
                
                start_time = time.time()
                try:
                    # Simulate processing
                    time.sleep(1.5)
                    
                    execution_time = time.time() - start_time
                    result = {
                        "domain": domain,
                        "level": level,
                        "concept": name,
                        "abstraction_level": concept['abstraction_level'],
                        "execution_time": execution_time,
                        "has_prerequisite": concept['prerequisite'] is not None,
                        "status": "completed",
                        "timestamp": datetime.now().isoformat()
                    }
                    results.append(result)
                    print(f"✅ Level {level} completed in {execution_time:.1f}s")
                    
                except Exception as e:
                    print(f"❌ Level {level} failed: {e}")
                    result = {
                        "domain": domain,
                        "level": level,
                        "concept": name,
                        "status": "failed",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                    results.append(result)
        
        # Save results
        with open(self.results_dir / "experiment2_scaffolded_learning.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📁 Results saved to: {self.results_dir / 'experiment2_scaffolded_learning.json'}")
        return results
    
    def run_all_experiments(self):
        """Run all experiments in sequence"""
        print("🚀 Starting All InsightSpike-AI Experiments")
        print("=" * 80)
        
        start_time = time.time()
        all_results = {}
        
        # Run experiments
        experiments = [
            ("experiment1", self.run_experiment_1),
            ("experiment2", self.run_experiment_2),
            # Add other experiments here when implemented
        ]
        
        for exp_name, exp_func in experiments:
            print(f"\n{'='*20} {exp_name.upper()} {'='*20}")
            try:
                results = exp_func()
                all_results[exp_name] = results
                print(f"✅ {exp_name} completed successfully")
            except Exception as e:
                print(f"❌ {exp_name} failed: {e}")
                all_results[exp_name] = {"status": "failed", "error": str(e)}
        
        total_time = time.time() - start_time
        
        # Generate summary
        summary = {
            "experiment_suite": "InsightSpike-AI Large-Scale Validation",
            "total_experiments": len(experiments),
            "total_execution_time": total_time,
            "timestamp": datetime.now().isoformat(),
            "results": all_results
        }
        
        with open(self.results_dir / "comprehensive_experiment_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "=" * 80)
        print("🎉 ALL EXPERIMENTS COMPLETED!")
        print("=" * 80)
        print(f"⏱️ Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"📁 Results saved to: {self.results_dir}")
        
        return summary

def main():
    """Main experiment runner"""
    parser = argparse.ArgumentParser(description="InsightSpike-AI Experiment Runner")
    parser.add_argument("--experiment", type=int, choices=[1, 2, 3, 4, 5], 
                       help="Run specific experiment (1-5)")
    parser.add_argument("--all", action="store_true", 
                       help="Run all experiments")
    parser.add_argument("--base-dir", default="experiments",
                       help="Base directory for experiments")
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(args.base_dir)
    
    if args.all:
        runner.run_all_experiments()
    elif args.experiment == 1:
        runner.run_experiment_1()
    elif args.experiment == 2:
        runner.run_experiment_2()
    else:
        print("Please specify --experiment [1-5] or --all")
        print("Available experiments:")
        print("  1: Paradox Resolution Task")
        print("  2: Scaffolded Learning Task")
        print("  3: Emergent Problem-Solving Task")
        print("  4: Baseline Comparison")
        print("  5: Real-time Insight Detection")

if __name__ == "__main__":
    main()
