#!/usr/bin/env python3
"""
Mathematical Concept Evolution Experiment
========================================

Tests episodic memory integration/separation through gradual mathematical learning.
"""

import json
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from insightspike.implementations.agents.main_agent import MainAgent
from insightspike.core.base.datastore import DataStore
from insightspike.config.models import InsightSpikeConfig
from insightspike.config.converter import ConfigConverter


class MathEvolutionExperiment:
    """Simulates mathematical concept learning from elementary to advanced"""
    
    def __init__(self):
        # Create configuration for experiment
        config_dict = {
            "memory": {
                "type": "enhanced",
                "embedding_model": "all-MiniLM-L6-v2",
                "max_episodes": 500
            },
            "graph": {
                "episode_merge_threshold": 0.75,  # Lower threshold to see more merges
                "episode_split_threshold": 0.35,  # Adjusted for concept conflicts
                "episode_prune_threshold": 0.1,
                "enable_auto_operations": True
            },
            "llm": {
                "provider": "mock",  # Using mock for consistent results
                "model": "mock-model"
            },
            "reasoning": {
                "max_iterations": 5,
                "spike_threshold": 0.5,
                "enable_graph_reasoning": True
            }
        }
        
        # Convert to proper config format
        self.config = InsightSpikeConfig(**config_dict)
        
        # Initialize agent directly
        self.agent = MainAgent(config=self.config)
        
        # Tracking
        self.memory_operations = []
        self.concept_evolution = {}
        self.phase_transitions = []
        
    def load_phase_data(self, phase: int) -> List[Dict]:
        """Load learning data for a specific phase"""
        phase_files = {
            1: "phase1_elementary.json",
            2: "phase2_middle_school.json",
            3: "phase3_high_school.json",
            4: "phase4_university.json"
        }
        
        file_path = Path(__file__).parent.parent / "data" / phase_files.get(phase, "phase1_elementary.json")
        
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def inject_concept(self, concept_data: Dict) -> Dict[str, Any]:
        """Inject a mathematical concept into the agent's memory"""
        # Format the concept as a learning episode
        episode_text = f"[{concept_data['concept']}] {concept_data['explanation']}"
        
        if 'examples' in concept_data:
            episode_text += f" 例: {', '.join(concept_data['examples'])}"
        
        if 'misconception' in concept_data:
            episode_text += f" (注意: {concept_data['misconception']})"
            
        # Add metadata for tracking
        metadata = {
            "concept": concept_data['concept'],
            "phase": concept_data['phase'],
            "timestamp": concept_data['timestamp'],
            "id": concept_data['id']
        }
        
        # Store current memory state
        memory_stats_before = self.agent.l2_memory.get_memory_stats()
        
        # Add knowledge through agent
        self.agent.add_knowledge(episode_text, metadata=metadata)
        
        # Check for memory operations
        memory_stats_after = self.agent.l2_memory.get_memory_stats()
        
        # Track any operations that occurred
        operation = self._detect_memory_operation(memory_stats_before, memory_stats_after)
        if operation:
            self.memory_operations.append({
                "timestamp": concept_data['timestamp'],
                "concept": concept_data['concept'],
                "operation": operation,
                "stats_before": memory_stats_before,
                "stats_after": memory_stats_after
            })
        
        return {
            "concept": concept_data['concept'],
            "stored": True,
            "operation": operation
        }
    
    def _detect_memory_operation(self, before: Dict, after: Dict) -> str:
        """Detect what memory operation occurred"""
        episodes_before = before.get('total_episodes', 0)
        episodes_after = after.get('total_episodes', 0)
        
        if episodes_after < episodes_before:
            return "merge"
        elif episodes_after > episodes_before + 1:
            return "split"
        elif episodes_after == episodes_before:
            return "merge_and_add"
        return "add"
    
    def process_phase_transition(self, from_phase: int, to_phase: int):
        """Process the transition between learning phases"""
        # Test understanding with questions that reveal conflicts
        transition_questions = {
            (1, 2): [
                "負の数を掛けるとはどういうことか？",
                "分数は本当にピザを切ることなのか？", 
                "文字式の意味は？"
            ],
            (2, 3): [
                "関数とは何か？",
                "微分と割り算の関係は？",
                "無限とは？"
            ]
        }
        
        questions = transition_questions.get((from_phase, to_phase), [])
        
        for question in questions:
            result = self.agent.process_question(question)
            
            # Check if this triggered any memory reorganization
            if hasattr(result, 'memory_operations'):
                self.phase_transitions.append({
                    "from_phase": from_phase,
                    "to_phase": to_phase,
                    "question": question,
                    "triggered_operations": result.memory_operations
                })
    
    def track_concept_evolution(self, concept_name: str):
        """Track how a specific concept evolves across phases"""
        # Search for all episodes related to this concept
        episodes = []
        
        for i, episode in enumerate(self.agent.l2_memory.episodes):
            if concept_name.lower() in episode.text.lower():
                episodes.append({
                    "index": i,
                    "text": episode.text,
                    "c_value": episode.c,
                    "metadata": episode.metadata
                })
        
        self.concept_evolution[concept_name] = episodes
    
    def run_experiment(self):
        """Run the full mathematical learning experiment"""
        print("=== Mathematical Concept Evolution Experiment ===\n")
        
        # Phase 1: Elementary concepts
        print("Phase 1: Elementary School Mathematics")
        phase1_data = self.load_phase_data(1)
        
        for concept in phase1_data[:5]:  # Start with first 5 concepts
            result = self.inject_concept(concept)
            print(f"  Learned: {concept['concept']} - {result['operation']}")
            time.sleep(0.1)  # Simulate temporal learning
        
        # Check specific concept evolution
        self.track_concept_evolution("fraction")
        self.track_concept_evolution("multiplication")
        
        # Analyze initial state
        print(f"\nMemory state after Phase 1:")
        stats = self.agent.l2_memory.get_memory_stats()
        print(f"  Total episodes: {stats['total_episodes']}")
        print(f"  Memory operations: {len(self.memory_operations)}")
        
        # Phase transition
        print("\n--- Phase Transition: Elementary → Middle School ---")
        self.process_phase_transition(1, 2)
        
        # Phase 2: Middle school concepts
        print("\nPhase 2: Middle School Mathematics")
        phase2_data = self.load_phase_data(2)
        
        for concept in phase2_data[:3]:  # First 3 middle school concepts
            result = self.inject_concept(concept)
            print(f"  Learned: {concept['concept']} - {result['operation']}")
            
            # Check if this conflicts with elementary understanding
            if 'conflicts_with' in concept:
                print(f"    → Conflicts with: {concept['conflicts_with']}")
        
        # Final analysis
        self.generate_report()
    
    def generate_report(self):
        """Generate experiment report"""
        print("\n=== Experiment Report ===\n")
        
        # Memory operations summary
        print("Memory Operations:")
        operation_counts = {}
        for op in self.memory_operations:
            op_type = op['operation']
            operation_counts[op_type] = operation_counts.get(op_type, 0) + 1
        
        for op_type, count in operation_counts.items():
            print(f"  {op_type}: {count}")
        
        # Concept evolution
        print("\nConcept Evolution Tracking:")
        for concept, episodes in self.concept_evolution.items():
            print(f"\n  {concept}:")
            for ep in episodes:
                phase = ep['metadata'].get('phase', 'unknown')
                print(f"    Phase {phase}: {ep['text'][:50]}...")
        
        # Save detailed results
        results = {
            "timestamp": datetime.now().isoformat(),
            "memory_operations": self.memory_operations,
            "concept_evolution": self.concept_evolution,
            "phase_transitions": self.phase_transitions,
            "final_stats": self.agent.l2_memory.get_memory_stats()
        }
        
        output_dir = Path(__file__).parent.parent / "results"
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"evolution_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    experiment = MathEvolutionExperiment()
    experiment.run_experiment()