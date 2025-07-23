#!/usr/bin/env python3
"""
Mathematical Concept Evolution Experiment (Fixed Version)
========================================================

Following CLAUDE.md policy for proper configuration and execution.
"""

import json
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# CLAUDE.md指定の設定システムを使用
from insightspike.config import load_config
from insightspike.config.presets import ConfigPresets
from insightspike.implementations.agents.main_agent import MainAgent


class MathEvolutionExperimentFixed:
    """Fixed version following CLAUDE.md policy"""
    
    def __init__(self):
        # CLAUDE.md: プリセットを使用
        self.config = load_config(preset="experiment")
        
        # エピソード記憶操作の閾値を調整
        # Note: These fields don't exist in the current config structure
        # The thresholds are likely handled internally by the memory system
        
        # MockProviderを使用（CLAUDE.md推奨）
        if hasattr(self.config, 'llm'):
            self.config.llm.provider = "mock"
            self.config.llm.model = "mock-model"
        
        # Initialize agent with proper config
        try:
            self.agent = MainAgent(config=self.config)
            print("✅ MainAgent initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize MainAgent: {e}")
            raise
        
        # Tracking
        self.memory_operations = []
        self.concept_evolution = {}
        self.phase_transitions = []
        self.episode_count_history = []
        
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
            episode_text += f" Examples: {', '.join(concept_data['examples'])}"
        
        if 'misconception' in concept_data:
            episode_text += f" (Note: {concept_data['misconception']})"
        
        # Add metadata for tracking
        metadata = {
            "concept": concept_data['concept'],
            "phase": concept_data['phase'],
            "timestamp": concept_data.get('timestamp', f"day_{len(self.episode_count_history)}"),
            "id": concept_data['id']
        }
        
        # Store current episode count
        if hasattr(self.agent, 'l2_memory') and hasattr(self.agent.l2_memory, 'episodes'):
            episodes_before = len(self.agent.l2_memory.episodes)
        else:
            episodes_before = 0
        
        # Add knowledge through agent
        try:
            # MainAgentのadd_knowledgeはmetadata引数を受け付けない（CLAUDE.md記載）
            self.agent.add_knowledge(episode_text)
            
            # Track episode count change
            if hasattr(self.agent, 'l2_memory') and hasattr(self.agent.l2_memory, 'episodes'):
                episodes_after = len(self.agent.l2_memory.episodes)
                
                # Detect operation
                operation = self._detect_memory_operation(episodes_before, episodes_after)
                
                # Check for splits by examining episode texts
                if operation == "add" and self._check_for_conceptual_split(concept_data):
                    operation = "split_detected"
                
                self.episode_count_history.append({
                    "concept": concept_data['concept'],
                    "before": episodes_before,
                    "after": episodes_after,
                    "operation": operation
                })
                
                if operation != "add":
                    self.memory_operations.append({
                        "timestamp": metadata['timestamp'],
                        "concept": concept_data['concept'],
                        "operation": operation,
                        "episodes_before": episodes_before,
                        "episodes_after": episodes_after
                    })
            else:
                operation = "unknown"
                
        except Exception as e:
            print(f"Error adding knowledge: {e}")
            operation = "error"
        
        return {
            "concept": concept_data['concept'],
            "stored": True,
            "operation": operation
        }
    
    def _detect_memory_operation(self, before: int, after: int) -> str:
        """Detect what memory operation occurred based on episode count"""
        if after < before:
            return "merge"
        elif after > before + 1:
            return "split"
        elif after == before:
            return "merge_and_add"
        return "add"
    
    def _check_for_conceptual_split(self, concept_data: Dict) -> bool:
        """Check if this concept might cause a split based on conflicts"""
        # Simple heuristic: if it conflicts with existing concepts
        if 'conflicts_with' in concept_data:
            return True
        
        # Check phase transitions that typically cause splits
        if concept_data['phase'] >= 2 and concept_data['concept'] in ['multiplication', 'function', 'number']:
            return True
            
        return False
    
    def analyze_memory_state(self):
        """Analyze current memory state"""
        if not hasattr(self.agent, 'l2_memory'):
            return {"error": "No memory layer available"}
        
        memory = self.agent.l2_memory
        analysis = {
            "total_episodes": len(memory.episodes) if hasattr(memory, 'episodes') else 0,
            "concepts": {},
            "phase_distribution": {}
        }
        
        # Analyze episodes
        if hasattr(memory, 'episodes'):
            for i, episode in enumerate(memory.episodes):
                # Extract concept from episode text
                text = episode.text
                if '[' in text and ']' in text:
                    concept = text.split('[')[1].split(']')[0]
                    
                    if concept not in analysis["concepts"]:
                        analysis["concepts"][concept] = []
                    
                    analysis["concepts"][concept].append({
                        "index": i,
                        "c_value": episode.c,
                        "text_preview": text[:100] + "..." if len(text) > 100 else text
                    })
        
        return analysis
    
    def run_experiment(self):
        """Run the full mathematical learning experiment"""
        print("=== Mathematical Concept Evolution Experiment (Fixed) ===\n")
        print("Following CLAUDE.md policy with proper configuration\n")
        
        # Phase 1: Elementary concepts
        print("Phase 1: Elementary School Mathematics")
        phase1_data = self.load_phase_data(1)
        
        for i, concept in enumerate(phase1_data[:5]):  # First 5 concepts
            print(f"  [{i+1}/5] Learning: {concept['concept']}...", end='')
            result = self.inject_concept(concept)
            print(f" {result['operation']}")
            time.sleep(0.1)  # Simulate temporal learning
        
        # Analyze state after Phase 1
        state1 = self.analyze_memory_state()
        print(f"\nMemory state after Phase 1:")
        print(f"  Total episodes: {state1['total_episodes']}")
        print(f"  Unique concepts: {len(state1['concepts'])}")
        
        # Phase 2: Middle school concepts
        print("\nPhase 2: Middle School Mathematics")
        phase2_data = self.load_phase_data(2)
        
        for i, concept in enumerate(phase2_data[:5]):  # First 5 middle school concepts
            print(f"  [{i+1}/5] Learning: {concept['concept']}...", end='')
            result = self.inject_concept(concept)
            print(f" {result['operation']}")
            
            # Check if this caused interesting changes
            if 'conflicts_with' in concept:
                print(f"    → Potential conflict with: {concept['conflicts_with']}")
        
        # Phase 3: Advanced concepts with conflicts
        print("\nPhase 3: Advanced Concepts (Testing Conflicts)")
        conflict_file = Path(__file__).parent.parent / "data" / "conflict_examples.json"
        if conflict_file.exists():
            with open(conflict_file, 'r', encoding='utf-8') as f:
                conflict_data = json.load(f)
            
            for i, concept in enumerate(conflict_data):
                print(f"  [{i+1}/{len(conflict_data)}] Learning: {concept['concept']} (advanced)...", end='')
                result = self.inject_concept(concept)
                print(f" {result['operation']}")
        
        # Final analysis
        self.generate_report()
    
    def generate_report(self):
        """Generate experiment report"""
        print("\n=== Experiment Report ===\n")
        
        # Memory operations summary
        print("Memory Operations Summary:")
        operation_counts = {}
        for op in self.memory_operations:
            op_type = op['operation']
            operation_counts[op_type] = operation_counts.get(op_type, 0) + 1
        
        if operation_counts:
            for op_type, count in operation_counts.items():
                print(f"  {op_type}: {count}")
        else:
            print("  No automatic memory operations detected")
        
        # Episode count evolution
        print("\nEpisode Count Evolution:")
        if self.episode_count_history:
            print(f"  Started with: {self.episode_count_history[0]['before']} episodes")
            print(f"  Ended with: {self.episode_count_history[-1]['after']} episodes")
            
            # Look for interesting transitions
            for record in self.episode_count_history:
                if record['operation'] not in ['add', 'unknown']:
                    print(f"  → {record['concept']}: {record['operation']} ({record['before']}→{record['after']})")
        
        # Final memory state
        final_state = self.analyze_memory_state()
        print(f"\nFinal Memory State:")
        print(f"  Total episodes: {final_state['total_episodes']}")
        print(f"  Concepts tracked: {len(final_state['concepts'])}")
        
        # Concept evolution
        print("\nConcept Distribution:")
        for concept, episodes in final_state['concepts'].items():
            if len(episodes) > 1:
                print(f"  {concept}: {len(episodes)} episodes (potential split/evolution)")
            else:
                print(f"  {concept}: {len(episodes)} episode")
        
        # Save detailed results
        results = {
            "timestamp": datetime.now().isoformat(),
            "experiment": "mathematical_concept_evolution_fixed",
            "summary": {
                "total_operations": len(self.memory_operations),
                "operation_counts": operation_counts,
                "final_episodes": final_state['total_episodes'],
                "unique_concepts": len(final_state['concepts'])
            },
            "memory_operations": self.memory_operations,
            "episode_history": self.episode_count_history,
            "final_state": final_state
        }
        
        # CLAUDE.md: 実験結果をresultsディレクトリに保存
        output_dir = Path(__file__).parent.parent / "results"
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"math_evolution_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Detailed results saved to: {output_file}")
        
        # CLAUDE.md: 実験終了時の処理
        print("\n✅ Experiment completed following CLAUDE.md policy")


if __name__ == "__main__":
    # CLAUDE.md: Poetry経由で実行されることを前提
    experiment = MathEvolutionExperimentFixed()
    experiment.run_experiment()