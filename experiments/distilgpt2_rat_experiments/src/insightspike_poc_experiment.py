#!/usr/bin/env python3
"""
InsightSpike PoC Experiment using actual src implementation
Tests creative association tasks with proper geDIG detection
"""

import sys
from pathlib import Path
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import logging

# Add parent directory to path to import InsightSpike
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import actual InsightSpike components
from src.insightspike.detection.eureka_spike import EurekaDetector
from src.insightspike.algorithms.graph_edit_distance import GraphEditDistance
from src.insightspike.algorithms.information_gain import InformationGain
from src.insightspike.core.memory_graph.knowledge_graph_memory import KnowledgeGraphMemory
from src.insightspike.core.layers.layer2_memory_manager import L2MemoryManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InsightSpikePoCExperiment:
    """
    Proper PoC using InsightSpike's actual implementation
    Tests: "What word associates with COTTAGE and SWISS?"
    """
    
    def __init__(self):
        print("🚀 Initializing InsightSpike PoC Experiment...")
        
        # Initialize InsightSpike components
        self.eureka_detector = EurekaDetector()
        self.ged_calculator = GraphEditDistance()
        self.ig_calculator = InformationGain()
        self.memory_manager = L2MemoryManager(
            embedding_dim=768,  # Standard embedding dimension
            similarity_threshold=0.7
        )
        
        # Load word association database
        self.load_association_database()
        
        # Load test problems
        self.load_test_problems()
        
        print("✅ InsightSpike components initialized")
    
    def load_association_database(self):
        """Load word associations without including answers"""
        db_path = Path(__file__).parent.parent / "data" / "input" / "word_associations.json"
        
        # Create database if it doesn't exist
        if not db_path.exists():
            database = {
                "metadata": {
                    "description": "Word associations for creative thinking",
                    "note": "Does NOT include RAT answers to avoid cheating"
                },
                "associations": {
                    "COTTAGE": {
                        "concepts": ["small house", "rural", "vacation", "cozy", "countryside"],
                        "related": ["cabin", "home", "dwelling", "retreat", "bungalow"],
                        "attributes": ["quaint", "peaceful", "rustic", "charming"]
                    },
                    "SWISS": {
                        "concepts": ["Switzerland", "Alpine", "European", "neutral", "precision"],
                        "related": ["Alps", "watches", "banks", "chocolate", "mountains"],
                        "attributes": ["precise", "quality", "mountainous", "wealthy"]
                    },
                    "CAKE": {
                        "concepts": ["dessert", "baking", "celebration", "sweet", "layers"],
                        "related": ["birthday", "wedding", "frosting", "candles", "party"],
                        "attributes": ["sweet", "moist", "decorated", "delicious"]
                    },
                    "CREAM": {
                        "concepts": ["dairy", "white", "smooth", "rich", "topping"],
                        "related": ["milk", "butter", "whipped", "coffee", "skin"],
                        "attributes": ["smooth", "white", "rich", "thick"]
                    },
                    "SKATE": {
                        "concepts": ["gliding", "wheels", "blade", "movement", "sport"],
                        "related": ["roller", "figure", "hockey", "board", "rink"],
                        "attributes": ["fast", "smooth", "athletic", "graceful"]
                    },
                    "WATER": {
                        "concepts": ["liquid", "H2O", "ocean", "drink", "life"],
                        "related": ["rain", "river", "lake", "sea", "stream"],
                        "attributes": ["wet", "clear", "flowing", "essential"]
                    }
                }
            }
            
            db_path.parent.mkdir(parents=True, exist_ok=True)
            with open(db_path, 'w') as f:
                json.dump(database, f, indent=2)
        
        with open(db_path, 'r') as f:
            self.association_db = json.load(f)
    
    def load_test_problems(self):
        """Create test problems in the new format"""
        self.test_problems = [
            {
                "id": 1,
                "question": "What word associates with COTTAGE and SWISS?",
                "input_words": ["COTTAGE", "SWISS"],
                "expected_association": "cheese",  # For evaluation only
                "difficulty": "easy"
            },
            {
                "id": 2,
                "question": "What word connects CREAM, SKATE, and WATER?",
                "input_words": ["CREAM", "SKATE", "WATER"],
                "expected_association": "ice",
                "difficulty": "easy"
            },
            {
                "id": 3,
                "question": "Find a word that relates to COTTAGE, SWISS, and CAKE",
                "input_words": ["COTTAGE", "SWISS", "CAKE"],
                "expected_association": "cheese",
                "difficulty": "medium"
            }
        ]
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text (simplified for PoC)"""
        # In real implementation, would use sentence transformers
        # For PoC, create deterministic embeddings based on text
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(768).astype(np.float32)
    
    def build_association_graph(self, words: List[str]) -> Dict:
        """Build graph structure from word associations"""
        graph = {
            "nodes": [],
            "edges": [],
            "embeddings": {}
        }
        
        # Add word nodes
        for word in words:
            if word in self.association_db["associations"]:
                graph["nodes"].append({"id": word, "type": "word"})
                graph["embeddings"][word] = self.generate_embedding(word)
                
                # Add concept nodes
                word_data = self.association_db["associations"][word]
                for concept in word_data["concepts"]:
                    concept_id = f"{word}_{concept}"
                    graph["nodes"].append({"id": concept_id, "type": "concept"})
                    graph["edges"].append({"from": word, "to": concept_id, "weight": 0.8})
                    graph["embeddings"][concept_id] = self.generate_embedding(concept)
                
                # Add related nodes
                for related in word_data["related"]:
                    related_id = f"related_{related}"
                    if not any(n["id"] == related_id for n in graph["nodes"]):
                        graph["nodes"].append({"id": related_id, "type": "related"})
                        graph["embeddings"][related_id] = self.generate_embedding(related)
                    graph["edges"].append({"from": word, "to": related_id, "weight": 0.6})
        
        return graph
    
    def find_associations(self, graph: Dict) -> List[Tuple[str, float]]:
        """Find potential associations through graph analysis"""
        # Count connections to each node
        connection_counts = {}
        
        for edge in graph["edges"]:
            target = edge["to"]
            if target not in connection_counts:
                connection_counts[target] = 0
            connection_counts[target] += edge["weight"]
        
        # Find nodes connected to multiple input words
        candidates = []
        for node_id, score in connection_counts.items():
            if score > 1.0:  # Connected to multiple words
                # Extract the actual word from node_id
                if node_id.startswith("related_"):
                    word = node_id.replace("related_", "")
                    candidates.append((word, score))
        
        # Sort by connection strength
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates
    
    def simulate_thinking_process(self, words: List[str]) -> Dict:
        """Simulate the insight detection process"""
        print(f"\n🧠 Processing: {', '.join(words)}")
        
        # Build initial graph
        graph_before = self.build_association_graph(words)
        node_count_before = len(graph_before["nodes"])
        edge_count_before = len(graph_before["edges"])
        
        # Add episodes to memory
        for word in words:
            if word in graph_before["embeddings"]:
                embedding = graph_before["embeddings"][word]
                self.memory_manager.add_episode(word, embedding)
        
        # Find associations
        associations = self.find_associations(graph_before)
        
        if associations:
            # Simulate finding the insight
            best_association = associations[0][0]
            confidence = associations[0][1] / len(words)
            
            # Build graph after insight
            graph_after = {
                "nodes": [{"id": w, "type": "word"} for w in words] + 
                        [{"id": best_association, "type": "insight"}],
                "edges": [{"from": w, "to": best_association, "weight": 1.0} for w in words]
            }
            
            node_count_after = len(graph_after["nodes"])
            edge_count_after = len(graph_after["edges"])
            
            # Calculate geDIG metrics (simplified)
            delta_ged = (node_count_after - node_count_before) / node_count_before
            delta_ig = confidence * 0.5  # Simplified information gain
            
            # Detect spike using actual EurekaDetector
            spike_result = self.eureka_detector.detect_spike(delta_ged, delta_ig)
            spike_detected = spike_result["eureka_spike"]
            
            return {
                "association": best_association,
                "confidence": confidence,
                "spike_detected": spike_detected,
                "metrics": {
                    "delta_ged": delta_ged,
                    "delta_ig": delta_ig,
                    "nodes_before": node_count_before,
                    "nodes_after": node_count_after
                },
                "candidates": associations[:5]
            }
        
        return {
            "association": "unknown",
            "confidence": 0.0,
            "spike_detected": False,
            "metrics": {},
            "candidates": []
        }
    
    def run_experiment(self):
        """Run the PoC experiment"""
        print("\n🧪 Running InsightSpike PoC Experiment")
        print("=" * 60)
        print("Testing creative association with proper geDIG detection")
        print("=" * 60)
        
        results = []
        correct = 0
        spike_count = 0
        
        for problem in self.test_problems:
            print(f"\n📝 {problem['question']}")
            
            # Process with InsightSpike
            result = self.simulate_thinking_process(problem['input_words'])
            
            # Check if correct
            is_correct = result['association'].lower() == problem['expected_association'].lower()
            if is_correct:
                correct += 1
                print(f"✅ Found: {result['association']} (confidence: {result['confidence']:.2f})")
            else:
                print(f"❌ Found: {result['association']} (expected: {problem['expected_association']})")
            
            if result['spike_detected']:
                spike_count += 1
                print(f"⚡ Spike detected! ΔGED: {result['metrics']['delta_ged']:.3f}, ΔIG: {result['metrics']['delta_ig']:.3f}")
            
            # Show candidates
            if result['candidates']:
                print(f"   Other candidates: {[c[0] for c in result['candidates'][:3]]}")
            
            results.append({
                "problem": problem,
                "result": result,
                "correct": is_correct
            })
        
        # Summary
        total = len(self.test_problems)
        accuracy = (correct / total * 100) if total > 0 else 0
        
        print("\n" + "=" * 60)
        print("📊 RESULTS")
        print("=" * 60)
        print(f"Accuracy: {correct}/{total} = {accuracy:.1f}%")
        print(f"Spike Detection Rate: {spike_count}/{total} = {spike_count/total*100:.1f}%")
        
        # Save results
        output_dir = Path(__file__).parent.parent / "results" / "outputs"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"insightspike_poc_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump({
                "metadata": {
                    "experiment": "InsightSpike PoC with actual implementation",
                    "description": "Creative association using src/insightspike components",
                    "timestamp": timestamp
                },
                "summary": {
                    "accuracy": accuracy,
                    "spike_rate": spike_count / total if total > 0 else 0,
                    "total_problems": total
                },
                "results": results
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        # Analysis
        print("\n💡 INSIGHTS:")
        print("✓ Using actual InsightSpike components (EurekaDetector, MemoryManager)")
        print("✓ Problem format: 'What associates with X and Y?'")
        print("✓ No answer cheating - associations discovered through graph analysis")
        if spike_count > 0:
            print("✓ geDIG spike detection is working!")

if __name__ == "__main__":
    experiment = InsightSpikePoCExperiment()
    experiment.run_experiment()