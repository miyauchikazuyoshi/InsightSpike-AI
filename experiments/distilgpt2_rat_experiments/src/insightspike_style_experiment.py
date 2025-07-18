#!/usr/bin/env python3
"""
InsightSpike-style experiment with proper problem format
"What word associates with COTTAGE and SWISS?"
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from collections import defaultdict
from tqdm import tqdm


class InsightSpikeStyleExperiment:
    """
    Implements InsightSpike logic without complex dependencies
    Focus: Problem format and geDIG detection
    """
    
    def __init__(self):
        print("🚀 Initializing InsightSpike-style Experiment...")
        
        # geDIG thresholds (from EurekaDetector defaults)
        self.ged_threshold = 0.5  # ΔGED threshold
        self.ig_threshold = 0.2   # ΔIG threshold
        
        # Load databases
        self.load_databases()
        print("✅ Ready for creative association tasks")
    
    def load_databases(self):
        """Load word associations and test problems"""
        # Word association database (no answers!)
        self.associations = {
            "COTTAGE": {
                "concepts": ["small house", "rural living", "vacation home", "cozy dwelling"],
                "contexts": ["countryside", "retreat", "peaceful", "rustic"],
                "compounds": []  # We don't list "cottage cheese" to avoid cheating
            },
            "SWISS": {
                "concepts": ["from Switzerland", "Alpine culture", "precision", "neutrality"],
                "contexts": ["mountains", "watches", "banks", "chocolate"],
                "compounds": []  # No "Swiss cheese" listed
            },
            "CAKE": {
                "concepts": ["baked dessert", "celebration food", "sweet treat", "layers"],
                "contexts": ["birthday", "wedding", "party", "bakery"],
                "compounds": []  # No "cheesecake" listed
            },
            "CREAM": {
                "concepts": ["dairy product", "thick liquid", "white color", "smooth texture"],
                "contexts": ["coffee", "dessert", "skin care", "whipped"],
                "compounds": []
            },
            "SKATE": {
                "concepts": ["gliding motion", "sport equipment", "wheels or blades"],
                "contexts": ["rink", "roller", "figure", "hockey"],
                "compounds": []
            },
            "WATER": {
                "concepts": ["H2O", "liquid", "essential for life", "transparent"],
                "contexts": ["ocean", "rain", "river", "drink"],
                "compounds": []
            },
            "DUCK": {
                "concepts": ["water bird", "webbed feet", "quacking sound"],
                "contexts": ["pond", "feathers", "swimming", "flying"],
                "compounds": []
            },
            "FOLD": {
                "concepts": ["bend", "crease", "doubling over", "collapsing"],
                "contexts": ["paper", "origami", "laundry", "poker"],
                "compounds": []
            },
            "DOLLAR": {
                "concepts": ["US currency", "money unit", "green paper"],
                "contexts": ["bank", "store", "economy", "wealth"],
                "compounds": []
            }
        }
        
        # Test problems in new format
        self.test_problems = [
            {
                "id": 1,
                "prompt": "What word associates with COTTAGE and SWISS?",
                "words": ["COTTAGE", "SWISS"],
                "expected": "CHEESE",  # For evaluation only
                "hint": "Think about food products"
            },
            {
                "id": 2,
                "prompt": "Find a word that connects CREAM, SKATE, and WATER",
                "words": ["CREAM", "SKATE", "WATER"],
                "expected": "ICE",
                "hint": "Think about states of matter"
            },
            {
                "id": 3,
                "prompt": "What concept links DUCK, FOLD, and DOLLAR?",
                "words": ["DUCK", "FOLD", "DOLLAR"],
                "expected": "BILL",
                "hint": "Think about multiple meanings"
            }
        ]
    
    def build_conceptual_graph(self, words: List[str]) -> Dict:
        """Build graph from word concepts"""
        graph = {
            "nodes": {},  # node_id -> {type, concepts}
            "edges": []   # {from, to, weight}
        }
        
        # Add word nodes
        for word in words:
            if word in self.associations:
                graph["nodes"][word] = {
                    "type": "word",
                    "concepts": self.associations[word]["concepts"]
                }
                
                # Add concept nodes
                for i, concept in enumerate(self.associations[word]["concepts"]):
                    concept_id = f"{word}_concept_{i}"
                    graph["nodes"][concept_id] = {
                        "type": "concept",
                        "text": concept
                    }
                    graph["edges"].append({
                        "from": word,
                        "to": concept_id,
                        "weight": 0.8
                    })
                
                # Add context nodes
                for i, context in enumerate(self.associations[word]["contexts"]):
                    context_id = f"context_{context}"
                    if context_id not in graph["nodes"]:
                        graph["nodes"][context_id] = {
                            "type": "context",
                            "text": context
                        }
                    graph["edges"].append({
                        "from": word,
                        "to": context_id,
                        "weight": 0.6
                    })
        
        return graph
    
    def find_conceptual_bridges(self, graph: Dict, words: List[str]) -> List[Tuple[str, float]]:
        """Find concepts that bridge multiple words"""
        # Count connections from input words to each node
        connection_strength = defaultdict(lambda: defaultdict(float))
        
        for edge in graph["edges"]:
            if edge["from"] in words:
                target = edge["to"]
                connection_strength[target][edge["from"]] = edge["weight"]
        
        # Find nodes connected to multiple input words
        bridges = []
        for node_id, connections in connection_strength.items():
            if len(connections) >= 2:  # Connected to at least 2 input words
                # Calculate bridge strength
                strength = sum(connections.values()) / len(words)
                
                # Extract meaningful word from node_id
                if node_id.startswith("context_"):
                    word = node_id.replace("context_", "").upper()
                    # Check if it's a valid single word
                    if " " not in word and len(word) > 2:
                        bridges.append((word, strength))
        
        # Sort by strength
        bridges.sort(key=lambda x: x[1], reverse=True)
        return bridges
    
    def detect_insight_spike(self, graph_before: Dict, graph_after: Dict, confidence: float) -> Dict:
        """Detect insight spike using geDIG logic"""
        # Calculate graph metrics
        nodes_before = len(graph_before["nodes"])
        edges_before = len(graph_before["edges"])
        nodes_after = len(graph_after["nodes"])
        edges_after = len(graph_after["edges"])
        
        # ΔGED: Negative means simplification (good for insights)
        if nodes_before > 0:
            delta_ged = -(nodes_before - nodes_after) / nodes_before
        else:
            delta_ged = 0
        
        # ΔIG: Information gain from convergence
        delta_ig = confidence * 0.7  # Scale confidence to IG
        
        # Spike detection (simplified EurekaDetector logic)
        ged_condition = delta_ged <= -self.ged_threshold
        ig_condition = delta_ig >= self.ig_threshold
        spike_detected = ged_condition and ig_condition
        
        return {
            "spike_detected": spike_detected,
            "delta_ged": delta_ged,
            "delta_ig": delta_ig,
            "ged_condition": ged_condition,
            "ig_condition": ig_condition
        }
    
    def solve_association(self, words: List[str]) -> Dict:
        """Solve creative association problem"""
        # Build conceptual graph
        graph_before = self.build_conceptual_graph(words)
        
        # Find bridges
        bridges = self.find_conceptual_bridges(graph_before, words)
        
        if bridges:
            answer = bridges[0][0]
            confidence = bridges[0][1]
            
            # Simulate graph after insight
            graph_after = {
                "nodes": {w: {"type": "word"} for w in words},
                "edges": []
            }
            graph_after["nodes"][answer] = {"type": "insight"}
            for word in words:
                graph_after["edges"].append({
                    "from": word,
                    "to": answer,
                    "weight": 1.0
                })
            
            # Detect spike
            spike_info = self.detect_insight_spike(graph_before, graph_after, confidence)
            
            return {
                "answer": answer,
                "confidence": confidence,
                "candidates": bridges[:5],
                "spike_info": spike_info,
                "graph_metrics": {
                    "nodes_before": len(graph_before["nodes"]),
                    "nodes_after": len(graph_after["nodes"]),
                    "simplification": len(graph_before["nodes"]) - len(graph_after["nodes"])
                }
            }
        
        return {
            "answer": "UNKNOWN",
            "confidence": 0.0,
            "candidates": [],
            "spike_info": {"spike_detected": False},
            "graph_metrics": {}
        }
    
    def run_experiment(self):
        """Run the experiment"""
        print("\n🧪 Running InsightSpike-style Creative Association")
        print("=" * 60)
        
        results = []
        correct = 0
        spike_count = 0
        
        for problem in self.test_problems:
            print(f"\n📝 Problem {problem['id']}: {problem['prompt']}")
            if problem["hint"]:
                print(f"   Hint: {problem['hint']}")
            
            # Solve
            solution = self.solve_association(problem["words"])
            
            # Check correctness
            is_correct = solution["answer"] == problem["expected"]
            if is_correct:
                correct += 1
                print(f"✅ Found: {solution['answer']} (confidence: {solution['confidence']:.2f})")
            else:
                print(f"❌ Found: {solution['answer']} (expected: {problem['expected']})")
            
            # Spike detection
            if solution["spike_info"]["spike_detected"]:
                spike_count += 1
                print(f"⚡ Insight spike detected!")
                print(f"   ΔGED: {solution['spike_info']['delta_ged']:.3f}")
                print(f"   ΔIG: {solution['spike_info']['delta_ig']:.3f}")
            
            # Show candidates
            if solution["candidates"]:
                candidate_words = [c[0] for c in solution["candidates"][:3]]
                print(f"   Other candidates: {candidate_words}")
            
            # Graph simplification
            if solution["graph_metrics"]:
                print(f"   Graph: {solution['graph_metrics']['nodes_before']} → {solution['graph_metrics']['nodes_after']} nodes")
            
            results.append({
                "problem": problem,
                "solution": solution,
                "correct": is_correct
            })
        
        # Summary
        total = len(self.test_problems)
        accuracy = (correct / total * 100) if total > 0 else 0
        
        print("\n" + "=" * 60)
        print("📊 RESULTS")
        print("=" * 60)
        print(f"Accuracy: {correct}/{total} = {accuracy:.1f}%")
        print(f"Spike Detection: {spike_count}/{total} = {spike_count/total*100:.1f}%")
        
        # Save results
        output_dir = Path(__file__).parent.parent / "results" / "outputs"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"insightspike_style_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump({
                "metadata": {
                    "experiment": "InsightSpike-style Creative Association",
                    "description": "Proper problem format with geDIG detection",
                    "timestamp": timestamp
                },
                "summary": {
                    "accuracy": accuracy,
                    "spike_rate": spike_count / total if total > 0 else 0,
                    "total": total
                },
                "results": results
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        # Insights
        print("\n💡 KEY INSIGHTS:")
        print("✓ Problem format: 'What associates with X and Y?'")
        print("✓ No answer cheating - discovered through conceptual bridging")
        print("✓ geDIG spike detection for true insights")
        print("✓ Graph simplification indicates conceptual convergence")
        
        if accuracy < 50:
            print("\n⚠️  Low accuracy shows the difficulty without explicit connections")
            print("   Real InsightSpike would use richer knowledge graphs")

if __name__ == "__main__":
    experiment = InsightSpikeStyleExperiment()
    experiment.run_experiment()