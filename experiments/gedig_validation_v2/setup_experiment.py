"""
geDIG理論検証実験 v2.0 セットアップスクリプト
=========================================

このスクリプトは実験環境を構築し、拡張知識ベースを作成します。
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# プロジェクトルートへのパスを設定
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from insightspike.core.config import Config

logger = logging.getLogger(__name__)


class ExperimentSetup:
    """実験環境のセットアップ"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.data_dir = base_dir / "data"
        self.results_dir = base_dir / "results"
        self.figures_dir = base_dir / "figures"
        
        # ディレクトリ作成
        for dir_path in [self.data_dir, self.results_dir, self.figures_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def create_knowledge_base(self) -> Dict[str, List[Dict[str, Any]]]:
        """5フェーズ×6エピソード = 30エピソードの知識ベース作成"""
        
        knowledge_base = {
            "Phase 1: Fundamental Concepts": [
                {
                    "id": "P1_E1",
                    "text": "Energy is the capacity to do work or cause change. It exists in various forms including kinetic, potential, thermal, and electromagnetic energy. The fundamental principle is that energy cannot be created or destroyed, only transformed from one form to another.",
                    "keywords": ["energy", "work", "transformation", "conservation"]
                },
                {
                    "id": "P1_E2",
                    "text": "Information represents patterns of organization that reduce uncertainty. In its most basic form, information is the resolution of uncertainty through the communication of meaningful patterns or structures.",
                    "keywords": ["information", "patterns", "uncertainty", "communication"]
                },
                {
                    "id": "P1_E3",
                    "text": "Structure refers to the arrangement and organization of parts within a whole. It determines how components relate to each other and influences the properties and behaviors of the system.",
                    "keywords": ["structure", "organization", "arrangement", "system"]
                },
                {
                    "id": "P1_E4",
                    "text": "Order is a state of arrangement where components follow predictable patterns or rules. It contrasts with disorder or chaos, representing lower entropy and higher organization.",
                    "keywords": ["order", "patterns", "entropy", "organization"]
                },
                {
                    "id": "P1_E5",
                    "text": "Change is the process by which things become different over time. It can be gradual or sudden, predictable or random, and is fundamental to all dynamic systems.",
                    "keywords": ["change", "process", "time", "dynamics"]
                },
                {
                    "id": "P1_E6",
                    "text": "Complexity arises from the interaction of multiple simple components. It is characterized by emergent properties that cannot be predicted from individual parts alone.",
                    "keywords": ["complexity", "interaction", "emergence", "components"]
                }
            ],
            
            "Phase 2: Mathematical Principles": [
                {
                    "id": "P2_E1",
                    "text": "Entropy in mathematics measures the amount of disorder or randomness in a system. Shannon entropy quantifies information content as H = -Σ p(x) log p(x), where higher entropy indicates more uncertainty.",
                    "keywords": ["entropy", "Shannon", "probability", "uncertainty"]
                },
                {
                    "id": "P2_E2",
                    "text": "Graph theory studies relationships between objects using nodes and edges. Graph structures can represent networks, dependencies, and complex relationships in abstract mathematical form.",
                    "keywords": ["graph", "nodes", "edges", "networks"]
                },
                {
                    "id": "P2_E3",
                    "text": "Probability theory provides a mathematical framework for quantifying uncertainty. It allows us to reason about random events and make predictions based on likelihood.",
                    "keywords": ["probability", "uncertainty", "random", "likelihood"]
                },
                {
                    "id": "P2_E4",
                    "text": "Optimization seeks to find the best solution among all possible alternatives. It involves maximizing or minimizing objective functions subject to constraints.",
                    "keywords": ["optimization", "maximum", "minimum", "constraints"]
                },
                {
                    "id": "P2_E5",
                    "text": "Topology studies properties that remain unchanged under continuous deformations. It reveals fundamental structures that persist despite superficial changes.",
                    "keywords": ["topology", "invariant", "deformation", "structure"]
                },
                {
                    "id": "P2_E6",
                    "text": "Fractals exhibit self-similarity at different scales. They demonstrate how simple recursive rules can generate infinitely complex patterns with fractional dimensions.",
                    "keywords": ["fractals", "self-similarity", "recursion", "dimension"]
                }
            ],
            
            "Phase 3: Physical Theories": [
                {
                    "id": "P3_E1",
                    "text": "Thermodynamics governs energy transformations in physical systems. The second law states that entropy always increases in isolated systems, defining the arrow of time.",
                    "keywords": ["thermodynamics", "entropy", "energy", "time"]
                },
                {
                    "id": "P3_E2",
                    "text": "Quantum mechanics reveals that information and physical reality are fundamentally connected. Quantum states encode information that determines measurable properties.",
                    "keywords": ["quantum", "information", "states", "measurement"]
                },
                {
                    "id": "P3_E3",
                    "text": "Statistical mechanics bridges microscopic and macroscopic descriptions. It shows how collective behavior emerges from individual particle interactions through statistical laws.",
                    "keywords": ["statistical", "microscopic", "macroscopic", "emergence"]
                },
                {
                    "id": "P3_E4",
                    "text": "Phase transitions occur when systems undergo qualitative changes in their properties. Critical points mark boundaries between different organizational states of matter.",
                    "keywords": ["phase", "transition", "critical", "states"]
                },
                {
                    "id": "P3_E5",
                    "text": "Conservation laws constrain physical processes. Energy, momentum, and information follow conservation principles that shape all physical interactions.",
                    "keywords": ["conservation", "constraints", "energy", "momentum"]
                },
                {
                    "id": "P3_E6",
                    "text": "Field theory describes how forces and interactions propagate through space. Fields carry both energy and information, mediating influences between distant objects.",
                    "keywords": ["field", "forces", "propagation", "space"]
                }
            ],
            
            "Phase 4: Biological Systems": [
                {
                    "id": "P4_E1",
                    "text": "Evolution operates through variation, selection, and heredity. It creates complex organisms by accumulating information in genetic sequences over generations.",
                    "keywords": ["evolution", "selection", "genetic", "information"]
                },
                {
                    "id": "P4_E2",
                    "text": "Neural networks process information through interconnected neurons. They demonstrate how simple processing units can collectively perform complex computations and learning.",
                    "keywords": ["neural", "networks", "processing", "learning"]
                },
                {
                    "id": "P4_E3",
                    "text": "Ecosystems exhibit complex interdependencies between organisms. Energy flows and information transfer create stable patterns and cycles in biological communities.",
                    "keywords": ["ecosystems", "interdependence", "energy", "cycles"]
                },
                {
                    "id": "P4_E4",
                    "text": "Adaptation allows organisms to optimize their fit to environments. It represents a form of information accumulation about environmental constraints and opportunities.",
                    "keywords": ["adaptation", "optimization", "environment", "fitness"]
                },
                {
                    "id": "P4_E5",
                    "text": "Metabolism transforms energy and matter to maintain living systems. It creates local order by increasing entropy in the surroundings, following thermodynamic laws.",
                    "keywords": ["metabolism", "energy", "order", "thermodynamics"]
                },
                {
                    "id": "P4_E6",
                    "text": "Self-organization in biological systems creates patterns without external control. It demonstrates how local interactions can produce global order and functionality.",
                    "keywords": ["self-organization", "patterns", "emergence", "order"]
                }
            ],
            
            "Phase 5: Information Theory": [
                {
                    "id": "P5_E1",
                    "text": "Information theory quantifies the capacity to transmit and store data. Channel capacity sets fundamental limits on communication rates given noise and bandwidth constraints.",
                    "keywords": ["information", "capacity", "communication", "limits"]
                },
                {
                    "id": "P5_E2",
                    "text": "Compression exploits redundancy to reduce information size. It reveals the true information content by removing predictable patterns while preserving essential data.",
                    "keywords": ["compression", "redundancy", "patterns", "efficiency"]
                },
                {
                    "id": "P5_E3",
                    "text": "Error correction adds controlled redundancy to protect information. It enables reliable communication over noisy channels by detecting and correcting corrupted data.",
                    "keywords": ["error", "correction", "redundancy", "reliability"]
                },
                {
                    "id": "P5_E4",
                    "text": "Mutual information measures the shared content between variables. It quantifies how much knowing one variable reduces uncertainty about another.",
                    "keywords": ["mutual", "information", "correlation", "uncertainty"]
                },
                {
                    "id": "P5_E5",
                    "text": "Algorithmic complexity defines information by the shortest program that generates it. This connects information theory to computation and compressibility.",
                    "keywords": ["algorithmic", "complexity", "computation", "program"]
                },
                {
                    "id": "P5_E6",
                    "text": "Information geometry treats probability distributions as points in a manifold. It provides geometric tools for understanding information relationships and transformations.",
                    "keywords": ["geometry", "probability", "manifold", "transformations"]
                }
            ]
        }
        
        # 知識ベースを保存
        kb_path = self.data_dir / "knowledge_base.json"
        with open(kb_path, 'w', encoding='utf-8') as f:
            json.dump(knowledge_base, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Knowledge base created with {sum(len(phase) for phase in knowledge_base.values())} episodes")
        return knowledge_base
    
    def create_question_sets(self) -> Dict[str, List[Dict[str, Any]]]:
        """30問の質問セット作成"""
        
        questions = {
            "baseline": [  # 単一ドメイン質問
                {"id": "B1", "text": "What is entropy?", "expected_phases": [2]},
                {"id": "B2", "text": "Define information theory", "expected_phases": [5]},
                {"id": "B3", "text": "Explain graph structures", "expected_phases": [2]},
                {"id": "B4", "text": "What is energy conservation?", "expected_phases": [1, 3]},
                {"id": "B5", "text": "Describe biological evolution", "expected_phases": [4]}
            ],
            
            "cross_domain": [  # クロスドメイン質問
                {"id": "C1", "text": "How does information relate to energy?", "expected_phases": [1, 3, 5]},
                {"id": "C2", "text": "What connects graphs and biological networks?", "expected_phases": [2, 4]},
                {"id": "C3", "text": "How do entropy and evolution interact?", "expected_phases": [2, 3, 4]},
                {"id": "C4", "text": "What links computation and thermodynamics?", "expected_phases": [2, 3, 5]},
                {"id": "C5", "text": "How do structures emerge from information?", "expected_phases": [1, 2, 5]},
                {"id": "C6", "text": "What unifies mathematical and physical entropy?", "expected_phases": [2, 3]},
                {"id": "C7", "text": "How does complexity arise in biological systems?", "expected_phases": [1, 4]},
                {"id": "C8", "text": "What role does information play in adaptation?", "expected_phases": [4, 5]},
                {"id": "C9", "text": "How do networks encode and process information?", "expected_phases": [2, 4, 5]},
                {"id": "C10", "text": "What connects order, entropy, and evolution?", "expected_phases": [1, 2, 3, 4]}
            ],
            
            "abstract": [  # 高次概念質問
                {"id": "A1", "text": "What is the nature of order and chaos?", "expected_phases": [1, 2, 3, 4]},
                {"id": "A2", "text": "How does complexity arise from simplicity?", "expected_phases": [1, 2, 4, 5]},
                {"id": "A3", "text": "What unifies discrete and continuous phenomena?", "expected_phases": [2, 3, 5]},
                {"id": "A4", "text": "What is the essence of emergence?", "expected_phases": [1, 3, 4]},
                {"id": "A5", "text": "How do patterns form in nature?", "expected_phases": [1, 2, 3, 4]},
                {"id": "A6", "text": "What is the relationship between structure and function?", "expected_phases": [1, 2, 3, 4]},
                {"id": "A7", "text": "How does information create reality?", "expected_phases": [1, 3, 5]},
                {"id": "A8", "text": "What drives self-organization?", "expected_phases": [1, 3, 4]},
                {"id": "A9", "text": "How do constraints enable creativity?", "expected_phases": [2, 3, 4, 5]},
                {"id": "A10", "text": "What is the fundamental nature of change?", "expected_phases": [1, 2, 3, 4, 5]}
            ],
            
            "edge_cases": [  # 境界条件テスト
                {"id": "E1", "text": "What has no information content?", "expected_phases": [1, 5]},
                {"id": "E2", "text": "Can entropy decrease globally?", "expected_phases": [2, 3]},
                {"id": "E3", "text": "Is there a limit to complexity?", "expected_phases": [1, 2, 5]},
                {"id": "E4", "text": "What defines a perfect structure?", "expected_phases": [1, 2]},
                {"id": "E5", "text": "How do paradoxes resolve?", "expected_phases": [2, 5]}
            ]
        }
        
        # 質問セットを保存
        questions_path = self.data_dir / "questions.json"
        with open(questions_path, 'w', encoding='utf-8') as f:
            json.dump(questions, f, indent=2, ensure_ascii=False)
        
        total_questions = sum(len(category) for category in questions.values())
        logger.info(f"Question set created with {total_questions} questions across {len(questions)} categories")
        
        return questions
    
    def create_experiment_config(self) -> Dict[str, Any]:
        """実験設定ファイルの作成"""
        
        config = {
            "experiment": {
                "name": "geDIG Theory Validation v2.0",
                "date": datetime.now().isoformat(),
                "random_seed": 42,
                "repetitions": 3
            },
            
            "models": {
                "llm": {
                    "name": "DistilGPT-2",
                    "provider": "local",
                    "temperature": 0.7,
                    "max_tokens": 256
                },
                "embedding": {
                    "name": "paraphrase-MiniLM-L6-v2",
                    "dimension": 384
                }
            },
            
            "parameters": {
                "spike_detection": {
                    "ged_threshold": -0.5,
                    "ig_threshold": 0.2,
                    "phase_threshold": 3,
                    "similarity_threshold": 0.3,
                    "confidence_threshold": 0.6
                },
                "retrieval": {
                    "top_k": 15,
                    "similarity_threshold": 0.25
                },
                "graph": {
                    "algorithm": "advanced",
                    "max_nodes": 1000,
                    "density_threshold": 0.8
                }
            },
            
            "logging": {
                "level": "DEBUG",
                "save_full_responses": True,
                "track_computational_cost": True,
                "log_graph_evolution": True
            },
            
            "analysis": {
                "metrics": [
                    "delta_ig", "delta_ged", "entropy_before", "entropy_after",
                    "graph_nodes", "graph_edges", "graph_density", "complexity_score"
                ],
                "statistical_tests": [
                    "chi_square", "anova", "tukey_hsd", "cohen_d"
                ],
                "visualization": [
                    "performance_comparison", "spike_scatter", "entropy_heatmap",
                    "graph_evolution", "parameter_sensitivity"
                ]
            }
        }
        
        # 設定を保存
        config_path = self.data_dir / "experiment_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info("Experiment configuration created")
        return config


def main():
    """メイン実行関数"""
    # ロギング設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 実験ディレクトリ
    experiment_dir = Path(__file__).parent
    
    # セットアップ実行
    setup = ExperimentSetup(experiment_dir)
    
    print("=== geDIG Theory Validation v2.0 Setup ===")
    print(f"Base directory: {experiment_dir}")
    
    # 1. 知識ベース作成
    print("\n1. Creating knowledge base...")
    knowledge_base = setup.create_knowledge_base()
    print(f"   ✓ Created {sum(len(phase) for phase in knowledge_base.values())} episodes across {len(knowledge_base)} phases")
    
    # 2. 質問セット作成
    print("\n2. Creating question sets...")
    questions = setup.create_question_sets()
    total_q = sum(len(cat) for cat in questions.values())
    print(f"   ✓ Created {total_q} questions in {len(questions)} categories")
    
    # 3. 実験設定作成
    print("\n3. Creating experiment configuration...")
    config = setup.create_experiment_config()
    print(f"   ✓ Configuration saved with {len(config['parameters'])} parameter groups")
    
    print("\n✅ Setup complete! Ready to run experiments.")
    print(f"\nNext step: python run_experiment.py")


if __name__ == "__main__":
    main()