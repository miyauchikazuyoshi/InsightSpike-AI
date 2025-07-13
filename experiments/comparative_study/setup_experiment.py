#!/usr/bin/env python3
"""
Large-scale Comparative Study: LLM vs RAG vs InsightSpike
=========================================================

Compare three approaches on complex reasoning tasks:
1. Baseline LLM (GPT-3.5/4 or local model)
2. LLM + Traditional RAG
3. LLM + InsightSpike with Query Transformation
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.insightspike.core.config import Config
from src.insightspike.core.agents.main_agent_optimized import MainAgentOptimized

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for comparative experiments"""
    
    name: str = "LLM_RAG_InsightSpike_Comparison"
    description: str = "Compare baseline LLM, RAG, and InsightSpike on complex reasoning"
    
    # Model settings
    llm_provider: str = "openai"  # or "anthropic", "local"
    llm_model: str = "gpt-3.5-turbo"  # or "gpt-4", "claude-2", "llama2"
    embedding_model: str = "text-embedding-ada-002"
    
    # Experiment settings
    num_questions: int = 50
    num_runs_per_question: int = 3  # For statistical significance
    timeout_seconds: int = 60
    
    # Evaluation metrics
    metrics: List[str] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [
                "correctness",
                "completeness", 
                "reasoning_depth",
                "response_time",
                "token_usage",
                "insight_quality"
            ]


@dataclass
class TestCase:
    """Individual test case for evaluation"""
    
    id: str
    category: str  # e.g., "causal", "comparative", "analytical"
    question: str
    context: List[str]  # Background knowledge
    expected_insights: List[str]  # Key insights that should be discovered
    difficulty: str  # "easy", "medium", "hard", "expert"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DatasetGenerator:
    """Generate comprehensive test dataset"""
    
    @staticmethod
    def create_test_cases() -> List[TestCase]:
        """Create diverse test cases across multiple domains"""
        
        test_cases = []
        
        # 1. Scientific Reasoning
        test_cases.extend([
            TestCase(
                id="sci_001",
                category="causal",
                question="Why does water expand when it freezes, unlike most other substances?",
                context=[
                    "Water molecules form hydrogen bonds",
                    "Ice has a crystalline structure with hexagonal symmetry",
                    "Hydrogen bonds in ice are more ordered than in liquid water",
                    "The density of ice is about 0.92 g/cm³ while water is 1.0 g/cm³",
                    "Most substances contract when they solidify"
                ],
                expected_insights=[
                    "Hydrogen bonding creates open hexagonal structure",
                    "Ordered structure takes more space than random liquid arrangement",
                    "This anomaly is crucial for life (ice floats)"
                ],
                difficulty="medium"
            ),
            
            TestCase(
                id="sci_002", 
                category="comparative",
                question="How do quantum tunneling and classical diffusion differ in their mechanisms?",
                context=[
                    "Quantum tunneling allows particles to pass through energy barriers",
                    "Classical diffusion follows Fick's laws",
                    "Tunneling probability depends on barrier width and height",
                    "Diffusion rate depends on temperature and concentration gradient",
                    "Both involve particle movement but different physics"
                ],
                expected_insights=[
                    "Quantum tunneling is probabilistic and instantaneous",
                    "Classical diffusion is deterministic and time-dependent",
                    "Energy conservation works differently in each case"
                ],
                difficulty="hard"
            ),
            
            TestCase(
                id="sci_003",
                category="analytical",
                question="What role does entropy play in both biological evolution and technological innovation?",
                context=[
                    "Entropy measures disorder in systems",
                    "Evolution creates complex organisms from simple ones",
                    "Technology tends toward increasing complexity",
                    "The second law of thermodynamics states entropy increases",
                    "Living systems maintain low entropy by consuming energy",
                    "Innovation often involves combining existing elements in new ways"
                ],
                expected_insights=[
                    "Both processes create local order at the cost of global entropy",
                    "Information and energy flow are crucial for maintaining complexity",
                    "Selection pressure drives both biological and technological evolution"
                ],
                difficulty="expert"
            )
        ])
        
        # 2. Historical Analysis
        test_cases.extend([
            TestCase(
                id="hist_001",
                category="causal",
                question="What factors led to the simultaneous invention of calculus by Newton and Leibniz?",
                context=[
                    "17th century saw rapid mathematical development",
                    "Both had access to works by Fermat, Descartes, and others",
                    "Scientific revolution created demand for new mathematical tools",
                    "They worked on different problems (physics vs philosophy)",
                    "Communication between scholars was limited but existed"
                ],
                expected_insights=[
                    "Mathematical prerequisites were in place",
                    "Different approaches led to same fundamental concepts",
                    "Scientific needs drove mathematical innovation"
                ],
                difficulty="medium"
            ),
            
            TestCase(
                id="hist_002",
                category="comparative",
                question="Compare the collapse of the Roman Empire with the fall of the Soviet Union",
                context=[
                    "Rome fell gradually over centuries, USSR collapsed rapidly",
                    "Both were vast multi-ethnic empires",
                    "Economic problems plagued both systems",
                    "Military overextension was a factor",
                    "Internal political instability",
                    "External pressures and competition"
                ],
                expected_insights=[
                    "Speed of collapse differs due to communication technology",
                    "Economic unsustainability is common factor",
                    "Ideological rigidity contributed to both collapses"
                ],
                difficulty="hard"
            )
        ])
        
        # 3. Technological Innovation
        test_cases.extend([
            TestCase(
                id="tech_001",
                category="analytical",
                question="How might quantum computing fundamentally change machine learning algorithms?",
                context=[
                    "Quantum computers can explore multiple states simultaneously",
                    "Current ML relies on gradient descent optimization",
                    "Quantum algorithms like Grover's offer quadratic speedup",
                    "Quantum entanglement could enable new types of correlations",
                    "NISQ devices have high error rates",
                    "Quantum machine learning is an emerging field"
                ],
                expected_insights=[
                    "Superposition enables parallel exploration of solution space",
                    "Quantum kernels could capture non-classical patterns",
                    "Hybrid classical-quantum algorithms most promising near-term"
                ],
                difficulty="expert"
            ),
            
            TestCase(
                id="tech_002",
                category="causal",
                question="Why did blockchain technology emerge when it did and not earlier?",
                context=[
                    "Cryptographic hash functions existed since 1970s",
                    "Public key cryptography developed in 1976",
                    "P2P networks emerged in late 1990s",
                    "2008 financial crisis created distrust in institutions",
                    "Internet penetration reached critical mass",
                    "Previous attempts like HashCash and b-money failed"
                ],
                expected_insights=[
                    "Technical prerequisites converged by 2008",
                    "Social/economic conditions created demand",
                    "Satoshi's innovation was combining existing technologies"
                ],
                difficulty="medium"
            )
        ])
        
        # 4. Philosophy and Ethics
        test_cases.extend([
            TestCase(
                id="phil_001",
                category="analytical",
                question="Is consciousness an emergent property of complexity or something fundamentally different?",
                context=[
                    "Emergent properties arise from complex interactions",
                    "Consciousness seems to have subjective qualities (qualia)",
                    "Information integration theory proposes consciousness as integrated information",
                    "The hard problem of consciousness remains unsolved",
                    "Some argue for panpsychism or dualism",
                    "Neuroscience shows correlation between brain states and consciousness"
                ],
                expected_insights=[
                    "Emergence alone may not explain subjective experience",
                    "Information integration provides measurable framework",
                    "The question may require new conceptual frameworks"
                ],
                difficulty="expert"
            )
        ])
        
        # 5. Interdisciplinary Connections
        test_cases.extend([
            TestCase(
                id="inter_001",
                category="comparative",
                question="How do network effects in social media parallel phase transitions in physics?",
                context=[
                    "Phase transitions occur at critical points",
                    "Network effects show non-linear growth",
                    "Both exhibit sudden qualitative changes",
                    "Critical mass concepts apply to both",
                    "Metcalfe's law describes network value",
                    "Percolation theory models connectivity"
                ],
                expected_insights=[
                    "Both show threshold behavior and criticality",
                    "Small changes can trigger large-scale transformations",
                    "Mathematical frameworks from physics apply to social phenomena"
                ],
                difficulty="hard"
            ),
            
            TestCase(
                id="inter_002",
                category="analytical",
                question="What can ant colony optimization teach us about human urban planning?",
                context=[
                    "Ants use pheromone trails for pathfinding",
                    "Stigmergic communication enables coordination",
                    "Ant colonies optimize for efficiency without central planning",
                    "Cities face traffic, resource distribution challenges",
                    "Decentralized systems can be robust and adaptive",
                    "Feedback loops reinforce successful paths"
                ],
                expected_insights=[
                    "Decentralized optimization can outperform central planning",
                    "Local information and feedback create global patterns",
                    "Adaptive systems handle uncertainty better"
                ],
                difficulty="medium"
            )
        ])
        
        return test_cases


class KnowledgeBaseBuilder:
    """Build comprehensive knowledge base for experiments"""
    
    @staticmethod
    def create_knowledge_base() -> Dict[str, List[str]]:
        """Create domain-specific knowledge bases"""
        
        knowledge_base = {
            "physics": [
                "Energy cannot be created or destroyed, only transformed",
                "Entropy always increases in isolated systems",
                "Quantum mechanics describes probabilistic behavior at small scales",
                "Relativity unifies space and time into spacetime",
                "Forces are mediated by exchange of virtual particles",
                "Wave-particle duality is fundamental to quantum mechanics",
                "Conservation laws arise from symmetries (Noether's theorem)",
                "Phase transitions occur at critical points",
                "Chaos theory describes sensitive dependence on initial conditions",
                "Thermodynamic equilibrium is a statistical concept"
            ],
            
            "computer_science": [
                "Algorithms transform input to output through defined steps",
                "Computational complexity measures resource requirements",
                "Data structures organize information for efficient access",
                "Machine learning extracts patterns from data",
                "Networks enable communication between distributed systems",
                "Cryptography ensures secure communication",
                "Parallel processing speeds up computation",
                "Abstraction hides implementation details",
                "Recursion solves problems by self-reference",
                "Optimization finds best solutions under constraints"
            ],
            
            "biology": [
                "Evolution operates through natural selection",
                "DNA encodes genetic information",
                "Cells are the basic unit of life",
                "Ecosystems involve complex interactions",
                "Homeostasis maintains stable internal conditions",
                "Energy flows through food chains",
                "Proteins perform most cellular functions",
                "Mutations provide genetic variation",
                "Symbiosis involves mutual benefit",
                "Emergent properties arise from biological complexity"
            ],
            
            "mathematics": [
                "Numbers form various algebraic structures",
                "Calculus studies continuous change",
                "Probability quantifies uncertainty",
                "Topology studies properties preserved under deformation",
                "Logic provides foundations for reasoning",
                "Set theory underlies modern mathematics",
                "Symmetry groups describe invariances",
                "Fractals show self-similarity at all scales",
                "Graph theory models relationships",
                "Category theory abstracts mathematical structures"
            ],
            
            "philosophy": [
                "Epistemology studies nature of knowledge",
                "Ethics examines moral principles",
                "Metaphysics explores nature of reality",
                "Logic analyzes valid reasoning",
                "Consciousness poses the hard problem",
                "Free will may be compatible with determinism",
                "Language shapes thought (Sapir-Whorf hypothesis)",
                "Empiricism emphasizes sensory experience",
                "Rationalism emphasizes reason",
                "Pragmatism judges ideas by consequences"
            ],
            
            "history": [
                "Civilizations rise and fall in patterns",
                "Technology drives social change",
                "Economic systems evolve over time",
                "Ideas spread through cultural exchange",
                "Revolutions occur when conditions align",
                "Geography influences development",
                "Trade networks connect civilizations",
                "Conflict shapes political boundaries",
                "Innovation clusters in specific times/places",
                "Historical events have multiple causes"
            ],
            
            "interdisciplinary": [
                "Complex systems show emergent behavior",
                "Feedback loops amplify or dampen changes",
                "Networks exhibit small-world properties",
                "Information theory applies across domains",
                "Optimization principles appear everywhere",
                "Symmetry breaking creates diversity",
                "Scaling laws describe size relationships",
                "Phase transitions mark qualitative changes",
                "Hierarchies organize complex systems",
                "Adaptation occurs at multiple levels"
            ]
        }
        
        return knowledge_base


def create_experiment_structure():
    """Create directory structure for experiments"""
    
    base_path = Path(__file__).parent
    
    # Create directories
    directories = [
        "data/test_cases",
        "data/knowledge_base",
        "results/baseline_llm",
        "results/rag_system",
        "results/insightspike",
        "analysis/figures",
        "analysis/reports",
        "logs"
    ]
    
    for dir_path in directories:
        (base_path / dir_path).mkdir(parents=True, exist_ok=True)
    
    # Save test cases
    test_cases = DatasetGenerator.create_test_cases()
    with open(base_path / "data/test_cases/all_cases.json", "w") as f:
        json.dump([tc.to_dict() for tc in test_cases], f, indent=2)
    
    # Save knowledge base
    knowledge_base = KnowledgeBaseBuilder.create_knowledge_base()
    with open(base_path / "data/knowledge_base/knowledge.json", "w") as f:
        json.dump(knowledge_base, f, indent=2)
    
    # Create experiment config
    config = ExperimentConfig()
    with open(base_path / "experiment_config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
    
    logger.info(f"Created experiment structure at {base_path}")
    logger.info(f"Generated {len(test_cases)} test cases")
    logger.info(f"Knowledge base contains {sum(len(v) for v in knowledge_base.values())} facts")
    
    return base_path


if __name__ == "__main__":
    experiment_path = create_experiment_structure()
    print(f"\n✅ Experiment setup complete!")
    print(f"📁 Location: {experiment_path}")
    print(f"📊 Next step: Run comparative_study.py to execute experiments")