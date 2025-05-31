#!/usr/bin/env python3
"""
True insight detection experiment - challenging datasets that require genuine reasoning leaps
================================================================================

This creates datasets where answers are NOT directly available in the training data,
requiring genuine insight and reasoning connections.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

def create_indirect_knowledge_base() -> List[str]:
    """Create knowledge base with INDIRECT information - no direct answers"""
    
    knowledge_base = []
    
    # Mathematics: Only basic facts, no solutions
    math_facts = [
        "Probability is a measure of uncertainty between 0 and 1.",
        "Independent events do not affect each other's outcomes.",
        "Conditional probability measures likelihood given prior information.",
        "Information theory quantifies the amount of information in data.",
        "Bayes' theorem relates prior and posterior probabilities.",
        "Game theory studies strategic decision-making between rational agents.",
        "Television game shows often involve probability and psychology.",
        "Contestants must make decisions under uncertainty.",
        "Host behavior in games may follow specific rules.",
        "Mathematical intuition can sometimes be misleading.",
        
        # Convergence concepts - but NOT about Zeno specifically
        "Mathematical series can converge to finite values.",
        "Infinite processes sometimes produce finite results.",
        "Ancient Greek philosophers posed various logical puzzles.",
        "Motion can be analyzed using mathematical tools.",
        "Discrete approximations can model continuous phenomena.",
        "Calculus deals with limits and infinitesimal changes.",
        "Racing involves relative speeds and distances.",
        "Geometric progressions have predictable patterns.",
        
        # Identity concepts - but NOT about Ship of Theseus
        "Objects persist through time despite changes.",
        "Philosophy examines the nature of identity.",
        "Physical and functional properties may differ in importance.",
        "Continuity can be spatial, temporal, or logical.",
        "Ancient vessels required periodic maintenance and repair.",
        "Replacement of parts is common in maintained objects.",
        "Identity criteria vary across different domains.",
    ]
    
    # Physics: General concepts, no paradigm shift explanations
    physics_facts = [
        "Classical mechanics assumes deterministic behavior.",
        "Particles have properties like position and momentum.",
        "Energy and matter are fundamental concepts in physics.",
        "Measurement is essential for experimental physics.",
        "Light travels at approximately 300,000 km per second.",
        "Mass is a measure of an object's inertia.",
        "Quantum effects become important at very small scales.",
        "Uncertainty principles exist in various physical systems.",
        "Wave-particle duality appears in many phenomena.",
        "Relativity theory was developed in the early 20th century.",
        "Scientific revolutions involve changing fundamental assumptions.",
        "New theories often contradict common sense initially.",
    ]
    
    # Completely unrelated domains to add noise
    unrelated_facts = [
        "Cooking requires precise timing and temperature control.",
        "Weather patterns are influenced by atmospheric pressure.",
        "Plants photosynthesize using sunlight and carbon dioxide.",
        "Historical documents provide insights into past civilizations.",
        "Art museums display works from various time periods.",
        "Transportation systems connect different geographical locations.",
        "Literature explores human emotions and experiences.",
        "Music composition involves rhythm, melody, and harmony.",
        "Architecture balances function with aesthetic considerations.",
        "Economics studies resource allocation and market behavior.",
        "Psychology examines human cognition and behavior.",
        "Sociology analyzes social structures and interactions.",
        "Biology investigates living organisms and their processes.",
        "Chemistry focuses on atomic and molecular interactions.",
        "Geology studies Earth's structure and formation.",
        "Astronomy explores celestial objects and phenomena.",
        "Computer science develops algorithms and data structures.",
        "Engineering applies scientific principles to practical problems.",
        "Medicine aims to diagnose and treat human ailments.",
        "Education facilitates knowledge transfer and skill development.",
    ]
    
    knowledge_base.extend(math_facts)
    knowledge_base.extend(physics_facts)
    knowledge_base.extend(unrelated_facts)
    
    return knowledge_base

def create_insight_requiring_questions() -> List[Dict[str, Any]]:
    """Create questions that require genuine insight - answers not directly in data"""
    
    questions = [
        {
            "id": "monty_hall_insight",
            "question": "A game show contestant picks door 1 of 3. The host, who knows what's behind each door, opens door 3 (empty). Should the contestant switch to door 2? Explain the mathematical reasoning.",
            "why_insight_required": "Must connect conditional probability, information theory, and Bayes' theorem concepts that exist separately in the knowledge base",
            "required_connections": [
                "conditional probability + game show rules",
                "information theory + host behavior", 
                "Bayes' theorem + decision making"
            ],
            "category": "cross_domain_reasoning",
            "difficulty": "high"
        },
        {
            "id": "zeno_resolution_insight", 
            "question": "If a runner must always cover half the remaining distance to the finish line, how can they ever finish the race? Resolve this paradox using modern mathematics.",
            "why_insight_required": "Must connect infinite series convergence with motion analysis - concepts separated in knowledge base",
            "required_connections": [
                "infinite series + motion analysis",
                "convergence concepts + discrete vs continuous",
                "ancient puzzles + modern calculus"
            ],
            "category": "paradigm_synthesis",
            "difficulty": "high"
        },
        {
            "id": "identity_persistence_insight",
            "question": "If every component of a complex system is gradually replaced while maintaining its function, what determines whether it remains the 'same' system? Analyze different identity criteria.",
            "why_insight_required": "Must synthesize identity philosophy with practical examples - no direct connection in data",
            "required_connections": [
                "identity criteria + object persistence",
                "physical vs functional properties",
                "continuity concepts + philosophical analysis"
            ],
            "category": "conceptual_synthesis", 
            "difficulty": "high"
        },
        {
            "id": "measurement_reality_insight",
            "question": "How might the act of measurement itself fundamentally change what we're trying to observe? What does this imply about the nature of reality?",
            "why_insight_required": "Must connect measurement concepts with reality questions - significant conceptual leap required",
            "required_connections": [
                "measurement principles + quantum effects",
                "uncertainty + wave-particle duality",
                "experimental physics + philosophical implications"
            ],
            "category": "paradigm_shift",
            "difficulty": "very_high"
        },
        {
            "id": "emergence_insight",
            "question": "How can deterministic rules at a micro level lead to unpredictable behavior at a macro level? Provide a framework for understanding this paradox.",
            "why_insight_required": "Must synthesize concepts across classical/quantum physics and complex systems - no direct path in data",
            "required_connections": [
                "deterministic behavior + uncertainty principles",
                "micro vs macro scales",
                "predictability + complex systems"
            ],
            "category": "emergence_reasoning",
            "difficulty": "very_high"
        },
        {
            "id": "control_baseline",
            "question": "What are the primary factors that influence daily weather patterns in temperate climates?",
            "why_insight_required": "Control question - should be answerable via direct retrieval",
            "required_connections": ["atmospheric pressure + weather patterns"],
            "category": "control",
            "difficulty": "low"
        }
    ]
    
    return questions

def main():
    print("🔬 Creating True Insight Detection Experiment")
    print("=" * 50)
    
    # Create directories
    for dir_path in [RAW_DIR, PROCESSED_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Generate indirect knowledge base
    print("📚 Creating indirect knowledge base...")
    knowledge_base = create_indirect_knowledge_base()
    
    # Save knowledge base
    knowledge_file = RAW_DIR / "indirect_knowledge.txt"
    with open(knowledge_file, "w", encoding="utf-8") as f:
        for fact in knowledge_base:
            f.write(fact + "\n")
    
    # Create insight-requiring questions
    print("❓ Creating insight-requiring questions...")
    questions = create_insight_requiring_questions()
    
    # Save questions
    questions_file = PROCESSED_DIR / "insight_questions.json"
    with open(questions_file, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    
    # Create metadata
    metadata = {
        "experiment_type": "true_insight_detection",
        "design_principle": "answers_not_directly_available",
        "knowledge_base": {
            "total_facts": len(knowledge_base),
            "direct_answers": 0,
            "requires_synthesis": True
        },
        "questions": {
            "total": len(questions),
            "insight_required": len([q for q in questions if q["category"] != "control"]),
            "control": len([q for q in questions if q["category"] == "control"]),
            "avg_difficulty": "high"
        },
        "validation_criteria": {
            "rag_baseline_should_fail": True,
            "insight_spike_should_succeed": True,
            "requires_cross_domain_synthesis": True
        }
    }
    
    metadata_file = PROCESSED_DIR / "insight_experiment_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Generated true insight detection experiment:")
    print(f"   📚 Knowledge base: {len(knowledge_base)} facts (no direct answers)")
    print(f"   ❓ Questions: {len(questions)} (requiring genuine synthesis)")
    print(f"   📊 Files saved:")
    print(f"      - {knowledge_file}")
    print(f"      - {questions_file}")
    print(f"      - {metadata_file}")
    
    print(f"\n🧠 Experiment validates TRUE insight detection:")
    print(f"   ❌ Standard RAG should struggle (no direct answers)")
    print(f"   ✅ InsightSpike should excel (synthesis capability)")

if __name__ == "__main__":
    main()
