#!/usr/bin/env python3
"""
Setup script for fixed metrics comparison experiment.
Copies necessary data from project root to experiment directory.
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime

def setup_experiment():
    """Setup experiment data following CLAUDE.md guidelines."""
    
    # Get paths
    experiment_root = Path(__file__).parent.parent
    project_root = experiment_root.parent.parent
    
    input_dir = experiment_root / "data" / "input"
    processed_dir = experiment_root / "data" / "processed"
    
    print(f"Setting up experiment at: {experiment_root}")
    print(f"Project root: {project_root}")
    
    # Create sample texts for testing
    sample_texts = {
        "quantum_bio": {
            "title": "Quantum Biology",
            "content": "Recent discoveries suggest quantum mechanics plays a role in biological processes. Photosynthesis uses quantum coherence for efficient energy transfer. Bird navigation may rely on quantum entanglement in cryptochromes. These findings bridge physics and biology in unexpected ways."
        },
        "consciousness": {
            "title": "Consciousness and Information",
            "content": "Consciousness might be understood as integrated information. The brain processes vast amounts of data, creating subjective experience through information integration. This connects neuroscience, information theory, and philosophy of mind."
        },
        "entropy_info": {
            "title": "Entropy and Information Theory",
            "content": "Shannon entropy measures information content. In thermodynamics, entropy represents disorder. The connection between physical and information entropy reveals deep principles about the universe, linking physics with communication theory."
        }
    }
    
    # Create test questions
    test_questions = [
        {
            "id": "q1",
            "question": "How might quantum mechanics relate to biological processes?",
            "domain": "cross-domain",
            "expected_insights": ["quantum coherence in photosynthesis", "quantum effects in bird navigation"]
        },
        {
            "id": "q2", 
            "question": "What connects entropy in physics and information theory?",
            "domain": "cross-domain",
            "expected_insights": ["Shannon entropy", "thermodynamic entropy", "information as physical quantity"]
        },
        {
            "id": "q3",
            "question": "How does information integration relate to consciousness?",
            "domain": "neuroscience-information",
            "expected_insights": ["integrated information theory", "subjective experience from information processing"]
        }
    ]
    
    # Create simple knowledge graph for validation
    knowledge_graph = {
        "nodes": [
            {"id": "quantum_mechanics", "type": "concept", "domain": "physics"},
            {"id": "biology", "type": "concept", "domain": "life_science"},
            {"id": "photosynthesis", "type": "process", "domain": "biology"},
            {"id": "quantum_coherence", "type": "phenomenon", "domain": "physics"},
            {"id": "information", "type": "concept", "domain": "information_theory"},
            {"id": "entropy", "type": "concept", "domain": "physics"},
            {"id": "consciousness", "type": "concept", "domain": "neuroscience"}
        ],
        "edges": [
            {"source": "quantum_coherence", "target": "photosynthesis", "type": "enables"},
            {"source": "quantum_mechanics", "target": "biology", "type": "applies_to"},
            {"source": "entropy", "target": "information", "type": "relates_to"},
            {"source": "information", "target": "consciousness", "type": "underlies"}
        ]
    }
    
    # Copy sample texts from project root if they exist
    project_samples_dir = project_root / "data" / "samples"
    if project_samples_dir.exists():
        print(f"Copying sample texts from {project_samples_dir}")
        for sample_file in project_samples_dir.glob("*.txt"):
            shutil.copy2(sample_file, input_dir / sample_file.name)
            print(f"  Copied: {sample_file.name}")
    
    # Save experiment data
    with open(input_dir / "sample_texts.json", "w") as f:
        json.dump(sample_texts, f, indent=2)
    print("Created sample_texts.json")
    
    with open(input_dir / "test_questions.json", "w") as f:
        json.dump(test_questions, f, indent=2)
    print("Created test_questions.json")
    
    with open(input_dir / "knowledge_graph.json", "w") as f:
        json.dump(knowledge_graph, f, indent=2)
    print("Created knowledge_graph.json")
    
    # Create metadata
    metadata = {
        "setup_date": datetime.now().isoformat(),
        "experiment_name": "fixed_metrics_comparison",
        "data_sources": {
            "sample_texts": "created",
            "test_questions": "created",
            "knowledge_graph": "created",
            "copied_files": [f.name for f in input_dir.glob("*.txt")]
        }
    }
    
    with open(input_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("Created metadata.json")
    
    print("\nExperiment setup complete!")
    print(f"Input data ready at: {input_dir}")

if __name__ == "__main__":
    setup_experiment()