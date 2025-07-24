#!/usr/bin/env python3
"""
Prepare data for the v2 experiment.
Generate questions and copy knowledge base.
"""

import json
import shutil
from pathlib import Path

# Add project root to path
import sys
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from question_generator import ExpandedQuestionGenerator


def prepare_experiment_data():
    """Prepare all data for the experiment."""
    experiment_dir = Path(__file__).parent.parent
    data_dir = experiment_dir / "data"
    
    # Create directory structure
    for subdir in ["input", "processed"]:
        (data_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    print("=== Preparing Experiment Data ===")
    
    # 1. Copy knowledge base from project root
    print("\n1. Copying knowledge base...")
    source_kb = project_root / "data" / "synthetic" / "mathematical_associations.json"
    dest_kb = data_dir / "input" / "knowledge_base.json"
    
    if source_kb.exists():
        shutil.copy2(source_kb, dest_kb)
        print(f"   Copied from: {source_kb}")
        print(f"   Copied to: {dest_kb}")
        
        # Show knowledge base stats
        with open(dest_kb, 'r') as f:
            kb_data = json.load(f)
        print(f"   Knowledge items: {len(kb_data.get('associations', []))}")
    else:
        print(f"   WARNING: Knowledge base not found at {source_kb}")
        print("   Creating minimal knowledge base...")
        
        # Create minimal knowledge base
        kb_data = {
            "associations": [
                {"id": "math_1", "text": "Addition is combining numbers to get a sum"},
                {"id": "math_2", "text": "Multiplication is repeated addition"},
                {"id": "math_3", "text": "Division is splitting into equal parts"},
                {"id": "math_4", "text": "Fractions represent parts of a whole"},
                {"id": "math_5", "text": "Decimals are another way to write fractions"},
                {"id": "sci_1", "text": "Water freezes at 0 degrees Celsius"},
                {"id": "sci_2", "text": "Plants need sunlight for photosynthesis"},
                {"id": "sci_3", "text": "Gravity pulls objects toward Earth"},
                {"id": "ling_1", "text": "Nouns name people, places, or things"},
                {"id": "ling_2", "text": "Verbs describe actions or states"}
            ]
        }
        
        with open(dest_kb, 'w') as f:
            json.dump(kb_data, f, indent=2)
        print(f"   Created minimal knowledge base with {len(kb_data['associations'])} items")
    
    # 2. Generate test questions
    print("\n2. Generating test questions...")
    generator = ExpandedQuestionGenerator(seed=42)
    
    # Generate 100 questions
    questions = generator.generate_questions(n_easy=25, n_medium=50, n_hard=25)
    
    # Save questions
    questions_path = data_dir / "input" / "test_questions.json"
    generator.save_questions(questions, str(questions_path))
    
    print(f"   Generated {len(questions)} questions")
    print(f"   Saved to: {questions_path}")
    
    # Show distribution
    easy_count = len([q for q in questions if q.difficulty == 'easy'])
    medium_count = len([q for q in questions if q.difficulty == 'medium'])
    hard_count = len([q for q in questions if q.difficulty == 'hard'])
    
    print(f"\n   Difficulty distribution:")
    print(f"   - Easy: {easy_count}")
    print(f"   - Medium: {medium_count}")
    print(f"   - Hard: {hard_count}")
    
    categories = set(q.category for q in questions)
    print(f"\n   Categories: {', '.join(sorted(categories))}")
    
    # 3. Show sample questions
    print("\n3. Sample questions:")
    for difficulty in ['easy', 'medium', 'hard']:
        sample = next(q for q in questions if q.difficulty == difficulty)
        print(f"\n   {difficulty.upper()}: {sample.text}")
        print(f"   Category: {sample.category}")
        print(f"   Expected associations: {', '.join(sample.expected_associations[:3])}...")
    
    print("\n=== Data Preparation Complete ===")
    print(f"\nReady to run experiment with:")
    print(f"  - Knowledge base: {dest_kb}")
    print(f"  - Test questions: {questions_path}")
    print(f"\nTo run the experiment:")
    print(f"  cd {experiment_dir}")
    print(f"  poetry run python src/run_experiment.py")


if __name__ == "__main__":
    prepare_experiment_data()