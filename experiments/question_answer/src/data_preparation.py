#!/usr/bin/env python3
"""
Data Preparation for Question-Answer Experiment

This script prepares the knowledge base and question sets for the experiment.
It can generate additional synthetic data if needed.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any
import argparse


class DataPreparation:
    """Prepare data for question-answer experiment."""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.input_dir = self.base_dir / "data" / "input"
        self.processed_dir = self.base_dir / "data" / "processed"
        
    def expand_knowledge_base(self, target_size: int = 100):
        """Expand knowledge base to target size with variations."""
        # Load existing knowledge
        kb_path = self.input_dir / "knowledge_base" / "sample_knowledge.json"
        with open(kb_path, 'r') as f:
            data = json.load(f)
            
        existing = data['knowledge_entries']
        
        # Templates for generating variations
        templates = {
            'science': [
                "The principle of {} states that {}",
                "In {}, we observe that {}",
                "{} is a process where {}",
                "The relationship between {} and {} is characterized by {}"
            ],
            'daily_life': [
                "When {}, it's important to {}",
                "The technique of {} involves {}",
                "{} can be optimized by {}",
                "Common practice in {} includes {}"
            ],
            'abstract': [
                "The concept of {} relates to {}",
                "{} can be understood as {}",
                "From a {} perspective, {}",
                "The theory of {} suggests that {}"
            ]
        }
        
        # Generate additional entries if needed
        while len(existing) < target_size:
            # Create variation from existing entry
            base_entry = random.choice(existing[:10])  # Use only original entries
            category = base_entry['category']
            
            if category in templates:
                template = random.choice(templates[category])
                # Simple variation generation (in real experiment, would be more sophisticated)
                new_content = template.format(
                    base_entry['subcategory'],
                    "various factors interact"
                )
                
                new_entry = {
                    'id': f"kb_{len(existing) + 1:03d}",
                    'category': category,
                    'subcategory': base_entry['subcategory'],
                    'content': new_content,
                    'related_concepts': base_entry['related_concepts'][:2]
                }
                
                existing.append(new_entry)
                
        # Save expanded knowledge base
        output_path = self.input_dir / "knowledge_base" / f"knowledge_base_{target_size}.json"
        with open(output_path, 'w') as f:
            json.dump({
                'domain': 'mixed',
                'size': target_size,
                'knowledge_entries': existing[:target_size]
            }, f, indent=2)
            
        print(f"Created knowledge base with {target_size} entries at {output_path}")
        
    def validate_questions(self):
        """Validate question set against knowledge base."""
        # Load questions
        q_path = self.input_dir / "questions" / "question_set.json"
        with open(q_path, 'r') as f:
            questions = json.load(f)
            
        # Load knowledge base
        kb_path = self.input_dir / "knowledge_base" / "sample_knowledge.json"
        with open(kb_path, 'r') as f:
            kb_data = json.load(f)
            
        kb_ids = {entry['id'] for entry in kb_data['knowledge_entries']}
        
        # Validate each question
        issues = []
        for category, cat_data in questions['question_categories'].items():
            for q in cat_data['questions']:
                # Check if expected knowledge exists
                expected = q.get('expected_knowledge', [])
                missing = [kb_id for kb_id in expected if kb_id not in kb_ids]
                
                if missing:
                    issues.append({
                        'question_id': q['id'],
                        'missing_knowledge': missing
                    })
                    
        if issues:
            print(f"Found {len(issues)} validation issues:")
            for issue in issues:
                print(f"  - {issue['question_id']}: Missing {issue['missing_knowledge']}")
        else:
            print("All questions validated successfully!")
            
        return len(issues) == 0
        
    def create_evaluation_template(self):
        """Create template for human evaluation."""
        template = {
            'evaluator': '',
            'date': '',
            'evaluations': {}
        }
        
        # Load questions to create evaluation structure
        q_path = self.input_dir / "questions" / "question_set.json"
        with open(q_path, 'r') as f:
            questions = json.load(f)
            
        for category, cat_data in questions['question_categories'].items():
            template['evaluations'][category] = []
            
            for q in cat_data['questions']:
                eval_entry = {
                    'question_id': q['id'],
                    'question': q['question'],
                    'baseline_response': '',
                    'insightspike_response': '',
                    'evaluation': {
                        'baseline': {
                            'correctness': 0,  # 1-5 scale
                            'completeness': 0,
                            'clarity': 0
                        },
                        'insightspike': {
                            'correctness': 0,
                            'completeness': 0,
                            'clarity': 0,
                            'creativity': 0,
                            'insight_quality': 0
                        },
                        'preferred_response': '',  # 'baseline' or 'insightspike'
                        'notes': ''
                    }
                }
                template['evaluations'][category].append(eval_entry)
                
        # Save template
        output_path = self.processed_dir / "evaluation_template.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(template, f, indent=2)
            
        print(f"Created evaluation template at {output_path}")
        
    def prepare_all(self):
        """Run all preparation steps."""
        print("=== Data Preparation ===")
        
        # Validate questions
        print("\n1. Validating questions...")
        self.validate_questions()
        
        # Create different sized knowledge bases
        print("\n2. Creating knowledge bases...")
        for size in [100, 500, 1000]:
            self.expand_knowledge_base(size)
            
        # Create evaluation template
        print("\n3. Creating evaluation template...")
        self.create_evaluation_template()
        
        print("\n✓ Data preparation complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Prepare data for experiment')
    parser.add_argument(
        '--prepare-all',
        action='store_true',
        help='Run all preparation steps'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate existing data'
    )
    parser.add_argument(
        '--kb-size',
        type=int,
        help='Create knowledge base of specific size'
    )
    
    args = parser.parse_args()
    
    prep = DataPreparation()
    
    if args.prepare_all:
        prep.prepare_all()
    elif args.validate_only:
        prep.validate_questions()
    elif args.kb_size:
        prep.expand_knowledge_base(args.kb_size)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()