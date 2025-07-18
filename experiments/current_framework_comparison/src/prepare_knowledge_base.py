#!/usr/bin/env python3
"""
Prepare knowledge base for current framework experiment
Convert english_knowledge_base.json to InsightSpike episode format
"""

import json
from pathlib import Path
from datetime import datetime
import uuid


def convert_to_episodes():
    """Convert english knowledge base to InsightSpike episode format"""
    
    # Load original knowledge base
    original_path = Path(__file__).parent.parent.parent / "english_insight_experiment/data/input/english_knowledge_base.json"
    with open(original_path, 'r', encoding='utf-8') as f:
        original_kb = json.load(f)
    
    # Phase mapping for semantic categories
    phase_to_category = {
        1: "basic_concepts",
        2: "relationships", 
        3: "deep_integration",
        4: "emergent_insights",
        5: "integration_circulation"
    }
    
    # Convert to episodes - using the format expected by InsightSpike
    episodes = []
    for item in original_kb['episodes']:
        # Create episode in the format expected by the framework
        episode = {
            'id': str(uuid.uuid4()),
            'content': item['text'],
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'original_id': item['id'],
                'phase': item['phase'],
                'category': phase_to_category.get(item['phase'], 'unknown'),
                'source': 'english_knowledge_base',
                'importance': 0.8 if item['phase'] >= 3 else 0.6,  # Higher importance for integration phases
                'c_value': 0.7 if item['phase'] >= 3 else 0.5  # Initial C-value based on phase
            }
        }
        episodes.append(episode)
    
    # Create InsightSpike knowledge base format
    insightspike_kb = {
        "metadata": {
            "created": datetime.now().isoformat(),
            "source": "english_insight_experiment",
            "total_episodes": len(episodes),
            "description": "English knowledge base converted for InsightSpike framework comparison"
        },
        "episodes": episodes
    }
    
    # Save to new location
    output_path = Path(__file__).parent.parent / "data/input/insightspike_knowledge_base.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(insightspike_kb, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Converted {len(episodes)} episodes to InsightSpike format")
    print(f"📁 Saved to: {output_path}")
    
    # Also save the original questions for reference
    questions = [
        "What is the relationship between energy and information?",
        "Why does consciousness emerge?",
        "What is the mechanism of creativity at the edge of chaos?",
        "What is entropy?",
        "Can you explain quantum entanglement?",
        "Is there a principle that unifies all phenomena?"
    ]
    
    questions_data = {
        "questions": questions,
        "metadata": {
            "source": "english_insight_experiment",
            "description": "Original 6 questions for comparison"
        }
    }
    
    questions_path = Path(__file__).parent.parent / "data/input/test_questions.json"
    with open(questions_path, 'w', encoding='utf-8') as f:
        json.dump(questions_data, f, ensure_ascii=False, indent=2)
    
    print(f"📝 Saved {len(questions)} test questions")
    
    return len(episodes)


if __name__ == "__main__":
    convert_to_episodes()