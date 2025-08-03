#!/usr/bin/env python3
"""
Convert knowledge format for the experiment
"""

import json


def convert_knowledge_format(input_file, output_file):
    """Convert knowledge_entries format to simple list format"""
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Extract knowledge entries
    knowledge_list = []
    for entry in data.get('knowledge_entries', []):
        knowledge_list.append({
            'id': entry['id'],
            'content': entry['content'],
            'category': entry.get('category', 'general')
        })
    
    # Save in new format
    with open(output_file, 'w') as f:
        json.dump(knowledge_list, f, indent=2)
    
    print(f"Converted {len(knowledge_list)} entries")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    convert_knowledge_format(
        "data/input/knowledge_base/sample_knowledge.json",
        "data/input/knowledge_base/sample_knowledge_converted.json"
    )