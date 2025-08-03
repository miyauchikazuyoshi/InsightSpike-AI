#!/usr/bin/env python3
"""
Run small-scale test with 2-hop GED/IG evaluation
"""

import json
import time
import subprocess

# Create small test data
knowledge_data = {
    "knowledge_entries": [
        {
            "id": f"k{i:03d}", 
            "content": f"Knowledge entry {i} about {['physics', 'biology', 'chemistry'][i%3]} fundamental particles interact through force carriers",
            "tags": [["physics", "biology", "chemistry"][i%3]],
            "difficulty": "basic",
            "related_concepts": ["particles", "energy", "matter"]
        } 
        for i in range(20)
    ]
}

questions_data = {
    "questions": [
        {"id": f"q{i:03d}", "question": f"Test question {i}?", "difficulty": "easy"} 
        for i in range(10)
    ]
}

# Save test data
with open('data/input/knowledge_base/test_knowledge.json', 'w') as f:
    json.dump(knowledge_data, f, indent=2)

with open('data/input/questions/test_questions.json', 'w') as f:
    json.dump(questions_data, f, indent=2)

# Update config to use test data
import yaml
with open('experiment_config_2hop.yaml', 'r') as f:
    config = yaml.safe_load(f)

config['data']['knowledge_base_path'] = 'data/input/knowledge_base/test_knowledge.json'
config['data']['questions_path'] = 'data/input/questions/test_questions.json'
config['parameters']['total_knowledge'] = 20
config['parameters']['total_questions'] = 10

with open('test_config_2hop.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print("Running small-scale test with 2-hop evaluation...")
start_time = time.time()

# Run the experiment
result = subprocess.run(
    ['poetry', 'run', 'python', 'src/run_experiment.py', '--config', 'test_config_2hop.yaml'],
    capture_output=True,
    text=True
)

elapsed = time.time() - start_time
print(f"\nCompleted in {elapsed:.1f} seconds")

# Show output
if result.returncode != 0:
    print(f"\nError (exit code {result.returncode}):")
    print(result.stderr)
else:
    print("\nOutput:")
    print(result.stdout)

# Check for results
import os
if os.path.exists('results/outputs/experiment_results.csv'):
    import pandas as pd
    df = pd.read_csv('results/outputs/experiment_results.csv')
    print(f"\nResults summary:")
    print(f"Total questions: {len(df)}")
    print(f"Spike detection rate: {df['has_spike'].mean():.1%}")
    print(f"\nGED/IG values:")
    print(df[['question_id', 'ged_value', 'ig_value']].head(10))