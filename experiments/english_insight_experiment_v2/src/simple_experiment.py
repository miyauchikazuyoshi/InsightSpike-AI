#!/usr/bin/env python3
"""
Simplified English Insight Experiment
====================================

A simplified version that tests the core InsightSpike functionality
without complex datastore operations.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_knowledge_base(data_dir: Path) -> List[Dict[str, Any]]:
    """Load knowledge base from JSON."""
    kb_path = data_dir / "input" / "english_knowledge_base.json"
    with open(kb_path, 'r') as f:
        data = json.load(f)
    return data['episodes']


def main():
    """Run simplified experiment."""
    experiment_dir = Path(__file__).parent.parent
    data_dir = experiment_dir / "data"
    results_dir = experiment_dir / "results"
    
    # Load knowledge base
    logger.info("Loading knowledge base...")
    knowledge_base = load_knowledge_base(data_dir)
    logger.info(f"Loaded {len(knowledge_base)} episodes")
    
    # Questions to test
    questions = [
        "What is the relationship between energy and information?",
        "Why does consciousness emerge?",
        "What is entropy?",
    ]
    
    # Phase analysis
    phase_names = {
        1: "Basic Concepts",
        2: "Relationships", 
        3: "Deep Integration",
        4: "Emergent Insights",
        5: "Integration and Circulation"
    }
    
    # Count episodes per phase
    phase_counts = {}
    for episode in knowledge_base:
        phase = episode['phase']
        phase_name = phase_names.get(phase, f"Phase {phase}")
        phase_counts[phase_name] = phase_counts.get(phase_name, 0) + 1
    
    logger.info("Phase distribution:")
    for phase, count in sorted(phase_counts.items()):
        logger.info(f"  {phase}: {count} episodes")
    
    # Simple analysis results
    results = {
        "experiment_date": datetime.now().isoformat(),
        "knowledge_base_stats": {
            "total_episodes": len(knowledge_base),
            "phase_distribution": phase_counts
        },
        "questions_tested": questions,
        "sample_episodes": []
    }
    
    # Sample some episodes from each phase
    for phase_num in range(1, 6):
        phase_episodes = [ep for ep in knowledge_base if ep['phase'] == phase_num]
        if phase_episodes:
            sample = phase_episodes[0]
            results["sample_episodes"].append({
                "phase": phase_num,
                "phase_name": phase_names[phase_num],
                "text": sample['text'],
                "id": sample['id']
            })
    
    # Save results
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "outputs" / "simple_experiment_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")
    
    # Generate simple report
    report_lines = [
        "# English Insight Experiment - Simple Analysis",
        "",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Knowledge Base Overview",
        "",
        f"Total episodes: {len(knowledge_base)}",
        "",
        "### Phase Distribution:",
        ""
    ]
    
    for phase, count in sorted(phase_counts.items()):
        report_lines.append(f"- {phase}: {count} episodes")
    
    report_lines.extend([
        "",
        "## Sample Episodes",
        ""
    ])
    
    for sample in results["sample_episodes"]:
        report_lines.extend([
            f"### {sample['phase_name']} (Phase {sample['phase']})",
            f"ID: {sample['id']}",
            f"Text: {sample['text']}",
            ""
        ])
    
    report_lines.extend([
        "## Questions for Testing",
        ""
    ])
    
    for i, q in enumerate(questions, 1):
        report_lines.append(f"{i}. {q}")
    
    report_lines.extend([
        "",
        "## Next Steps",
        "",
        "1. Implement proper embedding and retrieval",
        "2. Test with actual InsightSpike agent",
        "3. Compare with baseline approaches",
        "4. Generate visualizations"
    ])
    
    report_path = results_dir / "simple_analysis_report.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Report saved to {report_path}")
    logger.info("Simple experiment completed successfully!")


if __name__ == "__main__":
    main()