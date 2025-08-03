#!/usr/bin/env python3
"""
Small-scale test with Claude API to verify spike detection behavior
"""

import sys
import logging
import yaml
import json
from pathlib import Path
from datetime import datetime
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from insightspike.implementations.agents.main_agent import MainAgent
from insightspike.config.converter import ConfigConverter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_small_test():
    """Run test with first 5 questions"""
    
    # Load config
    with open('../experiment_config.yaml', 'r') as f:
        experiment_config = yaml.safe_load(f)
    
    # Convert to legacy format
    legacy_config = ConfigConverter.experimental_config_to_legacy(experiment_config)
    
    # Initialize agent
    logger.info("Initializing MainAgent with Claude API...")
    agent = MainAgent(legacy_config)
    
    # Load knowledge base (first 10 for quick test)
    with open('../data/input/knowledge_base/knowledge_500.json', 'r') as f:
        knowledge_data = json.load(f)
    
    logger.info("Loading first 10 knowledge entries...")
    for i, entry in enumerate(knowledge_data[:10]):
        agent.add_knowledge(entry['text'])
        logger.info(f"Loaded knowledge {i+1}/10")
    
    # Load questions (first 5)
    with open('../data/input/questions/questions_100.json', 'r') as f:
        questions_data = json.load(f)
    
    results = []
    
    # Test first 5 questions
    for i, q_data in enumerate(questions_data[:5]):
        question = q_data['text']
        logger.info(f"\n{'='*60}")
        logger.info(f"Question {i+1}: {question}")
        logger.info(f"Difficulty: {q_data['difficulty']}")
        
        start_time = time.time()
        
        # Process question
        result = agent.process_question(question)
        
        processing_time = time.time() - start_time
        
        # Extract results
        has_spike = result.has_spike if hasattr(result, 'has_spike') else result.get('has_spike', False)
        response = result.response if hasattr(result, 'response') else result.get('response', '')
        
        # Get geDIG metrics if available
        gedig_metrics = None
        if hasattr(result, 'gedig_metrics'):
            gedig_metrics = result.gedig_metrics
        elif isinstance(result, dict) and 'gedig_metrics' in result:
            gedig_metrics = result['gedig_metrics']
        
        logger.info(f"Has Spike: {has_spike}")
        if gedig_metrics:
            logger.info(f"geDIG Reward: {gedig_metrics.get('gedig', 'N/A')}")
            logger.info(f"GED: {gedig_metrics.get('ged', 'N/A')}")
            logger.info(f"IG: {gedig_metrics.get('ig', 'N/A')}")
            if 'normalized_metrics' in gedig_metrics:
                norm = gedig_metrics['normalized_metrics']
                logger.info(f"Normalized GED: {norm.get('ged_normalized', 'N/A')}")
                logger.info(f"IG Z-score: {norm.get('ig_z_score', 'N/A')}")
                logger.info(f"Conservation Sum: {norm.get('conservation_sum', 'N/A')}")
        
        logger.info(f"Response: {response[:200]}...")
        logger.info(f"Processing time: {processing_time:.2f}s")
        
        # Save result
        results.append({
            'question_id': q_data['id'],
            'question': question,
            'difficulty': q_data['difficulty'],
            'has_spike': has_spike,
            'response': response,
            'processing_time': processing_time,
            'gedig_metrics': gedig_metrics
        })
        
        # Wait a bit to avoid rate limiting
        time.sleep(1)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'../results/claude_test_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_file}")
    
    # Summary
    spike_count = sum(1 for r in results if r['has_spike'])
    logger.info(f"\nSummary:")
    logger.info(f"Total questions: {len(results)}")
    logger.info(f"Spike detections: {spike_count}")
    logger.info(f"Spike rate: {spike_count/len(results)*100:.1f}%")

if __name__ == "__main__":
    run_small_test()