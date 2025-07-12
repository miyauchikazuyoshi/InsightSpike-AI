#!/usr/bin/env python3
"""
Export Experiment Details to CSV
================================

Export questions, responses, insights, and prompts to CSV files
"""

import json
import csv
from pathlib import Path
from datetime import datetime

def load_results():
    """Load both experiment results"""
    results = {}
    
    # Load efficient experiment results
    efficient_files = list(Path('.').glob('experiment_v5_efficient_results_*.json'))
    if efficient_files:
        with open(sorted(efficient_files)[-1], 'r') as f:
            results['efficient'] = json.load(f)
    
    # Load full experiment results
    full_files = list(Path('.').glob('experiment_v5_full_results_*.json'))
    if full_files:
        with open(sorted(full_files)[-1], 'r') as f:
            results['full'] = json.load(f)
    
    return results

def export_questions_and_responses(results, experiment_type='full'):
    """Export questions and responses to CSV"""
    if experiment_type not in results:
        print(f"No {experiment_type} results found")
        return
    
    data = results[experiment_type]
    
    # Prepare CSV data
    csv_data = []
    
    for config_name, config_results in data['configurations'].items():
        for result in config_results:
            row = {
                'Configuration': config_name,
                'Question_ID': result['question_id'],
                'Question_Text': result['question_text'],
                'Question_Type': result['question_type'],
                'Question_Category': result.get('question_category', 'N/A'),
                'Response': result['response'].replace('\n', ' '),
                'Response_Length': result['response_length'],
                'Confidence': result['confidence'],
                'Processing_Time': f"{result['processing_time']:.2f}",
                'Documents_Retrieved': result['documents_retrieved'],
                'Spike_Detected': result.get('spike_detected', 'N/A'),
                'Delta_GED': f"{result.get('delta_ged', 0):.3f}" if result.get('delta_ged') else 'N/A',
                'Delta_IG': f"{result.get('delta_ig', 0):.3f}" if result.get('delta_ig') else 'N/A',
                'Insights_Count': len(result.get('insights', [])) if 'insights' in result else 0,
                'Phases_Integrated': ', '.join(map(str, result.get('phases_integrated', []))) if 'phases_integrated' in result else 'N/A'
            }
            csv_data.append(row)
    
    # Write to CSV
    filename = f'experiment_{experiment_type}_questions_responses_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    
    fieldnames = ['Configuration', 'Question_ID', 'Question_Text', 'Question_Type', 
                  'Question_Category', 'Response', 'Response_Length', 'Confidence', 
                  'Processing_Time', 'Documents_Retrieved', 'Spike_Detected', 
                  'Delta_GED', 'Delta_IG', 'Insights_Count', 'Phases_Integrated']
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"Exported questions and responses to: {filename}")
    return filename

def export_insights_and_prompts(results, experiment_type='full'):
    """Export insights and prompt details to CSV"""
    if experiment_type not in results:
        print(f"No {experiment_type} results found")
        return
    
    data = results[experiment_type]
    
    # Only export InsightSpike results
    if 'insightspike' not in data['configurations']:
        print("No InsightSpike results found")
        return
    
    csv_data = []
    
    for result in data['configurations']['insightspike']:
        if 'insights' in result and result['insights']:
            for idx, insight in enumerate(result['insights']):
                row = {
                    'Question_ID': result['question_id'],
                    'Question_Text': result['question_text'],
                    'Question_Category': result.get('question_category', 'N/A'),
                    'Insight_Number': idx + 1,
                    'Insight_Text': insight,
                    'Spike_Detected': result.get('spike_detected', False),
                    'Delta_GED': f"{result.get('delta_ged', 0):.3f}",
                    'Delta_IG': f"{result.get('delta_ig', 0):.3f}",
                    'Confidence': result['confidence'],
                    'Phases_Integrated': ', '.join(map(str, result.get('phases_integrated', []))) if 'phases_integrated' in result else 'N/A',
                    'Documents_Retrieved': result['documents_retrieved']
                }
                csv_data.append(row)
    
    # Write to CSV
    filename = f'experiment_{experiment_type}_insights_details_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    
    fieldnames = ['Question_ID', 'Question_Text', 'Question_Category', 'Insight_Number', 
                  'Insight_Text', 'Spike_Detected', 'Delta_GED', 'Delta_IG', 
                  'Confidence', 'Phases_Integrated', 'Documents_Retrieved']
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"Exported insights details to: {filename}")
    return filename

def export_knowledge_base_usage(experiment_type='full'):
    """Export knowledge base and its usage in experiments"""
    # Define knowledge bases
    if experiment_type == 'efficient':
        knowledge_base = [
            {"id": "K1", "text": "Entropy measures disorder in thermodynamic systems.", "domain": "thermodynamics"},
            {"id": "K2", "text": "The second law states that entropy always increases in isolated systems.", "domain": "thermodynamics"},
            {"id": "K3", "text": "Shannon entropy quantifies uncertainty in information.", "domain": "information"},
            {"id": "K4", "text": "Information processing requires energy according to Landauer's principle.", "domain": "information"},
            {"id": "K5", "text": "Living systems maintain low entropy by consuming energy.", "domain": "biology"},
            {"id": "K6", "text": "Organisms export entropy to their environment through metabolism.", "domain": "biology"}
        ]
    else:  # full
        knowledge_base = [
            # Phase 1
            {"id": "K1", "text": "Entropy is a measure of disorder or randomness in a system.", "domain": "fundamentals", "phase": 1},
            {"id": "K2", "text": "Energy is the capacity to do work and cannot be created or destroyed.", "domain": "fundamentals", "phase": 1},
            {"id": "K3", "text": "Information represents the reduction of uncertainty about a system's state.", "domain": "fundamentals", "phase": 1},
            # Phase 2
            {"id": "K4", "text": "Shannon entropy H(X) = -Σ p(x) log p(x) quantifies information content.", "domain": "mathematics", "phase": 2},
            {"id": "K5", "text": "Graph theory studies relationships between objects using nodes and edges.", "domain": "mathematics", "phase": 2},
            {"id": "K6", "text": "Probability distributions describe the likelihood of different outcomes.", "domain": "mathematics", "phase": 2},
            # Phase 3
            {"id": "K7", "text": "The second law of thermodynamics states that entropy always increases in isolated systems.", "domain": "physics", "phase": 3},
            {"id": "K8", "text": "Maxwell's demon thought experiment links information processing to thermodynamics.", "domain": "physics", "phase": 3},
            {"id": "K9", "text": "Landauer's principle: erasing one bit of information releases kT ln(2) of heat.", "domain": "physics", "phase": 3},
            # Phase 4
            {"id": "K10", "text": "Living organisms maintain low internal entropy by consuming free energy.", "domain": "biology", "phase": 4},
            {"id": "K11", "text": "DNA encodes hereditary information in a four-letter molecular alphabet.", "domain": "biology", "phase": 4},
            {"id": "K12", "text": "Metabolism allows organisms to export entropy to their environment.", "domain": "biology", "phase": 4},
            # Phase 5
            {"id": "K13", "text": "Information processing requires energy expenditure according to thermodynamic limits.", "domain": "information", "phase": 5},
            {"id": "K14", "text": "Error correction codes add redundancy to protect information from corruption.", "domain": "information", "phase": 5},
            {"id": "K15", "text": "Compression algorithms reduce data size by removing statistical redundancy.", "domain": "information", "phase": 5}
        ]
    
    # Write to CSV
    filename = f'experiment_{experiment_type}_knowledge_base_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    
    fieldnames = ['Knowledge_ID', 'Text', 'Domain', 'Phase'] if experiment_type == 'full' else ['Knowledge_ID', 'Text', 'Domain']
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for kb in knowledge_base:
            row = {
                'Knowledge_ID': kb['id'],
                'Text': kb['text'],
                'Domain': kb['domain']
            }
            if experiment_type == 'full' and 'phase' in kb:
                row['Phase'] = kb['phase']
            writer.writerow(row)
    
    print(f"Exported knowledge base to: {filename}")
    return filename

def create_prompt_examples():
    """Create examples of enhanced prompts used in the experiment"""
    examples = []
    
    # Example 1: Factual question (no spike)
    examples.append({
        'Question_ID': 'Q1',
        'Question': 'What is entropy?',
        'Prompt_Type': 'InsightSpike_NoSpike',
        'Enhanced_Prompt': """You are an AI assistant with access to structured knowledge and reasoning capabilities.

Analyze the provided information and generate a comprehensive response that goes beyond simple retrieval.

## 💡 Discovered Insights

### Integrated Understanding:
1. The knowledge elements form an interconnected system with mutually reinforcing relationships

## 📚 Supporting Knowledge
1. Entropy is a measure of disorder or randomness in a system.
2. Shannon entropy H(X) = -Σ p(x) log p(x) quantifies information content.
3. The second law of thermodynamics states that entropy always increases in isolated systems.

## 📊 Reasoning Analysis
• Structural simplification: 15%
• Information integration: 18%
• Reasoning confidence: 55%

## ❓ Question
"What is entropy?"

## 📝 Your Task
Based on the available information, provide the best possible answer. Note any limitations in the current knowledge."""
    })
    
    # Example 2: Cross-domain question (spike detected)
    examples.append({
        'Question_ID': 'Q4',
        'Question': 'How does information relate to energy?',
        'Prompt_Type': 'InsightSpike_WithSpike',
        'Enhanced_Prompt': """🧠 INSIGHT SPIKE DETECTED - Breakthrough Understanding Available

You are processing a query where graph neural network analysis has discovered significant conceptual connections through message-passing algorithms. Your response should emphasize these emergent insights.

## 💡 Discovered Insights

**🎯 Breakthrough Pattern Detected:**
Multiple knowledge pathways have converged, revealing hidden patterns and creating new possibilities for understanding.

### Integrated Understanding:
1. Previously separate concepts of information and energy are fundamentally connected through shared principles
2. Multiple knowledge fragments have been unified into a simpler, more elegant framework that explains more with less
3. The integration reveals emergent properties that were not apparent when considering each concept in isolation

### Cross-Domain Synthesis:
• Thermodynamic entropy and information entropy are mathematically equivalent, revealing deep unity between physics and information theory

## 📚 Supporting Knowledge
1. Information processing requires energy according to Landauer's principle.
2. Landauer's principle: erasing one bit of information releases kT ln(2) of heat.
3. Maxwell's demon thought experiment links information processing to thermodynamics.
4. Energy is the capacity to do work and cannot be created or destroyed.
5. Information represents the reduction of uncertainty about a system's state.

## 📊 Reasoning Analysis
• Structural simplification: 42%
• Information integration: 38%
• Reasoning confidence: 75%
• Status: **High-quality synthesis achieved**

## ❓ Question
"How does information relate to energy?"

## 📝 Your Task
Explain the breakthrough insight discovered through knowledge integration. Emphasize how the unified understanding transcends individual facts."""
    })
    
    # Example 3: Abstract question (high spike)
    examples.append({
        'Question_ID': 'Q7',
        'Question': 'What is the relationship between order and information?',
        'Prompt_Type': 'InsightSpike_HighSpike',
        'Enhanced_Prompt': """🧠 INSIGHT SPIKE DETECTED - Breakthrough Understanding Available

You are processing a query where graph neural network analysis has discovered significant conceptual connections through message-passing algorithms. Your response should emphasize these emergent insights.

## 💡 Discovered Insights

**🎯 Breakthrough Pattern Detected:**
A fundamental reorganization of conceptual structure has occurred. Previously disconnected ideas have merged into a unified theory that dramatically simplifies our understanding while increasing explanatory power.

### Integrated Understanding:
1. Multiple knowledge fragments have been unified into a simpler, more elegant framework that explains more with less
2. The integration reveals emergent properties that were not apparent when considering each concept in isolation

### Cross-Domain Synthesis:
• Living systems create local order by processing information and exporting entropy, demonstrating how information processing requires energy

## 📚 Supporting Knowledge
1. Entropy is a measure of disorder or randomness in a system.
2. Information represents the reduction of uncertainty about a system's state.
3. Living organisms maintain low internal entropy by consuming free energy.

## 📊 Reasoning Analysis
• Structural simplification: 65%
• Information integration: 52%
• Reasoning confidence: 88%
• Status: **High-quality synthesis achieved**

## ❓ Question
"What is the relationship between order and information?"

## 📝 Your Task
Explain the breakthrough insight discovered through knowledge integration. Emphasize how the unified understanding transcends individual facts."""
    })
    
    # Write to CSV
    filename = f'experiment_prompt_examples_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    
    fieldnames = ['Question_ID', 'Question', 'Prompt_Type', 'Enhanced_Prompt']
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(examples)
    
    print(f"Exported prompt examples to: {filename}")
    return filename

def main():
    """Export all experiment details to CSV files"""
    print("Loading experiment results...")
    results = load_results()
    
    print("\nExporting data...")
    
    # Export efficient experiment
    if 'efficient' in results:
        print("\n--- Efficient Experiment ---")
        export_questions_and_responses(results, 'efficient')
        export_insights_and_prompts(results, 'efficient')
        export_knowledge_base_usage('efficient')
    
    # Export full experiment
    if 'full' in results:
        print("\n--- Full Experiment ---")
        export_questions_and_responses(results, 'full')
        export_insights_and_prompts(results, 'full')
        export_knowledge_base_usage('full')
    
    # Export prompt examples
    print("\n--- Prompt Examples ---")
    create_prompt_examples()
    
    print("\n✅ All exports complete!")

if __name__ == "__main__":
    main()