#!/usr/bin/env python3
"""
Generate CSV comparison of actual responses from InsightSpike vs Traditional RAG
"""

import csv
import json
import os
from datetime import datetime
from typing import List, Dict
import pandas as pd

# Import the demo classes
from simple_baseline_demo import SimpleBaselineDemo


def generate_detailed_comparison():
    """Generate detailed comparison with actual responses"""
    
    # Run the demo
    demo = SimpleBaselineDemo()
    print("Running comparison to generate detailed responses...")
    
    # Extended test cases with more variety
    test_cases = [
        {
            "query": "What is the relationship between sleep and memory consolidation?",
            "documents": [
                "Sleep plays a crucial role in memory consolidation, particularly during REM stages.",
                "Studies show that REM sleep is associated with procedural memory enhancement.",
                "Memory formation occurs during specific sleep stages, with slow-wave sleep crucial for declarative memory.",
                "Lack of sleep significantly impairs both short-term and long-term memory performance.",
                "Exercise has been shown to improve sleep quality and memory function.",
                "The hippocampus transfers memories to the cortex during deep sleep phases."
            ],
            "expected_insights": ["sleep-memory connection", "REM importance", "hippocampus role"]
        },
        {
            "query": "How does physical exercise affect cognitive function and brain health?",
            "documents": [
                "Exercise increases BDNF (brain-derived neurotrophic factor) production in the hippocampus.",
                "Regular physical activity improves cognitive function across all age groups.",
                "Aerobic exercise enhances neuroplasticity and promotes neurogenesis.",
                "Studies link regular exercise to reduced risk of dementia and Alzheimer's disease.",
                "Exercise improves blood flow to the brain, delivering oxygen and nutrients.",
                "Physical activity reduces inflammation and oxidative stress in brain tissue."
            ],
            "expected_insights": ["BDNF mechanism", "neuroplasticity", "dementia prevention"]
        },
        {
            "query": "What are the mechanisms of neuroplasticity in learning?",
            "documents": [
                "Neuroplasticity involves structural and functional changes in neural connections.",
                "Long-term potentiation (LTP) strengthens synaptic connections during learning.",
                "Dendritic spine formation increases with repeated neural activation.",
                "Myelination of axons improves signal transmission speed in frequently used pathways.",
                "Neurotrophic factors like BDNF promote synaptic plasticity.",
                "Critical periods exist where plasticity is heightened for specific types of learning."
            ],
            "expected_insights": ["LTP mechanism", "structural changes", "critical periods"]
        },
        {
            "query": "How do stress hormones impact memory and learning?",
            "documents": [
                "Cortisol, the primary stress hormone, can impair memory formation at high levels.",
                "Acute stress can enhance memory consolidation for emotionally significant events.",
                "Chronic stress damages hippocampal neurons and reduces neurogenesis.",
                "The amygdala becomes hyperactive under stress, affecting emotional memory.",
                "Stress hormones interact with neurotransmitters to modulate synaptic plasticity.",
                "Meditation and exercise can reduce cortisol levels and protect memory function."
            ],
            "expected_insights": ["cortisol effects", "acute vs chronic stress", "protective factors"]
        },
        {
            "query": "What is the role of sleep in creativity and problem-solving?",
            "documents": [
                "REM sleep facilitates creative problem-solving by allowing novel connections.",
                "Dreams during REM can lead to insights and 'eureka' moments upon waking.",
                "Sleep deprivation significantly impairs divergent thinking abilities.",
                "The default mode network is active during sleep, promoting creative associations.",
                "Memory replay during sleep can reorganize information in innovative ways.",
                "Artists and scientists often report breakthrough ideas after sleep."
            ],
            "expected_insights": ["REM-creativity link", "memory reorganization", "default mode network"]
        }
    ]
    
    # Collect results
    comparison_data = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nProcessing test case {i}/{len(test_cases)}: {test_case['query'][:50]}...")
        
        # Run both systems
        rag_result = demo.simulate_traditional_rag(
            test_case["query"], 
            test_case["documents"]
        )
        
        spike_result = demo.simulate_insightspike(
            test_case["query"], 
            test_case["documents"]
        )
        
        # Create comparison row
        row = {
            "test_id": i,
            "query": test_case["query"],
            "num_documents": len(test_case["documents"]),
            "expected_insights": ", ".join(test_case["expected_insights"]),
            
            # Traditional RAG results
            "rag_response": rag_result["response"],
            "rag_time_ms": round(rag_result["time"] * 1000, 2),
            "rag_confidence": round(rag_result["confidence"], 3),
            "rag_docs_used": rag_result["docs_retrieved"],
            
            # InsightSpike results
            "spike_response": spike_result["response"],
            "spike_time_ms": round(spike_result["time"] * 1000, 2),
            "spike_confidence": round(spike_result["confidence"], 3),
            "spike_insight_detected": spike_result["insight_detected"],
            "spike_insight_type": spike_result["insight_type"] or "none",
            "spike_delta_ig": round(spike_result["metrics"]["delta_ig"], 3) if spike_result["insight_detected"] else 0,
            "spike_delta_ged": round(spike_result["metrics"]["delta_ged"], 3) if spike_result["insight_detected"] else 0,
            
            # Comparison metrics
            "confidence_improvement": round(spike_result["confidence"] - rag_result["confidence"], 3),
            "response_length_ratio": round(len(spike_result["response"]) / len(rag_result["response"]), 2),
            "processing_overhead_ms": round((spike_result["time"] - rag_result["time"]) * 1000, 2)
        }
        
        comparison_data.append(row)
    
    # Save to CSV
    csv_filename = f"experiments/results/response_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'test_id', 'query', 'num_documents', 'expected_insights',
            'rag_response', 'rag_time_ms', 'rag_confidence', 'rag_docs_used',
            'spike_response', 'spike_time_ms', 'spike_confidence', 
            'spike_insight_detected', 'spike_insight_type', 
            'spike_delta_ig', 'spike_delta_ged',
            'confidence_improvement', 'response_length_ratio', 'processing_overhead_ms'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_data)
    
    print(f"\nDetailed comparison saved to: {csv_filename}")
    
    # Also create a simplified version for easy reading
    simplified_filename = f"experiments/results/response_comparison_simple_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    simplified_data = []
    for row in comparison_data:
        simplified_row = {
            "質問": row["query"][:50] + "...",
            "従来RAG回答": row["rag_response"][:100] + "...",
            "InsightSpike回答": row["spike_response"][:100] + "...",
            "洞察検出": "✓" if row["spike_insight_detected"] else "✗",
            "洞察タイプ": row["spike_insight_type"],
            "信頼度向上": f"+{row['confidence_improvement']:.1%}" if row['confidence_improvement'] > 0 else f"{row['confidence_improvement']:.1%}"
        }
        simplified_data.append(simplified_row)
    
    # Save simplified version
    df = pd.DataFrame(simplified_data)
    df.to_csv(simplified_filename, index=False, encoding='utf-8-sig')  # utf-8-sig for Excel compatibility
    
    print(f"Simplified comparison saved to: {simplified_filename}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    total_cases = len(comparison_data)
    insights_found = sum(1 for row in comparison_data if row["spike_insight_detected"])
    avg_confidence_improvement = sum(row["confidence_improvement"] for row in comparison_data) / total_cases
    
    print(f"Total test cases: {total_cases}")
    print(f"Insights detected: {insights_found}/{total_cases} ({insights_found/total_cases:.1%})")
    print(f"Average confidence improvement: {avg_confidence_improvement:.1%}")
    
    # Response quality analysis
    print("\nRESPONSE QUALITY COMPARISON:")
    for i, row in enumerate(comparison_data, 1):
        print(f"\nCase {i}: {row['query'][:50]}...")
        print(f"  RAG: {row['rag_response'][:80]}...")
        print(f"  Spike: {row['spike_response'][:80]}...")
        if row["spike_insight_detected"]:
            print(f"  → Insight type: {row['spike_insight_type']}")
    
    return comparison_data, csv_filename, simplified_filename


def create_visual_comparison_html(comparison_data: List[Dict], filename: str):
    """Create an HTML file for visual side-by-side comparison"""
    
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>InsightSpike vs Traditional RAG Comparison</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .comparison { border: 1px solid #ddd; margin: 20px 0; padding: 15px; }
        .query { font-weight: bold; color: #333; margin-bottom: 10px; }
        .responses { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .response-box { padding: 10px; border-radius: 5px; }
        .rag { background-color: #f0f0f0; }
        .spike { background-color: #e8f4f8; }
        .insight { background-color: #ffffcc; padding: 5px; margin-top: 10px; }
        .metrics { font-size: 0.9em; color: #666; margin-top: 10px; }
        h3 { margin-top: 0; }
    </style>
</head>
<body>
    <h1>InsightSpike vs Traditional RAG: Response Comparison</h1>
"""
    
    for row in comparison_data:
        html_content += f"""
    <div class="comparison">
        <div class="query">Question: {row['query']}</div>
        <div class="responses">
            <div class="response-box rag">
                <h3>Traditional RAG</h3>
                <p>{row['rag_response']}</p>
                <div class="metrics">
                    Time: {row['rag_time_ms']}ms | 
                    Confidence: {row['rag_confidence']:.2f} | 
                    Docs used: {row['rag_docs_used']}
                </div>
            </div>
            <div class="response-box spike">
                <h3>InsightSpike</h3>
                <p>{row['spike_response']}</p>
                {f'<div class="insight">🎯 Insight detected: {row["spike_insight_type"]}</div>' if row['spike_insight_detected'] else ''}
                <div class="metrics">
                    Time: {row['spike_time_ms']}ms | 
                    Confidence: {row['spike_confidence']:.2f} | 
                    {f'ΔIG: {row["spike_delta_ig"]:.2f} | ΔGED: {row["spike_delta_ged"]:.2f}' if row['spike_insight_detected'] else ''}
                </div>
            </div>
        </div>
    </div>
"""
    
    html_content += """
</body>
</html>
"""
    
    html_filename = f"experiments/results/{filename}.html"
    with open(html_filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Visual comparison saved to: {html_filename}")
    return html_filename


if __name__ == "__main__":
    print("Generating detailed response comparison...")
    
    # Generate comparison data
    comparison_data, csv_file, simple_csv = generate_detailed_comparison()
    
    # Create visual HTML comparison
    html_file = create_visual_comparison_html(
        comparison_data, 
        f"visual_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    print(f"\n✅ Comparison complete!")
    print(f"   - Detailed CSV: {csv_file}")
    print(f"   - Simple CSV: {simple_csv}")
    print(f"   - Visual HTML: {html_file}")
    print("\nYou can now review the actual responses to judge quality!")