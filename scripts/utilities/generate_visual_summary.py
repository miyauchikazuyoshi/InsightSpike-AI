#!/usr/bin/env python3
"""
Visual summary generator for InsightSpike-AI experimental results.
Creates ASCII charts and visual representations of the experimental data.
"""

import json
import numpy as np
from pathlib import Path

def create_ascii_bar_chart(data: dict, title: str, width: int = 60) -> str:
    """Create an ASCII bar chart."""
    chart = f"\n{title}\n{'='*len(title)}\n"
    
    max_val = max(data.values()) if data.values() else 1
    
    for label, value in data.items():
        bar_length = int((value / max_val) * width) if max_val > 0 else 0
        bar = '█' * bar_length
        chart += f"{label:20} │{bar:<{width}} {value:.3f}\n"
    
    return chart

def create_comparison_chart(insight_vals: list, baseline_vals: list, labels: list, title: str) -> str:
    """Create a side-by-side comparison chart."""
    chart = f"\n{title}\n{'='*len(title)}\n"
    
    max_val = max(max(insight_vals), max(baseline_vals)) if insight_vals and baseline_vals else 1
    width = 25
    
    chart += f"{'Question':<20} │ {'InsightSpike':<{width}} │ {'Baseline':<{width}} │ Improvement\n"
    chart += f"{'-'*20}─┼─{'-'*width}─┼─{'-'*width}─┼─{'-'*12}\n"
    
    for i, label in enumerate(labels):
        if i < len(insight_vals) and i < len(baseline_vals):
            insight_val = insight_vals[i]
            baseline_val = baseline_vals[i]
            
            insight_bar_len = int((insight_val / max_val) * width) if max_val > 0 else 0
            baseline_bar_len = int((baseline_val / max_val) * width) if max_val > 0 else 0
            
            insight_bar = '█' * insight_bar_len
            baseline_bar = '█' * baseline_bar_len
            
            improvement = ((insight_val - baseline_val) / baseline_val * 100) if baseline_val > 0 else float('inf')
            improvement_str = f"{improvement:+.1f}%" if improvement != float('inf') else "∞"
            
            chart += f"{label[:20]:<20} │ {insight_bar:<{width}} │ {baseline_bar:<{width}} │ {improvement_str}\n"
    
    return chart

def create_metrics_dashboard(results: dict) -> str:
    """Create a comprehensive metrics dashboard."""
    analysis = results['analysis']
    
    dashboard = """
┌─────────────────────────────────────────────────────────────────────────────┐
│                    INSIGHTSPIKE-AI EXPERIMENTAL DASHBOARD                   │
├─────────────────────────────────────────────────────────────────────────────┤
"""
    
    # Key metrics
    dashboard += f"│ 🎯 RESPONSE QUALITY                                                         │\n"
    dashboard += f"│   InsightSpike: {analysis['response_quality']['insightspike_avg']:.3f}                                                 │\n"
    dashboard += f"│   Baseline:     {analysis['response_quality']['baseline_avg']:.3f}                                                 │\n"
    dashboard += f"│   Improvement:  {analysis['improvements']['response_quality_improvement_pct']:+.1f}%                                              │\n"
    dashboard += f"│                                                                             │\n"
    dashboard += f"│ 🧠 INSIGHT DETECTION                                                        │\n"
    dashboard += f"│   Detection Rate: {analysis['insight_detection']['insightspike_rate']*100:.0f}%                                                │\n"
    dashboard += f"│   False Positives: {analysis['insight_detection']['false_positive_rate']*100:.0f}%                                               │\n"
    dashboard += f"│                                                                             │\n"
    dashboard += f"│ ⚡ PROCESSING SPEED                                                          │\n"
    dashboard += f"│   InsightSpike: {analysis['processing_metrics']['avg_response_time_is']*1000:.2f}ms                                             │\n"
    dashboard += f"│   Baseline:     {analysis['processing_metrics']['avg_response_time_baseline']*1000:.1f}ms                                            │\n"
    dashboard += f"│   Speed Boost:  {(analysis['processing_metrics']['avg_response_time_baseline']/analysis['processing_metrics']['avg_response_time_is']):.0f}x faster                                            │\n"
    dashboard += f"└─────────────────────────────────────────────────────────────────────────────┘\n"
    
    return dashboard

def generate_spike_visualization(results: dict) -> str:
    """Generate a visualization of spike activity."""
    viz = "\n🌊 SPIKE ACTIVITY VISUALIZATION\n"
    viz += "="*35 + "\n\n"
    
    for result in results['insightspike_results']:
        question = result['question_id']
        spikes = result['spike_count']
        detected = result['insight_detected']
        
        # Create spike visualization
        spike_viz = '▲' * spikes if spikes > 0 else '─'
        status = "🔥 INSIGHT DETECTED" if detected else "💤 No insight"
        
        viz += f"{question:<20} │ {spike_viz:<20} │ {status}\n"
        
        if detected:
            viz += f"{'':21} │ ΔGED: {result['avg_delta_ged']:.3f}         │\n"
            viz += f"{'':21} │ ΔIG:  {result['avg_delta_ig']:.3f}         │\n"
        viz += "\n"
    
    return viz

def main():
    """Generate visual summaries."""
    print("📊 Generating Visual Summary...")
    
    # Load results
    results_path = "data/processed/experiment_results.json"
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Create visual summary
    visual_summary = ""
    
    # Dashboard
    visual_summary += create_metrics_dashboard(results)
    
    # Response quality comparison
    insight_qualities = [r['response_quality'] for r in results['insightspike_results']]
    baseline_qualities = [r['response_quality'] for r in results['baseline_results']]
    question_labels = [r['question_id'][:15] for r in results['insightspike_results']]
    
    visual_summary += create_comparison_chart(
        insight_qualities, baseline_qualities, question_labels,
        "📈 RESPONSE QUALITY COMPARISON"
    )
    
    # Spike activity
    visual_summary += generate_spike_visualization(results)
    
    # Category performance
    categories = {}
    for result in results['insightspike_results']:
        category = result['category']
        if category not in categories:
            categories[category] = []
        categories[category].append(result['response_quality'])
    
    category_avgs = {cat: np.mean(vals) for cat, vals in categories.items()}
    visual_summary += create_ascii_bar_chart(
        category_avgs,
        "🎯 PERFORMANCE BY CATEGORY"
    )
    
    # Processing time comparison
    time_data = {
        'InsightSpike': results['analysis']['processing_metrics']['avg_response_time_is'] * 1000,
        'Baseline': results['analysis']['processing_metrics']['avg_response_time_baseline'] * 1000
    }
    visual_summary += create_ascii_bar_chart(
        time_data,
        "⚡ PROCESSING TIME COMPARISON (ms)"
    )
    
    # Save visual summary
    output_path = "VISUAL_SUMMARY.txt"
    with open(output_path, 'w') as f:
        f.write(visual_summary)
    
    print(f"✅ Visual summary saved to: {output_path}")
    print("\n" + visual_summary)

if __name__ == "__main__":
    main()
