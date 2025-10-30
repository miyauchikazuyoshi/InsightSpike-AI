#!/usr/bin/env python3
"""Export initial knowledge base to CSV format."""

import csv
from pathlib import Path
from datetime import datetime
import pandas as pd

from run_experiment_improved import create_high_quality_knowledge_base


def export_knowledge_base_to_csv():
    """Export the initial knowledge base to CSV."""
    print("📚 Exporting Initial Knowledge Base")
    print("=" * 60)
    
    # Get knowledge base
    knowledge_base = create_high_quality_knowledge_base()
    
    # Prepare data for CSV
    kb_data = []
    
    for i, kb_item in enumerate(knowledge_base):
        # Extract information
        kb_data.append({
            'id': i + 1,
            'text': kb_item.text,
            'domain': kb_item.domain,
            'concepts': ', '.join(kb_item.concepts) if kb_item.concepts else '',
            'depth': kb_item.depth,
            'text_length': len(kb_item.text),
            'n_concepts': len(kb_item.concepts) if kb_item.concepts else 0
        })
    
    # Convert to DataFrame for better display
    df = pd.DataFrame(kb_data)
    
    # Save to CSV
    output_dir = Path("../results/knowledge_base")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"initial_knowledge_base_{timestamp}.csv"
    
    df.to_csv(csv_path, index=False, encoding='utf-8')
    
    print(f"✅ Saved knowledge base to: {csv_path}")
    
    # Display summary
    print("\n📊 Knowledge Base Summary:")
    print("-" * 60)
    print(f"Total items: {len(df)}")
    print(f"\nDomains:")
    for domain in df['domain'].unique():
        count = len(df[df['domain'] == domain])
        print(f"  • {domain}: {count} items")
    
    print(f"\nDepth distribution:")
    for depth in df['depth'].unique():
        count = len(df[df['depth'] == depth])
        print(f"  • {depth}: {count} items")
    
    print(f"\nText length statistics:")
    print(f"  • Mean: {df['text_length'].mean():.0f} characters")
    print(f"  • Min: {df['text_length'].min()} characters")
    print(f"  • Max: {df['text_length'].max()} characters")
    
    print(f"\nConcepts statistics:")
    print(f"  • Total unique concepts: {len(set(','.join(df['concepts']).split(', ')))}")
    print(f"  • Avg concepts per item: {df['n_concepts'].mean():.1f}")
    
    return df


def display_knowledge_base_content(df):
    """Display the knowledge base content."""
    print("\n" + "=" * 60)
    print("📖 KNOWLEDGE BASE CONTENT")
    print("=" * 60)
    
    for i, row in df.iterrows():
        print(f"\n{row['id']}. [{row['domain'].upper()}] {row['text'][:100]}...")
        print(f"   Depth: {row['depth']} | Concepts: {row['concepts'][:50]}...")
        print(f"   Length: {row['text_length']} chars")


def analyze_knowledge_relationships():
    """Analyze relationships between knowledge items."""
    print("\n" + "=" * 60)
    print("🔗 KNOWLEDGE RELATIONSHIPS")
    print("=" * 60)
    
    knowledge_base = create_high_quality_knowledge_base()
    
    # Build concept overlap matrix
    concept_overlaps = []
    
    for i, kb1 in enumerate(knowledge_base):
        for j, kb2 in enumerate(knowledge_base):
            if i < j:  # Only check upper triangle
                concepts1 = set(kb1.concepts) if kb1.concepts else set()
                concepts2 = set(kb2.concepts) if kb2.concepts else set()
                
                overlap = concepts1.intersection(concepts2)
                if overlap:
                    concept_overlaps.append({
                        'item1_id': i + 1,
                        'item1_text': kb1.text[:50] + "...",
                        'item2_id': j + 1,
                        'item2_text': kb2.text[:50] + "...",
                        'shared_concepts': ', '.join(overlap),
                        'n_shared': len(overlap)
                    })
    
    if concept_overlaps:
        print("\n📊 Concept Overlaps (potential edges):")
        print("-" * 60)
        
        # Sort by number of shared concepts
        concept_overlaps.sort(key=lambda x: x['n_shared'], reverse=True)
        
        for overlap in concept_overlaps[:10]:  # Show top 10
            print(f"\nItems {overlap['item1_id']} ↔ {overlap['item2_id']}:")
            print(f"  Item 1: {overlap['item1_text']}")
            print(f"  Item 2: {overlap['item2_text']}")
            print(f"  Shared: {overlap['shared_concepts']} ({overlap['n_shared']} concepts)")
        
        # Save relationships to CSV
        output_dir = Path("../results/knowledge_base")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        relationships_path = output_dir / f"knowledge_relationships_{timestamp}.csv"
        
        relationships_df = pd.DataFrame(concept_overlaps)
        relationships_df.to_csv(relationships_path, index=False, encoding='utf-8')
        
        print(f"\n✅ Saved relationships to: {relationships_path}")
        
        # Summary statistics
        print(f"\n📈 Relationship Statistics:")
        print(f"  • Total potential edges: {len(concept_overlaps)}")
        print(f"  • Average shared concepts: {relationships_df['n_shared'].mean():.1f}")
        print(f"  • Max shared concepts: {relationships_df['n_shared'].max()}")
    
    return concept_overlaps


def create_comprehensive_report():
    """Create a comprehensive report of the knowledge base."""
    output_dir = Path("../results/knowledge_base")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"knowledge_base_report_{timestamp}.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("INITIAL KNOWLEDGE BASE REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        knowledge_base = create_high_quality_knowledge_base()
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Items: {len(knowledge_base)}\n\n")
        
        f.write("DETAILED CONTENT:\n")
        f.write("-" * 70 + "\n\n")
        
        for i, kb in enumerate(knowledge_base, 1):
            f.write(f"{i}. {kb.text}\n")
            f.write(f"   Domain: {kb.domain}\n")
            f.write(f"   Depth: {kb.depth}\n")
            f.write(f"   Concepts: {', '.join(kb.concepts) if kb.concepts else 'None'}\n")
            f.write(f"   Length: {len(kb.text)} characters\n")
            f.write("\n")
        
        f.write("=" * 70 + "\n")
        f.write("END OF REPORT\n")
    
    print(f"\n✅ Saved comprehensive report to: {report_path}")
    
    return report_path


def main():
    """Main execution."""
    try:
        print("🚀 Starting Knowledge Base Export")
        print("=" * 60)
        
        # Export to CSV
        df = export_knowledge_base_to_csv()
        
        # Display content
        display_knowledge_base_content(df)
        
        # Analyze relationships
        relationships = analyze_knowledge_relationships()
        
        # Create comprehensive report
        report_path = create_comprehensive_report()
        
        print("\n" + "=" * 60)
        print("✅ EXPORT COMPLETE")
        print("=" * 60)
        print("\nGenerated files:")
        print("  • Knowledge base CSV")
        print("  • Relationships CSV")
        print("  • Comprehensive report TXT")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)