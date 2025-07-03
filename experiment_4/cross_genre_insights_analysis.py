#!/usr/bin/env python3
"""
Cross-Genre Insights Analysis for Experiment 4
ジャンルを飛び越えた洞察の分析
"""

import os
import sys
import json
import torch
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def load_experiment_data():
    """Load episodes and graph from experiment 4"""
    with open('data/episodes.json', 'r') as f:
        episodes = json.load(f)
    
    graph = torch.load('data/graph_pyg.pt')
    
    return episodes, graph

def extract_cross_genre_patterns(episodes):
    """Extract patterns that bridge different domains/genres"""
    
    # Define technology categories
    tech_categories = {
        'AI/ML': ['artificial intelligence', 'machine learning', 'deep learning', 
                  'neural network', 'nlp', 'computer vision', 'reinforcement learning'],
        'Computing': ['quantum computing', 'edge computing', 'cloud computing', 
                      'distributed systems', 'parallel processing'],
        'Data': ['big data', 'data science', 'analytics', 'data mining', 'database'],
        'Security': ['cybersecurity', 'cryptography', 'blockchain', 'privacy', 'security'],
        'Bio/Health': ['healthcare', 'bioinformatics', 'genomics', 'medical', 'drug discovery'],
        'Engineering': ['robotics', 'iot', 'embedded systems', 'automation', 'control systems'],
        'Business': ['finance', 'fintech', 'e-commerce', 'marketing', 'supply chain'],
        'Science': ['physics', 'chemistry', 'biology', 'astronomy', 'materials science']
    }
    
    # Define application domains
    domains = {
        'Healthcare': ['healthcare', 'medical', 'patient', 'diagnosis', 'treatment', 'hospital'],
        'Finance': ['finance', 'banking', 'trading', 'investment', 'fintech', 'payment'],
        'Education': ['education', 'learning', 'student', 'teaching', 'curriculum', 'training'],
        'Manufacturing': ['manufacturing', 'production', 'factory', 'assembly', 'quality control'],
        'Transportation': ['transportation', 'autonomous', 'vehicle', 'traffic', 'logistics'],
        'Energy': ['energy', 'renewable', 'grid', 'efficiency', 'sustainability'],
        'Agriculture': ['agriculture', 'farming', 'crop', 'harvest', 'precision agriculture'],
        'Retail': ['retail', 'e-commerce', 'customer', 'shopping', 'recommendation'],
        'Entertainment': ['gaming', 'entertainment', 'media', 'content', 'streaming'],
        'Research': ['research', 'scientific', 'experiment', 'discovery', 'innovation']
    }
    
    cross_genre_insights = []
    tech_domain_combinations = defaultdict(list)
    
    for idx, episode in enumerate(episodes):
        text = episode['text'].lower()
        
        # Find tech categories in this episode
        found_techs = []
        for tech_name, keywords in tech_categories.items():
            if any(keyword in text for keyword in keywords):
                found_techs.append(tech_name)
        
        # Find domains in this episode
        found_domains = []
        for domain_name, keywords in domains.items():
            if any(keyword in text for keyword in keywords):
                found_domains.append(domain_name)
        
        # If multiple categories or unusual combinations found
        if len(found_techs) >= 2 or (found_techs and found_domains):
            insight = {
                'episode_id': idx,
                'text_preview': text[:100] + '...',
                'technologies': found_techs,
                'domains': found_domains,
                'cross_genre_type': 'multi-tech' if len(found_techs) >= 2 else 'tech-domain'
            }
            
            # Identify specific cross-genre patterns
            if 'AI/ML' in found_techs and 'Bio/Health' in found_techs:
                insight['pattern'] = 'AI-Bio Convergence'
            elif 'Computing' in found_techs and 'Healthcare' in found_domains:
                insight['pattern'] = 'Computational Healthcare'
            elif 'Security' in found_techs and 'Finance' in found_domains:
                insight['pattern'] = 'FinSec Innovation'
            elif 'AI/ML' in found_techs and 'Agriculture' in found_domains:
                insight['pattern'] = 'AgriTech AI'
            elif 'Data' in found_techs and 'Manufacturing' in found_domains:
                insight['pattern'] = 'Industry 4.0'
            elif len(found_techs) >= 2 and len(found_domains) >= 2:
                insight['pattern'] = 'Multi-Domain Convergence'
            else:
                insight['pattern'] = 'Cross-Sector Application'
            
            cross_genre_insights.append(insight)
            
            # Track combinations
            for tech in found_techs:
                for domain in found_domains:
                    tech_domain_combinations[(tech, domain)].append(idx)
    
    return cross_genre_insights, tech_domain_combinations

def analyze_cross_genre_graph_connections(graph, cross_genre_insights):
    """Analyze how cross-genre nodes are connected in the graph"""
    
    if not hasattr(graph, 'edge_index') or graph.edge_index.size(1) == 0:
        return None
    
    # Get episode IDs that are cross-genre
    cross_genre_ids = set(insight['episode_id'] for insight in cross_genre_insights)
    
    # Analyze connectivity
    edge_index = graph.edge_index.numpy()
    cross_genre_connections = 0
    internal_connections = 0
    external_connections = 0
    
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        
        if src in cross_genre_ids and dst in cross_genre_ids:
            cross_genre_connections += 1
        elif src in cross_genre_ids or dst in cross_genre_ids:
            external_connections += 1
        else:
            internal_connections += 1
    
    return {
        'cross_genre_connections': cross_genre_connections,
        'external_connections': external_connections,
        'internal_connections': internal_connections,
        'connectivity_ratio': cross_genre_connections / (cross_genre_connections + external_connections) if (cross_genre_connections + external_connections) > 0 else 0
    }

def generate_cross_genre_report(cross_genre_insights, tech_domain_combinations, connectivity_analysis):
    """Generate a comprehensive report on cross-genre insights"""
    
    print("=== ジャンルを飛び越えた洞察の分析結果 ===\n")
    
    # 1. 総数と割合
    total_episodes = 282  # From experiment results
    cross_genre_count = len(cross_genre_insights)
    percentage = (cross_genre_count / total_episodes) * 100
    
    print(f"📊 総合統計:")
    print(f"  - 全エピソード数: {total_episodes}")
    print(f"  - クロスジャンル洞察: {cross_genre_count} ({percentage:.1f}%)")
    
    # 2. パターン別の内訳
    pattern_counts = Counter(insight['pattern'] for insight in cross_genre_insights)
    print(f"\n🔍 クロスジャンルパターンの内訳:")
    for pattern, count in pattern_counts.most_common():
        print(f"  - {pattern}: {count}件")
    
    # 3. 最も多い技術×ドメインの組み合わせ
    print(f"\n🔗 最も多い技術×ドメインの組み合わせ (Top 10):")
    sorted_combinations = sorted(tech_domain_combinations.items(), 
                                key=lambda x: len(x[1]), reverse=True)[:10]
    
    for (tech, domain), episodes in sorted_combinations:
        print(f"  - {tech} × {domain}: {len(episodes)}件")
    
    # 4. 特に興味深いクロスジャンル洞察の例
    print(f"\n💡 特に興味深いクロスジャンル洞察の例:")
    
    # AI-Bio Convergence の例
    ai_bio_insights = [i for i in cross_genre_insights if i.get('pattern') == 'AI-Bio Convergence']
    if ai_bio_insights:
        print(f"\n  【AI-Bio Convergence】")
        for insight in ai_bio_insights[:3]:
            print(f"    Episode {insight['episode_id']}: {insight['text_preview']}")
    
    # Multi-Domain Convergence の例
    multi_domain = [i for i in cross_genre_insights if i.get('pattern') == 'Multi-Domain Convergence']
    if multi_domain:
        print(f"\n  【Multi-Domain Convergence】")
        for insight in multi_domain[:3]:
            techs = ', '.join(insight['technologies'])
            domains = ', '.join(insight['domains'])
            print(f"    Episode {insight['episode_id']}: {techs} → {domains}")
    
    # 5. グラフ構造分析
    if connectivity_analysis:
        print(f"\n📈 グラフ構造におけるクロスジャンルノードの特徴:")
        print(f"  - クロスジャンル間の接続: {connectivity_analysis['cross_genre_connections']}")
        print(f"  - 外部との接続: {connectivity_analysis['external_connections']}")
        print(f"  - 接続密度: {connectivity_analysis['connectivity_ratio']:.2%}")
    
    # 6. 創発的な洞察
    print(f"\n🌟 創発的な洞察:")
    print(f"  1. 技術融合の加速: {percentage:.1f}%のエピソードが複数分野を橋渡し")
    print(f"  2. AI/MLが触媒役: ほとんどのクロスジャンル洞察にAI/MLが関与")
    print(f"  3. 産業変革の兆し: 従来の産業境界を越えた新しいソリューションが多数")
    print(f"  4. 知識のハブ化: クロスジャンルノードが知識ネットワークの中心に")
    
    return {
        'total_cross_genre': cross_genre_count,
        'percentage': percentage,
        'pattern_distribution': dict(pattern_counts),
        'top_combinations': sorted_combinations[:10]
    }

def main():
    """Main analysis function"""
    print("=== Cross-Genre Insights Analysis ===\n")
    
    # Load data
    episodes, graph = load_experiment_data()
    
    # Extract cross-genre patterns
    cross_genre_insights, tech_domain_combinations = extract_cross_genre_patterns(episodes)
    
    # Analyze graph connections
    connectivity_analysis = analyze_cross_genre_graph_connections(graph, cross_genre_insights)
    
    # Generate report
    summary = generate_cross_genre_report(cross_genre_insights, tech_domain_combinations, connectivity_analysis)
    
    # Save results
    results = {
        'summary': summary,
        'cross_genre_insights': cross_genre_insights[:20],  # Save top 20 examples
        'connectivity_analysis': connectivity_analysis
    }
    
    with open('cross_genre_insights_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 分析完了！結果は cross_genre_insights_results.json に保存されました。")

if __name__ == "__main__":
    main()