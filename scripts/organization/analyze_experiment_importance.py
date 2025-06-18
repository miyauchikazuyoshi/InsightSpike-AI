#!/usr/bin/env python3
"""
重要度分析・実験クリーンアップスクリプト
実験の重要度を分析し、削除候補を特定
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

def analyze_experiment_importance(experiments_dir: str = "experiments") -> Dict:
    """実験の重要度を分析"""
    
    results = {
        "critical_experiments": [],
        "important_experiments": [],
        "standard_experiments": [],
        "deprecated_candidates": [],
        "analysis_timestamp": datetime.now().isoformat()
    }
    
    # 最重要実験（革命的発見をした実験）
    critical_files = [
        "01_realtime_insight_experiments/detailed_logging_realtime_experiment.py",
        "02_comprehensive_experiments/comprehensive_analysis_detailed_logging.py"
    ]
    
    # 重要実験（現在のアーキテクチャと互換性があり、価値の高い実験）
    important_patterns = [
        "comprehensive_experiment_framework.py",
        "objective_evaluation_framework.py",
        "rag_enhanced_experiment_framework.py",
        "large_scale",
        "integration",
        "performance"
    ]
    
    # 削除候補パターン（古い設計・テスト・実験的なもの）
    deprecated_patterns = [
        "test_",  # テストファイル
        "debug_",  # デバッグファイル
        "_old",   # 明示的に古いファイル
        "_backup",  # バックアップファイル
        "experimental_",  # 実験的なファイル
        "prototype_",  # プロトタイプ
        "draft_"  # 下書き
    ]
    
    # 実験ディレクトリを走査
    exp_path = Path(experiments_dir)
    
    for category_dir in exp_path.iterdir():
        if not category_dir.is_dir() or category_dir.name.startswith('.'):
            continue
            
        category_name = category_dir.name
        category_files = []
        
        for file_path in category_dir.rglob("*.py"):
            if file_path.is_file():
                rel_path = str(file_path.relative_to(exp_path))
                file_size = file_path.stat().st_size
                
                # ファイル分析
                analysis = {
                    "path": rel_path,
                    "category": category_name,
                    "size": file_size,
                    "lines": count_lines(file_path),
                    "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                }
                
                # 重要度判定
                if rel_path in critical_files:
                    analysis["importance"] = "CRITICAL"
                    analysis["reason"] = "Revolutionary discovery - 81.6% insight detection achieved"
                    results["critical_experiments"].append(analysis)
                    
                elif any(pattern in rel_path.lower() for pattern in important_patterns):
                    analysis["importance"] = "IMPORTANT"
                    analysis["reason"] = "Current architecture compatible, high value"
                    results["important_experiments"].append(analysis)
                    
                elif any(pattern in rel_path.lower() for pattern in deprecated_patterns):
                    analysis["importance"] = "DEPRECATED"
                    analysis["reason"] = "Test/debug/old design - candidate for removal"
                    results["deprecated_candidates"].append(analysis)
                    
                else:
                    analysis["importance"] = "STANDARD"
                    analysis["reason"] = "Standard experiment file"
                    results["standard_experiments"].append(analysis)
    
    return results

def count_lines(file_path: Path) -> int:
    """ファイルの行数をカウント"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except:
        return 0

def generate_cleanup_report(analysis: Dict) -> str:
    """クリーンアップレポートを生成"""
    
    report = f"""# 実験重要度分析・クリーンアップレポート

## 📊 分析サマリー

| カテゴリ | ファイル数 | 合計サイズ |
|----------|------------|------------|
| 🚨 **CRITICAL** | {len(analysis['critical_experiments'])} | {sum(exp['size'] for exp in analysis['critical_experiments'])} bytes |
| ⭐ **IMPORTANT** | {len(analysis['important_experiments'])} | {sum(exp['size'] for exp in analysis['important_experiments'])} bytes |
| 📄 **STANDARD** | {len(analysis['standard_experiments'])} | {sum(exp['size'] for exp in analysis['standard_experiments'])} bytes |
| 🗑️ **DEPRECATED** | {len(analysis['deprecated_candidates'])} | {sum(exp['size'] for exp in analysis['deprecated_candidates'])} bytes |

## 🚨 **最重要実験（CRITICAL）** - 絶対に保持

"""
    
    for exp in analysis['critical_experiments']:
        report += f"- **{exp['path']}** ({exp['lines']} lines)\n"
        report += f"  - {exp['reason']}\n"
        report += f"  - サイズ: {exp['size']} bytes\n\n"
    
    report += f"""## ⭐ **重要実験（IMPORTANT）** - 現在のアーキテクチャで有用

"""
    
    for exp in sorted(analysis['important_experiments'], key=lambda x: x['size'], reverse=True)[:10]:
        report += f"- **{exp['path']}** ({exp['lines']} lines)\n"
        report += f"  - {exp['reason']}\n\n"
    
    if len(analysis['important_experiments']) > 10:
        report += f"... および他 {len(analysis['important_experiments']) - 10} ファイル\n\n"
    
    report += f"""## 🗑️ **削除候補（DEPRECATED）** - 古い設計・テストファイル

"""
    
    total_deprecated_size = sum(exp['size'] for exp in analysis['deprecated_candidates'])
    
    for exp in analysis['deprecated_candidates']:
        report += f"- **{exp['path']}** ({exp['lines']} lines)\n"
        report += f"  - {exp['reason']}\n\n"
    
    report += f"""## 💾 **ディスク節約効果**

削除候補を削除することで **{total_deprecated_size} bytes** ({total_deprecated_size/1024/1024:.1f} MB) のディスク容量を節約できます。

## 🎯 **推奨アクション**

1. **CRITICAL実験の完全バックアップ**: 詳細ログ実験とその分析結果
2. **DEPRECATED削除**: 古い設計のテスト・デバッグファイル削除
3. **IMPORTANT実験の整理**: 現在のアーキテクチャに合わせて更新
4. **STANDARD実験の評価**: 個別に価値を判定

---
*分析実行日時: {analysis['analysis_timestamp']}*
"""
    
    return report

def main():
    """メイン実行"""
    print("🔍 実験重要度分析を開始...")
    
    # 分析実行
    analysis = analyze_experiment_importance()
    
    # レポート生成
    report = generate_cleanup_report(analysis)
    
    # 結果保存
    output_dir = Path("experiments/00_data_backups")
    output_dir.mkdir(exist_ok=True)
    
    # JSON保存
    json_file = output_dir / f"experiment_importance_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    
    # Markdownレポート保存
    md_file = output_dir / f"cleanup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📊 分析完了!")
    print(f"📋 レポート保存: {md_file}")
    print(f"📄 JSON保存: {json_file}")
    
    # サマリー表示
    print(f"\n📊 **サマリー**")
    print(f"🚨 CRITICAL: {len(analysis['critical_experiments'])} files")
    print(f"⭐ IMPORTANT: {len(analysis['important_experiments'])} files")
    print(f"📄 STANDARD: {len(analysis['standard_experiments'])} files")
    print(f"🗑️ DEPRECATED: {len(analysis['deprecated_candidates'])} files")
    
    deprecated_size = sum(exp['size'] for exp in analysis['deprecated_candidates'])
    print(f"💾 削除可能サイズ: {deprecated_size/1024/1024:.1f} MB")

if __name__ == "__main__":
    main()
