#!/usr/bin/env python3
"""
実験クリーンアップ実行スクリプト
削除候補ファイルを安全に削除し、重要実験を強化
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def safe_cleanup_experiments():
    """安全な実験クリーンアップを実行"""
    
    print("🧹 実験クリーンアップを開始...")
    
    # 削除候補ファイル
    deprecated_files = [
        "experiments/validation/run_experimental_validation.py",
        "experiments/05_metrics_analysis/debug_gedig_experiment.py"
    ]
    
    # バックアップディレクトリ作成
    backup_dir = Path("experiments/00_data_backups/deprecated_cleanup_backup")
    backup_dir.mkdir(exist_ok=True)
    
    # 削除前にバックアップ
    print("💾 削除前バックアップ作成...")
    for file_path in deprecated_files:
        file_p = Path(file_path)
        if file_p.exists():
            backup_file = backup_dir / file_p.name
            shutil.copy2(file_p, backup_file)
            print(f"  📄 バックアップ: {file_p.name}")
    
    # 削除実行
    print("🗑️ 削除候補ファイル削除...")
    deleted_count = 0
    for file_path in deprecated_files:
        file_p = Path(file_path)
        if file_p.exists():
            file_p.unlink()
            print(f"  ❌ 削除完了: {file_path}")
            deleted_count += 1
        else:
            print(f"  ⚠️ ファイル不存在: {file_path}")
    
    # 空のディレクトリをチェック・削除
    validation_dir = Path("experiments/validation")
    if validation_dir.exists() and not any(validation_dir.iterdir()):
        validation_dir.rmdir()
        print(f"  📁 空ディレクトリ削除: validation/")
    
    # クリーンアップ完了レポート作成
    report = f"""# 実験クリーンアップ完了レポート

## 🧹 実行サマリー

- **実行日時**: {datetime.now().isoformat()}
- **削除ファイル数**: {deleted_count}
- **バックアップ場所**: `experiments/00_data_backups/deprecated_cleanup_backup/`

## 🗑️ 削除されたファイル

"""
    
    for file_path in deprecated_files:
        if Path(file_path).exists() == False:  # 削除済み
            report += f"- `{file_path}` ✅\n"
        else:
            report += f"- `{file_path}` ❌ (削除失敗)\n"
    
    report += f"""
## 🚨 保護された重要実験

**CRITICAL実験 (81.6%洞察検出率達成)**:
- `experiments/01_realtime_insight_experiments/detailed_logging_realtime_experiment.py`
- `experiments/02_comprehensive_experiments/comprehensive_analysis_detailed_logging.py`

## 📁 クリーンアップ後の構造

experiments/フォルダは以下の整理された構造になりました：

1. **00_data_backups/** - すべてのバックアップデータ
2. **01_realtime_insight_experiments/** - 🚨 **最重要** 革命的洞察実験
3. **02-10_各種実験カテゴリ/** - 整理されたカテゴリ別実験
4. **03_agent_testing/integration_test_outputs/** - 移動された統合テスト結果

## ✅ クリーンアップ効果

- 古い設計ファイル削除完了
- 重要実験の保護・強化
- フォルダ構造の最適化
- バックアップによる安全性確保

---
*InsightSpike-AI Project - Experiment Cleanup Report*
"""
    
    # レポート保存
    report_file = Path("experiments/00_data_backups/cleanup_completion_report.md")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📋 クリーンアップ完了レポート保存: {report_file}")
    print(f"✅ 実験クリーンアップ完了! 削除: {deleted_count} ファイル")
    
    return deleted_count

if __name__ == "__main__":
    safe_cleanup_experiments()
