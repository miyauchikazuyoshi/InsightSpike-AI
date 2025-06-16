#!/usr/bin/env python3
"""
実データベーステスト - InsightSpike-AI
=====================================

既存のDBを使用した実用テスト
"""

import sys
import sqlite3
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

def test_with_real_data():
    """実データを使用したテストを実行"""
    
    print("💾 実データベーステスト")
    print("=" * 35)
    
    from insightspike.core.agents.main_agent import MainAgent
    
    # データベースパス確認
    db_path = Path("data/insight_facts.db")
    if not db_path.exists():
        print(f"❌ データベースが見つかりません: {db_path}")
        return False
    
    # データベースから実データを読み込み
    print("📂 データベースからデータを読み込み中...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # テーブル構造確認
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"   利用可能なテーブル: {[t[0] for t in tables]}")
    
    # サンプルデータ取得
    if tables:
        table_name = tables[0][0]  # 最初のテーブルを使用
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 10")
        sample_data = cursor.fetchall()
        
        if sample_data:
            print(f"   {table_name}から{len(sample_data)}件のサンプルデータを取得")
            
            # エージェント初期化
            agent = MainAgent()
            
            # データをエピソードとして追加
            for i, row in enumerate(sample_data[:5]):  # 最初の5件のみ
                # rowを文字列として結合
                content = " | ".join(str(cell) for cell in row if cell is not None)
                if len(content) > 10:  # 有効なコンテンツのみ
                    vector = np.random.random(384).astype(np.float32)
                    agent.l2_memory.add_episode(vector, content[:200])  # 200文字に制限
                    print(f"   ✅ エピソード{i+1}: {content[:50]}...")
            
            # メモリ状態確認
            print(f"\n📊 メモリ状態:")
            print(f"   追加されたエピソード: {len(agent.l2_memory.episodes)}")
            print(f"   平均C-value: {sum(ep.c for ep in agent.l2_memory.episodes) / max(len(agent.l2_memory.episodes), 1):.3f}")
            
            conn.close()
            print("\n✅ 実データベーステスト完了")
            return True
        else:
            print("   ❌ テーブルにデータがありません")
    else:
        print("   ❌ 有効なテーブルがありません")
    
    conn.close()
    return False

if __name__ == "__main__":
    test_with_real_data()
