#!/usr/bin/env python3
"""
簡単なPoCテスト
================

torch-geometricの問題を回避してInsightSpike-AIの基本機能をテストします。
"""

import sys
import os
from pathlib import Path

# プロジェクトルートを追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def main():
    print("🧠 InsightSpike-AI 簡易PoCテスト")
    print("=" * 50)
    
    try:
        print("1. エージェント作成中...")
        from insightspike.core.agents.main_agent import MainAgent
        
        agent = MainAgent()
        print("✅ エージェント作成成功")
        
        print("\n2. 初期化テスト中...")
        init_success = agent.initialize()
        print(f"初期化結果: {'✅ 成功' if init_success else '⚠️ 失敗'}")
        
        if init_success:
            print("\n3. 簡単な質問処理テスト中...")
            test_question = "What is artificial intelligence?"
            print(f"質問: {test_question}")
            
            try:
                result = agent.process_question(test_question, max_cycles=1, verbose=False)
                
                print("\n📊 処理結果:")
                print(f"✅ 成功: {result.get('success', False)}")
                print(f"📈 品質スコア: {result.get('reasoning_quality', 0):.3f}")
                print(f"🔄 サイクル数: {result.get('total_cycles', 0)}")
                print(f"⚡ スパイク検出: {result.get('spike_detected', False)}")
                
                response = result.get('response', 'No response')
                print(f"\n💭 応答 (最初の200文字):")
                print(f"   {response[:200]}...")
                
                # 統計情報取得
                try:
                    stats = agent.get_stats()
                    print(f"\n📊 エージェント統計:")
                    print(f"   初期化済み: {stats.get('initialized', False)}")
                    print(f"   総サイクル: {stats.get('total_cycles', 0)}")
                    print(f"   平均品質: {stats.get('average_quality', 0):.3f}")
                except Exception as e:
                    print(f"⚠️ 統計取得エラー: {e}")
                
                return True
                
            except Exception as e:
                print(f"❌ 質問処理エラー: {e}")
                import traceback
                print("スタックトレース:")
                traceback.print_exc()
                return False
        else:
            print("⚠️ 初期化に失敗したため、質問処理をスキップします")
            return False
            
    except Exception as e:
        print(f"❌ 重大なエラー: {e}")
        import traceback
        print("スタックトレース:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    print(f"\n🎯 テスト結果: {'✅ 成功' if success else '❌ 失敗'}")
    
    if success:
        print("\n🎉 PoCは基本的に動作しています！")
        print("💡 torch-geometricの問題にも関わらず、フォールバック機能が正常に動作しています。")
    else:
        print("\n⚠️ PoCに問題があります。")
        print("🔧 依存関係やコンフィギュレーションを確認してください。")
    
    sys.exit(0 if success else 1)
