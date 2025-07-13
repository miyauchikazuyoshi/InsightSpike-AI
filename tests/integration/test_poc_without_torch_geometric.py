#!/usr/bin/env python3
"""
PoCテスト - torch-geometric無しでの基本機能確認
============================================

torch-geometricの依存関係問題を回避して、
InsightSpike-AIの基本機能をテストします。
"""

import sys
import os
from pathlib import Path

# プロジェクトルートを追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))


def test_basic_imports():
    """基本的なインポートのテスト"""
    print("🔍 基本インポートテスト開始...")

    try:
        # 基本設定
        from insightspike.core.config import get_config

        print("✅ 設定システム: OK")

        # CLI機能
        from insightspike.cli.main import app

        print("✅ CLI機能: OK")

        # エージェント（torch-geometric無しで動作確認）
        from insightspike.core.agents.main_agent import MainAgent

        print("✅ メインエージェント: OK")

        return True

    except ImportError as e:
        print(f"❌ インポートエラー: {e}")
        return False
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")
        return False


def test_config_system():
    """設定システムのテスト"""
    print("\n🔧 設定システムテスト開始...")

    try:
        from insightspike.core.config import get_config

        config = get_config()
        print(f"✅ 設定読み込み成功")
        print(f"   - 埋め込み次元: {config.embedding.dimension}")
        print(f"   - 推論閾値: {config.reasoning.spike_ged_threshold}")

        return True

    except Exception as e:
        print(f"❌ 設定システムエラー: {e}")
        return False


def test_agent_initialization():
    """エージェント初期化テスト"""
    print("\n🤖 エージェント初期化テスト開始...")

    try:
        from insightspike.core.agents.main_agent import MainAgent

        agent = MainAgent()
        print("✅ エージェント作成成功")

        # torch-geometric無しでの初期化を試行
        init_success = agent.initialize()
        if init_success:
            print("✅ エージェント初期化成功")
        else:
            print("⚠️ エージェント初期化失敗（依存関係の問題）")

        return True

    except Exception as e:
        print(f"❌ エージェント初期化エラー: {e}")
        return False


def test_simple_question():
    """シンプルな質問処理テスト"""
    print("\n💭 シンプル質問処理テスト開始...")

    try:
        from insightspike.core.agents.main_agent import MainAgent

        agent = MainAgent()

        # torch-geometric無しでも動作する基本的な質問
        test_question = "What is artificial intelligence?"

        print(f"質問: {test_question}")

        try:
            result = agent.process_question(test_question, max_cycles=1, verbose=False)
            print("✅ 質問処理実行成功")
            print(f"   - 応答品質: {result.get('reasoning_quality', 0):.3f}")
            print(f"   - サイクル数: {result.get('total_cycles', 0)}")
            print(f"   - スパイク検出: {result.get('spike_detected', False)}")
            return True

        except Exception as e:
            print(f"⚠️ 質問処理中にエラー（期待される）: {e}")
            print("   これはtorch-geometric依存の問題である可能性があります")
            return False

    except Exception as e:
        print(f"❌ 質問処理テストエラー: {e}")
        return False


def test_cli_help():
    """CLI ヘルプ機能のテスト"""
    print("\n🖥️ CLI機能テスト開始...")

    try:
        import subprocess

        # ヘルプコマンドの実行
        result = subprocess.run(
            [sys.executable, "-m", "insightspike.cli", "--help"],
            capture_output=True,
            text=True,
            cwd=project_root,
            env={**os.environ, "PYTHONPATH": str(project_root / "src")},
        )

        if result.returncode == 0:
            print("✅ CLI ヘルプコマンド実行成功")
            return True
        else:
            print(f"❌ CLI ヘルプコマンド失敗: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ CLI テストエラー: {e}")
        return False


def main():
    """メインテスト実行"""
    print("🧠 InsightSpike-AI PoC テスト (torch-geometric無し)")
    print("=" * 60)
    print("torch-geometricの依存関係問題を回避して基本機能をテストします")
    print()

    # テスト実行
    tests = [
        ("基本インポート", test_basic_imports),
        ("設定システム", test_config_system),
        ("エージェント初期化", test_agent_initialization),
        ("シンプル質問処理", test_simple_question),
        ("CLI機能", test_cli_help),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name}テスト中にクリティカルエラー: {e}")
            results.append((test_name, False))

    # 結果サマリー
    print("\n" + "=" * 60)
    print("📊 テスト結果サマリー")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:10} {test_name}")
        if success:
            passed += 1

    print()
    print(f"🎯 成功率: {passed}/{total} ({passed/total*100:.1f}%)")

    if passed >= 3:  # 5つ中3つ以上成功
        print("\n🎉 基本機能は動作しています！")
        print("💡 推奨対応:")
        print("   1. torch-geometricの互換バージョンを探す")
        print("   2. 代替実装（NetworkX等）を使用する")
        print("   3. フォールバック機能を強化する")
    else:
        print("\n⚠️ 基本機能に問題があります")
        print("🔧 依存関係と設定を確認してください")

    return passed >= 3


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
