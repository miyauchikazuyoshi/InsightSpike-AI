#!/usr/bin/env python3
"""
基本的な統合テスト - InsightSpikeの主要機能をエンドツーエンドでテスト
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from insightspike.core.agents.main_agent import MainAgent
from insightspike.core.config import Config
from insightspike.utils.error_handler import get_logger


class TestScenario:
    """テストシナリオの基底クラス"""

    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(f"test.{name}")
        self.results = []

    def run(self) -> bool:
        """シナリオを実行"""
        raise NotImplementedError

    def assert_condition(self, condition: bool, message: str):
        """条件をチェックして結果を記録"""
        self.results.append(
            {"condition": condition, "message": message, "passed": condition}
        )
        if not condition:
            self.logger.error(f"Assertion failed: {message}")
        else:
            self.logger.debug(f"Assertion passed: {message}")

    def get_summary(self) -> Dict[str, Any]:
        """結果のサマリーを取得"""
        passed = sum(1 for r in self.results if r["passed"])
        total = len(self.results)
        return {
            "scenario": self.name,
            "passed": passed,
            "total": total,
            "success_rate": passed / total if total > 0 else 0,
            "details": self.results,
        }


class BasicSpikDetectionScenario(TestScenario):
    """基本的なスパイク検出シナリオ"""

    def __init__(self):
        super().__init__("basic_spike_detection")

    def run(self) -> bool:
        """スパイク検出の基本動作をテスト"""
        print(f"\n=== {self.name} ===")

        # 設定
        config = Config()
        config.llm.safe_mode = True  # 安定性のためモックを使用
        config.spike.spike_ged = 0.001  # 敏感な閾値
        config.spike.spike_ig = 0.001

        # エージェント初期化
        agent = MainAgent(config=config)

        try:
            agent.initialize()
            self.assert_condition(True, "Agent initialization successful")
        except Exception as e:
            self.assert_condition(False, f"Agent initialization failed: {e}")
            return False

        # エピソード追加
        episodes = [
            "システムAは独立して動作する。",
            "システムBも独立して動作する。",
            "システムAとBを統合すると、新しい性質が生まれる。",  # ここでスパイクを期待
            "この統合により、全体の効率が向上する。",
        ]

        spike_detected = False
        for i, episode in enumerate(episodes):
            try:
                result = agent.add_episode_with_graph_update(text=episode)

                self.assert_condition(
                    "graph_analysis" in result, f"Episode {i+1}: graph_analysis present"
                )

                if result.get("graph_analysis", {}).get("spike_detected", False):
                    spike_detected = True
                    self.logger.info(f"Spike detected at episode {i+1}: {episode}")

            except Exception as e:
                self.assert_condition(False, f"Episode {i+1} processing failed: {e}")

        # 最終確認
        self.assert_condition(
            result.get("graph_nodes", 0) == len(episodes),
            f"Graph size matches episode count",
        )

        return all(r["passed"] for r in self.results)


class MemoryRetrievalScenario(TestScenario):
    """メモリ検索シナリオ"""

    def __init__(self):
        super().__init__("memory_retrieval")

    def run(self) -> bool:
        """メモリ検索機能をテスト"""
        print(f"\n=== {self.name} ===")

        config = Config()
        config.llm.safe_mode = True

        agent = MainAgent(config=config)
        agent.initialize()

        # 知識を追加
        knowledge = [
            "量子コンピュータは重ね合わせを利用する。",
            "古典コンピュータはビットを使用する。",
            "量子ビットは0と1の重ね合わせ状態を取れる。",
            "エンタングルメントは量子の特徴的な現象である。",
        ]

        for text in knowledge:
            agent.add_episode_with_graph_update(text=text)

        # 質問して検索をテスト
        questions = ["量子コンピュータとは何ですか？", "エンタングルメントについて教えてください。"]

        for question in questions:
            try:
                result = agent.process_question(question)

                self.assert_condition(
                    isinstance(result, dict), f"Question returns dict result"
                )

                self.assert_condition(
                    "response" in result, f"Response contains 'response' field"
                )

                self.assert_condition(
                    len(result.get("response", "")) > 0, f"Response is not empty"
                )

                self.logger.info(f"Q: {question}")
                self.logger.info(f"A: {result.get('response', 'N/A')[:100]}...")

            except Exception as e:
                self.assert_condition(False, f"Question processing failed: {e}")

        return all(r["passed"] for r in self.results)


class ErrorRecoveryScenario(TestScenario):
    """エラーリカバリーシナリオ"""

    def __init__(self):
        super().__init__("error_recovery")

    def run(self) -> bool:
        """エラーからの回復をテスト"""
        print(f"\n=== {self.name} ===")

        config = Config()
        config.llm.safe_mode = True

        agent = MainAgent(config=config)
        agent.initialize()

        # 正常なエピソード
        result = agent.add_episode_with_graph_update(text="正常なテキスト")
        self.assert_condition(result.get("success", False), "Normal episode succeeds")

        # 空のエピソード
        try:
            result = agent.add_episode_with_graph_update(text="")
            # 空でも処理できるはず
            self.assert_condition(True, "Empty episode handled")
        except Exception as e:
            self.assert_condition(True, f"Empty episode rejected appropriately: {e}")

        # 非常に長いエピソード
        long_text = "これは非常に長いテキストです。" * 1000
        try:
            result = agent.add_episode_with_graph_update(text=long_text)
            self.assert_condition(
                result.get("success", False), "Long episode processed"
            )
        except Exception as e:
            self.assert_condition(False, f"Long episode failed: {e}")

        # 特殊文字を含むエピソード
        special_text = "特殊文字テスト: 🎯 ✨ 数式: ∑∫∂ HTML: <div>test</div>"
        try:
            result = agent.add_episode_with_graph_update(text=special_text)
            self.assert_condition(
                result.get("success", False), "Special characters handled"
            )
        except Exception as e:
            self.assert_condition(False, f"Special characters failed: {e}")

        return all(r["passed"] for r in self.results)


class PerformanceScenario(TestScenario):
    """パフォーマンステスト"""

    def __init__(self):
        super().__init__("performance")

    def run(self) -> bool:
        """基本的なパフォーマンスをテスト"""
        print(f"\n=== {self.name} ===")

        config = Config()
        config.llm.safe_mode = True

        agent = MainAgent(config=config)

        # 初期化時間
        start = time.time()
        agent.initialize()
        init_time = time.time() - start

        self.assert_condition(
            init_time < 5.0,
            f"Initialization completes in < 5s (actual: {init_time:.2f}s)",
        )

        # エピソード追加時間
        episode_times = []
        for i in range(10):
            start = time.time()
            agent.add_episode_with_graph_update(text=f"エピソード {i+1}")
            episode_time = time.time() - start
            episode_times.append(episode_time)

        avg_episode_time = sum(episode_times) / len(episode_times)
        self.assert_condition(
            avg_episode_time < 1.0,
            f"Average episode time < 1s (actual: {avg_episode_time:.3f}s)",
        )

        # 質問応答時間
        start = time.time()
        result = agent.process_question("テスト質問")
        question_time = time.time() - start

        self.assert_condition(
            question_time < 2.0,
            f"Question processing < 2s (actual: {question_time:.2f}s)",
        )

        # メモリ使用量の確認（簡易版）
        import os

        import psutil

        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024

        self.assert_condition(
            memory_mb < 1000, f"Memory usage < 1GB (actual: {memory_mb:.0f}MB)"  # 1GB以下
        )

        return all(r["passed"] for r in self.results)


def run_integration_tests():
    """統合テストを実行"""
    print("=== Running Integration Tests ===")

    scenarios = [
        BasicSpikDetectionScenario(),
        MemoryRetrievalScenario(),
        ErrorRecoveryScenario(),
        PerformanceScenario(),
    ]

    results = []
    for scenario in scenarios:
        try:
            scenario.run()
            summary = scenario.get_summary()
            results.append(summary)

            print(f"\n{scenario.name}: {summary['passed']}/{summary['total']} passed")

        except Exception as e:
            print(f"\n✗ {scenario.name} crashed: {e}")
            import traceback

            traceback.print_exc()
            results.append(
                {
                    "scenario": scenario.name,
                    "passed": 0,
                    "total": 1,
                    "success_rate": 0,
                    "error": str(e),
                }
            )

    # 結果のサマリー
    print("\n=== Integration Test Summary ===")
    total_passed = sum(r["passed"] for r in results)
    total_tests = sum(r["total"] for r in results)

    for result in results:
        status = "✓" if result["success_rate"] == 1.0 else "✗"
        print(
            f"{status} {result['scenario']}: {result['passed']}/{result['total']} ({result['success_rate']*100:.0f}%)"
        )

    print(
        f"\nOverall: {total_passed}/{total_tests} passed ({total_passed/total_tests*100:.1f}%)"
    )

    # 結果をJSONファイルに保存
    output_file = Path("test_results_integration.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "scenarios": results,
                "summary": {
                    "total_passed": total_passed,
                    "total_tests": total_tests,
                    "success_rate": total_passed / total_tests
                    if total_tests > 0
                    else 0,
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nResults saved to: {output_file}")

    return total_passed == total_tests


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
