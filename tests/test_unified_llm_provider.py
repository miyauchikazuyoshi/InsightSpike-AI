#!/usr/bin/env python3
"""
統一LLMプロバイダーのテスト
"""

import sys
from pathlib import Path

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from insightspike.core.config import Config
from insightspike.core.layers.unified_llm_provider import (
    LocalLLMProvider,
    MockLLMProvider,
    UnifiedLLMProvider,
)


def test_mock_provider():
    """モックプロバイダーのテスト"""
    print("\n=== MockLLMProvider Test ===")

    config = Config()
    config.llm.safe_mode = True

    provider = UnifiedLLMProvider.create(config)
    assert isinstance(provider, MockLLMProvider)

    # 初期化
    assert provider.initialize() == True

    # 生成テスト
    context = {"retrieved_documents": [{"text": "カオス理論について"}]}
    result = provider.generate_response(context, "カオスの縁と創造性の関係は？")

    print(f"Success: {result['success']}")
    print(f"Response: {result['response'][:100]}...")
    assert result["success"] == True
    assert len(result["response"]) > 0

    print("✓ MockLLMProvider test passed")


def test_local_provider_validation():
    """ローカルプロバイダーの検証テスト"""
    print("\n=== LocalLLMProvider Validation Test ===")

    config = Config()
    config.llm.safe_mode = False
    config.llm.provider = "local"
    config.llm.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    config.llm.device = "cpu"

    # 設定の検証
    is_valid = UnifiedLLMProvider.validate_config(config)
    print(f"Config validation: {is_valid}")

    # プロバイダー作成
    provider = UnifiedLLMProvider.create(config)
    assert isinstance(provider, LocalLLMProvider)

    print("✓ LocalLLMProvider validation passed")


def test_error_handling():
    """エラーハンドリングのテスト"""
    print("\n=== Error Handling Test ===")

    config = Config()
    config.llm.safe_mode = False
    config.llm.provider = "invalid_provider"

    # 無効なプロバイダーでもモックにフォールバック
    provider = UnifiedLLMProvider.create(config)
    assert isinstance(provider, MockLLMProvider)

    print("✓ Error handling test passed")


def test_fallback_response():
    """フォールバック応答のテスト"""
    print("\n=== Fallback Response Test ===")

    config = Config()
    config.llm.safe_mode = True

    # モックプロバイダーを使用して、強制的にエラーを発生させる
    provider = MockLLMProvider(config)

    # generateメソッドを強制的にエラーにする
    def error_generate(prompt, **kwargs):
        raise Exception("Test error")

    provider.initialize()
    provider.generate = error_generate

    result = provider.generate_response({}, "テスト質問")

    print(f"Success: {result['success']}")
    print(f"Error: {result.get('error', 'N/A')}")
    print(f"Fallback response: {result['response']}")

    assert result["success"] == False
    assert "error" in result
    assert len(result["response"]) > 0

    print("✓ Fallback response test passed")


def test_prompt_building():
    """プロンプト構築のテスト"""
    print("\n=== Prompt Building Test ===")

    config = Config()
    config.llm.safe_mode = True

    provider = UnifiedLLMProvider.create(config)
    provider.initialize()

    # コンテキストありの場合
    context = {
        "retrieved_documents": [{"text": "エントロピーは無秩序の尺度"}, {"text": "カオスの縁は創造的な領域"}]
    }

    result = provider.generate_response(context, "エントロピーとは？")

    print(f"Prompt preview:\n{result['prompt'][:200]}...")
    assert "コンテキスト" in result["prompt"]
    assert "エントロピー" in result["prompt"]

    print("✓ Prompt building test passed")


def run_all_tests():
    """すべてのテストを実行"""
    print("=== Running Unified LLM Provider Tests ===")

    tests = [
        test_mock_provider,
        test_local_provider_validation,
        test_error_handling,
        test_fallback_response,
        test_prompt_building,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n✗ {test.__name__} failed: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print(f"\n=== Test Summary ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {len(tests)}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
