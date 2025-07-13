#!/usr/bin/env python3
"""
エラーハンドリングのテスト
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from insightspike.utils.error_handler import (
    ConfigurationError,
    InitializationError,
    InsightSpikeError,
    ModelNotFoundError,
    ProcessingError,
    get_logger,
    handle_error,
    validate_config,
    with_error_handling,
)


def test_custom_exceptions():
    """カスタム例外のテスト"""
    print("\n=== Custom Exceptions Test ===")

    # 基底例外
    try:
        raise InsightSpikeError("Base error")
    except InsightSpikeError as e:
        assert str(e) == "Base error"
        print("✓ Base exception works")

    # 各種例外
    exceptions = [
        (ConfigurationError, "Config error"),
        (ModelNotFoundError, "Model not found"),
        (InitializationError, "Init failed"),
        (ProcessingError, "Processing failed"),
    ]

    for exc_class, msg in exceptions:
        try:
            raise exc_class(msg)
        except exc_class as e:
            assert str(e) == msg
            assert isinstance(e, InsightSpikeError)

    print("✓ All custom exceptions work correctly")


def test_logging():
    """ロギング機能のテスト"""
    print("\n=== Logging Test ===")

    # ロガー取得
    logger = get_logger("test")
    assert logger.name == "insightspike.test"

    # 各レベルでログ出力
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

    # ログファイルの存在確認
    log_dir = Path.home() / ".insightspike" / "logs"
    assert log_dir.exists()

    log_files = list(log_dir.glob("insightspike_*.log"))
    assert len(log_files) > 0

    print(f"✓ Log file created: {log_files[0]}")


def test_handle_error():
    """エラーハンドリング関数のテスト"""
    print("\n=== Handle Error Test ===")

    logger = get_logger("test_handler")

    # 通常の例外
    try:
        1 / 0
    except ZeroDivisionError as e:
        error_info = handle_error(e, context={"operation": "division"}, logger=logger)

    assert error_info["error_type"] == "ZeroDivisionError"
    assert "traceback" in error_info
    assert error_info["context"]["operation"] == "division"
    print("✓ Regular exception handled")

    # カスタム例外
    try:
        raise ConfigurationError("Bad config")
    except ConfigurationError as e:
        error_info = handle_error(e, logger=logger)

    assert error_info["error_type"] == "ConfigurationError"
    assert "設定エラー" in error_info["user_message"]
    print("✓ Custom exception handled")


def test_error_decorator():
    """エラーハンドリングデコレーターのテスト"""
    print("\n=== Error Decorator Test ===")

    # 正常な関数
    @with_error_handling(default_return=42)
    def good_function(x):
        return x * 2

    result = good_function(5)
    assert result == 10
    print("✓ Normal function works")

    # エラーを起こす関数
    @with_error_handling(default_return=None, error_class=ProcessingError)
    def bad_function(x):
        return 1 / x

    try:
        bad_function(0)
        assert False, "Should have raised an error"
    except ProcessingError as e:
        assert "Error in bad_function" in str(e)
        print("✓ Error wrapped correctly")


def test_validate_config():
    """設定検証のテスト"""
    print("\n=== Config Validation Test ===")

    # 正しい設定
    config = {"name": "test", "value": 42, "enabled": True}

    required = {"name": str, "value": int, "enabled": bool}

    validate_config(config, required)
    print("✓ Valid config passed")

    # 不足フィールド
    try:
        validate_config({"name": "test"}, required)
        assert False, "Should have raised ConfigurationError"
    except ConfigurationError as e:
        assert "Required field 'value' is missing" in str(e)
        print("✓ Missing field detected")

    # 型エラー
    try:
        bad_config = {"name": "test", "value": "not an int", "enabled": True}
        validate_config(bad_config, required)
        assert False, "Should have raised ConfigurationError"
    except ConfigurationError as e:
        assert "must be of type int" in str(e)
        print("✓ Type error detected")


def test_debug_mode():
    """デバッグモードのテスト"""
    print("\n=== Debug Mode Test ===")

    # デバッグモードを有効化
    os.environ["INSIGHTSPIKE_DEBUG"] = "true"

    # error_handlerモジュールを再インポート
    import importlib

    import insightspike.utils.error_handler as eh

    importlib.reload(eh)

    logger = eh.get_logger("debug_test")

    # デバッグレベルのログが出力されることを確認
    import logging

    assert logging.getLogger("insightspike").level == logging.DEBUG

    print("✓ Debug mode enabled correctly")

    # クリーンアップ
    del os.environ["INSIGHTSPIKE_DEBUG"]


def run_all_tests():
    """すべてのテストを実行"""
    print("=== Running Error Handler Tests ===")

    tests = [
        test_custom_exceptions,
        test_logging,
        test_handle_error,
        test_error_decorator,
        test_validate_config,
        test_debug_mode,
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
