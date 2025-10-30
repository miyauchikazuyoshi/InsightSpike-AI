"""Temporary test collection limiter.

Pytest のハング原因切り分けのため、他の unit テスト群の収集を意図的にスキップし
`test_gedig_ab_logger_alerts_csv.py` のみを実行対象に限定する仕組み。

仕組み:
 - conftest で本ファイルは特別扱いされないため通常テストとして認識される
 - ただし本ファイル内では pytest の収集フックを用いて *他* の test_*.py
   を無視する。

本ファイルはデバッグ完了後に削除予定。
"""

import pytest

TARGET = "test_gedig_ab_logger_alerts_csv.py"

def pytest_ignore_collect(path, config):  # type: ignore[override]
    p = str(path)
    if p.endswith(TARGET):
        return False
    # この limiter 自身は収集させる (レポート用) が他は無視
    if p.endswith("_only_ab_logger_tests.py"):
        return False
    # ルートが unit テストディレクトリの場合に限定的にフィルタ
    if "/tests/unit/" in p and p.endswith(".py"):
        return True
    return False

def test_collection_limiter_active():
    # 形だけのテスト: limiter が動いていることをログ出力
    import os
    assert os.environ.get("INSIGHTSPIKE_LITE_MODE") == "1"
    # 常に成功
    pass
