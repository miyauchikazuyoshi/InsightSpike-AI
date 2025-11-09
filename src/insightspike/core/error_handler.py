"""
統一エラーハンドリングとロギング
================================

InsightSpike全体で使用する共通のエラーハンドリングとロギング機能
"""

import json
import logging
import os
import traceback
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional


# カスタム例外クラス
class InsightSpikeError(Exception):
    """InsightSpike基底例外"""

    pass


class ConfigurationError(InsightSpikeError):
    """設定関連のエラー"""

    pass


class ModelNotFoundError(InsightSpikeError):
    """モデルが見つからない"""

    pass


class InitializationError(InsightSpikeError):
    """初期化エラー"""

    pass


class ProcessingError(InsightSpikeError):
    """処理中のエラー"""

    pass


class InsightSpikeLogger:
    """統一ロギングクラス"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._setup_logging()
            self._initialized = True

    def _setup_logging(self):
        """ロギングの設定（クラウド安全・相対パス優先）"""
        # 1) 決定順: 環境変数 -> プロジェクト相対 results/logs -> ホーム
        env_dir = os.getenv("INSIGHTSPIKE_LOG_DIR")
        log_dir: Path
        log_file: Path
        resolved = None
        if env_dir:
            try:
                # 可能ならプロジェクト相対に解決
                try:
                    from ..utils.path_utils import resolve_project_relative  # type: ignore
                except Exception:  # フォールバック
                    resolve_project_relative = lambda p: str(Path(p))  # type: ignore
                resolved = Path(resolve_project_relative(env_dir))
            except Exception:
                resolved = Path(env_dir).expanduser()
        if not resolved:
            try:
                from ..utils.path_utils import resolve_project_relative  # type: ignore
                resolved = Path(resolve_project_relative("results/logs"))
            except Exception:
                resolved = Path.cwd() / "results" / "logs"
        log_dir = resolved

        # 2) ディレクトリ作成は失敗しても致命にしない
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # 最後のフォールバック: ホーム配下。それもダメならコンソールのみ。
            try:
                home_dir = Path.home() / ".insightspike" / "logs"
                home_dir.mkdir(parents=True, exist_ok=True)
                log_dir = home_dir
            except Exception:
                log_dir = None  # type: ignore

        # 3) ロガーベース設定
        self.logger = logging.getLogger("insightspike")
        self.logger.setLevel(logging.DEBUG)

        # コンソールハンドラー（重要ログ）
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        simple_formatter = logging.Formatter("%(levelname)s: %(message)s")
        ch.setFormatter(simple_formatter)

        # 4) ファイルハンドラー（Lite/Minモードではスキップ可）
        fh = None
        lite_mode = os.getenv("INSIGHTSPIKE_LITE_MODE") == "1" or os.getenv("INSIGHTSPIKE_MIN_IMPORT") == "1"
        if not lite_mode and log_dir is not None:
            try:
                log_file = log_dir / f"insightspike_{datetime.now().strftime('%Y%m%d')}.log"
                fh = logging.FileHandler(log_file, encoding="utf-8")
                fh.setLevel(logging.DEBUG)
                detailed_formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
                )
                fh.setFormatter(detailed_formatter)
            except Exception:
                fh = None  # コンソールのみ

        # 5) ハンドラーの追加（既存のものは削除）
        self.logger.handlers.clear()
        if fh is not None:
            self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        # 初期メッセージ（ファイル出力がない場合も明示）
        if fh is not None:
            self.logger.info(f"InsightSpike logging initialized. Log dir: {log_dir}")
        else:
            self.logger.info("InsightSpike logging initialized (console-only mode)")

    def get_logger(self, name: str) -> logging.Logger:
        """名前付きロガーを取得"""
        return logging.getLogger(f"insightspike.{name}")


# グローバルロガーインスタンス
_logger_instance = InsightSpikeLogger()


def get_logger(name: str) -> logging.Logger:
    """ロガーを取得する便利関数"""
    return _logger_instance.get_logger(name)


def handle_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
    user_message: Optional[str] = None,
) -> Dict[str, Any]:
    """
    エラーを処理して構造化された応答を返す

    Args:
        error: 発生した例外
        context: エラーのコンテキスト情報
        logger: 使用するロガー（Noneの場合はデフォルト）
        user_message: ユーザーに表示するメッセージ

    Returns:
        エラー情報を含む辞書
    """
    if logger is None:
        logger = get_logger("error_handler")

    # エラーの詳細情報を収集
    error_info = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "timestamp": datetime.now().isoformat(),
        "context": context or {},
    }

    # スタックトレースを取得
    tb = traceback.format_exc()
    error_info["traceback"] = tb

    # ログに記録
    if isinstance(error, InsightSpikeError):
        # 既知のエラー
        logger.warning(f"{error_info['error_type']}: {error}")
        if context:
            logger.debug(f"Context: {json.dumps(context, ensure_ascii=False)}")
    else:
        # 予期しないエラー
        logger.error(f"Unexpected error: {error}")
        logger.error(tb)

    # ユーザー向けメッセージ
    if user_message:
        error_info["user_message"] = user_message
    else:
        if isinstance(error, ConfigurationError):
            error_info["user_message"] = f"設定エラー: {error}"
        elif isinstance(error, ModelNotFoundError):
            error_info["user_message"] = f"モデルが見つかりません: {error}"
        elif isinstance(error, InitializationError):
            error_info["user_message"] = f"初期化エラー: {error}"
        elif isinstance(error, ProcessingError):
            error_info["user_message"] = f"処理エラー: {error}"
        else:
            error_info["user_message"] = "予期しないエラーが発生しました。ログを確認してください。"

    return error_info


def with_error_handling(
    default_return: Any = None,
    error_class: type = ProcessingError,
    log_errors: bool = True,
):
    """
    エラーハンドリングデコレーター

    使用例:
        @with_error_handling(default_return={}, error_class=ProcessingError)
        def process_data(data):
            # 処理
            return result
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)

            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    handle_error(
                        e,
                        context={
                            "function": func.__name__,
                            "args": str(args)[:100],
                            "kwargs": str(kwargs)[:100],
                        },
                        logger=logger,
                    )

                # InsightSpikeErrorでなければ、指定されたエラークラスで再発生
                if not isinstance(e, InsightSpikeError):
                    raise error_class(f"Error in {func.__name__}: {e}") from e
                else:
                    raise

        return wrapper

    return decorator


def validate_config(config: Dict[str, Any], required_fields: Dict[str, type]) -> None:
    """
    設定の検証を行う

    Args:
        config: 検証する設定辞書
        required_fields: 必須フィールドとその型

    Raises:
        ConfigurationError: 設定が不正な場合
    """
    logger = get_logger("config_validator")

    for field, expected_type in required_fields.items():
        if field not in config:
            raise ConfigurationError(f"Required field '{field}' is missing")

        if not isinstance(config[field], expected_type):
            raise ConfigurationError(
                f"Field '{field}' must be of type {expected_type.__name__}, "
                f"got {type(config[field]).__name__}"
            )

    logger.debug(f"Config validation passed for fields: {list(required_fields.keys())}")


def setup_debug_mode():
    """デバッグモードの設定"""
    # 環境変数でデバッグモードを制御
    if os.environ.get("INSIGHTSPIKE_DEBUG", "").lower() == "true":
        logger = get_logger("debug")
        logger.info("Debug mode enabled")

        # すべてのログレベルをDEBUGに
        logging.getLogger("insightspike").setLevel(logging.DEBUG)

        # より詳細なフォーマッター
        for handler in logging.getLogger("insightspike").handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(logging.DEBUG)


# モジュール初期化時にデバッグモードをチェック
setup_debug_mode()
