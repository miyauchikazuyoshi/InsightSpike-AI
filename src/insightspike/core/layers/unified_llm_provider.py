"""
Unified LLM Provider - シンプルで堅牢なLLMインターフェース
========================================================

すべてのLLMプロバイダーを統一的に扱い、エラーに強い実装を提供します。
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Iterator
import os

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """LLMプロバイダーの基底クラス"""

    def __init__(self, config: Any):
        self.config = config
        self._initialized = False
        self.model_name = config.llm.model_name
        self.device = config.llm.device or "cpu"
        self.max_tokens = config.llm.max_tokens or 256
        self.temperature = config.llm.temperature or 0.3

    def ensure_initialized(self):
        """初期化を保証"""
        if not self._initialized:
            self.initialize()

    @abstractmethod
    def initialize(self) -> bool:
        """プロバイダーの初期化"""
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """テキスト生成"""
        pass

    def generate_response(
        self, context: Dict[str, Any], question: str
    ) -> Dict[str, Any]:
        """統一的なレスポンス生成インターフェース"""
        try:
            self.ensure_initialized()

            # プロンプト構築
            prompt = self._build_prompt(context, question)

            # 生成
            response = self.generate(prompt)

            return {
                "response": response,
                "success": True,
                "prompt": prompt,
                "model": self.model_name,
                "confidence": self._estimate_confidence(response),
            }

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {
                "response": self._get_fallback_response(question),
                "success": False,
                "error": str(e),
                "model": self.model_name,
            }

    def _build_prompt(self, context: Dict[str, Any], question: str) -> str:
        """プロンプトの構築"""
        prompt_parts = []

        # コンテキストがある場合
        if context.get("retrieved_documents"):
            prompt_parts.append("以下のコンテキストを参考に質問に答えてください。\n")
            prompt_parts.append("コンテキスト:\n")

            for i, doc in enumerate(context["retrieved_documents"][:5]):
                text = doc.get("text", str(doc))
                prompt_parts.append(f"{i+1}. {text}\n")

            prompt_parts.append("\n")

        # 質問
        prompt_parts.append(f"質問: {question}\n")
        prompt_parts.append("回答: ")

        return "".join(prompt_parts)

    def _estimate_confidence(self, response: str) -> float:
        """信頼度の推定"""
        if not response:
            return 0.0

        # 応答の長さによる基本スコア
        length_score = min(1.0, len(response) / 100)

        # 不確実性キーワードのチェック
        uncertain_words = ["かもしれません", "おそらく", "不明", "わかりません"]
        uncertainty_penalty = sum(0.1 for word in uncertain_words if word in response)

        confidence = max(0.1, min(1.0, length_score - uncertainty_penalty))
        return confidence

    def _get_fallback_response(self, question: str) -> str:
        """フォールバック応答"""
        return f"申し訳ありません。「{question[:30]}...」に対する応答を生成できませんでした。"


class MockLLMProvider(BaseLLMProvider):
    """テスト用モックプロバイダー"""

    def initialize(self) -> bool:
        """即座に成功"""
        self._initialized = True
        logger.info("MockLLMProvider initialized")
        return True

    def generate(self, prompt: str, **kwargs) -> str:
        """定型応答を返す"""
        if "カオス" in prompt and "創造性" in prompt:
            return "カオスの縁では、秩序と無秩序のバランスが創造性を生み出します。"
        elif "エントロピー" in prompt:
            return "エントロピーは系の無秩序さを表す指標です。"
        else:
            return "興味深い質問です。提供されたコンテキストに基づいて考察します。"


class LocalLLMProvider(BaseLLMProvider):
    """ローカルモデルプロバイダー（改善版）"""

    def __init__(self, config: Any):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.pipeline = None

    def initialize(self) -> bool:
        """Transformersモデルの初期化"""
        try:
            # 必要なライブラリのインポート
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            except ImportError:
                logger.error(
                    "transformers not installed. Run: pip install transformers torch"
                )
                return False

            # CPUを明示的に使用
            device = "cpu"
            torch.set_num_threads(4)  # CPU使用時のスレッド数を制限

            logger.info(f"Loading model: {self.model_name}")

            # トークナイザーの読み込み
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True, use_fast=True
            )

            # パディングトークンの設定
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # モデルの読み込み（CPU用に最適化）
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32,  # CPUではfloat32を使用
                low_cpu_mem_usage=True,
                device_map={"": device},
            )

            # パイプラインの作成
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            self._initialized = True
            logger.info(f"Successfully loaded {self.model_name} on {device}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize local model: {e}")
            return False

    def generate(self, prompt: str, **kwargs) -> str:
        """ローカルモデルでの生成"""
        if not self._initialized:
            raise RuntimeError("Provider not initialized")

        try:
            # TinyLlama用のプロンプトフォーマット
            formatted_prompt = f"<|system|>\nYou are a helpful assistant.\n<|user|>\n{prompt}\n<|assistant|>\n"

            # 生成
            outputs = self.pipeline(
                formatted_prompt,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False,
                return_full_text=False,
            )

            generated_text = outputs[0]["generated_text"]

            # クリーンアップ
            generated_text = self._clean_response(generated_text)

            return generated_text

        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise

    def _clean_response(self, text: str) -> str:
        """応答のクリーンアップ"""
        # 特殊トークンの除去
        stop_tokens = ["<|", "</s>", "<s>", "[END]", "[STOP]", "\n\n\n"]
        for token in stop_tokens:
            if token in text:
                text = text.split(token)[0]

        return text.strip()


class UnifiedLLMProvider:
    """統一LLMプロバイダー - すべてのプロバイダーを管理"""

    @staticmethod
    def create(config: Any) -> BaseLLMProvider:
        """設定に基づいて適切なプロバイダーを作成"""

        # safe_modeまたはテストモード
        if config.llm.safe_mode or os.environ.get("INSIGHTSPIKE_TEST_MODE"):
            logger.info("Using MockLLMProvider (safe_mode)")
            return MockLLMProvider(config)

        # プロバイダーの選択
        provider_type = config.llm.provider.lower()

        if provider_type == "local":
            logger.info("Using LocalLLMProvider")
            return LocalLLMProvider(config)
        elif provider_type == "openai":
            # 将来の実装用
            logger.warning("OpenAI provider not implemented, falling back to mock")
            return MockLLMProvider(config)
        else:
            logger.warning(f"Unknown provider {provider_type}, using mock")
            return MockLLMProvider(config)

    @staticmethod
    def validate_config(config: Any) -> bool:
        """設定の検証"""
        required_fields = ["provider", "model_name", "device"]

        for field in required_fields:
            if not hasattr(config.llm, field):
                logger.error(f"Missing required config field: llm.{field}")
                return False

        # モデルの存在確認（ローカルの場合）
        if config.llm.provider == "local" and not config.llm.safe_mode:
            try:
                from transformers import AutoTokenizer

                # トークナイザーだけ試しに読み込んでみる
                AutoTokenizer.from_pretrained(config.llm.model_name)
                return True
            except Exception as e:
                logger.error(f"Model {config.llm.model_name} not available: {e}")
                return False

        return True


# 既存コードとの互換性のため
def get_llm_provider(config: Any) -> BaseLLMProvider:
    """互換性のための関数"""
    return UnifiedLLMProvider.create(config)
