#!/usr/bin/env python3
"""
LLMプロバイダーの使い方サンプル
================================
"""

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.insightspike.config import load_config
from src.insightspike.implementations.datastore.sqlite_store import SQLiteDataStore
from src.insightspike.implementations.agents.datastore_agent import DataStoreMainAgent


def example_openai():
    """OpenAI (GPT)を使う例"""
    print("=== OpenAI Provider Example ===\n")
    
    # 1. 環境変数を設定
    # export OPENAI_API_KEY="your-api-key"
    
    # 2. 設定を読み込み
    config = load_config(config_path="./config_openai.yaml")
    
    # 3. DataStoreとAgentを初期化
    datastore = SQLiteDataStore("./data/sqlite/openai_example.db")
    agent = DataStoreMainAgent(datastore, config)
    
    # 4. 使ってみる
    result = agent.process("量子コンピューティングの基本原理について説明してください")
    
    print(f"Response: {result.get('response', 'No response')[:200]}...")
    print(f"Has spike: {result.get('has_spike', False)}")


def example_anthropic():
    """Anthropic (Claude)を使う例"""
    print("\n=== Anthropic Provider Example ===\n")
    
    # 1. 環境変数を設定
    # export ANTHROPIC_API_KEY="your-api-key"
    
    # 2. 設定を読み込み
    config = load_config(config_path="./config_anthropic.yaml")
    
    # 3. DataStoreとAgentを初期化
    datastore = SQLiteDataStore("./data/sqlite/anthropic_example.db")
    agent = DataStoreMainAgent(datastore, config)
    
    # 4. 使ってみる
    result = agent.process("機械学習と深層学習の違いは何ですか？")
    
    print(f"Response: {result.get('response', 'No response')[:200]}...")
    print(f"Has spike: {result.get('has_spike', False)}")


def example_with_custom_settings():
    """カスタム設定での例"""
    print("\n=== Custom Settings Example ===\n")
    
    # Pythonコードで設定を上書き
    from src.insightspike.config.models import InsightSpikeConfig, LLMConfig
    
    config = InsightSpikeConfig(
        llm=LLMConfig(
            provider="openai",
            model="gpt-4",  # GPT-4を使用
            temperature=0.2,  # より決定的な回答
            max_tokens=2000,  # より長い回答
            api_key=os.getenv("OPENAI_API_KEY")
        )
    )
    
    datastore = SQLiteDataStore("./data/sqlite/custom_example.db")
    agent = DataStoreMainAgent(datastore, config)
    
    result = agent.process("InsightSpikeのアーキテクチャを説明してください")
    print(f"Response: {result.get('response', 'No response')[:200]}...")


def example_cli_usage():
    """CLIでの使い方"""
    print("\n=== CLI Usage Examples ===\n")
    
    print("# OpenAIを使う場合:")
    print("export OPENAI_API_KEY='your-key'")
    print("spike query --config config_openai.yaml '質問内容'")
    print()
    
    print("# Anthropicを使う場合:")
    print("export ANTHROPIC_API_KEY='your-key'")
    print("spike query --config config_anthropic.yaml '質問内容'")
    print()
    
    print("# 環境変数で直接指定:")
    print("OPENAI_API_KEY='your-key' spike query --llm-provider openai '質問内容'")


def check_providers():
    """利用可能なプロバイダーをチェック"""
    print("\n=== Available Providers ===\n")
    
    from src.insightspike.providers import ProviderFactory
    
    providers = ['mock', 'openai', 'anthropic']
    
    for provider in providers:
        try:
            p = ProviderFactory.create(provider)
            print(f"✓ {provider}: Available")
        except ImportError:
            print(f"✗ {provider}: Dependencies not installed")
        except ValueError:
            print(f"⚠ {provider}: API key required")
        except Exception as e:
            print(f"✗ {provider}: {e}")


if __name__ == "__main__":
    # プロバイダーの状態をチェック
    check_providers()
    
    # 使い方の例を表示
    example_cli_usage()
    
    # 実際に動かす例（APIキーが必要）
    if os.getenv("OPENAI_API_KEY"):
        example_openai()
    else:
        print("\n⚠️  OPENAI_API_KEY not set, skipping OpenAI example")
    
    if os.getenv("ANTHROPIC_API_KEY"):
        example_anthropic()
    else:
        print("\n⚠️  ANTHROPIC_API_KEY not set, skipping Anthropic example")