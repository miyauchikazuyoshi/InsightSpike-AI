#!/usr/bin/env python3
"""
最適化された実験実行スクリプト
==============================

実験の成功率を最大化するための実行スクリプト
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.insightspike.config import load_config
from src.insightspike.implementations.datastore.sqlite_store import SQLiteDataStore
from src.insightspike.implementations.agents.datastore_agent import DataStoreMainAgent


def prepare_experiment(experiment_name: str):
    """実験の準備"""
    print(f"=== 実験準備: {experiment_name} ===\n")
    
    # 1. 実験用ディレクトリの作成
    exp_dir = Path(f"experiments/{experiment_name}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. データベースの準備
    db_path = exp_dir / "data" / "experiment.db"
    db_path.parent.mkdir(exist_ok=True)
    
    # 3. 結果保存先の準備
    results_dir = exp_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    return exp_dir, db_path, results_dir


def run_insightspike_experiment(db_path: Path, results_dir: Path):
    """InsightSpike実験の実行"""
    print("\n=== InsightSpike実験 ===\n")
    
    # 1. 設定の読み込み
    config = load_config(config_path="./config_experiment_optimized.yaml")
    
    # 2. DataStoreとAgentの初期化
    datastore = SQLiteDataStore(str(db_path))
    agent = DataStoreMainAgent(datastore, config)
    
    # 3. 実験データの準備
    experiment_data = [
        # 基礎知識の投入
        "量子力学では、粒子は観測されるまで複数の状態の重ね合わせとして存在する",
        "ニューラルネットワークは、脳のシナプス結合を模倣した計算モデルである",
        "進化は、自然選択によって有利な形質が次世代に受け継がれるプロセスである",
        "意識は、主観的な経験と自己認識を含む複雑な現象である",
        "情報理論では、エントロピーは情報の不確実性を測る尺度である",
        
        # 関連する概念
        "量子もつれは、複数の粒子が相関を持つ非局所的な現象である",
        "深層学習は、多層のニューラルネットワークを用いた機械学習手法である",
        "創発とは、個々の要素の相互作用から予測不可能な性質が生まれることである",
        "量子コンピュータは、量子の重ね合わせを利用した新しい計算パラダイムである",
        "脳の可塑性により、ニューラルネットワークは経験に基づいて変化する",
    ]
    
    results = []
    
    # 4. 知識の投入と処理
    print("知識を投入中...")
    for i, text in enumerate(experiment_data):
        start_time = time.time()
        result = agent.process(text)
        processing_time = time.time() - start_time
        
        results.append({
            'index': i,
            'text': text,
            'has_spike': result.get('has_spike', False),
            'spike_confidence': result.get('spike_info', {}).get('confidence', 0),
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat()
        })
        
        if result.get('has_spike'):
            print(f"  ✨ インサイト検出: {text[:50]}...")
        else:
            print(f"  📝 知識追加: {text[:50]}...")
    
    # 5. 洞察を引き出す質問
    print("\n洞察を引き出す質問を処理中...")
    insight_questions = [
        "量子もつれと意識の間に関連性はありますか？",
        "ニューラルネットワークと量子力学の類似点は何ですか？",
        "創発現象は意識の発生を説明できますか？",
        "量子コンピュータは人工意識の実現に貢献できますか？",
        "情報エントロピーと意識の複雑性に関係はありますか？"
    ]
    
    for question in insight_questions:
        start_time = time.time()
        result = agent.process(question)
        processing_time = time.time() - start_time
        
        results.append({
            'index': len(results),
            'text': question,
            'has_spike': result.get('has_spike', False),
            'spike_confidence': result.get('spike_info', {}).get('confidence', 0),
            'response': result.get('response', ''),
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"\n質問: {question}")
        if result.get('has_spike'):
            print(f"✨ インサイトスパイク検出！（確信度: {result.get('spike_info', {}).get('confidence', 0):.2f}）")
        print(f"回答: {result.get('response', 'No response')[:200]}...")
    
    # 6. 結果の保存
    results_file = results_dir / f"insightspike_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n結果を保存しました: {results_file}")
    
    # 7. 統計情報
    total_spikes = sum(1 for r in results if r.get('has_spike'))
    avg_confidence = sum(r.get('spike_confidence', 0) for r in results if r.get('has_spike')) / max(total_spikes, 1)
    
    print(f"\n=== 実験統計 ===")
    print(f"総処理数: {len(results)}")
    print(f"インサイトスパイク数: {total_spikes}")
    print(f"平均確信度: {avg_confidence:.2f}")
    print(f"データベースサイズ: {datastore.get_stats()['db_size_bytes'] / 1024 / 1024:.2f} MB")
    
    return results


def compare_with_baseline(results_dir: Path):
    """ベースラインとの比較（オプション）"""
    print("\n=== ベースライン比較 ===\n")
    
    # ここではモックプロバイダーと比較
    from src.insightspike.config.models import InsightSpikeConfig, LLMConfig
    
    mock_config = InsightSpikeConfig(
        llm=LLMConfig(provider="mock", model="mock-model")
    )
    
    # 簡単な比較実装...
    print("（ベースライン比較は別途実装）")


def main():
    """メイン実行関数"""
    print("InsightSpike最適化実験")
    print("=" * 50)
    
    # APIキーの確認
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("\n⚠️  警告: LLM APIキーが設定されていません")
        print("MockProviderで実行されます")
        print("\n最良の結果を得るには:")
        print("export OPENAI_API_KEY='your-key'")
        print("または")
        print("export ANTHROPIC_API_KEY='your-key'")
        input("\nEnterキーで続行...")
    
    # 実験名の設定
    experiment_name = f"optimized_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 実験の準備
    exp_dir, db_path, results_dir = prepare_experiment(experiment_name)
    
    # 実験の実行
    results = run_insightspike_experiment(db_path, results_dir)
    
    # ベースラインとの比較（オプション）
    # compare_with_baseline(results_dir)
    
    print(f"\n✅ 実験完了！")
    print(f"結果は以下に保存されています: {exp_dir}")


if __name__ == "__main__":
    main()