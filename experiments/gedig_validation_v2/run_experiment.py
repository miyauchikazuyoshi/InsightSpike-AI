"""
geDIG理論検証実験 v2.0 実行スクリプト
==================================

詳細なログ記録と物理量の追跡を含む実験実行。
"""

import os
import json
import time
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# プロジェクトルートへのパスを設定
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from insightspike.core.agents.main_agent import MainAgent
from insightspike.core.config import Config
from insightspike.algorithms.information_gain import InformationGain, EntropyMethod
from insightspike.algorithms.graph_edit_distance import GraphEditDistance
from insightspike.algorithms.entropy_calculator import EntropyCalculator

# ロギング設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PhysicsMetricsCalculator:
    """物理量の計算と記録"""
    
    def __init__(self):
        self.ig_calculator = InformationGain(method=EntropyMethod.SHANNON)
        self.ged_calculator = GraphEditDistance()
        self.entropy_calculator = EntropyCalculator()
    
    def calculate_metrics(self, before_state: Any, after_state: Any) -> Dict[str, Dict[str, Any]]:
        """物理量を計算して単位付きで返す"""
        
        metrics = {}
        
        # 情報利得の計算
        try:
            ig_result = self.ig_calculator.calculate(before_state, after_state)
            metrics["delta_ig"] = {
                "value": float(ig_result.ig_value),
                "unit": "bits",
                "method": "shannon_entropy",
                "entropy_before": float(ig_result.entropy_before),
                "entropy_after": float(ig_result.entropy_after)
            }
        except Exception as e:
            logger.error(f"IG calculation failed: {e}")
            metrics["delta_ig"] = {"value": 0.0, "unit": "bits", "error": str(e)}
        
        # グラフ編集距離の計算（簡易版）
        try:
            # ここでは仮の実装（実際のグラフ構造が必要）
            metrics["delta_ged"] = {
                "value": 0.0,  # 実際の計算は後で実装
                "unit": "nodes",
                "method": "approximation"
            }
        except Exception as e:
            logger.error(f"GED calculation failed: {e}")
            metrics["delta_ged"] = {"value": 0.0, "unit": "nodes", "error": str(e)}
        
        # 統合エントロピーの計算
        try:
            delta, before_result, after_result = self.entropy_calculator.calculate_delta_entropy(
                before_state, after_state
            )
            metrics["unified_entropy"] = {
                "delta": float(delta),
                "before": before_result.to_dict(),
                "after": after_result.to_dict()
            }
        except Exception as e:
            logger.error(f"Unified entropy calculation failed: {e}")
            metrics["unified_entropy"] = {"delta": 0.0, "error": str(e)}
        
        return metrics


class DetailedLogger:
    """詳細なログ記録システム"""
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.log_entries = []
        self.physics_calculator = PhysicsMetricsCalculator()
    
    def log_episode(self, episode_data: Dict[str, Any]) -> None:
        """エピソードの詳細ログを記録"""
        
        log_entry = {
            "episode_id": episode_data.get("id", "unknown"),
            "timestamp": datetime.now().isoformat(),
            "question": episode_data.get("question", {}),
            "knowledge_retrieval": episode_data.get("retrieval", {}),
            "spike_detection": episode_data.get("spike", {}),
            "graph_evolution": episode_data.get("graph", {}),
            "response_analysis": episode_data.get("response", {}),
            "computational_cost": episode_data.get("cost", {}),
            "physics_metrics": episode_data.get("physics", {})
        }
        
        self.log_entries.append(log_entry)
        
        # 個別ファイルにも保存
        episode_file = self.results_dir / f"episode_{log_entry['episode_id']}.json"
        with open(episode_file, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)
    
    def save_full_log(self):
        """全ログを保存"""
        full_log_path = self.results_dir / "full_experiment_log.json"
        with open(full_log_path, 'w', encoding='utf-8') as f:
            json.dump(self.log_entries, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Full log saved to {full_log_path}")


class ExperimentRunner:
    """実験実行クラス"""
    
    def __init__(self, experiment_dir: Path):
        self.experiment_dir = experiment_dir
        self.data_dir = experiment_dir / "data"
        self.results_dir = experiment_dir / "results"
        
        # 知識ベースと質問の読み込み
        self.knowledge_base = self._load_json(self.data_dir / "knowledge_base.json")
        self.questions = self._load_json(self.data_dir / "questions.json")
        self.config = self._load_json(self.data_dir / "experiment_config.json")
        
        # エージェントとロガーの初期化
        self.agent = None
        self.direct_llm = None
        self.logger = DetailedLogger(self.results_dir)
        
        # 結果を格納
        self.results = {
            "direct": [],
            "rag": [],
            "insight": []
        }
    
    def _load_json(self, path: Path) -> Dict:
        """JSONファイルを読み込む"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def setup_models(self):
        """モデルの初期化"""
        logger.info("Initializing models...")
        
        # InsightSpikeエージェントの初期化
        config = Config()
        config.llm.safe_mode = True  # Mockプロバイダーを使用
        config.llm.model_name = self.config["models"]["llm"]["name"]
        
        self.agent = MainAgent(config)
        if not self.agent.initialize():
            raise RuntimeError("Failed to initialize MainAgent")
        
        # 知識ベースをエージェントに追加
        for phase_name, episodes in self.knowledge_base.items():
            for episode in episodes:
                self.agent.add_document(
                    text=f"[{phase_name}] {episode['text']}",
                    c_value=0.5,
                    metadata={"phase": phase_name, "id": episode["id"]}
                )
        
        # Direct LLMの初期化（L4LLMProviderを使用）
        from insightspike.core.layers.layer4_llm_provider import get_llm_provider
        self.direct_llm = get_llm_provider(config, safe_mode=True)  # Mockプロバイダーを使用
        
        logger.info("Models initialized successfully")
    
    def run_single_question(self, question_data: Dict) -> Dict[str, Any]:
        """単一の質問を実行"""
        question_id = question_data["id"]
        question_text = question_data["text"]
        
        logger.info(f"\nProcessing question {question_id}: {question_text}")
        
        episode_results = {
            "id": question_id,
            "question": question_data,
            "results": {}
        }
        
        # 1. Direct LLM
        logger.info("Running Direct LLM...")
        start_time = time.time()
        try:
            direct_response = self.direct_llm.generate_response(
                {"reasoning_quality": 0.5}, 
                question_text
            )
            direct_time = time.time() - start_time
            
            episode_results["results"]["direct"] = {
                "response": direct_response.get("response", ""),
                "time": direct_time,
                "success": direct_response.get("success", False)
            }
        except Exception as e:
            logger.error(f"Direct LLM failed: {e}")
            episode_results["results"]["direct"] = {
                "response": f"Error: {e}",
                "time": 0.0,
                "success": False
            }
        
        # 2. Standard RAG (簡易版 - InsightSpikeの信頼度を下げて実行)
        logger.info("Running Standard RAG...")
        start_time = time.time()
        try:
            # 一時的に閾値を変更
            original_threshold = self.agent.config.reasoning.confidence_threshold
            self.agent.config.reasoning.confidence_threshold = 1.0  # 常にRAGモード
            
            rag_result = self.agent.process_question(question_text, max_cycles=1)
            rag_time = time.time() - start_time
            
            # 閾値を戻す
            self.agent.config.reasoning.confidence_threshold = original_threshold
            
            episode_results["results"]["rag"] = {
                "response": rag_result.get("response", ""),
                "time": rag_time,
                "success": rag_result.get("success", False),
                "documents_retrieved": len(rag_result.get("documents", []))
            }
        except Exception as e:
            logger.error(f"RAG failed: {e}")
            episode_results["results"]["rag"] = {
                "response": f"Error: {e}",
                "time": 0.0,
                "success": False
            }
        
        # 3. InsightSpike
        logger.info("Running InsightSpike...")
        start_time = time.time()
        try:
            insight_result = self.agent.process_question(question_text, max_cycles=3)
            insight_time = time.time() - start_time
            
            # 物理量の計算
            physics_metrics = {}
            if insight_result.get("graph"):
                # グラフの変化から物理量を計算（簡易版）
                physics_metrics = {
                    "delta_ig": {
                        "value": insight_result.get("metrics", {}).get("delta_ig", 0.0),
                        "unit": "bits"
                    },
                    "delta_ged": {
                        "value": insight_result.get("metrics", {}).get("delta_ged", 0.0),
                        "unit": "nodes"
                    }
                }
            
            episode_results["results"]["insight"] = {
                "response": insight_result.get("response", ""),
                "time": insight_time,
                "success": insight_result.get("success", False),
                "spike_detected": insight_result.get("spike_detected", False),
                "reasoning_quality": insight_result.get("reasoning_quality", 0.0),
                "documents_retrieved": len(insight_result.get("documents", [])),
                "physics_metrics": physics_metrics,
                "graph_metrics": insight_result.get("metrics", {})
            }
            
            # 詳細ログの記録
            detailed_episode = {
                "id": question_id,
                "question": question_data,
                "retrieval": {
                    "documents": insight_result.get("documents", []),
                    "phase_distribution": self._analyze_phase_distribution(
                        insight_result.get("documents", [])
                    )
                },
                "spike": {
                    "detected": insight_result.get("spike_detected", False),
                    "confidence": insight_result.get("reasoning_quality", 0.0)
                },
                "graph": insight_result.get("metrics", {}),
                "response": {
                    "full_text": insight_result.get("response", ""),
                    "word_count": len(insight_result.get("response", "").split())
                },
                "cost": {
                    "total_time": insight_time,
                    "cycles": insight_result.get("total_cycles", 1)
                },
                "physics": physics_metrics
            }
            
            self.logger.log_episode(detailed_episode)
            
        except Exception as e:
            logger.error(f"InsightSpike failed: {e}")
            episode_results["results"]["insight"] = {
                "response": f"Error: {e}",
                "time": 0.0,
                "success": False
            }
        
        return episode_results
    
    def _analyze_phase_distribution(self, documents: List[Dict]) -> Dict[str, int]:
        """取得文書のフェーズ分布を分析"""
        phase_counts = {}
        for doc in documents:
            text = doc.get("text", "")
            for phase_name in self.knowledge_base.keys():
                if f"[{phase_name}]" in text:
                    phase_counts[phase_name] = phase_counts.get(phase_name, 0) + 1
        return phase_counts
    
    def run_experiment(self):
        """実験を実行"""
        logger.info("Starting experiment...")
        
        # モデルの初期化
        self.setup_models()
        
        all_results = []
        
        # 各カテゴリの質問を処理
        for category_name, questions in self.questions.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing category: {category_name}")
            logger.info(f"{'='*50}")
            
            for question in questions:
                try:
                    result = self.run_single_question(question)
                    result["category"] = category_name
                    all_results.append(result)
                    
                    # 中間結果を保存
                    self._save_intermediate_results(all_results)
                    
                except Exception as e:
                    logger.error(f"Failed to process question {question['id']}: {e}")
                    continue
        
        # 最終結果の保存
        self._save_final_results(all_results)
        
        # ログの保存
        self.logger.save_full_log()
        
        logger.info("\nExperiment completed!")
        
        return all_results
    
    def _save_intermediate_results(self, results: List[Dict]):
        """中間結果を保存"""
        intermediate_path = self.results_dir / "intermediate_results.json"
        with open(intermediate_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    def _save_final_results(self, results: List[Dict]):
        """最終結果を保存"""
        final_path = self.results_dir / "final_results.json"
        with open(final_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # サマリーも作成
        summary = self._create_summary(results)
        summary_path = self.results_dir / "experiment_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    def _create_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """実験結果のサマリーを作成"""
        summary = {
            "total_questions": len(results),
            "categories": {},
            "methods": {
                "direct": {"success": 0, "total_time": 0.0},
                "rag": {"success": 0, "total_time": 0.0},
                "insight": {"success": 0, "total_time": 0.0, "spikes": 0}
            },
            "physics_summary": {
                "average_delta_ig": [],
                "average_delta_ged": []
            }
        }
        
        # 結果を集計
        for result in results:
            category = result.get("category", "unknown")
            if category not in summary["categories"]:
                summary["categories"][category] = {"total": 0, "spikes": 0}
            
            summary["categories"][category]["total"] += 1
            
            # 各手法の結果を集計
            for method in ["direct", "rag", "insight"]:
                if method in result.get("results", {}):
                    method_result = result["results"][method]
                    if method_result.get("success", False):
                        summary["methods"][method]["success"] += 1
                    summary["methods"][method]["total_time"] += method_result.get("time", 0.0)
                    
                    if method == "insight" and method_result.get("spike_detected", False):
                        summary["methods"]["insight"]["spikes"] += 1
                        summary["categories"][category]["spikes"] += 1
                        
                        # 物理量を記録
                        physics = method_result.get("physics_metrics", {})
                        if "delta_ig" in physics:
                            summary["physics_summary"]["average_delta_ig"].append(
                                physics["delta_ig"].get("value", 0.0)
                            )
                        if "delta_ged" in physics:
                            summary["physics_summary"]["average_delta_ged"].append(
                                physics["delta_ged"].get("value", 0.0)
                            )
        
        # 平均を計算
        if summary["physics_summary"]["average_delta_ig"]:
            summary["physics_summary"]["average_delta_ig"] = float(
                np.mean(summary["physics_summary"]["average_delta_ig"])
            )
        else:
            summary["physics_summary"]["average_delta_ig"] = 0.0
            
        if summary["physics_summary"]["average_delta_ged"]:
            summary["physics_summary"]["average_delta_ged"] = float(
                np.mean(summary["physics_summary"]["average_delta_ged"])
            )
        else:
            summary["physics_summary"]["average_delta_ged"] = 0.0
        
        return summary


def main():
    """メイン実行関数"""
    experiment_dir = Path(__file__).parent
    
    print("=== geDIG Theory Validation v2.0 ===")
    print(f"Experiment directory: {experiment_dir}")
    print(f"Start time: {datetime.now()}")
    
    # 実験実行
    runner = ExperimentRunner(experiment_dir)
    results = runner.run_experiment()
    
    print(f"\nEnd time: {datetime.now()}")
    print(f"Total questions processed: {len(results)}")
    
    # サマリーを表示
    summary_path = experiment_dir / "results" / "experiment_summary.json"
    if summary_path.exists():
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        print("\n=== Summary ===")
        print(f"InsightSpike detection rate: {summary['methods']['insight']['spikes']}/{summary['total_questions']} "
              f"({summary['methods']['insight']['spikes']/summary['total_questions']*100:.1f}%)")
        print(f"Average ΔIG: {summary['physics_summary']['average_delta_ig']:.3f} bits")
        print(f"Average ΔGED: {summary['physics_summary']['average_delta_ged']:.3f} nodes")
    
    print("\n✅ Experiment completed successfully!")
    print(f"Results saved in: {experiment_dir / 'results'}")


if __name__ == "__main__":
    main()