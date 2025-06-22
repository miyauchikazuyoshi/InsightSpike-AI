"""
実験評価指標計算ユーティリティ

各実験で共通して使用される性能評価指標の計算機能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error,
    ndcg_score, average_precision_score
)
from scipy.stats import pearsonr, spearmanr
import time
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """性能評価指標計算クラス"""
    
    def __init__(self):
        self.metrics_history = []
    
    def calculate_classification_metrics(self, y_true: List[int], y_pred: List[int],
                                       y_prob: Optional[List[float]] = None) -> Dict[str, float]:
        """分類タスクの評価指標計算"""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average='macro', zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average='macro', zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average='macro', zero_division=0),
            "precision_micro": precision_score(y_true, y_pred, average='micro', zero_division=0),
            "recall_micro": recall_score(y_true, y_pred, average='micro', zero_division=0),
            "f1_micro": f1_score(y_true, y_pred, average='micro', zero_division=0)
        }
        
        # ROC-AUC（確率が提供された場合）
        if y_prob is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_prob, average='macro', multi_class='ovr')
            except ValueError:
                logger.warning("ROC-AUC calculation failed")
        
        return metrics
    
    def calculate_qa_metrics(self, predicted_answers: List[str], 
                           ground_truth_answers: List[str]) -> Dict[str, float]:
        """QAタスクの評価指標計算"""
        exact_matches = []
        f1_scores = []
        
        for pred, truth in zip(predicted_answers, ground_truth_answers):
            # Exact Match
            exact_matches.append(1.0 if pred.strip().lower() == truth.strip().lower() else 0.0)
            
            # Token-level F1 Score
            pred_tokens = set(pred.strip().lower().split())
            truth_tokens = set(truth.strip().lower().split())
            
            if len(pred_tokens) == 0 and len(truth_tokens) == 0:
                f1_scores.append(1.0)
            elif len(pred_tokens) == 0 or len(truth_tokens) == 0:
                f1_scores.append(0.0)
            else:
                common_tokens = pred_tokens.intersection(truth_tokens)
                precision = len(common_tokens) / len(pred_tokens)
                recall = len(common_tokens) / len(truth_tokens)
                
                if precision + recall == 0:
                    f1_scores.append(0.0)
                else:
                    f1_scores.append(2 * (precision * recall) / (precision + recall))
        
        return {
            "exact_match": np.mean(exact_matches),
            "f1_score": np.mean(f1_scores),
            "exact_match_std": np.std(exact_matches),
            "f1_score_std": np.std(f1_scores)
        }
    
    def calculate_retrieval_metrics(self, query_results: Dict[str, List[Tuple[str, float]]],
                                  ground_truth: Dict[str, List[str]],
                                  k_values: List[int] = [1, 5, 10, 20]) -> Dict[str, float]:
        """情報検索タスクの評価指標計算"""
        precision_at_k = {f"precision@{k}": [] for k in k_values}
        recall_at_k = {f"recall@{k}": [] for k in k_values}
        ndcg_at_k = {f"ndcg@{k}": [] for k in k_values}
        map_scores = []
        
        for query_id, results in query_results.items():
            if query_id not in ground_truth:
                continue
            
            relevant_docs = set(ground_truth[query_id])
            retrieved_docs = [doc_id for doc_id, _ in results]
            
            # Average Precision
            ap = self._calculate_average_precision(retrieved_docs, relevant_docs)
            map_scores.append(ap)
            
            # Precision@K, Recall@K, NDCG@K
            for k in k_values:
                retrieved_k = retrieved_docs[:k]
                relevant_retrieved = len(set(retrieved_k).intersection(relevant_docs))
                
                # Precision@K
                precision_at_k[f"precision@{k}"].append(
                    relevant_retrieved / min(k, len(retrieved_k)) if retrieved_k else 0.0
                )
                
                # Recall@K
                recall_at_k[f"recall@{k}"].append(
                    relevant_retrieved / len(relevant_docs) if relevant_docs else 0.0
                )
                
                # NDCG@K
                relevance_scores = [1.0 if doc in relevant_docs else 0.0 for doc in retrieved_k]
                if relevance_scores:
                    ideal_scores = sorted([1.0] * len(relevant_docs), reverse=True)[:k]
                    if ideal_scores:
                        ndcg_at_k[f"ndcg@{k}"].append(
                            ndcg_score([ideal_scores], [relevance_scores], k=k)
                        )
                    else:
                        ndcg_at_k[f"ndcg@{k}"].append(0.0)
                else:
                    ndcg_at_k[f"ndcg@{k}"].append(0.0)
        
        # 平均値計算
        metrics = {}
        metrics["map"] = np.mean(map_scores) if map_scores else 0.0
        
        for k in k_values:
            metrics[f"precision@{k}"] = np.mean(precision_at_k[f"precision@{k}"])
            metrics[f"recall@{k}"] = np.mean(recall_at_k[f"recall@{k}"])
            metrics[f"ndcg@{k}"] = np.mean(ndcg_at_k[f"ndcg@{k}"])
        
        return metrics
    
    def _calculate_average_precision(self, retrieved_docs: List[str], 
                                   relevant_docs: set) -> float:
        """Average Precision計算"""
        if not relevant_docs:
            return 0.0
        
        precision_sum = 0.0
        relevant_count = 0
        
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                relevant_count += 1
                precision_sum += relevant_count / (i + 1)
        
        return precision_sum / len(relevant_docs) if relevant_docs else 0.0
    
    def calculate_pathfinding_metrics(self, solutions: List[Dict[str, Any]]) -> Dict[str, float]:
        """経路探索アルゴリズムの評価指標計算"""
        path_lengths = []
        computation_times = []
        memory_usages = []
        optimality_ratios = []
        success_rates = []
        
        for solution in solutions:
            if solution.get("success", False):
                success_rates.append(1.0)
                path_lengths.append(solution.get("path_length", float('inf')))
                computation_times.append(solution.get("computation_time", float('inf')))
                memory_usages.append(solution.get("memory_usage", float('inf')))
                
                # 最適性比率（最短経路長 / 実際の経路長）
                optimal_length = solution.get("optimal_path_length", solution.get("path_length", 1))
                optimality_ratios.append(optimal_length / solution.get("path_length", 1))
            else:
                success_rates.append(0.0)
        
        metrics = {
            "success_rate": np.mean(success_rates),
            "avg_path_length": np.mean(path_lengths) if path_lengths else float('inf'),
            "avg_computation_time": np.mean(computation_times) if computation_times else float('inf'),
            "avg_memory_usage": np.mean(memory_usages) if memory_usages else float('inf'),
            "avg_optimality_ratio": np.mean(optimality_ratios) if optimality_ratios else 0.0,
            "std_path_length": np.std(path_lengths) if path_lengths else 0.0,
            "std_computation_time": np.std(computation_times) if computation_times else 0.0,
            "std_memory_usage": np.std(memory_usages) if memory_usages else 0.0
        }
        
        return metrics
    
    def calculate_memory_efficiency_metrics(self, memory_measurements: List[Dict[str, float]]) -> Dict[str, float]:
        """メモリ効率性評価指標計算"""
        construction_times = [m.get("construction_time", 0) for m in memory_measurements]
        memory_usages = [m.get("memory_usage", 0) for m in memory_measurements]
        retrieval_accuracies = [m.get("retrieval_accuracy", 0) for m in memory_measurements]
        knowledge_densities = [m.get("knowledge_density", 0) for m in memory_measurements]
        
        return {
            "avg_construction_time": np.mean(construction_times),
            "avg_memory_usage": np.mean(memory_usages),
            "avg_retrieval_accuracy": np.mean(retrieval_accuracies),
            "avg_knowledge_density": np.mean(knowledge_densities),
            "efficiency_score": np.mean(retrieval_accuracies) / (np.mean(memory_usages) + 1e-8),
            "speed_score": 1.0 / (np.mean(construction_times) + 1e-8),
            "std_construction_time": np.std(construction_times),
            "std_memory_usage": np.std(memory_usages),
            "std_retrieval_accuracy": np.std(retrieval_accuracies)
        }


class PerformanceAnalyzer:
    """パフォーマンス分析クラス"""
    
    def __init__(self):
        self.baseline_metrics = {}
        self.improvement_thresholds = {
            "significant": 0.05,  # 5%改善
            "substantial": 0.15,  # 15%改善
            "breakthrough": 0.30  # 30%改善
        }
    
    def set_baseline(self, method_name: str, metrics: Dict[str, float]):
        """ベースライン性能の設定"""
        self.baseline_metrics[method_name] = metrics
        logger.info(f"Baseline set for {method_name}")
    
    def analyze_improvement(self, method_name: str, new_metrics: Dict[str, float],
                          baseline_method: str = None) -> Dict[str, Any]:
        """性能改善の分析"""
        if baseline_method is None:
            baseline_method = list(self.baseline_metrics.keys())[0] if self.baseline_metrics else None
        
        if baseline_method not in self.baseline_metrics:
            return {"error": f"Baseline method '{baseline_method}' not found"}
        
        baseline = self.baseline_metrics[baseline_method]
        improvements = {}
        significance_levels = {}
        
        for metric, new_value in new_metrics.items():
            if metric in baseline:
                baseline_value = baseline[metric]
                if baseline_value != 0:
                    improvement = (new_value - baseline_value) / abs(baseline_value)
                    improvements[metric] = improvement
                    
                    # 改善の有意性レベル判定
                    if abs(improvement) >= self.improvement_thresholds["breakthrough"]:
                        significance_levels[metric] = "breakthrough"
                    elif abs(improvement) >= self.improvement_thresholds["substantial"]:
                        significance_levels[metric] = "substantial"
                    elif abs(improvement) >= self.improvement_thresholds["significant"]:
                        significance_levels[metric] = "significant"
                    else:
                        significance_levels[metric] = "marginal"
        
        return {
            "method": method_name,
            "baseline": baseline_method,
            "improvements": improvements,
            "significance_levels": significance_levels,
            "overall_improvement": np.mean(list(improvements.values())) if improvements else 0.0,
            "metrics_improved": sum(1 for imp in improvements.values() if imp > 0),
            "total_metrics": len(improvements)
        }
    
    def compare_methods(self, results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """複数手法の比較分析"""
        comparison_df = pd.DataFrame(results).T
        
        # ランキング追加
        for metric in comparison_df.columns:
            if metric.endswith(('_time', '_usage', '_error')):
                # 小さいほど良い指標（昇順）
                comparison_df[f"{metric}_rank"] = comparison_df[metric].rank(ascending=True)
            else:
                # 大きいほど良い指標（降順）
                comparison_df[f"{metric}_rank"] = comparison_df[metric].rank(ascending=False)
        
        # 総合ランキング
        rank_columns = [col for col in comparison_df.columns if col.endswith('_rank')]
        comparison_df['overall_rank'] = comparison_df[rank_columns].mean(axis=1)
        
        return comparison_df.sort_values('overall_rank')
    
    def statistical_significance_test(self, results_a: List[float], 
                                    results_b: List[float],
                                    test_type: str = "t_test") -> Dict[str, Any]:
        """統計的有意性検定"""
        from scipy.stats import ttest_rel, wilcoxon, mannwhitneyu
        
        if len(results_a) != len(results_b):
            logger.warning("Sample sizes don't match, using Mann-Whitney U test")
            test_type = "mannwhitney"
        
        try:
            if test_type == "t_test":
                statistic, p_value = ttest_rel(results_a, results_b)
            elif test_type == "wilcoxon":
                statistic, p_value = wilcoxon(results_a, results_b)
            elif test_type == "mannwhitney":
                statistic, p_value = mannwhitneyu(results_a, results_b, alternative='two-sided')
            else:
                raise ValueError(f"Unknown test type: {test_type}")
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(results_a) + np.var(results_b)) / 2)
            effect_size = (np.mean(results_a) - np.mean(results_b)) / pooled_std if pooled_std > 0 else 0
            
            return {
                "test_type": test_type,
                "statistic": float(statistic),
                "p_value": float(p_value),
                "effect_size": float(effect_size),
                "significant_05": p_value < 0.05,
                "significant_01": p_value < 0.01,
                "significant_001": p_value < 0.001
            }
            
        except Exception as e:
            logger.error(f"Statistical test failed: {e}")
            return {"error": str(e)}
    
    def generate_performance_summary(self, all_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """パフォーマンス総合サマリー生成"""
        if not all_results:
            return {"error": "No results provided"}
        
        methods = list(all_results.keys())
        metrics = list(all_results[methods[0]].keys())
        
        summary = {
            "methods_compared": len(methods),
            "metrics_evaluated": len(metrics),
            "best_method_per_metric": {},
            "worst_method_per_metric": {},
            "method_rankings": {},
            "performance_gaps": {}
        }
        
        # 各指標での最良・最悪手法
        for metric in metrics:
            metric_values = {method: results[metric] for method, results in all_results.items()}
            
            if metric.endswith(('_time', '_usage', '_error')):
                # 小さいほど良い
                best_method = min(metric_values, key=metric_values.get)
                worst_method = max(metric_values, key=metric_values.get)
            else:
                # 大きいほど良い
                best_method = max(metric_values, key=metric_values.get)
                worst_method = min(metric_values, key=metric_values.get)
            
            summary["best_method_per_metric"][metric] = {
                "method": best_method,
                "value": metric_values[best_method]
            }
            summary["worst_method_per_metric"][metric] = {
                "method": worst_method,
                "value": metric_values[worst_method]
            }
            
            # パフォーマンスギャップ
            best_value = metric_values[best_method]
            worst_value = metric_values[worst_method]
            if worst_value != 0:
                gap = abs(best_value - worst_value) / abs(worst_value)
                summary["performance_gaps"][metric] = gap
        
        # 手法ランキング
        comparison_df = self.compare_methods(all_results)
        summary["method_rankings"] = comparison_df["overall_rank"].to_dict()
        
        return summary
