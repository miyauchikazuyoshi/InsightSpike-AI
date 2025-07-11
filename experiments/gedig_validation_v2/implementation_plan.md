# 実装計画書

## レビューからの改善点反映

### 1. 理論指標の定量ログ化
```python
# 各質問処理時に記録すべき物理量
physics_metrics = {
    "delta_ig": {
        "value": float,
        "unit": "bits",
        "method": "shannon_entropy",
        "description": "情報利得（エントロピー減少量）"
    },
    "delta_ged": {
        "value": float,
        "unit": "nodes",
        "method": "advanced_ged",
        "description": "グラフ編集距離の変化"
    },
    "entropy_before": {
        "value": float,
        "unit": "bits",
        "description": "統合前の情報エントロピー"
    },
    "entropy_after": {
        "value": float,
        "unit": "bits",
        "description": "統合後の情報エントロピー"
    },
    "temperature": {
        "value": float,
        "unit": "dimensionless",
        "description": "システム温度パラメータ"
    }
}
```

### 2. 詳細ログ設計
```python
detailed_log = {
    "episode_id": str,
    "timestamp": datetime,
    "question": {
        "text": str,
        "category": str,
        "expected_insight": bool
    },
    "knowledge_retrieval": {
        "phase_distribution": Dict[int, List[str]],  # フェーズ別取得知識
        "similarity_scores": List[float],
        "episode_ids": List[str],  # 参照したエピソードID
        "retrieval_time": float
    },
    "spike_detection": {
        "detected": bool,
        "confidence": float,
        "phase_count": int,
        "trigger_conditions": {
            "delta_ged": float,
            "delta_ig": float,
            "similarity_threshold": float
        }
    },
    "graph_evolution": {
        "before": {
            "nodes": int,
            "edges": int,
            "density": float,
            "diameter": int,
            "clustering_coefficient": float,
            "components": int
        },
        "after": {
            "nodes": int,
            "edges": int,
            "density": float,
            "diameter": int,
            "clustering_coefficient": float,
            "components": int
        },
        "changes": {
            "node_change": float,  # percentage
            "edge_change": float,
            "density_change": float,
            "complexity_change": float
        }
    },
    "response_analysis": {
        "full_text": str,  # 完全な回答テキスト
        "word_count": int,
        "new_concepts": List[str],
        "concept_origins": Dict[str, str],  # 概念の由来フェーズ
        "readability_score": float,
        "technical_terms": List[str]
    },
    "computational_cost": {
        "total_time": float,
        "retrieval_time": float,
        "integration_time": float,
        "generation_time": float,
        "memory_usage": float
    }
}
```

### 3. 拡張実験セット

#### A. 質問バンク（30問に拡張）
```python
question_bank = {
    "baseline": [  # 単一ドメイン
        "What is entropy?",
        "Define information theory",
        "Explain graph structures",
        "What is energy conservation?",
        "Describe biological evolution"
    ],
    "cross_domain": [  # 2-3ドメイン統合
        "How does information relate to energy?",
        "What connects graphs and biological networks?",
        "How do entropy and evolution interact?",
        "What links computation and thermodynamics?",
        "How do structures emerge from information?"
    ],
    "abstract": [  # 高次概念統合
        "What is the nature of order and chaos?",
        "How does complexity arise from simplicity?",
        "What unifies discrete and continuous?",
        "What is the essence of emergence?",
        "How do patterns form in nature?"
    ],
    "edge_cases": [  # 境界条件テスト
        "What has no information content?",
        "Can entropy decrease globally?",
        "Is there a limit to complexity?",
        "What defines a perfect structure?",
        "How do paradoxes resolve?"
    ]
}
```

#### B. パラメータ感度マトリクス
```python
parameter_grid = {
    "spike_ged_threshold": np.linspace(-1.0, 0.0, 11),
    "spike_ig_threshold": np.linspace(0.0, 0.5, 11),
    "phase_threshold": range(1, 6),
    "similarity_threshold": np.linspace(0.1, 0.5, 9),
    "confidence_threshold": np.linspace(0.3, 0.8, 11),
    "temperature": np.logspace(-1, 1, 10)  # 0.1 to 10
}

# グリッドサーチで最適パラメータ探索
from itertools import product
param_combinations = list(product(*parameter_grid.values()))
```

### 4. 人間評価プロトコル

#### A. 評価者募集基準
```python
evaluator_criteria = {
    "minimum_count": 10,
    "expertise_distribution": {
        "ai_researchers": 3,
        "domain_experts": 3,
        "graduate_students": 2,
        "general_users": 2
    },
    "training_required": True,
    "calibration_questions": 3
}
```

#### B. ブラインド評価インターフェース
```python
class BlindEvaluationInterface:
    def prepare_responses(self, responses):
        """回答をシャッフル・匿名化"""
        anonymized = []
        for idx, (method, response) in enumerate(responses):
            anonymized.append({
                "id": f"RESP_{idx:04d}",
                "text": response["text"],
                "hidden_method": method  # 評価後に開示
            })
        return random.shuffle(anonymized)
    
    def collect_evaluation(self, evaluator_id, response_id):
        """5段階評価を収集"""
        return {
            "evaluator": evaluator_id,
            "response": response_id,
            "scores": {
                "novelty": int(input("新規性 (1-5): ")),
                "usefulness": int(input("有用性 (1-5): ")),
                "coherence": int(input("一貫性 (1-5): ")),
                "depth": int(input("深さ (1-5): ")),
                "integration": int(input("統合度 (1-5): "))
            },
            "confidence": int(input("評価の確信度 (1-5): ")),
            "comments": input("コメント（任意）: ")
        }
```

### 5. 可視化実装

#### A. リアルタイム監視ダッシュボード
```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_monitoring_dashboard():
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Spike Detection Rate', 'ΔIG Distribution', 'ΔGED Distribution',
            'Quality Scores', 'Processing Time', 'Graph Evolution'
        )
    )
    
    # リアルタイム更新
    return fig
```

#### B. 論文用図表生成
```python
def generate_publication_figures():
    figures = {
        "performance_comparison": create_bar_chart_comparison(),
        "spike_scatter": create_ig_ged_scatter(),
        "entropy_heatmap": create_phase_entropy_heatmap(),
        "graph_evolution": create_before_after_graphs(),
        "human_eval_radar": create_radar_charts(),
        "ablation_results": create_ablation_bar_chart()
    }
    
    # LaTeX用にエクスポート
    for name, fig in figures.items():
        fig.write_image(f"figures/{name}.pdf", format="pdf")
        fig.write_image(f"figures/{name}.png", format="png", dpi=300)
```

### 6. 統計分析パイプライン
```python
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

class StatisticalAnalysis:
    def __init__(self, results):
        self.results = results
    
    def test_spike_detection_rates(self):
        """洞察検出率の検定"""
        contingency_table = self._create_contingency_table()
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        # 効果量も計算
        cramers_v = np.sqrt(chi2 / (self.n_samples * (min(contingency_table.shape) - 1)))
        
        return {
            "test": "Chi-square",
            "statistic": chi2,
            "p_value": p_value,
            "effect_size": cramers_v,
            "interpretation": self._interpret_p_value(p_value)
        }
    
    def test_quality_scores(self):
        """品質スコアのANOVA"""
        # 正規性検定
        normality_tests = {}
        for method in ['direct', 'rag', 'insight']:
            stat, p = stats.shapiro(self.results[method]['quality_scores'])
            normality_tests[method] = {"statistic": stat, "p_value": p}
        
        # ANOVA
        f_stat, p_value = stats.f_oneway(
            self.results['direct']['quality_scores'],
            self.results['rag']['quality_scores'],
            self.results['insight']['quality_scores']
        )
        
        # Post-hoc (Tukey HSD)
        if p_value < 0.05:
            tukey_results = self._perform_tukey_hsd()
        
        return {
            "normality": normality_tests,
            "anova": {"f_statistic": f_stat, "p_value": p_value},
            "post_hoc": tukey_results if p_value < 0.05 else None
        }
```

### 7. エラー処理とロバスト性
```python
class RobustExperimentRunner:
    def __init__(self, config):
        self.config = config
        self.checkpoints = []
    
    def run_with_checkpointing(self):
        """中断からの再開をサポート"""
        last_checkpoint = self._load_last_checkpoint()
        
        for idx, question in enumerate(self.questions[last_checkpoint:]):
            try:
                result = self._process_question(question)
                self._save_checkpoint(idx, result)
            except Exception as e:
                logger.error(f"Failed at question {idx}: {e}")
                self._save_error_state(idx, e)
                
                if self.config.fail_fast:
                    raise
                else:
                    continue
    
    def validate_results(self):
        """結果の整合性チェック"""
        validations = {
            "all_metrics_present": self._check_metrics_completeness(),
            "spike_consistency": self._check_spike_detection_consistency(),
            "graph_validity": self._check_graph_structures(),
            "statistical_assumptions": self._check_statistical_assumptions()
        }
        
        return all(validations.values()), validations
```

### 8. 実行スクリプト
```python
# experiments/gedig_validation_v2/run_experiment.py

def main():
    # 設定読み込み
    config = load_experiment_config()
    
    # 実験実行
    runner = RobustExperimentRunner(config)
    results = runner.run_with_checkpointing()
    
    # 分析
    analyzer = StatisticalAnalysis(results)
    stats_results = analyzer.run_all_tests()
    
    # 可視化
    visualizer = ExperimentVisualizer(results, stats_results)
    visualizer.generate_all_figures()
    
    # レポート生成
    reporter = ExperimentReporter(results, stats_results)
    reporter.generate_latex_report()
    reporter.generate_markdown_summary()
    
    print("実験完了！")

if __name__ == "__main__":
    main()
```

これにより、レビューで指摘された全ての改善点を反映した、
論文レベルの厳密な実験が可能になります。