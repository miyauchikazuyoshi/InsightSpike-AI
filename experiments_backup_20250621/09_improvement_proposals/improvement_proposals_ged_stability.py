#!/usr/bin/env python3
"""
GED急落現象の改善策実装提案
=========================

83件のGED急落現象を分析し、安定化処理を提案
"""

def improve_ged_stability():
    """
    GED急落現象を防ぐための改善処理
    """
    improvements = {
        "exponential_smoothing": {
            "description": "指数平滑化による安定化",
            "implementation": """
            class ExponentialGEDSmoothing:
                def __init__(self, alpha=0.3):
                    self.alpha = alpha  # 平滑化係数
                    self.smoothed_ged = None
                
                def update(self, raw_ged):
                    if self.smoothed_ged is None:
                        self.smoothed_ged = raw_ged
                    else:
                        # 指数平滑化
                        self.smoothed_ged = (self.alpha * raw_ged + 
                                           (1 - self.alpha) * self.smoothed_ged)
                    
                    return self.smoothed_ged
                
                def detect_anomaly(self, raw_ged, threshold=0.15):
                    # 急激な変化を検出
                    if self.smoothed_ged is not None:
                        change = abs(raw_ged - self.smoothed_ged)
                        return change > threshold
                    return False
            """
        },
        
        "moving_window_stabilization": {
            "description": "移動窓による安定化",
            "implementation": """
            class MovingWindowGEDStabilizer:
                def __init__(self, window_size=5):
                    self.window_size = window_size
                    self.ged_history = []
                
                def stabilize_ged(self, raw_ged):
                    self.ged_history.append(raw_ged)
                    
                    # 窓サイズを超えた場合は古いデータを削除
                    if len(self.ged_history) > self.window_size:
                        self.ged_history.pop(0)
                    
                    # 移動平均を計算
                    stabilized_ged = np.mean(self.ged_history)
                    
                    # 外れ値検出と修正
                    median_ged = np.median(self.ged_history)
                    mad = np.median(np.abs(np.array(self.ged_history) - median_ged))
                    
                    # MAD基準での外れ値判定
                    if abs(raw_ged - median_ged) > 3 * mad:
                        # 外れ値の場合は中央値で置換
                        return median_ged
                    
                    return stabilized_ged
            """
        },
        
        "adaptive_learning_rate": {
            "description": "適応的学習率制御",
            "implementation": """
            class AdaptiveLearningRateController:
                def __init__(self, base_lr=0.001, stability_factor=0.8):
                    self.base_lr = base_lr
                    self.stability_factor = stability_factor
                    self.ged_variance_history = []
                
                def adjust_learning_rate(self, recent_ged_values):
                    # 最近のGED値の分散を計算
                    current_variance = np.var(recent_ged_values)
                    self.ged_variance_history.append(current_variance)
                    
                    # 分散が高い場合は学習率を下げる
                    if len(self.ged_variance_history) > 10:
                        avg_variance = np.mean(self.ged_variance_history[-10:])
                        
                        if current_variance > avg_variance * 1.5:
                            # 不安定な場合は学習率を下げる
                            adjusted_lr = self.base_lr * self.stability_factor
                        else:
                            # 安定している場合は通常の学習率
                            adjusted_lr = self.base_lr
                        
                        return adjusted_lr
                    
                    return self.base_lr
            """
        },
        
        "robust_graph_metrics": {
            "description": "ロバストなグラフ指標計算",
            "implementation": """
            def robust_ged_calculation(graph1, graph2, method='hybrid'):
                # 複数手法での計算とアンサンブル
                ged_methods = {}
                
                # 1. 標準GED
                try:
                    ged_methods['standard'] = calculate_standard_ged(graph1, graph2)
                except:
                    ged_methods['standard'] = None
                
                # 2. 近似GED（高速版）
                try:
                    ged_methods['approximate'] = calculate_approximate_ged(graph1, graph2)
                except:
                    ged_methods['approximate'] = None
                
                # 3. グラフカーネル基準
                try:
                    ged_methods['kernel_based'] = calculate_kernel_ged(graph1, graph2)
                except:
                    ged_methods['kernel_based'] = None
                
                # 有効な結果の統合
                valid_geds = [ged for ged in ged_methods.values() if ged is not None]
                
                if not valid_geds:
                    return 0.0  # フォールバック値
                
                if method == 'median':
                    return np.median(valid_geds)
                elif method == 'mean':
                    return np.mean(valid_geds)
                elif method == 'robust_mean':
                    # 外れ値を除外した平均
                    q1, q3 = np.percentile(valid_geds, [25, 75])
                    iqr = q3 - q1
                    filtered_geds = [ged for ged in valid_geds 
                                   if q1 - 1.5*iqr <= ged <= q3 + 1.5*iqr]
                    return np.mean(filtered_geds) if filtered_geds else np.median(valid_geds)
                else:  # hybrid
                    # 分散が小さい場合は平均、大きい場合は中央値
                    if np.std(valid_geds) < 0.1:
                        return np.mean(valid_geds)
                    else:
                        return np.median(valid_geds)
            """
        }
    }
    
    return improvements

# 実装例
print("=== GED急落現象改善策 ===")
improvements = improve_ged_stability()
for key, value in improvements.items():
    print(f"\n{key.upper()}:")
    print(f"説明: {value['description']}")
    print("実装例:")
    print(value['implementation'])
