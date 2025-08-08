#!/usr/bin/env python3
"""
実験結果のビジュアライゼーション
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Dict, List, Optional
import seaborn as sns

# スタイル設定
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ResultVisualizer:
    """実験結果の可視化"""
    
    def __init__(self, experiment_path: str):
        """
        Args:
            experiment_path: 実験結果ディレクトリのパス
        """
        self.experiment_path = Path(experiment_path)
        self.figures_path = self.experiment_path / "figures"
        self.figures_path.mkdir(exist_ok=True)
    
    def visualize_navigation(self, maze_name: str):
        """ナビゲーション結果を可視化"""
        # データ読み込み
        maze = np.load(self.experiment_path / f"{maze_name}_maze.npy")
        
        with open(self.experiment_path / f"{maze_name}_path.json", 'r') as f:
            path_data = json.load(f)
        
        with open(self.experiment_path / f"{maze_name}_result.json", 'r') as f:
            result = json.load(f)
        
        path = path_data['path']
        visit_counts = {
            tuple(map(int, k.split(','))): v 
            for k, v in path_data['visit_counts'].items()
        }
        
        # 図作成
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. 迷路とパス
        self._plot_maze_and_path(axes[0], maze, path, result['success'])
        
        # 2. 訪問頻度ヒートマップ
        self._plot_visit_heatmap(axes[1], maze, visit_counts)
        
        # 3. エピソード密度
        self._plot_episode_density(axes[2], maze, path, result)
        
        # タイトル
        success_str = "SUCCESS" if result['success'] else "FAILED"
        fig.suptitle(
            f"{maze_name} - {success_str}\n"
            f"Steps: {result['steps']}, Wall hits: {result['wall_hits']} "
            f"({result['wall_hit_rate']:.1%})",
            fontsize=14
        )
        
        plt.tight_layout()
        
        # 保存
        fig_path = self.figures_path / f"{maze_name}_navigation.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {fig_path}")
    
    def _plot_maze_and_path(self, ax, maze, path, success):
        """迷路とパスを描画"""
        # 迷路を表示
        ax.imshow(maze, cmap='binary', alpha=0.3)
        
        # パスを描画
        if path:
            path_array = np.array(path)
            ax.plot(path_array[:, 1], path_array[:, 0],
                   'b-', linewidth=2, alpha=0.7, label='Path')
            
            # パスの進行を色で表現
            colors = plt.cm.viridis(np.linspace(0, 1, len(path)))
            for i in range(len(path)-1):
                ax.plot(path[i:i+2, 1], path[i:i+2, 0],
                       color=colors[i], linewidth=2, alpha=0.8)
        
        # スタートとゴール
        ax.plot(1, 1, 'go', markersize=12, label='Start')
        goal_y, goal_x = maze.shape[0]-2, maze.shape[1]-2
        
        marker_color = 'ro' if not success else 'g*'
        ax.plot(goal_x, goal_y, marker_color, markersize=15, label='Goal')
        
        ax.set_title('Maze & Navigation Path')
        ax.legend(loc='upper right')
        ax.axis('equal')
        ax.set_xlim(-0.5, maze.shape[1]-0.5)
        ax.set_ylim(maze.shape[0]-0.5, -0.5)
    
    def _plot_visit_heatmap(self, ax, maze, visit_counts):
        """訪問頻度のヒートマップ"""
        visit_map = np.zeros_like(maze, dtype=float)
        
        for (y, x), count in visit_counts.items():
            if 0 <= y < maze.shape[0] and 0 <= x < maze.shape[1]:
                visit_map[y, x] = count
        
        # マスク（壁は表示しない）
        masked_visits = np.ma.masked_where(maze == 1, visit_map)
        
        im = ax.imshow(masked_visits, cmap='hot', interpolation='nearest')
        plt.colorbar(im, ax=ax, label='Visit Count')
        
        ax.set_title('Visit Frequency Heatmap')
        ax.axis('equal')
    
    def _plot_episode_density(self, ax, maze, path, result):
        """エピソード密度を表示"""
        # エピソードの空間分布を推定
        episode_map = np.zeros_like(maze, dtype=float)
        
        # パス上の各点でエピソードが生成される
        for y, x in path:
            if 0 <= y < maze.shape[0] and 0 <= x < maze.shape[1]:
                # 各位置で5エピソード（移動1+視覚4）
                episode_map[y, x] += 5
        
        # ガウシアンフィルタでスムージング
        from scipy.ndimage import gaussian_filter
        episode_density = gaussian_filter(episode_map, sigma=1.0)
        
        # マスク
        masked_density = np.ma.masked_where(maze == 1, episode_density)
        
        im = ax.imshow(masked_density, cmap='viridis', interpolation='nearest')
        plt.colorbar(im, ax=ax, label='Episode Density')
        
        ax.set_title(f'Episode Distribution\n(Total: {result["total_episodes"]} episodes)')
        ax.axis('equal')
    
    def visualize_learning_progress(self, summary_path: str = "experiment_summary.json"):
        """学習進行状況を可視化"""
        # サマリー読み込み
        with open(self.experiment_path / summary_path, 'r') as f:
            summary = json.load(f)
        
        results = summary['results']
        
        # 図作成
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 成功率の推移（サイズ別）
        self._plot_success_rate(axes[0, 0], results)
        
        # 2. 壁衝突率の分布
        self._plot_wall_hit_distribution(axes[0, 1], results)
        
        # 3. ステップ数の比較
        self._plot_steps_comparison(axes[1, 0], results)
        
        # 4. 深度使用パターン
        self._plot_depth_usage(axes[1, 1], results)
        
        fig.suptitle('Learning Progress Analysis', fontsize=14)
        plt.tight_layout()
        
        # 保存
        fig_path = self.figures_path / "learning_progress.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {fig_path}")
    
    def _plot_success_rate(self, ax, results):
        """成功率をプロット"""
        # サイズごとに集計
        by_size = {}
        for r in results:
            size_key = f"{r['maze_size'][0]}x{r['maze_size'][1]}"
            if size_key not in by_size:
                by_size[size_key] = []
            by_size[size_key].append(r['success'])
        
        sizes = []
        success_rates = []
        for size, successes in by_size.items():
            sizes.append(size)
            success_rates.append(sum(successes) / len(successes))
        
        # バープロット
        bars = ax.bar(sizes, success_rates, alpha=0.7)
        
        # 色分け
        for i, bar in enumerate(bars):
            if success_rates[i] >= 0.7:
                bar.set_color('green')
            elif success_rates[i] >= 0.5:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        ax.axhline(y=0.7, color='g', linestyle='--', alpha=0.5, label='Target (70%)')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rate by Maze Size')
        ax.legend()
    
    def _plot_wall_hit_distribution(self, ax, results):
        """壁衝突率の分布"""
        wall_hit_rates = [r['wall_hit_rate'] for r in results]
        
        ax.hist(wall_hit_rates, bins=20, alpha=0.7, edgecolor='black')
        ax.axvline(x=0.3, color='r', linestyle='--', alpha=0.5, 
                  label='Target (<30%)')
        
        mean_rate = np.mean(wall_hit_rates)
        ax.axvline(x=mean_rate, color='b', linestyle='-', alpha=0.7,
                  label=f'Mean ({mean_rate:.1%})')
        
        ax.set_xlabel('Wall Hit Rate')
        ax.set_ylabel('Frequency')
        ax.set_title('Wall Hit Rate Distribution')
        ax.legend()
    
    def _plot_steps_comparison(self, ax, results):
        """ステップ数の比較"""
        # サイズごとに成功事例のステップ数
        by_size = {}
        for r in results:
            if r['success']:
                size_key = f"{r['maze_size'][0]}x{r['maze_size'][1]}"
                if size_key not in by_size:
                    by_size[size_key] = []
                by_size[size_key].append(r['steps'])
        
        # ボックスプロット
        data = []
        labels = []
        for size, steps in by_size.items():
            if steps:
                data.append(steps)
                labels.append(size)
        
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            
            # 色付け
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)
        
        ax.set_ylabel('Steps to Goal')
        ax.set_title('Steps Required (Successful Runs Only)')
        ax.grid(True, alpha=0.3)
    
    def _plot_depth_usage(self, ax, results):
        """深度使用パターン"""
        # 全結果の深度使用を集計
        total_usage = {}
        for r in results:
            if 'depth_usage' in r:
                for depth, count in r['depth_usage'].items():
                    if depth not in total_usage:
                        total_usage[depth] = 0
                    total_usage[depth] += count
        
        if total_usage:
            depths = sorted(total_usage.keys(), key=lambda x: int(x))
            counts = [total_usage[d] for d in depths]
            
            bars = ax.bar(depths, counts, alpha=0.7, color='green')
            
            ax.set_xlabel('Message Passing Depth')
            ax.set_ylabel('Usage Count')
            ax.set_title('Multi-hop Depth Usage Pattern')
            ax.grid(True, alpha=0.3)
    
    def visualize_all(self):
        """全ての可視化を実行"""
        print("Visualizing results...")
        
        # 各迷路の結果を可視化
        for maze_file in self.experiment_path.glob("*_maze.npy"):
            maze_name = maze_file.stem.replace("_maze", "")
            try:
                self.visualize_navigation(maze_name)
            except Exception as e:
                print(f"Error visualizing {maze_name}: {e}")
        
        # 学習進行状況
        if (self.experiment_path / "experiment_summary.json").exists():
            self.visualize_learning_progress()
        
        print(f"\n✅ Visualizations saved to: {self.figures_path}")


def main():
    """メイン実行"""
    import sys
    
    if len(sys.argv) > 1:
        experiment_path = sys.argv[1]
    else:
        # 最新の実験結果を使用
        results_dir = Path("../results")
        if results_dir.exists():
            experiments = sorted(results_dir.glob("pure_memory_*"))
            if experiments:
                experiment_path = experiments[-1]
                print(f"Using latest experiment: {experiment_path}")
            else:
                print("No experiment results found")
                return
        else:
            print("Results directory not found")
            return
    
    # 可視化実行
    visualizer = ResultVisualizer(experiment_path)
    visualizer.visualize_all()


if __name__ == "__main__":
    main()