# Visualization Files Organization Report
Date: 2025-08-03

## 概要
トップディレクトリに散らばっていた実験のビジュアライゼーション結果ファイルを、適切な実験ディレクトリに整理しました。

## 移動したファイル

### 1. Episodic Message Passing実験 (GeDIG関連)
**移動先**: `experiments/episodic-message-passing/results/`

| 元ファイル名 | 新ファイル名 | 生成日 | 関連プログラム |
|------------|------------|--------|--------------|
| gedig_episode_graph.png | gedig_episode_graph_20250802.png | 2025-08-02 | gedig_episode_navigator.py |
| gedig_frontier_graph.png | gedig_frontier_graph_20250802.png | 2025-08-02 | gedig_frontier_navigator.py |
| gedig_gnn_visualization.png | gedig_gnn_visualization_20250802.png | 2025-08-02 | gedig_gnn_navigator.py |
| gedig_graph_visualization.png | gedig_graph_visualization_20250802.png | 2025-08-02 | gedig_based_navigator.py |
| pure_gedig_visualization.png | pure_gedig_visualization_20250802.png | 2025-08-02 | gedig_pure_navigator.py |
| balanced_gedig_visualization.png | balanced_gedig_visualization_20250802.png | 2025-08-02 | gedig_balanced_navigator.py |
| final_gedig_visualization.png | final_gedig_visualization_20250802.png | 2025-08-02 | gedig_final_navigator.py |
| episode_graph_visualization.png | episode_graph_visualization_20250802.png | 2025-08-02 | episode_navigator.py |
| episode_graph_visualization_0.png | episode_graph_visualization_0_20250802.png | 2025-08-02 | episode_navigator.py |

### 2. Maze Agent Integration実験 (Large Maze)
**移動先**: `experiments/maze-agent-integration/results/large_maze_experiments/`

| 元ファイル名 | 新ファイル名 | 生成日 | 関連プログラム |
|------------|------------|--------|--------------|
| large_maze_progress_5000.png | large_maze_progress_5000_20250801.png | 2025-08-01 | large_maze_experiment.py |
| large_maze_progress_10000.png | large_maze_progress_10000_20250801.png | 2025-08-01 | large_maze_experiment.py |
| large_maze_progress_15000.png | large_maze_progress_15000_20250801.png | 2025-08-01 | large_maze_experiment.py |
| large_maze_progress_20000.png | large_maze_progress_20000_20250801.png | 2025-08-01 | large_maze_experiment.py |
| large_maze_progress_25000.png | large_maze_progress_25000_20250801.png | 2025-08-01 | large_maze_experiment.py |
| large_maze_final.png | large_maze_final_20250801.png | 2025-08-01 | large_maze_experiment.py |

### 3. Maze Agent Integration実験 (Query Generation)
**移動先**: `experiments/maze-agent-integration/results/`

| 元ファイル名 | 新ファイル名 | 生成日 | 関連プログラム |
|------------|------------|--------|--------------|
| query_generation_at_junction.png | query_generation_at_junction_20250802.png | 2025-08-02 | (maze_agent_integration内の実験) |

### 4. Wake Mode実験
**移動先**: `experiments/wake-mode/results/`

| 元ファイル名 | 新ファイル名 | 生成日 | 関連プログラム |
|------------|------------|--------|--------------|
| wake_mode_comparison.png | wake_mode_comparison_20250802.png | 2025-08-02 | wake_mode関連実験 |
| wake_mode_convergence.png | wake_mode_convergence_20250802.png | 2025-08-02 | wake_mode関連実験 |

## 作成したディレクトリ
- `experiments/episodic-message-passing/results/`
- `experiments/maze-agent-integration/results/large_maze_experiments/`
- `experiments/wake-mode/results/`

## 整理後の状態
トップディレクトリからすべてのPNGファイルが適切な実験ディレクトリに移動され、生成日と実行プログラムが紐付けられました。

## 今後の推奨事項
1. 実験実行時は、結果を直接適切な実験ディレクトリに保存するよう設定
2. ファイル名に日付を含めることで、バージョン管理を容易に
3. 各実験ディレクトリにREADME.mdを作成し、ファイルの説明を記載