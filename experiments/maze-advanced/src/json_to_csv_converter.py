#!/usr/bin/env python3
"""
JSON実験ログをCSVに変換
========================
"""

import json
import csv
from pathlib import Path
import sys

def convert_json_to_csv(json_path: str):
    """JSONログをCSVに変換"""
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 出力ファイル名
    base_name = Path(json_path).stem
    output_dir = Path(json_path).parent
    
    # 1. 初期エピソードCSV
    initial_csv_path = output_dir / f"{base_name}_initial_episodes.csv"
    with open(initial_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode_id", "type", "position_x", "position_y", 
            "action", "action_name", "result",
            "vector_x", "vector_y", "vector_action", "vector_result", "vector_visits",
            "description"
        ])
        
        for ep in data["initial_episodes"]:
            writer.writerow([
                ep["episode_id"],
                ep["type"],
                ep["position"][0] if "position" in ep else "",
                ep["position"][1] if "position" in ep else "",
                ep.get("action", ""),
                ep.get("action_name", ""),
                ep.get("result", ""),
                ep["vector"][0],
                ep["vector"][1],
                ep["vector"][2],
                ep["vector"][3],
                ep["vector"][4],
                ep["description"]
            ])
    
    print(f"初期エピソード保存: {initial_csv_path}")
    
    # 2. ステップごとの詳細CSV
    steps_csv_path = output_dir / f"{base_name}_steps.csv"
    with open(steps_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "step", "position_x", "position_y", 
            "input_vector_x", "input_vector_y", "input_vector_action", "input_vector_result", "input_vector_visits",
            "goal_distance", "selected_action", "selected_action_name", "result",
            "donut_inner_count", "donut_candidate_count", "donut_outer_count"
        ])
        
        for step in data["steps"]:
            proc = step["processing"]
            writer.writerow([
                step["step"],
                step["position"][0],
                step["position"][1],
                proc["current_vector"][0],
                proc["current_vector"][1],
                proc["current_vector"][2],
                proc["current_vector"][3],
                proc["current_vector"][4],
                proc["goal_distance"],
                proc["selected_action"],
                proc["selected_action_name"],
                step["result"],
                proc["donut_search"]["inner_count"],
                proc["donut_search"]["candidate_count"],
                proc["donut_search"]["outer_count"]
            ])
    
    print(f"ステップ詳細保存: {steps_csv_path}")
    
    # 3. 各ステップの行動評価CSV
    actions_csv_path = output_dir / f"{base_name}_action_evaluations.csv"
    with open(actions_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "step", "action", "action_name", "target_x", "target_y",
            "predicted_vector_x", "predicted_vector_y", "predicted_vector_action", 
            "predicted_vector_result", "predicted_vector_visits",
            "goal_distance", "visit_penalty", "total_score", "selected"
        ])
        
        for step in data["steps"]:
            for eval_action in step["processing"]["action_evaluations"]:
                writer.writerow([
                    step["step"],
                    eval_action["action"],
                    eval_action["action_name"],
                    eval_action["target_position"][0],
                    eval_action["target_position"][1],
                    eval_action["predicted_vector"][0],
                    eval_action["predicted_vector"][1],
                    eval_action["predicted_vector"][2],
                    eval_action["predicted_vector"][3],
                    eval_action["predicted_vector"][4],
                    eval_action["goal_distance"],
                    eval_action["visit_penalty"],
                    eval_action["total_score"],
                    "YES" if eval_action["action"] == step["processing"]["selected_action"] else "NO"
                ])
    
    print(f"行動評価保存: {actions_csv_path}")
    
    # 4. サマリー情報
    summary_path = output_dir / f"{base_name}_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"=== 実験サマリー ===\n")
        f.write(f"迷路サイズ: {data['metadata']['maze_size'][0]}x{data['metadata']['maze_size'][1]}\n")
        f.write(f"スタート位置: {data['metadata']['start_pos']}\n")
        f.write(f"ゴール位置: {data['metadata']['goal_pos']}\n")
        f.write(f"総ステップ数: {len(data['steps'])}\n")
        f.write(f"実験日時: {data['metadata']['timestamp']}\n\n")
        
        f.write("=== 経路 ===\n")
        for i, step in enumerate(data["steps"]):
            f.write(f"Step {i}: {step['previous_position'] if step['previous_position'] else 'Start'} "
                   f"→ {step['action_name']} → {step['position']} ({step['result']})\n")
    
    print(f"サマリー保存: {summary_path}")


def main():
    """メイン処理"""
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        # デフォルトで最新のファイルを処理
        results_dir = Path("results")
        json_files = list(results_dir.glob("maze_experiment_*.json"))
        if json_files:
            # 3x3と5x5のファイルを処理
            for json_file in sorted(json_files):
                print(f"\n処理中: {json_file}")
                convert_json_to_csv(str(json_file))
        else:
            print("JSONファイルが見つかりません")
            return
    
    if len(sys.argv) > 1:
        convert_json_to_csv(json_path)


if __name__ == "__main__":
    main()