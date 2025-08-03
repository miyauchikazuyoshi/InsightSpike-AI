#!/usr/bin/env python3
"""
Query-based Maze Agent
======================

MainAgentスタイルのクエリベース迷路エージェント
"""

import json
from typing import Dict, List, Tuple

class QueryBasedMazeAgent:
    """クエリベースの迷路エージェント"""
    
    def __init__(self):
        self.position = (0, 0)
        self.visited = []
        
    def generate_query(self) -> Dict:
        """現在状態からクエリを生成"""
        # MainAgentスタイルのクエリ
        query = {
            "type": "maze_navigation",
            "current_state": {
                "position": self.position,
                "visited_positions": self.visited,
                "visit_count": self.visited.count(self.position)
            },
            "request": "次の移動方向を決定してください",
            "context": {
                "maze_size": (3, 3),
                "goal_position": (2, 2),
                "available_actions": self._get_available_actions()
            }
        }
        return query
    
    def generate_natural_language_query(self) -> str:
        """自然言語クエリを生成"""
        x, y = self.position
        visits = self.visited.count(self.position)
        
        query = f"""
現在位置: ({x}, {y})
訪問回数: {visits}回
ゴール: (2, 2)
利用可能な方向: {self._get_available_actions_names()}

次にどの方向に進むべきですか？
"""
        return query.strip()
    
    def generate_vector_query(self) -> List[float]:
        """5次元ベクトルクエリを生成"""
        x, y = self.position
        last_action = self._get_last_action()
        last_result = self._get_last_result()
        visits = self.visited.count(self.position)
        
        # 5次元ベクトル: [X, Y, action, result, visits]
        return [
            x / 2.0,  # 3x3迷路なので2で正規化
            y / 2.0,
            last_action * 0.25 if last_action is not None else 0.5,
            last_result,
            min(visits / 10.0, 1.0)
        ]
    
    def _get_available_actions(self) -> List[int]:
        """利用可能な行動のリスト"""
        # 簡略化のため、全方向を返す
        return [0, 1, 2, 3]  # 上、右、下、左
    
    def _get_available_actions_names(self) -> List[str]:
        """利用可能な行動の名前"""
        actions = self._get_available_actions()
        names = ['上', '右', '下', '左']
        return [names[a] for a in actions]
    
    def _get_last_action(self) -> int:
        """最後の行動を取得"""
        # 実装省略
        return None
    
    def _get_last_result(self) -> float:
        """最後の結果を取得"""
        # 実装省略
        return 0.0


def demonstrate_queries():
    """クエリの例を表示"""
    agent = QueryBasedMazeAgent()
    
    print("=== クエリベース迷路エージェントのデモ ===\n")
    
    # 1. 構造化クエリ
    print("1. 構造化クエリ（JSON形式）:")
    query_dict = agent.generate_query()
    print(json.dumps(query_dict, indent=2, ensure_ascii=False))
    
    # 2. 自然言語クエリ
    print("\n2. 自然言語クエリ:")
    nl_query = agent.generate_natural_language_query()
    print(nl_query)
    
    # 3. ベクトルクエリ
    print("\n3. 5次元ベクトルクエリ:")
    vector_query = agent.generate_vector_query()
    print(f"Vector: {vector_query}")
    print(f"  X位置: {vector_query[0]:.2f}")
    print(f"  Y位置: {vector_query[1]:.2f}")
    print(f"  行動: {vector_query[2]:.2f}")
    print(f"  結果: {vector_query[3]:.2f}")
    print(f"  訪問: {vector_query[4]:.2f}")


if __name__ == "__main__":
    demonstrate_queries()