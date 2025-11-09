"""
GraphManager with NO COPY optimization.
グラフコピーを完全に排除した実装。
旧版はgraph_manager_legacy.pyを参照。
"""

from typing import List, Optional, Dict, Any, Set
import os
import numpy as np
import networkx as nx
from core.episode_manager import Episode
from core.gedig_evaluator import GeDIGEvaluator


class GraphManager:
    """グラフコピーなしでgeDIG計算を行うGraphManager"""
    
    def __init__(self, gedig_evaluator: Optional[GeDIGEvaluator] = None, max_snapshots: Optional[int] = None):
        self.graph = nx.Graph()
        self.gedig_evaluator = gedig_evaluator or GeDIGEvaluator()
        self.edge_logs: List[Dict[str, Any]] = []
        self.graph_history: List[nx.Graph] = []
        self.edge_creation_log = []
        # L1ランク履歴（観測ローカル/グローバル双方）
        # 各要素: {step:int, mode:'obs'|'global', source:int, candidates:[{id, dv, visit, obs, dir, is_wall, pos}], meta:{...}}
        self.l1_history: List[Dict[str, Any]] = []
        self.max_snapshots = max_snapshots  # 互換性のため
        # Keep a trajectory-only edge set for strict cycle detection
        self._trajectory_edges: Set[tuple[int,int]] = set()
        
        # グラフの状態を記録（geDIG計算用）
        self._graph_state_cache = {
            'num_nodes': 0,
            'num_edges': 0,
            'degrees': {}
        }
        # ランタイム調整用（高速化のためのガード）
        try:
            self._wiring_window_max = int(os.environ.get('MAZE_WIRING_WINDOW', '8'))
        except Exception:
            self._wiring_window_max = 8
        try:
            self._spatial_gate = int(os.environ.get('MAZE_SPATIAL_GATE', '0'))  # 0: 無効
        except Exception:
            self._spatial_gate = 0
        try:
            self._early_accept_margin = float(os.environ.get('MAZE_EARLY_ACCEPT_MARGIN', '0.02'))
        except Exception:
            self._early_accept_margin = 0.02
        # Wiring enhancements (top-k acceptance, trajectory edge, min-accept fallback)
        try:
            self._wiring_topk = int(os.environ.get('MAZE_WIRING_TOPK', '1'))
        except Exception:
            self._wiring_topk = 1
        try:
            self._wiring_force_prev = os.environ.get('MAZE_WIRING_FORCE_PREV', '0').strip() not in ("0","false","False","")
        except Exception:
            self._wiring_force_prev = False
        try:
            self._wiring_min_accept = os.environ.get('MAZE_WIRING_MIN_ACCEPT', '0').strip() not in ("0","false","False","")
        except Exception:
            self._wiring_min_accept = False
        # Fallback: if no edge accepted under threshold, force-connect the nearest one once
        try:
            self._force_nearest_on_empty = os.environ.get('MAZE_WIRING_FORCE_NEAREST_ON_EMPTY', '1').strip() not in ("0","false","False","")
        except Exception:
            self._force_nearest_on_empty = True
        # Step3: observation-based L1 wiring (current-step, passable & unvisited)
        try:
            self._obs_l1_enable = os.environ.get('MAZE_OBS_L1_ENABLE', '1').strip() not in ("0","false","False","")
        except Exception:
            self._obs_l1_enable = True
        try:
            self._obs_l1_mode = os.environ.get('MAZE_OBS_L1_MODE', 'dv_only').strip().lower()  # 'dv_only'|'filtered'
        except Exception:
            self._obs_l1_mode = 'dv_only'
        try:
            self._obs_l1_tau = float(os.environ.get('MAZE_OBS_L1_TAU', os.environ.get('MAZE_L1_NORM_TAU', '0.6')))
        except Exception:
            self._obs_l1_tau = 0.6
        try:
            self._obs_l1_topk = int(os.environ.get('MAZE_OBS_L1_TOPK', '3'))
        except Exception:
            self._obs_l1_topk = 3
        # Observation wiring guards
        try:
            self._obs_l1_skip_same_pos = os.environ.get('MAZE_OBS_L1_SKIP_SAME_POS', '1').strip() not in ("0","false","False","")
        except Exception:
            self._obs_l1_skip_same_pos = True
        try:
            self._obs_l1_allow_walls = os.environ.get('MAZE_OBS_L1_ALLOW_WALLS', '0').strip() not in ("0","false","False","")
        except Exception:
            self._obs_l1_allow_walls = False
        # Force-accept edges by L1 distance (frontier recall)
        try:
            self._wiring_force_l1 = os.environ.get('MAZE_WIRING_FORCE_L1', '0').strip() not in ("0","false","False","")
        except Exception:
            self._wiring_force_l1 = False
        try:
            self._wiring_force_l1_tau = float(os.environ.get('MAZE_WIRING_FORCE_L1_TAU', 'nan'))
        except Exception:
            self._wiring_force_l1_tau = float('nan')
        try:
            self._wiring_force_l1_topk = int(os.environ.get('MAZE_WIRING_FORCE_L1_TOPK', '0'))  # 0 = unlimited within candidates
        except Exception:
            self._wiring_force_l1_topk = 0
        # Layer1 norm search (embedding distance) gate
        try:
            self._l1_norm_search_enabled = os.environ.get('MAZE_L1_NORM_SEARCH', '0').strip() not in ("0", "false", "False", "")
        except Exception:
            self._l1_norm_search_enabled = False
        try:
            self._l1_norm_tau = float(os.environ.get('MAZE_L1_NORM_TAU', '0.75'))
        except Exception:
            self._l1_norm_tau = 0.75
        # Weighted L2 option for norm search
        try:
            self._l1_weighted = os.environ.get('MAZE_L1_WEIGHTED', '0').strip() not in ("0", "false", "False", "")
        except Exception:
            self._l1_weighted = False
        # Optional: unit-norm after weighting (stabilize τ scale)
        try:
            self._l1_unit_norm = os.environ.get('MAZE_L1_UNIT_NORM', '0').strip() not in ("0", "false", "False", "")
        except Exception:
            self._l1_unit_norm = False
        # Optional: hard-filter unvisited only (visit_count == 0)
        try:
            self._l1_filter_unvisited = os.environ.get('MAZE_L1_FILTER_UNVISITED', '0').strip() not in ("0","false","False","")
        except Exception:
            self._l1_filter_unvisited = False
        # Simple L1-distance threshold wiring (query~episode) independent of geDIG
        try:
            self._l1_thresh_enable = os.environ.get('MAZE_WIRING_L1_ENABLE', '0').strip() not in ("0","false","False","")
        except Exception:
            self._l1_thresh_enable = False
        try:
            self._l1_thresh_tau = float(os.environ.get('MAZE_WIRING_L1_DV_TAU', '0.0'))
        except Exception:
            self._l1_thresh_tau = 0.0
        try:
            self._l1_thresh_topk = int(os.environ.get('MAZE_WIRING_L1_DV_TOPK', '0'))  # 0=no cap
        except Exception:
            self._l1_thresh_topk = 0
        # Optional: use external candidate provider (vector index) to fetch Top-K L1 candidates in O(k)
        self._cand_provider = None  # type: ignore[var-annotated]
        try:
            self._l1_index_search = os.environ.get('MAZE_L1_INDEX_SEARCH', '0').strip() not in ("0","false","False","")
        except Exception:
            self._l1_index_search = False
        # NA-triggered re-evaluation (experimental)
        try:
            self._escalate_reeval_enabled = os.environ.get('MAZE_ESCALATE_REEVAL', '0').strip() not in ("0","false","False","")
        except Exception:
            self._escalate_reeval_enabled = False
        try:
            self._escalate_ring = int(os.environ.get('MAZE_ESCALATE_RING', '0'))
        except Exception:
            self._escalate_ring = 0
        # Active flag set by Navigator per-step
        self._escalate_reeval_active = False
        # Optional: query hub (experimental)
        try:
            self._use_query_hub = os.environ.get('MAZE_USE_QUERY_HUB', '0').strip() not in ("0","false","False","")
        except Exception:
            self._use_query_hub = False
        # Query hub id policy: per_episode (stable) | per_step (always new hub each step)
        try:
            self._query_hub_mode = os.environ.get('MAZE_QUERY_HUB_MODE', 'per_episode').strip().lower()
        except Exception:
            self._query_hub_mode = 'per_episode'
        try:
            self._query_hub_persist = os.environ.get('MAZE_QUERY_HUB_PERSIST', '0').strip() not in ("0","false","False","")
        except Exception:
            self._query_hub_persist = False
        try:
            self._query_hub_connect_current = os.environ.get('MAZE_QUERY_HUB_CONNECT_CURRENT', '1').strip() not in ("0","false","False","")
        except Exception:
            self._query_hub_connect_current = True
        self._last_query_hub = None  # type: ignore[var-annotated]
        # Optional: cap number of L1 candidates considered per decision (Top-K by distance)
        try:
            self._l1_cand_topk = int(os.environ.get('MAZE_L1_CAND_TOPK', '0'))  # 0 = no cap
        except Exception:
            self._l1_cand_topk = 0
        # Track L1 candidate counts for diagnostics
        self._l1_candidate_counts: List[int] = []
        # Default weights aligned with maze-unified-v2 (x,y,dx,dy,wall,visits,success,goal)
        default_weights = [1.0, 1.0, 0.0, 0.0, 3.0, 2.0, 0.0, 0.0]
        w_str = os.environ.get('MAZE_L1_WEIGHTS', '')
        if w_str:
            try:
                parts = [float(x) for x in w_str.split(',')]
                if len(parts) == 8:
                    self._l1_weights = np.array(parts, dtype=float)
                else:
                    self._l1_weights = np.array(default_weights, dtype=float)
            except Exception:
                self._l1_weights = np.array(default_weights, dtype=float)
        else:
            self._l1_weights = np.array(default_weights, dtype=float)

        # Observation (same-step) norm search gating and merged candidate cap
        try:
            self._obs_norm_search_enabled = os.environ.get('MAZE_OBS_NORM_SEARCH', '1').strip() not in ("0","false","False","")
        except Exception:
            self._obs_norm_search_enabled = True
        try:
            self._obs_norm_tau = float(os.environ.get('MAZE_OBS_NORM_TAU', str(self._l1_norm_tau)))
        except Exception:
            self._obs_norm_tau = float(self._l1_norm_tau)
        try:
            self._eval_cand_topk = int(os.environ.get('MAZE_EVAL_CAND_TOPK', '0'))  # 0 = no cap
        except Exception:
            self._eval_cand_topk = 0

        # Position-based L2 observation wiring (force-connect observed passable/unvisited episodes)
        try:
            self._l2_obs_enable = os.environ.get('MAZE_FORCE_L2_OBS', '0').strip() not in ("0","false","False","")
        except Exception:
            self._l2_obs_enable = False
        try:
            self._l2_obs_tau = float(os.environ.get('MAZE_FORCE_L2_TAU', '1.5'))
        except Exception:
            self._l2_obs_tau = 1.5
        try:
            self._l2_obs_topk = int(os.environ.get('MAZE_FORCE_L2_TOPK', '0'))  # 0=no cap
        except Exception:
            self._l2_obs_topk = 0
        try:
            self._l2_obs_only_unvisited = os.environ.get('MAZE_FORCE_L2_ONLY_UNVISITED', '1').strip() not in ("0","false","False","")
        except Exception:
            self._l2_obs_only_unvisited = True
        try:
            self._l2_obs_only_passage = os.environ.get('MAZE_FORCE_L2_ONLY_PASSAGE', '1').strip() not in ("0","false","False","")
        except Exception:
            self._l2_obs_only_passage = True

        # Multi-hop logging (observation only; does not affect decisions)
        self._log_hops: list[int] = []
        hops_env = os.environ.get('MAZE_LOG_HOPS', '').strip()
        if hops_env:
            try:
                self._log_hops = [int(h) for h in hops_env.split(',') if h.strip().isdigit()]
            except Exception:
                self._log_hops = [0, 1, 2]
        try:
            self._max_hops_obs = int(os.environ.get('MAZE_MAX_HOPS', '2'))
        except Exception:
            self._max_hops_obs = 2
        # Hop-based decision knobs (ensure attributes always exist even without candidate provider)
        try:
            self._use_hop_decision = os.environ.get('MAZE_USE_HOP_DECISION', '0').strip() not in ("0","false","False","")
        except Exception:
            self._use_hop_decision = False
        self._hop_decision_level = os.environ.get('MAZE_HOP_DECISION_LEVEL', 'min').strip()  # '0','1','2','min'
        try:
            self._hop_decision_max = int(os.environ.get('MAZE_HOP_DECISION_MAX', '2'))
        except Exception:
            self._hop_decision_max = 2

    # External L1 candidate provider injection
    # Callback signature: (current_ep: Episode, topk: int) -> List[int]
    # Returns episode_ids (preferably sorted by increasing distance). GraphManager will re-evaluate dv/threshold locally.
    def set_candidate_provider(self, provider) -> None:  # type: ignore[no-untyped-def]
        self._cand_provider = provider

        # Optional: use hop-based decision scoring
        try:
            self._use_hop_decision = os.environ.get('MAZE_USE_HOP_DECISION', '0').strip() not in ("0","false","False","")
        except Exception:
            self._use_hop_decision = False
        self._hop_decision_level = os.environ.get('MAZE_HOP_DECISION_LEVEL', 'min').strip()  # '0','1','2','min'
        try:
            self._hop_decision_max = int(os.environ.get('MAZE_HOP_DECISION_MAX', '2'))
        except Exception:
            self._hop_decision_max = 2
    
    def add_episode_node(self, episode: Episode) -> None:
        """Add episode node to graph."""
        self.graph.add_node(
            episode.episode_id,
            position=episode.position,
            timestamp=episode.timestamp
        )
        self._update_state_cache()
    
    def _update_state_cache(self) -> None:
        """グラフの状態をキャッシュ（geDIG計算の高速化）"""
        self._graph_state_cache['num_nodes'] = self.graph.number_of_nodes()
        self._graph_state_cache['num_edges'] = self.graph.number_of_edges()
        self._graph_state_cache['degrees'] = dict(self.graph.degree())

    def _bump_state_cache_edge(self, u, v) -> None:  # type: ignore[no-untyped-def]
        """軽量更新: エッジ(u,v)追加をキャッシュに反映（ノード数は不変と仮定）。"""
        try:
            c = self._graph_state_cache
            c['num_edges'] = int(c.get('num_edges', 0)) + 1
            degs = c.get('degrees')
            if not isinstance(degs, dict):
                degs = {}
            degs[u] = int(degs.get(u, 0)) + 1
            degs[v] = int(degs.get(v, 0)) + 1
            c['degrees'] = degs
        except Exception:
            # フォールバック（安全側）
            self._update_state_cache()
    
    def _wire_with_gedig_nocopy(
        self,
        episodes: List[Episode],
        threshold: float = -0.05  # 実測値に基づく適切な閾値
    ) -> None:
        """
        グラフコピーなしでgeDIG配線を実行
        
        トリック：
        1. エッジを一時的に追加
        2. geDIG計算（g_before=元の状態をエミュレート、g_after=現在の状態）
        3. 閾値を満たさなければエッジを削除
        """
        if len(episodes) < 2:
            return
        
        sorted_episodes = sorted(episodes, key=lambda e: e.timestamp)
        # id -> episode quick lookup
        id_map: Dict[int, Episode] = {ep.episode_id: ep for ep in sorted_episodes}
        # Frontier-like positions (any non-wall episode at that position with visit_count==0)
        frontier_pos: Set[tuple[int,int]] = set()
        try:
            for ep in sorted_episodes:
                try:
                    if not getattr(ep, 'is_wall', False) and getattr(ep, 'visit_count', 0) == 0:
                        frontier_pos.add(tuple(ep.position))  # type: ignore[arg-type]
                except Exception:
                    continue
        except Exception:
            frontier_pos = set()
        
        # Precompute quick lookup for id->index (for provider path)
        id_to_index: Dict[int, int] = {ep.episode_id: idx for idx, ep in enumerate(sorted_episodes)}

        for i in range(1, len(sorted_episodes)):
            current = sorted_episodes[i]
            prev_ep = sorted_episodes[i-1]
            best_connection = None
            best_gedig = None
            
            # 探索範囲を限定（パフォーマンス向上）
            # Use provider-based Top-K if available; otherwise fall back to all prior episodes
            indices: List[int]
            provider_used = False
            if self._l1_index_search and self._cand_provider is not None:
                try:
                    # Oversample to leave room for later dv/visit_count/frontier filters
                    oversample = max(0, (self._l1_cand_topk or 10))
                    cand_ids = self._cand_provider(current, max(5, oversample + 5))  # type: ignore[misc]
                    # Map to local indices and ensure only prior episodes are included
                    indices = []
                    for eid in cand_ids or []:
                        j = id_to_index.get(int(eid))
                        if j is None or j >= i:
                            continue
                        indices.append(int(j))
                    # Fallback if provider returned nothing
                    if not indices:
                        indices = list(range(0, i))
                    else:
                        provider_used = True
                except Exception:
                    indices = list(range(0, i))
            else:
                indices = list(range(0, i))
            # Layer1 候補（ノルム検索 + 空間ゲート）を先に集約（局所正規化の分母に使用）
            l1_candidates: List[int] = []
            cand_with_dist: List[tuple[int, float]] = []
            for j in indices:
                other = sorted_episodes[j]
                if self.graph.has_edge(current.episode_id, other.episode_id):
                    continue
                # Effective spatial gate (expand when NA re-eval active)
                eff_gate = self._spatial_gate + (self._escalate_ring if (self._escalate_reeval_enabled and self._escalate_reeval_active) else 0)
                if eff_gate > 0:
                    try:
                        cx, cy = current.position  # type: ignore[attr-defined]
                        ox, oy = other.position    # type: ignore[attr-defined]
                        if abs(cx - ox) + abs(cy - oy) > eff_gate:
                            continue
                    except Exception:
                        pass
                dv_val: float | None = None
                # Optional: filter out already visited memory episodes entirely
                if self._l1_filter_unvisited:
                    try:
                        if int(getattr(other, 'visit_count', 0)) != 0:
                            continue
                    except Exception:
                        pass
                if self._l1_norm_search_enabled:
                    try:
                        # Allow walls as L1 recall candidates (they can be filtered by distance/weights).
                        # Actual wiring to walls is prevented at acceptance stage below.
                        # Drop explicit unvisited-only filter to allow recalling known nodes and frontier carriers
                        # Build a query-like vector from current episode vector:
                        #  - keep normalized position as-is (dims 0,1)
                        #  - set direction neutral (dims 2,3)=0
                        #  - prefer passage (dim4)=+1
                        #  - prefer unvisited (dim5)=0
                        #  - neutral success/goal (dims 6,7)=0
                        v_cur = np.asarray(current.vector, dtype=float).copy()
                        if v_cur.shape[0] >= 8:
                            v_cur[2] = 0.0; v_cur[3] = 0.0
                            v_cur[4] = 1.0
                            v_cur[5] = 0.0
                            v_cur[6] = 0.0; v_cur[7] = 0.0
                        v_oth = np.asarray(other.vector, dtype=float)
                        if self._l1_weighted:
                            v_cur = v_cur * self._l1_weights
                            v_oth = v_oth * self._l1_weights
                        if self._l1_unit_norm:
                            n1 = float(np.linalg.norm(v_cur)) or 1.0
                            n2 = float(np.linalg.norm(v_oth)) or 1.0
                            v_cur = v_cur / n1
                            v_oth = v_oth / n2
                        dv = float(np.linalg.norm(v_cur - v_oth))
                        dv_val = dv
                        if dv > self._l1_norm_tau:
                            continue
                    except Exception:
                        # 距離計算失敗時はスキップしない
                        pass
                l1_candidates.append(j)
                cand_with_dist.append((j, dv_val if dv_val is not None else float('inf')))

            # Optional: reduce to Top-K nearest by dv (ascending)
            if self._l1_cand_topk and len(cand_with_dist) > self._l1_cand_topk:
                try:
                    cand_with_dist.sort(key=lambda t: t[1])
                    cand_with_dist = cand_with_dist[: self._l1_cand_topk]
                    l1_candidates = [j for j, _ in cand_with_dist]
                except Exception:
                    l1_candidates = l1_candidates[: self._l1_cand_topk]

            # Observed-now candidates (same-step episodes) — include into evaluation set (with optional norm gating)
            obs_now_with_dist: List[tuple[int, float]] = []
            try:
                ts_cur = int(getattr(current, 'timestamp', -1))
            except Exception:
                ts_cur = -1
            if ts_cur >= 0:
                for j2, ep2 in enumerate(sorted_episodes):
                    if j2 == i:
                        continue
                    try:
                        if int(getattr(ep2, 'timestamp', -2)) != ts_cur:
                            continue
                        if self.graph.has_edge(current.episode_id, ep2.episode_id):
                            continue
                        # Optional: restrict to passages (avoid wall)
                        if getattr(ep2, 'is_wall', False):
                            continue
                        dv2 = float('inf')
                        if self._obs_norm_search_enabled:
                            try:
                                v_cur = np.asarray(current.vector, dtype=float).copy()
                                if v_cur.shape[0] >= 8:
                                    v_cur[2] = 0.0; v_cur[3] = 0.0
                                    v_cur[4] = 1.0
                                    v_cur[5] = 0.0
                                    v_cur[6] = 0.0; v_cur[7] = 0.0
                                v_obs = np.asarray(ep2.vector, dtype=float)
                                if self._l1_weighted:
                                    v_cur = v_cur * self._l1_weights
                                    v_obs = v_obs * self._l1_weights
                                if self._l1_unit_norm:
                                    n1 = float(np.linalg.norm(v_cur)) or 1.0
                                    n2 = float(np.linalg.norm(v_obs)) or 1.0
                                    v_cur = v_cur / n1; v_obs = v_obs / n2
                                dv2 = float(np.linalg.norm(v_cur - v_obs))
                                if dv2 > float(self._obs_norm_tau):
                                    continue
                            except Exception:
                                # if norm gating fails, keep candidate (conservative)
                                dv2 = float('inf')
                        obs_now_with_dist.append((j2, dv2))
                    except Exception:
                        continue
            # Merge candidate indices with distances (prefer smaller dv per index)
            eval_with_dist: List[tuple[int, float]] = []
            best_dv: Dict[int, float] = {}
            for j, dv in (cand_with_dist + obs_now_with_dist):
                dvv = float('inf') if (dv is None) else float(dv)
                if (j not in best_dv) or (dvv < best_dv[j]):
                    best_dv[j] = dvv
            eval_with_dist = [(j, d) for j, d in best_dv.items()]
            # Optional cap by dv ascending
            if self._eval_cand_topk and len(eval_with_dist) > self._eval_cand_topk:
                try:
                    eval_with_dist.sort(key=lambda t: t[1])
                    eval_with_dist = eval_with_dist[: self._eval_cand_topk]
                except Exception:
                    eval_with_dist = eval_with_dist[: self._eval_cand_topk]
            # If current step has any unvisited observation, optionally suppress memory candidates
            try:
                suppress_mem = bool(self._suppress_memory_on_obs_unvisited)
            except Exception:
                suppress_mem = True
            # Compute unvisited-now flag once
            has_unvisited_now = False
            try:
                ts_cur2 = int(getattr(current, 'timestamp', -1))
            except Exception:
                ts_cur2 = -1
            obs_now_indices_set: set[int] = set(j for j, _ in obs_now_with_dist)
            if ts_cur2 >= 0:
                for e in episodes:
                    try:
                        if int(getattr(e, 'timestamp', -2)) != ts_cur2:
                            continue
                        if bool(getattr(e, 'is_wall', False)):
                            continue
                        if int(getattr(e, 'visit_count', 0)) == 0:
                            has_unvisited_now = True
                            break
                    except Exception:
                        continue
            # Allow relaxation when NA re-eval is active (Navigator signaled NA in previous step),
            # but only when local unvisited does NOT exist (dead-end like state)
            relax_on_na = False
            try:
                relax_on_na = (os.environ.get('MAZE_SUPPRESS_MEMORY_RELAX_ON_NA', '1').strip() not in ("0","false","False","")) \
                               and bool(getattr(self, '_escalate_reeval_active', False)) and (not has_unvisited_now)
            except Exception:
                relax_on_na = False
            if suppress_mem and (not relax_on_na):
                if has_unvisited_now and obs_now_indices_set:
                    eval_with_dist = [(j, d) for (j, d) in eval_with_dist if j in obs_now_indices_set]
            cand_eval_indices: List[int] = [j for j, _ in eval_with_dist]

            # Record candidate count for diagnostics
            try:
                self._l1_candidate_counts.append(len(cand_eval_indices))
            except Exception:
                pass

            # Optional: force-accept extremely close L1 neighbors (skip walls)
            if self._wiring_force_l1 and cand_with_dist:
                try:
                    force_tau = (self._wiring_force_l1_tau if (self._wiring_force_l1_tau==self._wiring_force_l1_tau) else self._l1_norm_tau)
                    # sort by distance ascending
                    cand_with_dist.sort(key=lambda t: (float('inf') if t[1] is None else t[1]))
                    forced = []
                    for j, dv in cand_with_dist:
                        if dv is None:
                            continue
                        # Only force-accept if it's frontier-like (promote unexplored branches)
                        other = sorted_episodes[j]
                        is_frontier = False
                        try:
                            if getattr(other,'visit_count',0)==0:
                                is_frontier = True
                            else:
                                pos = tuple(other.position)  # type: ignore[arg-type]
                                if pos in frontier_pos:
                                    is_frontier = True
                        except Exception:
                            is_frontier = False
                        if dv <= force_tau and is_frontier:
                            forced.append((j, dv))
                    if self._wiring_force_l1_topk and len(forced) > self._wiring_force_l1_topk:
                        forced = forced[: self._wiring_force_l1_topk]
                    # Wire forced edges (non-wall only)
                    for j, dv in forced:
                        other = sorted_episodes[j]
                        if not self.graph.has_edge(current.episode_id, other.episode_id):
                            # Pre-add: compute potential cycle info
                            cycle_info = self._compute_cycle_info_before_add(current.episode_id, other.episode_id)
                            self.graph.add_edge(current.episode_id, other.episode_id)
                            self.edge_logs.append({'from': current.episode_id, 'to': other.episode_id, 'gedig': None, 'threshold': threshold, 'l1_force': float(dv)})
                            # 軽量キャッシュ更新
                            self._bump_state_cache_edge(current.episode_id, other.episode_id)
                            self._log_edge_creation(current.episode_id, other.episode_id, f'l1_force:{dv:.3f}')
                            # Attach cycle info
                            try:
                                if cycle_info is not None and self.edge_creation_log:
                                    self.edge_creation_log[-1]['cycle'] = cycle_info
                                    self.edge_logs[-1]['delta_sp'] = cycle_info.get('delta_sp')
                            except Exception:
                                pass
                except Exception:
                    pass

            # Experimental: Add transient "query hub" connected to l1 candidates
            hub_id = None
            if self._use_query_hub:
                try:
                    # Clean previous hub if not persistent
                    if not self._query_hub_persist and getattr(self, '_last_query_hub', None) and self.graph.has_node(self._last_query_hub):
                        self.graph.remove_node(self._last_query_hub)
                    # Use numeric hub id to avoid mixed-type node sorting in core (negative id namespace)
                    if getattr(self, '_query_hub_mode', 'per_episode') in ('per_step','step','time'):
                        hub_id = int(-1000000 - int(getattr(self, '_current_step', 0)))
                    else:
                        hub_id = int(-1000000 - int(current.episode_id))
                    if not self.graph.has_node(hub_id):
                        self.graph.add_node(hub_id)
                    if self._query_hub_connect_current and not self.graph.has_edge(hub_id, current.episode_id):
                        self.graph.add_edge(hub_id, current.episode_id)
                        self._log_edge_creation(hub_id, current.episode_id, 'query_hub')
                    for j in l1_candidates:
                        eid = sorted_episodes[j].episode_id
                        if not self.graph.has_edge(hub_id, eid):
                            self.graph.add_edge(hub_id, eid)
                            self._log_edge_creation(hub_id, eid, 'query_hub')
                    self._last_query_hub = hub_id
                except Exception:
                    hub_id = None

            # Step3: observation-based L1 wiring for current-step episodes (passable & unvisited)
            try:
                if self._obs_l1_enable and hasattr(current, 'timestamp'):
                    ts_cur = int(getattr(current, 'timestamp', -1))
                    if ts_cur >= 0:
                        # Build dv to same-step episodes using the SAME weighting/unit-norm policy
                        # as provider-based L1 search to keep consistency.
                        import numpy as _np
                        def _prep(vec):
                            v = _np.asarray(vec, dtype=float)
                            try:
                                if self._l1_weighted and isinstance(self._l1_weights, _np.ndarray):
                                    v = v * self._l1_weights
                                if self._l1_unit_norm:
                                    n = float(_np.linalg.norm(v)) or 1.0
                                    v = v / n
                            except Exception:
                                # Fallback: raw vector
                                pass
                            return v
                        cv = _prep(getattr(current, 'vector'))
                        cand_now: list[tuple[int, float]] = []
                        cand_now_details: list[Dict[str, Any]] = []
                        for e in episodes:
                            try:
                                if not isinstance(e, Episode):
                                    continue
                                if int(getattr(e, 'timestamp', -2)) != ts_cur:
                                    continue
                                # dv_only モードでは is_wall / visit_count で除外しない（純粋な距離ゲート）
                                if self._obs_l1_mode != 'dv_only':
                                    if bool(getattr(e, 'is_wall', False)):
                                        continue
                                    if int(getattr(e, 'visit_count', 0)) != 0:
                                        continue
                                # 追加ガード: 同一座標への配線はスキップ（微小サイクル抑止）
                                if self._obs_l1_skip_same_pos:
                                    try:
                                        ex, ey = int(e.position[0]), int(e.position[1])
                                        if (ex, ey) == (cx, cy):
                                            continue
                                    except Exception:
                                        pass
                                # 明示的に壁を除外（dv_onlyでも）
                                if (not self._obs_l1_allow_walls) and bool(getattr(e, 'is_wall', False)):
                                    continue
                                if self.graph.has_edge(current.episode_id, e.episode_id):
                                    continue
                                ev = _prep(getattr(e, 'vector'))
                                dv = float(_np.linalg.norm(cv - ev))
                                cand_now.append((int(e.episode_id), dv))
                                # 収集（観測回数/訪問回数など）
                                try:
                                    cand_now_details.append({
                                        'id': int(e.episode_id),
                                        'dv': float(dv),
                                        'visit': int(getattr(e, 'visit_count', 0)),
                                        'obs': int(getattr(e, 'observation_count', 0)),
                                        'dir': str(getattr(e, 'direction', None)),
                                        'is_wall': bool(getattr(e, 'is_wall', False)),
                                        'pos': [int(e.position[0]), int(e.position[1])] if hasattr(e, 'position') else None,
                                    })
                                except Exception:
                                    pass
                            except Exception:
                                continue
                        cand_now.sort(key=lambda t: t[1])
                        # 履歴ログ（TopK適用前で全候補、ただし後段でTopK表示用フィールドも添付）
                        try:
                            self.l1_history.append({
                                'step': int(getattr(self, '_current_step', -1)),
                                'mode': 'obs',
                                'source': int(getattr(current, 'episode_id', -1)),
                                'candidates': cand_now_details,
                                'meta': {
                                    'tau': float(self._obs_l1_tau),
                                    'topk': int(self._obs_l1_topk),
                                    'weighted': bool(self._l1_weighted),
                                    'unit_norm': bool(self._l1_unit_norm),
                                    'skip_same_pos': bool(self._obs_l1_skip_same_pos),
                                    'allow_walls': bool(self._obs_l1_allow_walls),
                                }
                            })
                        except Exception:
                            pass
                        if self._obs_l1_topk and len(cand_now) > self._obs_l1_topk:
                            cand_now = cand_now[: self._obs_l1_topk]
                        for eid, dv in cand_now:
                            if dv > float(self._obs_l1_tau):
                                continue
                            if not self.graph.has_edge(current.episode_id, eid):
                                cycle_info = self._compute_cycle_info_before_add(current.episode_id, eid)
                                self.graph.add_edge(current.episode_id, eid)
                                # Log dv with flags to aid debugging (weight/unit norm used)
                                self.edge_logs.append({
                                    'from': current.episode_id,
                                    'to': eid,
                                    'dv': float(dv),
                                    'strategy': 'obs_l1',
                                    'l1_weighted': bool(self._l1_weighted),
                                    'l1_unit_norm': bool(self._l1_unit_norm),
                                    'obs_l1_tau': float(self._obs_l1_tau),
                                })
                                self._bump_state_cache_edge(current.episode_id, eid)
                                self._log_edge_creation(current.episode_id, eid, f'obs_l1:{dv:.3f}')
                                try:
                                    if cycle_info is not None and self.edge_creation_log:
                                        self.edge_creation_log[-1]['cycle'] = cycle_info
                                        self.edge_logs[-1]['delta_sp'] = cycle_info.get('delta_sp')
                                except Exception:
                                    pass
            except Exception:
                pass

            # 評価ループ（同一候補集合で評価）
            evaluated: List[tuple[float, int]] = []
            for j in cand_eval_indices:
                other = sorted_episodes[j]
                # === コピーなしでgeDIG計算 ===
                if self._use_hop_decision:
                    decision_value, hop_obs = self._score_candidate_decision(current.episode_id, other.episode_id, l1_count=len(l1_candidates))
                    gedig_value = decision_value
                else:
                    self.graph.add_edge(current.episode_id, other.episode_id)
                    gedig_value = self._calculate_gedig_nocopy(current.episode_id, other.episode_id, l1_count=len(l1_candidates))
                    self.graph.remove_edge(current.episode_id, other.episode_id)
                evaluated.append((float(gedig_value), int(other.episode_id)))
                if best_gedig is None or gedig_value < best_gedig:
                    best_gedig = gedig_value
                    best_connection = other.episode_id
            
            # Top-K 受理（しきい値以下）
            accepted: List[tuple[float, int]] = []
            if evaluated:
                evaluated.sort(key=lambda x: x[0])
                for val, eid in evaluated:
                    if val <= threshold:
                        accepted.append((val, eid))
                        if len(accepted) >= max(1, self._wiring_topk):
                            break
                # Fallback: しきい値下回りゼロで許可されている場合、最良の1本を受理
                # 統一F制御のため、NAリラックス時の無条件受理は行わない
                if not accepted and (self._wiring_min_accept or self._force_nearest_on_empty):
                    accepted = [evaluated[0]]
            # エッジ追加
            added_targets: List[int] = []
            for val, eid in accepted:
                if not self.graph.has_edge(current.episode_id, eid):
                    # Pre-add: compute potential cycle info (existing u-v path)
                    cycle_info = self._compute_cycle_info_before_add(current.episode_id, eid)
                    self.graph.add_edge(current.episode_id, eid)
                    self.edge_logs.append({'from': current.episode_id, 'to': eid, 'gedig': val, 'threshold': threshold, 'l1_count': len(l1_candidates)})
                    # 軽量キャッシュ更新
                    self._bump_state_cache_edge(current.episode_id, eid)
                    self._log_edge_creation(current.episode_id, eid, f'gedig_nocopy (value={val:.3f})')
                    # Attach cycle info to last creation log (if any) with threshold
                    try:
                        if cycle_info is not None and self.edge_creation_log:
                            try:
                                attach_tau = float(os.environ.get('MAZE_CYCLE_ATTACH_TAU', '1.3'))
                            except Exception:
                                attach_tau = 1.3
                            if float(cycle_info.get('delta_sp', 0.0)) >= attach_tau:
                                self.edge_creation_log[-1]['cycle'] = cycle_info
                                # Also mirror minimal fields to edge_logs for unified analysis
                                self.edge_logs[-1]['delta_sp'] = cycle_info.get('delta_sp')
                    except Exception:
                        pass
                    if self._log_hops or self._use_hop_decision:
                        try:
                            hop_metrics = self._observe_hops(current.episode_id, eid, l1_count=len(l1_candidates))
                            if hop_metrics:
                                self.edge_logs[-1]['hop_metrics'] = hop_metrics
                        except Exception:
                            pass
                    added_targets.append(eid)

            # Position-based L2 observation wiring (force)
            if self._l2_obs_enable:
                try:
                    cx, cy = (int(current.position[0]), int(current.position[1])) if hasattr(current, 'position') else (None, None)
                except Exception:
                    cx, cy = (None, None)
                wired = 0
                if cx is not None and cy is not None:
                    cand_pos: List[tuple[int, float]] = []
                    for j in range(0, i):  # only prior episodes
                        other = sorted_episodes[j]
                        if self.graph.has_edge(current.episode_id, other.episode_id):
                            continue
                        # Filters
                        if self._l2_obs_only_passage and getattr(other, 'is_wall', False):
                            continue
                        if self._l2_obs_only_unvisited and getattr(other, 'visit_count', 0) != 0:
                            continue
                        try:
                            ox, oy = int(other.position[0]), int(other.position[1])
                        except Exception:
                            continue
                        dx = float(ox - cx); dy = float(oy - cy)
                        l2 = (dx*dx + dy*dy) ** 0.5
                        if l2 <= self._l2_obs_tau:
                            cand_pos.append((j, l2))
                    cand_pos.sort(key=lambda t: t[1])
                    if self._l2_obs_topk and len(cand_pos) > self._l2_obs_topk:
                        cand_pos = cand_pos[: self._l2_obs_topk]
                    for j, l2 in cand_pos:
                        other = sorted_episodes[j]
                        if self.graph.has_edge(current.episode_id, other.episode_id):
                            continue
                        cycle_info = self._compute_cycle_info_before_add(current.episode_id, other.episode_id)
                        self.graph.add_edge(current.episode_id, other.episode_id)
                        self.edge_logs.append({'from': current.episode_id, 'to': other.episode_id, 'l2': float(l2), 'strategy': 'l2_obs'})
                        self._bump_state_cache_edge(current.episode_id, other.episode_id)
                        self._log_edge_creation(current.episode_id, other.episode_id, f'l2_obs:{l2:.3f}')
                        try:
                            if cycle_info is not None and self.edge_creation_log:
                                try:
                                    attach_tau = float(os.environ.get('MAZE_CYCLE_ATTACH_TAU', '1.3'))
                                except Exception:
                                    attach_tau = 1.3
                                if float(cycle_info.get('delta_sp', 0.0)) >= attach_tau:
                                    self.edge_creation_log[-1]['cycle'] = cycle_info
                                    self.edge_logs[-1]['delta_sp'] = cycle_info.get('delta_sp')
                        except Exception:
                            pass
                        wired += 1
                        if self._l2_obs_topk and wired >= self._l2_obs_topk:
                            break
                    # Also consider episodes observed at the same step (current observation), which may not be in prior set
                    try:
                        ts_cur = int(getattr(current, 'timestamp', -1))
                    except Exception:
                        ts_cur = -1
                    if ts_cur >= 0:
                        cand_now: List[tuple[int, float]] = []
                        for e in episodes:  # all known episodes
                            if not isinstance(e, Episode):
                                continue
                            if self.graph.has_edge(current.episode_id, e.episode_id):
                                continue
                            try:
                                if int(getattr(e, 'timestamp', -2)) != ts_cur:
                                    continue
                                if self._l2_obs_only_passage and getattr(e, 'is_wall', False):
                                    continue
                                if self._l2_obs_only_unvisited and getattr(e, 'visit_count', 0) != 0:
                                    continue
                                ox, oy = int(e.position[0]), int(e.position[1])
                                dx = float(ox - cx); dy = float(oy - cy)
                                l2 = (dx*dx + dy*dy) ** 0.5
                                if l2 <= self._l2_obs_tau:
                                    cand_now.append((e.episode_id, l2))
                            except Exception:
                                continue
                        cand_now.sort(key=lambda t: t[1])
                        # Limit to remaining budget under topk
                        remain = max(0, (self._l2_obs_topk - wired)) if self._l2_obs_topk else len(cand_now)
                        cand_now = cand_now[:remain]
                        for eid, l2 in cand_now:
                            if self.graph.has_edge(current.episode_id, eid):
                                continue
                            cycle_info = self._compute_cycle_info_before_add(current.episode_id, eid)
                            self.graph.add_edge(current.episode_id, eid)
                            self.edge_logs.append({'from': current.episode_id, 'to': eid, 'l2': float(l2), 'strategy': 'l2_obs_now'})
                            self._bump_state_cache_edge(current.episode_id, eid)
                            self._log_edge_creation(current.episode_id, eid, f'l2_obs_now:{l2:.3f}')
                            try:
                                if cycle_info is not None and self.edge_creation_log:
                                    self.edge_creation_log[-1]['cycle'] = cycle_info
                                    self.edge_logs[-1]['delta_sp'] = cycle_info.get('delta_sp')
                            except Exception:
                                pass
                            wired += 1
                            if self._l2_obs_topk and wired >= self._l2_obs_topk:
                                break

            # L2/memory fallback: when no observation L1 edges were accepted, optionally add exactly one
            # non-wall past episode edge chosen to maximize potential cycle (ΔSP), subject to L2 tau.
            try:
                enable_fb = os.environ.get('MAZE_L2_FALLBACK_ON_OBS_EMPTY', '1').strip() not in ("0","false","False","")
            except Exception:
                enable_fb = True
            if enable_fb:
                try:
                    fb_tau = float(os.environ.get('MAZE_L2_FALLBACK_TAU', '2.5'))
                except Exception:
                    fb_tau = 2.5
                try:
                    fb_require_cycle = os.environ.get('MAZE_L2_FALLBACK_REQUIRE_CYCLE', '1').strip() not in ("0","false","False","")
                except Exception:
                    fb_require_cycle = True
                try:
                    fb_topk = int(os.environ.get('MAZE_L2_FALLBACK_TOPK', '1'))
                except Exception:
                    fb_topk = 1
                # Count how many obs_l1 edges were accepted earlier in this loop by scanning edge_logs for current step/source
                obs_now_accepts = 0
                try:
                    step_now = int(getattr(self, '_current_step', -1))
                except Exception:
                    step_now = -1
                try:
                    for rec in (self.edge_creation_log[-50:] if self.edge_creation_log else []):
                        if not isinstance(rec, dict):
                            continue
                        if int(rec.get('step', -1)) != step_now:
                            continue
                        if rec.get('strategy','').startswith('obs_l1') and int(rec.get('source', -1)) == int(current.episode_id):
                            obs_now_accepts += 1
                except Exception:
                    pass
                if obs_now_accepts == 0:
                    # Collect candidate past episodes within L2 tau
                    try:
                        cx, cy = int(current.position[0]), int(current.position[1])
                    except Exception:
                        cx, cy = (None, None)
                    if cx is not None and cy is not None:
                        best = None  # (delta_sp, l2, j)
                        for j in range(0, i):  # only prior episodes
                            other = sorted_episodes[j]
                            if self.graph.has_edge(current.episode_id, other.episode_id):
                                continue
                            if getattr(other, 'is_wall', False):
                                continue
                            try:
                                ox, oy = int(other.position[0]), int(other.position[1])
                            except Exception:
                                continue
                            dx = float(ox - cx); dy = float(oy - cy)
                            l2 = (dx*dx + dy*dy) ** 0.5
                            if l2 > fb_tau:
                                continue
                            # Estimate cycle gain if we add (current, other)
                            cyc = self._compute_cycle_info_before_add(current.episode_id, other.episode_id)
                            dsp = float(cyc.get('delta_sp', 0.0)) if isinstance(cyc, dict) else 0.0
                            if fb_require_cycle and dsp <= 0.0:
                                continue
                            if (best is None) or (dsp > best[0]) or (dsp == best[0] and l2 < best[1]):
                                best = (dsp, l2, j, cyc)
                        if best is not None:
                            _, l2d, jb, cyc_info = best
                            other = sorted_episodes[int(jb)]
                            if not self.graph.has_edge(current.episode_id, other.episode_id):
                                self.graph.add_edge(current.episode_id, other.episode_id)
                                self.edge_logs.append({'from': current.episode_id, 'to': other.episode_id, 'l2': float(l2d), 'strategy': 'l2_fallback'})
                                self._bump_state_cache_edge(current.episode_id, other.episode_id)
                                self._log_edge_creation(current.episode_id, other.episode_id, f'l2_fallback:{l2d:.3f}')
                                try:
                                    if isinstance(cyc_info, dict) and self.edge_creation_log:
                                        attach_tau = float(os.environ.get('MAZE_CYCLE_ATTACH_TAU', '1.3'))
                                        if float(cyc_info.get('delta_sp', 0.0)) >= attach_tau:
                                            self.edge_creation_log[-1]['cycle'] = cyc_info
                                            self.edge_logs[-1]['delta_sp'] = cyc_info.get('delta_sp')
                                except Exception:
                                    pass

            # Optional: L1-distance threshold wiring (pure memory relation)
            if self._l1_thresh_enable and cand_with_dist:
                try:
                    cd_sorted = sorted(cand_with_dist, key=lambda t: (float('inf') if t[1] is None else float(t[1])))
                    wired_l1 = 0
                    for j, dv in cd_sorted:
                        if dv is None:
                            continue
                        if self._l1_thresh_topk and wired_l1 >= self._l1_thresh_topk:
                            break
                        if float(dv) <= float(self._l1_thresh_tau):
                            other = sorted_episodes[j]
                            if not self.graph.has_edge(current.episode_id, other.episode_id):
                                # Pre-add: compute potential cycle info
                                cycle_info = self._compute_cycle_info_before_add(current.episode_id, other.episode_id)
                                self.graph.add_edge(current.episode_id, other.episode_id)
                                self.edge_logs.append({'from': current.episode_id, 'to': other.episode_id, 'dv': float(dv), 'strategy': 'l1_sim'})
                                # 軽量キャッシュ更新（全再計算は避ける）
                                self._bump_state_cache_edge(current.episode_id, other.episode_id)
                                self._log_edge_creation(current.episode_id, other.episode_id, f'l1_sim:{dv:.3f}')
                                # Attach cycle info
                                try:
                                    if cycle_info is not None and self.edge_creation_log:
                                        try:
                                            attach_tau = float(os.environ.get('MAZE_CYCLE_ATTACH_TAU', '1.3'))
                                        except Exception:
                                            attach_tau = 1.3
                                        if float(cycle_info.get('delta_sp', 0.0)) >= attach_tau:
                                            self.edge_creation_log[-1]['cycle'] = cycle_info
                                            self.edge_logs[-1]['delta_sp'] = cycle_info.get('delta_sp')
                                except Exception:
                                    pass
                                wired_l1 += 1
                        else:
                            break
                except Exception:
                    pass
            # 直前エピソードへの軌跡エッジ（常時）
            if prev_ep and not self.graph.has_edge(current.episode_id, prev_ep.episode_id):
                # Pre-add: cycle detection
                cycle_info = self._compute_cycle_info_before_add(current.episode_id, prev_ep.episode_id)
                self.graph.add_edge(current.episode_id, prev_ep.episode_id)
                self._bump_state_cache_edge(current.episode_id, prev_ep.episode_id)
                self._log_edge_creation(current.episode_id, prev_ep.episode_id, 'trajectory')
                try:
                    if cycle_info is not None and self.edge_creation_log:
                        self.edge_creation_log[-1]['cycle'] = cycle_info
                except Exception:
                    pass

            # Remove transient hub if not persistent
            if self._use_query_hub and not self._query_hub_persist:
                try:
                    if hub_id and self.graph.has_node(hub_id):
                        self.graph.remove_node(hub_id)
                        if getattr(self, '_last_query_hub', None) == hub_id:
                            self._last_query_hub = None
                except Exception:
                    pass
    
    def _calculate_gedig_nocopy(self, node1: int, node2: int, l1_count: Optional[int] = None) -> float:
        """
        コピーなしでgeDIG値を計算
        
        現在のグラフにはすでにエッジが追加されている状態。
        「前の状態」を仮想的に作成してgeDIG計算。
        """
        # 高速化: NetworkXの number_of_edges()/degree() 合算は O(E) になりうるため、
        # GraphManager が保持するキャッシュを用いて O(1) で計算する。
        # 評価時は「一時的にエッジが1本追加された後」の状態なので、
        # キャッシュ（追加前）から e_now = e_prev + 1 を再構成する。

        cache = self._graph_state_cache
        n_prev = int(cache.get('num_nodes', self.graph.number_of_nodes()))
        e_prev = int(cache.get('num_edges', self.graph.number_of_edges()))
        # 追加後の状態（現在の実グラフには一時エッジが載っている想定）
        n_now = n_prev
        e_now = e_prev + 1

        # 構造改善度（フォールバック実装と同等）
        dn = 0  # ノード数は変わらない
        de = 1  # エッジは1増える
        denom = (n_prev + n_now + 1)
        structural_improvement = -(dn + 0.5 * de) / denom if denom > 0 else 0.0

        # 接続性評価：次数は「追加前」の値を使う（従来コードは after-1 で同等）
        deg_cache: Dict[int, int] = cache.get('degrees', {}) or {}
        actual_deg1 = int(deg_cache.get(node1, 0))
        actual_deg2 = int(deg_cache.get(node2, 0))

        # 平均次数は after 状態から算出（sum(deg)=2E を利用）
        if n_now > 0:
            avg_degree = (2.0 * float(e_now)) / float(n_now)
        else:
            avg_degree = 0.0

        if avg_degree > 0.0:
            degree_factor = 1.0 - (float(actual_deg1 + actual_deg2) / (2.0 * avg_degree))
        else:
            degree_factor = 1.0

        # geDIG値（構造改善度 × 次数補正）
        base_gedig = structural_improvement * max(0.5, degree_factor)
        # スケーリング（従来相当）
        gedig_value = base_gedig * 5.5
        return float(gedig_value)
    
    def _calculate_gedig_accurate(self, node1: int, node2: int, l1_count: Optional[int] = None) -> float:
        """
        より正確なgeDIG計算（必要に応じて本物のGeDIGEvaluatorを使用）
        
        注意：これは一時的な仮想グラフを作成するが、
        メイングラフのコピーは作らない。
        """
        # 小さな仮想グラフを作成（エッジ追加前後）
        g_before = nx.Graph()
        g_after = nx.Graph()
        
        # 関連するノードとエッジのみをコピー（局所的）
        nodes = {node1, node2}
        for node in [node1, node2]:
            if self.graph.has_node(node):
                nodes.update(self.graph.neighbors(node))
        
        # 仮想グラフを構築
        for node in nodes:
            g_before.add_node(node)
            g_after.add_node(node)
        
        for n1, n2 in self.graph.edges():
            if n1 in nodes and n2 in nodes:
                if not (n1 == node1 and n2 == node2) and not (n1 == node2 and n2 == node1):
                    g_before.add_edge(n1, n2)
                g_after.add_edge(n1, n2)
        
        # 本物のgeDIG計算（局所的なグラフで）
        try:
            result = self.gedig_evaluator.calculate(g_before, g_after, l1_candidates=l1_count)
        except TypeError:
            # Fallback for older evaluator signature without l1_candidates
            result = self.gedig_evaluator.calculate(g_before, g_after)
        
        if hasattr(result, 'gedig_value'):
            return result.gedig_value
        elif hasattr(result, 'structural_improvement'):
            return result.structural_improvement
        else:
            return float(result)

    def _observe_hops(self, node1: int, node2: int, l1_count: Optional[int] = None) -> Dict[int, Dict[str, float]]:
        """Compute hop0..H metrics (ged, ig, gedig) on a small local subgraph around (node1,node2).
        Observation only: does not affect decision. Uses GeDIGCore with enable_multihop=True.
        """
        # Build local before/after graphs as in _calculate_gedig_accurate
        g_before = nx.Graph(); g_after = nx.Graph()
        nodes = {node1, node2}
        for node in [node1, node2]:
            if self.graph.has_node(node):
                nodes.update(self.graph.neighbors(node))
        for n in nodes:
            g_before.add_node(n); g_after.add_node(n)
        for a, b in self.graph.edges():
            if a in nodes and b in nodes:
                if not ((a == node1 and b == node2) or (a == node2 and b == node1)):
                    g_before.add_edge(a, b)
                g_after.add_edge(a, b)
        # Construct a local GeDIGCore for multihop observation
        try:
            # Ensure root src on path (mirror evaluator adapter behavior)
            import sys as _sys, os as _os
            _ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '../../../..', 'src'))
            if _ROOT not in _sys.path:
                _sys.path.insert(0, _ROOT)
            from insightspike.algorithms.gedig_core import GeDIGCore  # type: ignore
        except Exception:
            return {}
        # Mirror evaluator flags from environment for consistency
        use_local_norm = os.environ.get('MAZE_GEDIG_LOCAL_NORM', '0').strip() not in ("0","false","False","")
        local_mode = os.environ.get('MAZE_GEDIG_LOCAL_MODE', 'layer1')
        use_sp_gain = os.environ.get('MAZE_GEDIG_SP_GAIN', '0').strip() not in ("0","false","False","")
        sp_mode = os.environ.get('MAZE_GEDIG_SP_MODE', 'relative')
        core = GeDIGCore(
            enable_multihop=True,
            max_hops=max(0, min(10, self._max_hops_obs)),
            decay_factor=0.7,
            adaptive_hops=False,
            use_refactored_reward=True,
            use_local_normalization=use_local_norm,
            local_norm_mode=local_mode,
            use_multihop_sp_gain=use_sp_gain,
            sp_norm_mode=sp_mode,
        )
        try:
            from insightspike.algorithms.linkset_adapter import build_linkset_info as _build_ls  # type: ignore
        except Exception:
            _build_ls = None  # type: ignore
        ls = _build_ls(s_link=[], candidate_pool=[], decision={}, query_vector=None, base_mode="link") if _build_ls else None
        if ls is not None:
            res = core.calculate(g_prev=g_before, g_now=g_after, l1_candidates=l1_count, linkset_info=ls)
        else:
            res = core.calculate(g_prev=g_before, g_now=g_after, l1_candidates=l1_count)
        out: Dict[int, Dict[str, float]] = {}
        hops = sorted(res.hop_results.keys()) if res.hop_results else []  # type: ignore
        for h in hops:
            if self._log_hops and (h not in self._log_hops):
                continue
            hr = res.hop_results[h]
            out[h] = {'ged': float(getattr(hr, 'ged', 0.0)),
                      'ig': float(getattr(hr, 'ig', 0.0)),
                      'gedig': float(getattr(hr, 'gedig', 0.0))}
        return out

    def _score_candidate_decision(self, node1: int, node2: int, l1_count: Optional[int] = None) -> tuple[float, Dict[int, float]]:
        """Return decision score using hop-based evaluation and recorded hop scores.
        Builds small local graphs and uses evaluator.evaluate_escalating.
        """
        # Build local subgraphs
        g_before = nx.Graph(); g_after = nx.Graph()
        nodes = {node1, node2}
        for node in [node1, node2]:
            if self.graph.has_node(node):
                nodes.update(self.graph.neighbors(node))
        for n in nodes:
            g_before.add_node(n); g_after.add_node(n)
        for a, b in self.graph.edges():
            if a in nodes and b in nodes:
                if not ((a == node1 and b == node2) or (a == node2 and b == node1)):
                    g_before.add_edge(a, b)
                g_after.add_edge(a, b)
        # Evaluate escalating
        try:
            res = self.gedig_evaluator.evaluate_escalating(g_before, g_after, escalation_threshold=None, max_hops=self._hop_decision_max, l1_candidates=l1_count)
        except Exception:
            # fallback to nocopy heuristic
            self.graph.add_edge(node1, node2)
            val = self._calculate_gedig_nocopy(node1, node2, l1_count=l1_count)
            self.graph.remove_edge(node1, node2)
            return float(val), {}
        base = float(res.get('score', 0.0))
        multihop = res.get('multihop') or {}
        # Decide
        level = self._hop_decision_level
        if level in ('1','2'):
            h = int(level)
            val = float(multihop.get(h, base))
        else:  # 'min' or default
            candidates = [base]
            for h in range(1, min(self._hop_decision_max, 2) + 1):
                if h in multihop and isinstance(multihop[h], (int, float)):
                    candidates.append(float(multihop[h]))
            val = min(candidates) if candidates else base
        # Flatten hop scores for logging
        hop_flat: Dict[int, float] = {}
        for h,v in (multihop.items() if isinstance(multihop, dict) else []):
            try:
                hop_flat[int(h)] = float(v)
            except Exception:
                continue
        return float(val), hop_flat
    
    def wire_edges(self, episodes: List[Episode], strategy: str = 'simple') -> None:
        """Wire episodes with specified strategy."""
        if strategy == 'simple':
            self._wire_simple(episodes)
        elif strategy == 'gedig':
            self._wire_with_gedig(episodes)
        elif strategy == 'gedig_nocopy':
            self._wire_with_gedig_nocopy(episodes)
        elif strategy == 'temporal':
            self._wire_temporal(episodes)
        elif strategy == 'spatial':
            self._wire_spatial(episodes)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _wire_with_gedig(self, episodes: List[Episode], threshold: float = -0.045) -> None:
        """標準のgeDIG配線（NoCopy版）"""
        self._wire_with_gedig_nocopy(episodes, threshold)
    
    def wire_with_gedig(self, episodes: List[Episode], threshold: float = -0.045) -> None:
        """外部から呼ばれるgeDIG配線メソッド（互換性）"""
        self._wire_with_gedig_nocopy(episodes, threshold)
    
    def _wire_temporal(self, episodes: List[Episode]) -> None:
        """時間的配線戦略"""
        sorted_episodes = sorted(episodes, key=lambda e: e.timestamp)
        for i in range(len(sorted_episodes)):
            current = sorted_episodes[i]
            for j in range(max(0, i-3), min(len(sorted_episodes), i+4)):
                if i != j:
                    other = sorted_episodes[j]
                    if abs(current.timestamp - other.timestamp) <= 3:
                        self.graph.add_edge(current.episode_id, other.episode_id)
                        self._log_edge_creation(current.episode_id, other.episode_id, 'temporal')
    
    def _wire_spatial(self, episodes: List[Episode]) -> None:
        """空間的配線戦略"""
        position_groups = {}
        for ep in episodes:
            if hasattr(ep, 'position'):
                if ep.position not in position_groups:
                    position_groups[ep.position] = []
                position_groups[ep.position].append(ep)
        
        for position, group in position_groups.items():
            for i in range(len(group)):
                for j in range(i+1, len(group)):
                    self.graph.add_edge(group[i].episode_id, group[j].episode_id)
                    self._log_edge_creation(group[i].episode_id, group[j].episode_id, 'spatial')
    
    def _wire_simple(self, episodes: List[Episode]) -> None:
        """Simple sequential wiring."""
        for i in range(1, len(episodes)):
            self.graph.add_edge(
                episodes[i].episode_id,
                episodes[i-1].episode_id
            )
            self._log_edge_creation(
                episodes[i].episode_id,
                episodes[i-1].episode_id,
                'simple'
            )
    
    def _log_edge_creation(self, source: int, target: int, strategy: str) -> None:
        """Log edge creation."""
        self.edge_creation_log.append({
            'source': source,
            'target': target,
            'strategy': strategy,
            'step': int(getattr(self, '_current_step', -1)) if isinstance(getattr(self, '_current_step', -1), (int,)) else None,
            'graph_size': self.graph.number_of_nodes(),
            'edge_count': self.graph.number_of_edges()
        })
        # Track trajectory-only edges for strict cycle detection
        try:
            if strategy and str(strategy).startswith('trajectory'):
                u, v = int(source), int(target)
                key = (u, v) if u <= v else (v, u)
                self._trajectory_edges.add(key)
        except Exception:
            pass

    def _compute_cycle_info_before_add(self, u: int, v: int) -> Optional[Dict[str, Any]]:
        """If a u-v path already exists, adding (u,v) closes a cycle.

        Returns a dict with:
          - node_path: list of node ids along existing path from u to v
          - delta_sp: len(path)-1 - 1  (shortest-path gain if closing with one edge)
          - position_path: list of positions along the path (if available)
        Otherwise returns None.
        """
        try:
            if not (self.graph.has_node(u) and self.graph.has_node(v)):
                return None
            # Optional: restrict cycle detection to trajectory-only subgraph (hybrid mode)
            hybrid = os.environ.get('MAZE_CYCLE_HYBRID', '1').strip() not in ("0","false","False","")
            if hybrid:
                # If frontier (unvisited obs) exists now, use trajectory-only; else use full graph
                use_traj = bool(getattr(self, '_has_unvisited_obs', False))
            else:
                use_traj = os.environ.get('MAZE_CYCLE_USE_TRAJECTORY_ONLY', '1').strip() not in ("0","false","False","")
            if use_traj and self._trajectory_edges:
                g_t = nx.Graph()
                g_t.add_nodes_from(self.graph.nodes(data=True))
                for a, b in self._trajectory_edges:
                    if g_t.has_node(a) and g_t.has_node(b):
                        g_t.add_edge(a, b)
                base_graph = g_t
            else:
                base_graph = self.graph
            # If no path exists yet, closing edge does not form a cycle
            path = nx.shortest_path(base_graph, source=u, target=v)
            if not path or len(path) < 2:
                return None
            # Current path length in edges
            sp_len = len(path) - 1
            delta_sp = max(0, sp_len - 1)
            # Map to positions if present
            pos_path: List[Any] = []
            for nid in path:
                try:
                    data = self.graph.nodes[nid]
                    pos = data.get('position') if isinstance(data, dict) else None
                    pos_path.append(tuple(pos) if pos is not None else None)
                except Exception:
                    pos_path.append(None)
            # Filter out degenerate cycles where all positions collapse to the same coordinate
            try:
                uniq = {p for p in pos_path if p is not None}
                # Require at least N unique positions to treat as a meaningful loop
                try:
                    _min_up = int(os.environ.get('MAZE_BT_SP_MIN_UNIQUE_POS', '3'))
                except Exception:
                    _min_up = 3
                if len(uniq) < max(2, _min_up):
                    return None
            except Exception:
                pass
            return {
                'node_path': path,
                'delta_sp': float(delta_sp),
                'position_path': pos_path,
            }
        except nx.NetworkXNoPath:
            return None
        except Exception:
            return None
    
    # 互換性メソッド
    def get_graph_snapshot(self) -> nx.Graph:
        return self.graph.copy()
    
    def save_snapshot(self) -> None:
        self.graph_history.append(self.get_graph_snapshot())
    
    def get_connected_episodes(self, episode_id: int) -> List[int]:
        if episode_id not in self.graph:
            return []
        return list(self.graph.neighbors(episode_id))
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph) if self.graph.number_of_nodes() > 0 else 0,
            'num_components': nx.number_connected_components(self.graph),
            'is_connected': nx.is_connected(self.graph) if self.graph.number_of_nodes() > 0 else False
        }
        
        if stats['is_connected'] and self.graph.number_of_nodes() > 0:
            stats['diameter'] = nx.diameter(self.graph)
            stats['radius'] = nx.radius(self.graph)
            stats['average_shortest_path'] = nx.average_shortest_path_length(self.graph)
        
        if self.graph.number_of_nodes() > 0:
            degrees = [d for n, d in self.graph.degree()]
            stats['average_degree'] = sum(degrees) / len(degrees)
            stats['max_degree'] = max(degrees)
            stats['min_degree'] = min(degrees)
        
        return stats
