"""Maze navigation / graph exploration orchestrator (clean version with backtracking & reverse trace)."""

from __future__ import annotations

from typing import Tuple, Optional, Dict, List, Any, Union
from enum import Enum
import os, sys, time, heapq
import numpy as np
import networkx as nx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.episode_manager import EpisodeManager  # type: ignore
from core.graph_manager import GraphManager      # type: ignore
from core.vector_processor import VectorProcessor  # type: ignore
from core.gedig_evaluator import GeDIGEvaluator  # type: ignore
try:
    from indexes.vector_index import VectorIndex, InMemoryIndex  # type: ignore
except Exception:  # 軽量フォールバック (古いブランチ互換)
    VectorIndex = Any  # type: ignore
    InMemoryIndex = None  # type: ignore
from navigation.decision_engine import DecisionEngine  # type: ignore
from navigation.branch_detector import BranchDetector as BacktrackDetector  # type: ignore
try:
    from indexes.datastore_index import DataStoreIndex  # type: ignore
except Exception:
    DataStoreIndex = None  # type: ignore


class MazeNavigator:
    class EventType(str, Enum):
        BRANCH_ENTRY = 'branch_entry'
        BRANCH_COMPLETION = 'branch_completion'
        SHORTCUT_CANDIDATE = 'shortcut_candidate'
        BACKTRACK_TRIGGER = 'backtrack_trigger'
        BACKTRACK_STEP = 'backtrack_step'
        DEAD_END = 'dead_end_detected'
        ANALYSIS = 'branch_analysis'
        SNAPSHOT = 'snapshot_saved'
        EDGE_WIRE = 'edge_wire'
        FLUSH_SCORE = 'flush_score_probe'
        FLUSH_EVICT = 'flush_eviction'
        CATALOG_COMPACT = 'catalog_compact'
        START = 'start'
        GOAL = 'goal'
        TIMEOUT = 'timeout'
        WALL_SELECTED = 'wall_selected'
        BACKTRACK_PLAN = 'backtrack_plan'
        # Extended / advanced events
        REVERSE_TRACE = 'reverse_trace'
        REVERSE_TRACE_ERROR = 'reverse_trace_error'
        BRANCH_REMINDER = 'branch_reminder'
        FLUSH_ERROR = 'flush_error'
        REHYDRATION = 'rehydration'
        FALLBACK_MOVE = 'fallback_move'
        FALLBACK_FAILED = 'fallback_failed'
        CATALOG_COMPACT_FAILED = 'catalog_compact_failed'
        STUCK = 'stuck'
        ANN_INIT = 'ann_init'
        ANN_INIT_FAILED = 'ann_init_failed'
        ANN_UPGRADE = 'ann_upgrade'
        ANN_UPGRADE_FAILED = 'ann_upgrade_failed'
        # Backwards compatibility aliases removed (access via EventType only)
    def __init__(self, maze: np.ndarray, start_pos: Tuple[int, int], goal_pos: Tuple[int, int],
                 weights: Optional[np.ndarray] = None, temperature: float = 0.1,
                 gedig_threshold: float = -0.15, backtrack_threshold: float = -0.2,
                 wiring_strategy: str = 'simple', simple_mode: bool = False,
                 backtrack_debounce: bool = True, use_escalation: bool = True,
                 escalation_threshold: float | None | str = None,
                 dynamic_escalation: bool = False, dynamic_offset: float = 0.06,
                 dynamic_warmup: int = 10, dynamic_window: int = 25,
                 global_recall_enabled: bool = False, recall_score_threshold: float = 0.01,
                 wiring_top_k: int = 4, max_graph_snapshots: Optional[int] = None,
                 dense_metric_interval: int = 1, snapshot_skip_idle: bool = False,
                 verbosity: int = 0, progress_interval: int = 100,
                 enable_diameter_metrics: bool = True, diameter_node_cap: int = 900,
                 diameter_time_budget_ms: float = 40.0,
                 vector_index: Any | None = None, index_auto_add: bool = True,
                 enable_flush: bool = False, flush_interval: int = 200,
                 max_in_memory: int = 10000, max_in_memory_positions: int | None = None,
                 persistence_dir: str | None = None, evicted_catalog_max: int | None = 5000,
                 ann_backend: str | None = None, ann_m: int = 16,
                 ann_ef_construction: int = 100, ann_ef_search: int = 100,
                 ann_upgrade_threshold: int = 600, catalog_compaction_on_close: bool = False,
                 force_multihop: bool = False, gedig_mode: str = 'core_raw', gedig_scale: float = 25.0,
                 gedig_ig_weight: float = 0.1, gedig_allow_hop1_approx: bool | None = None,
                 eviction_policy: str | None = None,
                 passive_heuristics: bool = False,
                 backtrack_target_strategy: str = 'heuristic',
                 macro_target_analysis: bool = True) -> None:
        # Core config
        self.maze = maze; self.h, self.w = maze.shape
        self.start_pos = start_pos; self.goal_pos = goal_pos; self.current_pos = start_pos
        self.wiring_strategy = wiring_strategy; self.simple_mode = simple_mode
        # Passive heuristics mode: detectors log, but no backtrack plans / fallback moves are executed
        self.passive_heuristics = passive_heuristics
        # Backtrack target selection strategy: 'heuristic' (BranchDetector) or 'semantic'
        # Note: semantic is deprecated for experiment fairness; coerce to 'heuristic'
        _bt_strategy = (backtrack_target_strategy or 'heuristic').lower()
        if _bt_strategy == 'semantic':
            _bt_strategy = 'heuristic'
        self.backtrack_target_strategy = _bt_strategy
        self.backtrack_debounce = backtrack_debounce; self.use_escalation = use_escalation
        self.escalation_threshold = escalation_threshold; self.dynamic_escalation = dynamic_escalation
        self.dynamic_offset = dynamic_offset; self._dynamic_warmup = dynamic_warmup; self._dynamic_window = dynamic_window
        self.global_recall_enabled = global_recall_enabled; self.recall_score_threshold = recall_score_threshold
        # Engines
        self.episode_manager = EpisodeManager(self.w, self.h)
        self.vector_processor = VectorProcessor(self.w, self.h)
        self.gedig_mode = gedig_mode
        self.gedig_evaluator = GeDIGEvaluator(always_multihop=force_multihop, mode=gedig_mode, scale=gedig_scale, ig_weight=gedig_ig_weight)
        
        # Use optimized graph manager for gedig strategy
        if wiring_strategy == 'gedig_optimized':
            from core.graph_manager_optimized import OptimizedGraphManager
            self.graph_manager = OptimizedGraphManager(self.gedig_evaluator)
        else:
            self.graph_manager = GraphManager(self.gedig_evaluator, max_graph_snapshots)
        self.decision_engine = DecisionEngine(self.vector_processor, weights, temperature, include_walls=True)
        self.backtrack_detector = BacktrackDetector(backtrack_threshold)
        # Backtrack target policy (selection scoring): 'gedig' | 'heuristic'
        try:
            _pol = os.environ.get('MAZE_BT_TARGET_POLICY', 'gedig').strip().lower()
        except Exception:
            _pol = 'gedig'
        # Allow extended policies
        if _pol not in ('gedig', 'heuristic', 'gedig_l1', 'gedig_branches', 'gedig_legacy'):
            _pol = 'gedig'
        self._bt_target_policy = _pol
        # Weights for geDIG-like target scoring: F = w1 * travel_cost - kT * ig_gain
        try:
            self._bt_w1 = float(os.environ.get('MAZE_BT_W1', '1.0'))
        except Exception:
            self._bt_w1 = 1.0
        try:
            self._bt_kT = float(os.environ.get('MAZE_BT_KT', '1.0'))
        except Exception:
            self._bt_kT = 1.0
        try:
            self._bt_cand_max = int(os.environ.get('MAZE_BT_CAND_MAX', '32'))
        except Exception:
            self._bt_cand_max = 32
        # Keep last ranking for diagnostics/export
        self._last_bt_ranking = None
        try:
            from core.eviction_policy import get_policy  # type: ignore
            self._eviction_policy = get_policy(eviction_policy)
        except Exception:
            self._eviction_policy = None
        # State / history
        self.gedig_threshold = gedig_threshold; self.backtrack_threshold = backtrack_threshold
        self.path = [start_pos]; self.step_count = 0; self.is_goal_reached = False
        self.gedig_history = []
        self.gedig_structural = []
        self.decision_history = []
        self.event_log = []
        self._last_g0: Optional[float] = None
        self._last_gmin: Optional[float] = None
        self._in_run_loop = False; self._escalated_no_growth_streak = 0; self._stagnation_window = 25
        self._recent_positions = []; self._stagnation_resets = 0
        self.branch_completion_windows = []; self._pending_completion_windows = []
        self._pending_backtrack_plan = []; self._last_backtrack_step = None; self._bt_target_current = None
        # Plan-freeze (keep executing BT plan until arrival; avoid local fallback)
        try:
            self._bt_freeze_enabled = os.environ.get('MAZE_BT_PLAN_FREEZE', '1').strip() not in ("0","false","False","")
        except Exception:
            self._bt_freeze_enabled = True
        try:
            self._bt_replan_stuck_n = max(1, int(os.environ.get('MAZE_BT_REPLAN_STUCK_N', '3')))
        except Exception:
            self._bt_replan_stuck_n = 3
        self._bt_freeze_prev_remaining = None
        self._bt_freeze_stuck_count = 0
        self.backtrack_cooldown = 60; self.dynamic_backtrack_enabled = True
        # Optional: override backtrack cooldown via environment
        try:
            _cd = os.environ.get('MAZE_BACKTRACK_COOLDOWN', '')
            if _cd:
                self.backtrack_cooldown = max(0, int(_cd))
        except Exception:
            pass
        # Optional: disable dynamic backtrack (stagnation/dynamic threshold) via env
        try:
            _dyn = os.environ.get('MAZE_BT_DYNAMIC', '')
            if _dyn:
                self.dynamic_backtrack_enabled = (_dyn.strip() not in ("0","false","False","off"))
        except Exception:
            pass
        self._backtrack_window = 40; self._backtrack_min_samples = 12; self._dynamic_bt_margin = 0.002
        # Query wiring / NN
        self.query_wiring_k = wiring_top_k; self.query_wiring_max_dist = 6.0
        self.query_wiring_include_walls = False; self.query_wiring_force_previous = True
        self._current_query = None; self._query_generated_count = 0
        # Decision / semantic logging context
        self._last_decision_context = None
        self._last_semantic_candidates = None
        # Backtrack instrumentation
        self._static_last_backtrack_step = None; self._backtrack_triggers = 0
        self._dead_end_positions = set()
        # NN degeneracy detection parameters
        self.nn_degeneracy_enabled = False
        self.nn_deg_var_thresh = 1e-4
        self.nn_deg_range_thresh = 5e-4
        self.nn_deg_min_unvisited_ratio = 0.2
        self.nn_deg_min_window_no_growth = 5
        # Simple degeneracy mode flag
        self.nn_degeneracy_simple_mode = False
        self._nn_deg_last_trigger_step = None
        self._nn_degeneracy_triggers = 0
        self._nn_last_ranked_snapshot = None
        # Adaptive exploration tuning
        self._auto_visit_scale = False; self._auto_visit_scale_interval = 200; self._auto_visit_scale_max = 6.0
        # Graph growth logging
        self._log_graph_growth = False; self._graph_growth_interval = 100
        self._last_growth_nodes = 0; self._last_growth_edges = 0
        # Exploration instrumentation
        self._new_cells = 1; self._revisit_moves = 0; self._wall_selections = 0; self._dead_end_events = 0
        try:
            self._open_cells = int((self.maze == 0).sum())
        except Exception:
            self._open_cells = None
        # Frontier jump
        self._frontier_jump_window = 120; self._frontier_novelty_threshold = 0.16; self._frontier_cooldown = 160
        self._last_frontier_jump_step = None
        # Frontier / exploration bias instrumentation
        self._frontier_bias_records = []  # list of tuples: (step, prob_new_sum, has_new, chosen_is_new)
        # Timing / growth tracking
        self._snapshot_skip_idle = snapshot_skip_idle
        self._last_snapshot_nodes = None; self._last_snapshot_edges = None; self._idle_growth_streak = 0
        self._dense_metric_interval = max(1, dense_metric_interval); self._weights_version = 0
        self._timing = {k: [] for k in ['wiring_ms','gedig_ms','snapshot_ms','recall_ms','flush_ms','rehydration_ms']}
        self._no_growth_streak = 0; self._broken_pipe = False
        # Flush / persistence
        self._vector_index = vector_index; self._enable_flush = enable_flush
        self._flush_interval = max(1, flush_interval)
        self._max_in_memory = max(1, max_in_memory)
        self._max_in_memory_positions = (max_in_memory_positions if max_in_memory_positions is None else max(1, max_in_memory_positions))
        self._flush_events = 0; self._episodes_evicted_total = 0; self._episodes_rehydrated_total = 0
        self._episode_eviction_events = 0; self._position_eviction_events = 0; self._rehydration_events = 0
        self._rehydrated_positions = set()
        from collections import OrderedDict
        self._persistence_dir = persistence_dir
        self._evicted_catalog_max = (None if evicted_catalog_max is None else max(1, evicted_catalog_max))
        self._evicted_catalog = OrderedDict()
        if self._persistence_dir:
            try:
                os.makedirs(self._persistence_dir, exist_ok=True)
                self._evicted_catalog_path = os.path.join(self._persistence_dir,'evicted_catalog.jsonl')
            except Exception:
                self._persistence_dir = None
        if self._vector_index is None and self._enable_flush and DataStoreIndex:
            try:
                self._vector_index = DataStoreIndex()  # type: ignore
            except Exception:
                pass
        self._index_auto_add = index_auto_add
        # ANN backend
        self._ann_backend = ann_backend; self._ann_init_error = None; self._ann_upgrade_threshold = max(1, ann_upgrade_threshold)
        self._catalog_compaction_on_close = catalog_compaction_on_close; self._catalog_last_bytes = None
        if self._vector_index is None and ann_backend and ann_backend.lower()=='hnsw':
            try:
                from indexes.hnsw_index import HNSWLibIndex  # type: ignore
                self._vector_index = HNSWLibIndex(dim=8, M=ann_m, ef_construction=ann_ef_construction, ef_search=ann_ef_search)
                self.emit_event(self.EventType.ANN_INIT, {'backend':'hnsw','dim':8,'M':ann_m,'ef_construction':ann_ef_construction,'ef_search':ann_ef_search})
            except Exception as e:
                self._ann_init_error = str(e); self.emit_event(self.EventType.ANN_INIT_FAILED, str(e)); self._ann_backend = None
        # Verbosity / performance gates
        self.verbosity = verbosity; self.progress_interval = max(1, progress_interval)
        env_override = bool(os.environ.get('NAV_DISABLE_DIAMETER', ''))
        self.enable_diameter_metrics = (enable_diameter_metrics and not env_override)
        self._diameter_node_cap = max(10, diameter_node_cap); self._diameter_time_budget_ms = diameter_time_budget_ms
        # Index weighting mode (True=store weighted vectors, False=store raw vectors)
        self._index_use_weighted = True
        # Transition decomposition (Stage0 observer – optional lazy import)
        self._transition_observer = None
        self._transition_observer_enabled = False
        self._transition_move_hooks = 0
        self._transition_on_step_calls = 0
        try:
            from experiments.maze_transition_decomposition.observer import TransitionObserver  # type: ignore
            self._transition_observer = TransitionObserver(self)
            self._transition_observer_enabled = True
        except Exception:
            self._transition_observer = None
            self._transition_observer_enabled = False
        # Macro target planner (observation phase)
        try:
            from experiments.maze_transition_decomposition.macro_target_planner import MacroTargetPlanner  # type: ignore
            self._macro_target_planner = MacroTargetPlanner(self)
            self._macro_target_enabled = True
        except Exception:
            self._macro_target_planner = None
            self._macro_target_enabled = False
        # Macro target analyzer enable flag & timing
        self._macro_target_analysis_enabled = bool(macro_target_analysis)
        self._macro_target_analysis_times = []  # list[float]
        # Global SP last average (for delta using single compute per step)
        self._sp_last_avg = None
        # Backtrack path planning mode: 'memory_graph' | 'visited' | 'maze'
        try:
            self._bt_path_mode = os.environ.get('MAZE_BT_PATH_MODE', 'memory_graph').strip().lower()
        except Exception:
            self._bt_path_mode = 'memory_graph'
        try:
            # Default ON: prefer nearest-L2 target selection over composite scoring
            self._bt_sem_force_nearest_l2 = os.environ.get('MAZE_BT_SEM_FORCE_NEAREST_L2', '1').strip() not in ("0","false","False","")
        except Exception:
            self._bt_sem_force_nearest_l2 = True

        # Memory-triggered backtrack (pure memory signals; avoid maze-based dead-end triggers)
        try:
            self._bt_use_memory_trigger = os.environ.get('MAZE_BT_USE_MEMORY_TRIGGER', '0').strip() not in ("0","false","False","")
        except Exception:
            self._bt_use_memory_trigger = False
        try:
            self._bt_mem_require_na = os.environ.get('MAZE_BT_MEM_REQUIRE_NA', '1').strip() not in ("0","false","False","")
        except Exception:
            self._bt_mem_require_na = True
        try:
            self._bt_mem_streak = int(os.environ.get('MAZE_BT_MEM_STREAK', '3'))
        except Exception:
            self._bt_mem_streak = 3
        try:
            self._bt_mem_cand_max = int(os.environ.get('MAZE_BT_MEM_CAND_MAX', '2'))
        except Exception:
            self._bt_mem_cand_max = 2
        try:
            self._bt_mem_min_votes = int(os.environ.get('MAZE_BT_MEM_MIN_VOTES', '2'))
        except Exception:
            self._bt_mem_min_votes = 2
        try:
            self._bt_use_deadend_trigger = os.environ.get('MAZE_BT_USE_DEADEND', '0').strip() not in ("0","false","False","")
        except Exception:
            self._bt_use_deadend_trigger = False
        # Combined-ranking-driven BT (obs vs mem dv) knobs
        try:
            self._bt_from_combined = os.environ.get('MAZE_BT_FROM_COMBINED', '0').strip() not in ("0","false","False","")
        except Exception:
            self._bt_from_combined = False
        try:
            self._bt_require_na_general = os.environ.get('MAZE_BT_REQUIRE_NA', '1').strip() not in ("0","false","False","")
        except Exception:
            self._bt_require_na_general = True
        try:
            self._bt_dv_margin = float(os.environ.get('MAZE_BT_DV_MARGIN', '0.02'))
        except Exception:
            self._bt_dv_margin = 0.02

        # Provide L1 candidate search via vector index to GraphManager when enabled
        try:
            use_index_cands = os.environ.get('MAZE_L1_INDEX_SEARCH', '0').strip() not in ("0","false","False","")
        except Exception:
            use_index_cands = False
        if use_index_cands and hasattr(self.graph_manager, 'set_candidate_provider'):
            def _cand_provider(current_ep, topk: int):  # type: ignore[no-redef]
                try:
                    if self._vector_index is None or topk <= 0:
                        return []
                    # Build query vector near current position; prefer unexplored
                    q = self.decision_engine.create_query(current_ep.position, prefer_unexplored=True)
                    if self._index_use_weighted:
                        q_search = self.vector_processor.apply_weights(q, self.decision_engine.weights)
                    else:
                        q_search = q
                    # slight oversample; GraphManager will filter/time-order
                    K = max(1, int(topk))
                    search_res = self._vector_index.search(q_search, K)
                    out: list[int] = []
                    # Filter out self and future episodes by timestamp
                    for eid, _dist in search_res:
                        ep = self.episode_manager.episodes_by_id.get(int(eid))
                        if not ep:
                            continue
                        if ep.episode_id == current_ep.episode_id:
                            continue
                        if ep.timestamp >= current_ep.timestamp:
                            continue
                        out.append(int(eid))
                    return out
                except Exception:
                    return []
            try:
                self.graph_manager.set_candidate_provider(_cand_provider)  # type: ignore[attr-defined]
                self.emit_event(self.EventType.ANALYSIS, {'kind':'l1_index_provider','enabled': True})
            except Exception:
                pass

        # Global SP (average shortest path) metrics toggles (optional)
        # Env knobs:
        #  - MAZE_SP_GLOBAL_ENABLE=1 to enable
        #  - MAZE_SP_GLOBAL_FULL=1 for full average (connected component)
        #  - MAZE_SP_GLOBAL_SAMPLES=200 for sampled pairs
        #  - MAZE_SP_GLOBAL_BUDGET_MS=40 time budget for sampling
        try:
            def _env_bool(k: str, default: bool = False) -> bool:
                v = os.environ.get(k)
                if v is None:
                    return default
                return v.strip() not in ("0","false","False","")
            def _env_int(k: str, default: int) -> int:
                try:
                    return int(os.environ.get(k, str(default)))
                except Exception:
                    return default
            def _env_float(k: str, default: float) -> float:
                try:
                    return float(os.environ.get(k, str(default)))
                except Exception:
                    return default
            self._sp_global_enabled = _env_bool('MAZE_SP_GLOBAL_ENABLE', False)
            self._sp_global_full = _env_bool('MAZE_SP_GLOBAL_FULL', False)
            self._sp_global_poslevel = _env_bool('MAZE_SP_POSLEVEL', False)
            self._sp_global_samples = max(0, _env_int('MAZE_SP_GLOBAL_SAMPLES', 200))
            self._sp_global_budget_ms = max(1.0, _env_float('MAZE_SP_GLOBAL_BUDGET_MS', 40.0))
            # Force SP on NA steps (optional)
            self._sp_force_on_na = _env_bool('MAZE_SP_FORCE_ON_NA', False)
            self._sp_force_samples = max(1, _env_int('MAZE_SP_FORCE_SAMPLES', max(200, self._sp_global_samples)))
            self._sp_force_budget_ms = max(self._sp_global_budget_ms, _env_float('MAZE_SP_FORCE_BUDGET_MS', self._sp_global_budget_ms * 2.0))
        except Exception:
            self._sp_global_enabled = False
            self._sp_global_full = False
            self._sp_global_poslevel = False
            self._sp_global_samples = 0
            self._sp_global_budget_ms = 40.0
            self._sp_force_on_na = False
            self._sp_force_samples = 200
            self._sp_force_budget_ms = 80.0

        # Insight threshold (theta_detect) dynamic control via quantile (env knobs)
        try:
            self._theta_dynamic_enabled = os.environ.get('MAZE_THETA_DYNAMIC', '0').strip() not in ("0","false","False","")
        except Exception:
            self._theta_dynamic_enabled = False
        try:
            self._theta_target_rate = float(os.environ.get('MAZE_THETA_TARGET_RATE', '0.08'))  # 8% default
        except Exception:
            self._theta_target_rate = 0.08
        try:
            self._theta_window = int(os.environ.get('MAZE_THETA_WINDOW', '100'))
        except Exception:
            self._theta_window = 100
        # NA (0-hop) gating: detect "moyamoya" and escalate to multi-hop when high
        try:
            self._na_enable = os.environ.get('MAZE_NA_ENABLE', '0').strip() not in ("0","false","False","")
        except Exception:
            self._na_enable = False
        try:
            self._na_window = int(os.environ.get('MAZE_NA_WINDOW', '100'))
        except Exception:
            self._na_window = 100
        try:
            self._na_target_rate = float(os.environ.get('MAZE_NA_TARGET_RATE', '0.08'))
        except Exception:
            self._na_target_rate = 0.08
        try:
            self._na_use_struct = os.environ.get('MAZE_NA_USE_STRUCT', '1').strip() not in ("0","false","False","")
        except Exception:
            self._na_use_struct = True
        try:
            self._na_ge0 = os.environ.get('MAZE_NA_GE0', '0').strip() not in ("0","false","False","")
        except Exception:
            self._na_ge0 = False
        try:
            _t = os.environ.get('MAZE_NA_GE_THRESH')
            self._na_ge_thresh = (float(_t) if (_t is not None and _t.strip() != '') else None)
        except Exception:
            self._na_ge_thresh = None
        self._na_scores: list[float] = []
        # Dead-end instant drop tuning (env)
        try:
            self._dead_end_penalty = float(os.environ.get('MAZE_GEDIG_DE_PENALTY', '0.0'))
        except Exception:
            self._dead_end_penalty = 0.0
        try:
            self._nogrowth_penalty = float(os.environ.get('MAZE_GEDIG_NOGROWTH_PENALTY', '0.0'))
        except Exception:
            self._nogrowth_penalty = 0.0

    def set_index_weight_mode(self, use_weighted: bool) -> None:
        """Switch between storing weighted vs raw vectors in the vector index.

        If changing mode at runtime, the existing index (if small/in-memory) is rebuilt
        best-effort. ANN backends are skipped (user can trigger manual rebuild if needed).
        """
        if use_weighted == self._index_use_weighted:
            return
        self._index_use_weighted = use_weighted
        if self._vector_index is None:
            return
        # Only attempt lightweight rebuild for simple in-memory index types
        name = type(self._vector_index).__name__.lower()
        if 'memory' not in name:  # skip HNSW or datastore for now
            return
        try:
            # Clear existing index (assumes simple API: clear())
            if hasattr(self._vector_index, 'clear'):
                self._vector_index.clear()
            # Reinsert all passable episodes
            import numpy as _np
            for ep in self.episode_manager.episodes.values():
                if getattr(ep, 'is_wall', False):
                    continue
                if use_weighted:
                    vec = ep.get_weighted_vector(self.decision_engine.weights, self._weights_version, self.vector_processor.apply_weights)
                else:
                    vec = ep.vector
                self._vector_index.add([ep.episode_id], _np.asarray(vec, dtype=float).reshape(1, -1))
            self.emit_event(self.EventType.ANALYSIS, {'kind':'index_rebuild','mode':'weighted' if use_weighted else 'raw','size': len(self._vector_index)})
        except Exception:
            pass

    def _latest_g0(self) -> Optional[float]:
        if self._last_g0 is not None:
            return self._last_g0
        if self.gedig_history:
            try:
                return float(self.gedig_history[-1])
            except Exception:
                return None
        return None

    def _latest_gmin(self) -> Optional[float]:
        if self._last_gmin is not None:
            return self._last_gmin
        if self.gedig_structural:
            try:
                payload = self.gedig_structural[-1]
                if isinstance(payload, dict):
                    gmin = payload.get('gmin')
                    return float(gmin) if gmin is not None else None
            except Exception:
                return None
        return None

    def run(self, max_steps: int = 1000) -> bool:
        self.emit_event(self.EventType.START, f'Start @ {self.start_pos}')
        for step in range(max_steps):
            self._in_run_loop = True
            self.step_count = step
            if self.step():
                self.emit_event(self.EventType.GOAL, f'Goal reached in {step} steps')
                self.is_goal_reached = True
                if self._enable_flush:
                    self._memory_guard_pass()
                if self._catalog_compaction_on_close:
                    try:
                        self.compact_eviction_catalog()
                    except Exception:
                        pass
                return True
            if self.verbosity >= 1 and step and step % self.progress_interval == 0:
                self._print_progress()
        self.emit_event(self.EventType.TIMEOUT, f'max_steps={max_steps}')
        if self._enable_flush:
            self._memory_guard_pass()
        if self._catalog_compaction_on_close:
            try:
                self.compact_eviction_catalog()
            except Exception:
                pass
        return False

    def step(self) -> bool:
        """Execute one navigation step. Returns True if goal reached."""
        # If invoked outside run() (tests), advance step counter
        if not getattr(self, '_in_run_loop', False):
            self.step_count += 1
        # Expose current step to GraphManager for logging (e.g., cycle events on edge add)
        try:
            if hasattr(self, 'graph_manager'):
                setattr(self.graph_manager, '_current_step', int(self.step_count))
        except Exception:
            pass
        # Capture a "before" snapshot for within-step delta evaluation (prev_graph)
        try:
            self._step_before_graph = self.graph_manager.get_graph_snapshot()
        except Exception:
            self._step_before_graph = None
        self.episode_manager.increment_step()
        # Lazy rehydrate for current position if any evicted metadata exists
        if self._enable_flush:
            try:
                self._maybe_rehydrate_position(self.current_pos)
            except Exception:
                pass
        episodes = self.episode_manager.observe(self.current_pos, self.maze)
        # Register new episodes & optionally index (skip walls)
        for ep in episodes.values():
            if ep.episode_id not in self.graph_manager.graph:
                self.graph_manager.add_episode_node(ep)
            # Optionally include walls into the index (for L1 candidate recall parity)
            include_walls = os.environ.get('MAZE_INDEX_INCLUDE_WALLS', '0').strip() not in ("0","false","False","")
            if (self._vector_index is not None) and self._index_auto_add and (include_walls or not getattr(ep, 'is_wall', False)):
                try:
                    import numpy as _np
                    if self._index_use_weighted:
                        vec_to_add = ep.get_weighted_vector(self.decision_engine.weights, self._weights_version, self.vector_processor.apply_weights)
                    else:
                        vec_to_add = ep.vector
                    self._vector_index.add([ep.episode_id], _np.asarray(vec_to_add, dtype=float).reshape(1, -1))
                    if self.step_count < 5:
                        try:
                            self.emit_event(self.EventType.ANALYSIS, {'dbg':'vector_index_add','ep':ep.episode_id,'size':len(self._vector_index),'mode':'weighted' if self._index_use_weighted else 'raw'})
                        except Exception:
                            pass
                except Exception:
                    pass
        # Local frontier exhaustion probe (instrumentation only)
        try:
            open_new = [1 for e in episodes.values() if not getattr(e,'is_wall',False) and getattr(e,'visit_count',0)==0]
            # Expose flag for gating loop-driven BT (skip when local unvisited exists)
            try:
                self._has_unvisited_obs = bool(open_new)
            except Exception:
                self._has_unvisited_obs = False
            # Propagate frontier flag to GraphManager for hybrid cycle detection
            try:
                if hasattr(self, 'graph_manager'):
                    setattr(self.graph_manager, '_has_unvisited_obs', bool(self._has_unvisited_obs))
            except Exception:
                pass
            if not open_new:
                # current cell has no unvisited immediate neighbor; check if any global frontier exists elsewhere
                visited_set = set(self.path)
                dirs = [(-1,0),(1,0),(0,-1),(0,1)]
                frontier_exists = False
                for cell in visited_set:
                    x,y = cell
                    for dx,dy in dirs:
                        nx_, ny_ = x+dx, y+dy
                        if 0 <= nx_ < self.w and 0 <= ny_ < self.h and self.maze[ny_, nx_] == 0 and (nx_, ny_) not in visited_set:
                            frontier_exists = True; break
                    if frontier_exists:
                        break
                if frontier_exists:
                    self.emit_event(self.EventType.ANALYSIS, {'kind':'local_frontier_exhausted','pending_plan': bool(self._pending_backtrack_plan),'simple_mode': self.simple_mode,'backtrack_strategy': self.backtrack_target_strategy})
        except Exception:
            pass
        # Query (一元化): 毎ステップ生成し reuse。simple_mode では統計カウント。
        self._current_query = self.decision_engine.create_query(self.current_pos, prefer_unexplored=True)
        self._query_generated_count += 1

        # --- Combined ranking (obs + memory) for analysis/verification ---
        # 目的: 現在の観測4方向と過去メモリエピソードを同一ノルム(dv)で結合し、単一順位表を作成
        # 実行コストを抑えるため、メモリ側は TopK のみを保持
        try:
            def _env_bool(k: str, default: bool = False) -> bool:
                v = os.environ.get(k)
                if v is None: return default
                return v.strip() not in ("0","false","False","")
            def _env_int(k: str, default: int) -> int:
                try:
                    return int(os.environ.get(k, str(default)))
                except Exception:
                    return default
            # 軽量化: デフォルトは無効。必要時のみ有効化
            if _env_bool('MAZE_COMBINED_RANK_ENABLE', False):
                topk_mem = _env_int('MAZE_COMBINED_TOPK', 32)
                allow_walls = _env_bool('MAZE_COMBINED_ALLOW_WALLS', False)
                visited_only = _env_bool('MAZE_COMBINED_VISITED_ONLY', False)
                # NOTE: frontier 要求は解析用途のため簡略化 (visit_count==0 を frontier相当として扱う)
                require_frontier = _env_bool('MAZE_COMBINED_REQUIRE_FRONTIER', False)
                spatial_gate = _env_int('MAZE_COMBINED_SPATIAL_GATE', 0)
                unit_norm = _env_bool('MAZE_COMBINED_UNIT_NORM', False)
                include_obs = _env_bool('MAZE_COMBINED_INCLUDE_OBS', True)
                include_mem = _env_bool('MAZE_COMBINED_INCLUDE_MEM', True)
                # 重み（未指定なら DecisionEngine の既定）
                w_str = os.environ.get('MAZE_COMBINED_WEIGHTS', '')
                if w_str:
                    try:
                        parts = [float(x) for x in w_str.split(',')]
                        weights = np.array(parts, dtype=float) if len(parts) == 8 else self.decision_engine.weights
                    except Exception:
                        weights = self.decision_engine.weights
                else:
                    weights = self.decision_engine.weights
                # 準備: クエリの重み付け
                q = self._current_query
                qw = self.vector_processor.apply_weights(q, weights)
                if unit_norm:
                    n = float(np.linalg.norm(qw)) or 1.0
                    qw = qw / n

                combined: list[dict] = []
                # 観測4方向
                if include_obs and episodes:
                    for d, ep in episodes.items():
                        try:
                            if (not allow_walls) and getattr(ep, 'is_wall', False):
                                continue  # 解析では壁を除外
                            ev = ep.get_weighted_vector(weights, getattr(self, '_weights_version', 0), self.vector_processor.apply_weights)
                            if unit_norm:
                                nn = float(np.linalg.norm(ev)) or 1.0
                                ev = ev / nn
                            dv = float(np.linalg.norm(qw - ev))
                            combined.append({
                                'dv': float(dv),
                                'id': int(ep.episode_id),
                                'pos': [int(ep.position[0]), int(ep.position[1])],
                                'dir': str(getattr(ep, 'direction', None)),
                                'ts': int(getattr(ep, 'timestamp', 0)),
                                'visit': int(getattr(ep, 'visit_count', 0)),
                                'is_wall': bool(getattr(ep, 'is_wall', False)),
                                'source': 'obs'
                            })
                        except Exception:
                            continue
                # メモリエピソード（過去のみ）
                import heapq as _hq
                if include_mem and topk_mem > 0:
                    mem_heap: list[tuple[float, dict]] = []  # store as (-dv, rec) to cap by TopK
                    cx, cy = self.current_pos
                    for ep in self.episode_manager.episodes_by_id.values():
                        try:
                            if int(getattr(ep, 'timestamp', 0)) >= int(self.step_count):
                                continue  # 現在ステップ生成分は除外
                            if (not allow_walls) and getattr(ep, 'is_wall', False):
                                continue
                            if visited_only and int(getattr(ep, 'visit_count', 0)) <= 0:
                                continue
                            if require_frontier and int(getattr(ep, 'visit_count', 0)) != 0:
                                continue
                            if spatial_gate > 0:
                                px, py = ep.position
                                if abs(px - cx) + abs(py - cy) > spatial_gate:
                                    continue
                            ev = ep.get_weighted_vector(weights, getattr(self, '_weights_version', 0), self.vector_processor.apply_weights)
                            if unit_norm:
                                nn = float(np.linalg.norm(ev)) or 1.0
                                ev = ev / nn
                            dv = float(np.linalg.norm(qw - ev))
                            rec = {
                                'dv': float(dv),
                                'id': int(ep.episode_id),
                                'pos': [int(ep.position[0]), int(ep.position[1])],
                                'dir': str(getattr(ep, 'direction', None)),
                                'ts': int(getattr(ep, 'timestamp', 0)),
                                'visit': int(getattr(ep, 'visit_count', 0)),
                                'is_wall': bool(getattr(ep, 'is_wall', False)),
                                'source': 'mem'
                            }
                            if len(mem_heap) < topk_mem:
                                _hq.heappush(mem_heap, (-dv, rec))
                            else:
                                if mem_heap and (-mem_heap[0][0]) > dv:
                                    _hq.heapreplace(mem_heap, (-dv, rec))
                        except Exception:
                            continue
                    mem_top = [r for (_neg, r) in mem_heap]
                    mem_top.sort(key=lambda r: r.get('dv', 1e9))
                    combined.extend(mem_top)
                # 結合順位表（dv昇順）
                combined.sort(key=lambda r: r.get('dv', 1e9))
                # 代表値（obs vs memの最良dv）
                best_obs = min((r['dv'] for r in combined if r.get('source') == 'obs'), default=None)
                best_mem = min((r['dv'] for r in combined if r.get('source') == 'mem'), default=None)
                # ログ出力はTopMのみに制限
                log_topm = _env_int('MAZE_COMBINED_LOG_TOPK', 16)
                to_log = combined[:log_topm] if log_topm > 0 else combined
                try:
                    self.emit_event(self.EventType.ANALYSIS, {
                        'kind': 'combined_ranking',
                        'best_obs_dv': (float(best_obs) if best_obs is not None else None),
                        'best_mem_dv': (float(best_mem) if best_mem is not None else None),
                        'topk': to_log
                    })
                except Exception:
                    pass
                # 必要に応じて保持（エクスポータ参照用）
                try:
                    self._last_combined_ranking = combined  # type: ignore[attr-defined]
                except Exception:
                    pass
                # L1履歴へ統合ランクを記録（HTML/JSON 可視化用）。GraphManager 側に集約。
                try:
                    if hasattr(self, 'graph_manager') and hasattr(self.graph_manager, 'l1_history'):
                        self.graph_manager.l1_history.append({
                            'step': int(self.step_count),
                            'mode': 'combined',
                            'source_pos': [int(cx), int(cy)],
                            'candidates': to_log,  # each contains dv/id/pos/dir/visit/is_wall/source
                            'meta': {
                                'topk_mem': int(topk_mem),
                                'unit_norm': bool(unit_norm),
                                'spatial_gate': int(spatial_gate),
                                'allow_walls': bool(allow_walls),
                                'visited_only': bool(visited_only),
                                'require_frontier': bool(require_frontier)
                            }
                        })
                except Exception:
                    pass
        except Exception:
            pass
        # Wire graph (timed)
        t_w = time.perf_counter()
        if self.wiring_strategy == 'query':
            self._wire_episodes_query_based(self._current_query)
        else:
            # NA-triggered candidate re-evaluation (one-shot): expand ring for this step
            try:
                if getattr(self, '_na_recheck_flag', False) and getattr(self.graph_manager, '_escalate_reeval_enabled', False):
                    self.graph_manager._escalate_reeval_active = True  # type: ignore[attr-defined]
            except Exception:
                pass
            self._wire_episodes()
            # clear
            try:
                if getattr(self.graph_manager, '_escalate_reeval_active', False):
                    self.graph_manager._escalate_reeval_active = False  # type: ignore[attr-defined]
                self._na_recheck_flag = False
            except Exception:
                pass
        self._timing['wiring_ms'].append((time.perf_counter() - t_w) * 1000.0)
        # Optional graph growth logging (after wiring, before other analytics)
        if (self._log_graph_growth or self.verbosity >= 2):
            if (self.step_count % getattr(self, '_graph_growth_interval', 1) == 0):
                try:
                    g = self.graph_manager.graph
                    nodes = g.number_of_nodes(); edges = g.number_of_edges()
                    dn = nodes - self._last_growth_nodes; de = edges - self._last_growth_edges
                    novelty_ratio = (self._new_cells / (self.step_count + 1)) if self.step_count >= 0 else None
                    revisit_ratio = (self._revisit_moves / (self.step_count + 1)) if self.step_count >= 0 else None
                    coverage = None
                    if getattr(self, '_open_cells', None):
                        try:
                            coverage = len(set(self.path)) / max(1, self._open_cells)
                        except Exception:
                            coverage = None
                    self.emit_event(self.EventType.ANALYSIS, {
                        'kind': 'graph_growth',
                        'step': self.step_count,
                        'nodes': nodes,
                        'edges': edges,
                        'nodes_added_since_last': dn,
                        'edges_added_since_last': de,
                        'novelty_ratio': novelty_ratio,
                        'revisit_ratio': revisit_ratio,
                        'coverage': coverage
                    })
                    self._last_growth_nodes = nodes; self._last_growth_edges = edges
                except Exception:
                    pass
        # Branch detection
        if self.backtrack_detector.detect_branch_entry(self.current_pos, self.maze):
            self.emit_event(self.EventType.BRANCH_ENTRY, str(self.current_pos))
        if self.backtrack_detector.detect_branch_completion(self.current_pos, self.maze, visited_positions=set(self.path)):
            self.emit_event(self.EventType.BRANCH_COMPLETION, str(self.current_pos))
            self._analyze_branch_completion()
            self._pending_completion_windows.append((self.step_count, self.step_count + 5))
        # geDIG capture (interval)
        if (self.step_count % self._dense_metric_interval) == 0:
            t_g = time.perf_counter()
            self._capture_gedig()
            self._timing['gedig_ms'].append((time.perf_counter() - t_g) * 1000.0)
        # Snapshot (conditional)
        t_s = time.perf_counter()
        self._maybe_save_snapshot()
        self._timing['snapshot_ms'].append((time.perf_counter() - t_s) * 1000.0)
        # Global recall (timed)
        if self.global_recall_enabled and not self._pending_backtrack_plan:
            t_r = time.perf_counter()
            self._maybe_global_recall(self._current_query)
            self._timing['recall_ms'].append((time.perf_counter() - t_r) * 1000.0)
        # Action selection / backtrack plan
        forced = False
        if self._pending_backtrack_plan:
            # Drop any no-op head equal to current position (robustness)
            try:
                while self._pending_backtrack_plan and tuple(self._pending_backtrack_plan[0]) == tuple(self.current_pos):
                    self._pending_backtrack_plan.pop(0)
            except Exception:
                pass
            if self._pending_backtrack_plan:
                next_pos = self._pending_backtrack_plan[0]
                direction = self._direction_towards(next_pos)
            else:
                direction = None
            if direction is None:
                # Re-plan towards the same target using multiple memory graph modes (freeze)
                tgt = getattr(self, '_bt_target_current', None)
                if tgt and tgt != self.current_pos:
                    original_mode = getattr(self, '_bt_path_mode', 'memory_graph')
                    tried_modes = ['episode_graph','memory_positions','memory_adjacent']
                    plan = None
                    for m in tried_modes:
                        try:
                            self._bt_path_mode = m
                            plan = self._plan_path_to(tgt)
                            if plan and len(plan) > 1:
                                break
                        except Exception:
                            plan = None
                            continue
                    self._bt_path_mode = original_mode
                    if plan and len(plan) > 1:
                        self._pending_backtrack_plan = plan[1:]
                        try:
                            while self._pending_backtrack_plan and tuple(self._pending_backtrack_plan[0]) == tuple(self.current_pos):
                                self._pending_backtrack_plan.pop(0)
                        except Exception:
                            pass
                        if self._pending_backtrack_plan:
                            next_pos = self._pending_backtrack_plan[0]
                            direction = self._direction_towards(next_pos)
                        else:
                            direction = None
                if direction is None and not self._bt_freeze_enabled:
                    # Non-freeze mode: allow local fallback
                    direction = self.decision_engine.select_action(episodes, query=self._current_query)
            else:
                forced = True
            # Emit step progress for BT
            if forced:
                remaining = max(0, len(self._pending_backtrack_plan))
                try:
                    self.emit_event(self.EventType.BACKTRACK_STEP, {
                        'to': next_pos,
                        'remaining': remaining,
                        'target': getattr(self, '_bt_target_current', None),
                        'path_mode': getattr(self, '_bt_path_mode', None)
                    })
                except Exception:
                    pass
                # Freeze: if remaining does not decrease for N consecutive steps, replan
                try:
                    prev = self._bt_freeze_prev_remaining
                    if prev is None or remaining < prev:
                        self._bt_freeze_prev_remaining = remaining
                        self._bt_freeze_stuck_count = 0
                    else:
                        self._bt_freeze_stuck_count += 1
                        if self._bt_freeze_enabled and self._bt_freeze_stuck_count >= self._bt_replan_stuck_n:
                            tgt = getattr(self, '_bt_target_current', None)
                            if tgt and tgt != self.current_pos:
                                original_mode = getattr(self, '_bt_path_mode', 'memory_graph')
                                for m in ['episode_graph','memory_positions','memory_adjacent']:
                                    try:
                                        self._bt_path_mode = m
                                        plan = self._plan_path_to(tgt)
                                        if plan and len(plan) > 1:
                                            self._pending_backtrack_plan = plan[1:]
                                            self._bt_freeze_prev_remaining = len(self._pending_backtrack_plan)
                                            self._bt_freeze_stuck_count = 0
                                            try:
                                                self.emit_event(self.EventType.BACKTRACK_PLAN, {'target': tgt,'length': len(plan),'plan': plan,'reason': 'freeze_replan','path_mode': m})
                                            except Exception:
                                                pass
                                            break
                                    except Exception:
                                        continue
                                self._bt_path_mode = original_mode
                except Exception:
                    pass
        else:
            direction = self.decision_engine.select_action(episodes, query=self._current_query)
        if direction is None:
            # In freeze mode, avoid local fallback when a BT plan exists; simply mark stuck
            self.emit_event(self.EventType.STUCK, 'no_action')
        else:
            # オプション分析 (確率含む)
            analysis = self.decision_engine.analyze_options(episodes)
            self.decision_history.append({
                'step': self.step_count,
                'pos': self.current_pos,
                'dir': direction,
                'options': analysis
            })
            try:
                # Cache compact decision context for enrichment
                self._last_decision_context = {
                    'step': self.step_count,
                    'pos': self.current_pos,
                    'dir': direction,
                    'options': analysis.get('options', {}) if isinstance(analysis, dict) else {}
                }
            except Exception:
                self._last_decision_context = None
            # フロンティア(未訪問)選好確率の集計
            try:
                opts = analysis.get('options', {})
                prob_new_sum = 0.0; has_new = False; chosen_is_new = False
                for d, info in opts.items():
                    # 訪問回数0かつ壁でないものをフロンティアとする
                    if (not info.get('is_wall')) and info.get('visit_count', 1) == 0:
                        has_new = True
                        prob_new_sum += info.get('probability', 0.0)
                        if d == direction:
                            chosen_is_new = True
                self._frontier_bias_records.append((self.step_count, prob_new_sum, has_new, chosen_is_new))
            except Exception:
                pass
            selected = episodes.get(direction)
            if selected and getattr(selected, 'is_wall', False):
                self.emit_event(self.EventType.WALL_SELECTED, direction)
                self._wall_selections += 1
            moved_ok = self._move(direction)
            if moved_ok:
                if forced:
                    # consume the step from the pending plan if it matches where we moved
                    try:
                        if self._pending_backtrack_plan:
                            nxt = self._pending_backtrack_plan[0]
                            if isinstance(nxt, (list, tuple)) and tuple(nxt) == tuple(self.current_pos):
                                self._pending_backtrack_plan.pop(0)
                        # If we reached the target, clear plan/target
                        if getattr(self, '_bt_target_current', None) and tuple(self.current_pos) == tuple(self._bt_target_current):
                            self._pending_backtrack_plan = []
                            self._bt_target_current = None
                    except Exception:
                        pass
                    # Emit a post-move progress tick
                    try:
                        self.emit_event(self.EventType.BACKTRACK_STEP, {
                            'pos': self.current_pos,
                            'remaining': len(self._pending_backtrack_plan),
                            'target': getattr(self, '_bt_target_current', None),
                            'path_mode': getattr(self, '_bt_path_mode', None)
                        })
                    except Exception:
                        pass
                if self.current_pos == self.goal_pos:
                    return True
        # After action selection & movement but before stagnation handling, invoke flush check
        if self._enable_flush and (self.step_count % self._flush_interval == 0) and self.step_count > 0:
            self._memory_guard_pass()
        # Stagnation handling
        self._handle_stagnation(episodes)
        # Frontier jump (simple_mode only) after movement & stagnation fallback
        if self.simple_mode:
            try:
                self._maybe_frontier_jump()
            except Exception:
                pass
        # Auto visit-weight scaling (reduce wandering if high revisit ratio & low novelty)
        if self._auto_visit_scale and self.step_count > 0 and (self.step_count % self._auto_visit_scale_interval == 0):
            revisit_ratio = self._revisit_moves / (self.step_count + 1)
            novelty_ratio = self._new_cells / (self.step_count + 1)
            if revisit_ratio > 0.55 and novelty_ratio < 0.45 and self._visit_weight_scale < self._auto_visit_scale_max:
                new_scale = min(self._auto_visit_scale_max, self._visit_weight_scale + 0.3)
                prev_scale = self._visit_weight_scale
                eff_w = self.set_visit_weight_scale(new_scale)
                self.emit_event(self.EventType.ANALYSIS, {
                    'kind': 'visit_scale_auto',
                    'prev_scale': prev_scale,
                    'new_scale': new_scale,
                    'effective_visit_weight': eff_w,
                    'revisit_ratio': revisit_ratio,
                    'novelty_ratio': novelty_ratio
                })
        # Verbose per-step output
        if self.verbosity >= 2:
            unique_positions = len(set(self.path))
            qps = (self._query_generated_count / (self.step_count + 1)) if self.step_count >= 0 else 0.0
            last_gedig = self.gedig_history[-1] if self.gedig_history else 'NA'
            self._safe_print(f"[STEP] s={self.step_count} pos={self.current_pos} path={len(self.path)} unique={unique_positions} gedig_last={last_gedig} qps={qps:.3f}")
        return False

    def _memory_guard_pass(self) -> None:
        """Phase5 eviction pass.

        Steps:
          1. If within capacity -> return.
          2. Compute score per passable episode (walls excluded) combining
             recency rank (older worse), inverse visit, Manhattan distance.
          3. Select number_to_evict = current - max_in_memory (capped safety 5% each pass min 1).
          4. Remove chosen episodes from EpisodeManager + vector index (best-effort) and graph.
          5. Emit events + update counters.
        Rehydration: not yet implemented (pending persistence layer) – placeholder events only.
        """
        t0 = time.perf_counter()
        try:
            episodes_dict = self.episode_manager.episodes
            passable_eps = [ep for ep in episodes_dict.values() if not getattr(ep, 'is_wall', False)]
            # --- Position-based capacity mode (optional) ---
            if self._max_in_memory_positions is not None:
                # Group by position
                pos_groups: Dict[tuple[int,int], List[Any]] = {}
                for ep in passable_eps:
                    pos_groups.setdefault(ep.position, []).append(ep)
                total_positions = len(pos_groups)
                if total_positions <= self._max_in_memory_positions:
                    if len(passable_eps) <= self._max_in_memory:
                        return  # already within both caps
                else:
                    # Score each position (use oldest episode recency rank + inverse mean visit + distance)
                    timestamps_all = [(ep.timestamp, ep) for ep in passable_eps]
                    timestamps_all.sort(key=lambda x: x[0])
                    rank_map = {ep.episode_id: i for i, (ts, ep) in enumerate(timestamps_all)}
                    cx, cy = self.current_pos
                    pos_candidates: List[tuple[float, tuple[int,int]]] = []
                    for pos, eps in pos_groups.items():
                        oldest = min(eps, key=lambda e: e.timestamp)
                        rec_rank = rank_map.get(oldest.episode_id, 0)
                        mean_visit = float(np.mean([e.visit_count for e in eps]))
                        inv_visit = 1.0 / (1.0 + mean_visit)
                        ex, ey = pos
                        dist = abs(ex - cx) + abs(ey - cy)
                        score = 0.6*rec_rank + 1.2*inv_visit + 0.2*dist
                        if pos in {self.current_pos, self.start_pos, self.goal_pos}:
                            score += 1e9  # never evict key anchors
                        pos_candidates.append((score, pos))
                    pos_candidates.sort(key=lambda x: x[0])
                    evict_pos: List[tuple[int,int]] = []
                    remaining_over = total_positions - self._max_in_memory_positions
                    idx = 0
                    while remaining_over > 0 and idx < len(pos_candidates):
                        evict_pos.append(pos_candidates[idx][1])
                        remaining_over -= 1
                        idx += 1
                    evict_ids: List[int] = []
                    for pos in evict_pos:
                        for ep in pos_groups[pos]:
                            evict_ids.append(ep.episode_id)
                    if evict_ids:
                        sample_ids = evict_ids[:min(3, len(evict_ids))]
                        self.emit_event(self.EventType.FLUSH_SCORE, {
                            'mode': 'position',
                            'total_positions': total_positions,
                            'original_over': total_positions - self._max_in_memory_positions,
                            'sample_positions': evict_pos[:min(3,len(evict_pos))],
                            'sample_episode_ids': sample_ids
                        })
                        for k, ep in list(episodes_dict.items()):
                            if ep.episode_id in evict_ids:
                                self._record_eviction_metadata(ep)
                                del episodes_dict[k]
                        if (self._vector_index is not None) and hasattr(self._vector_index, 'remove'):
                            try:
                                self._vector_index.remove(evict_ids)
                            except Exception:
                                pass
                        g = self.graph_manager.graph
                        removed_nodes: List[int] = []
                        for node in list(g.nodes()):
                            if node in evict_ids:
                                g.remove_node(node)
                                removed_nodes.append(node)
                        self._episodes_evicted_total += len(evict_ids)
                        self._flush_events += 1
                        remaining_positions = len({ep.position for ep in self.episode_manager.episodes.values() if not getattr(ep,'is_wall',False)})
                        self._position_eviction_events += 1
                        self.emit_event(self.EventType.FLUSH_EVICT, {
                            'mode': 'position',
                            'total_positions_before': total_positions,
                            'target_cap': self._max_in_memory_positions,
                            'evicted_positions': len(evict_pos),
                            'remaining_positions': remaining_positions,
                            'evicted_episode_ids': evict_ids,
                            'removed_nodes': removed_nodes,
                        })
                        if remaining_positions > self._max_in_memory_positions:
                            # Safety fallback: second pass
                            self._memory_guard_pass()
                            return
                        passable_eps = [ep for ep in self.episode_manager.episodes.values() if not getattr(ep,'is_wall',False)]
            # --- Episode-level cap enforcement ---
            passable_eps = [ep for ep in self.episode_manager.episodes.values() if not getattr(ep,'is_wall',False)]
            total_passable = len(passable_eps)
            if total_passable <= self._max_in_memory:
                return
            over_by = total_passable - self._max_in_memory
            # Use pluggable eviction policy if available
            if self._eviction_policy is not None:
                context = { 'current_pos': self.current_pos }
                evict_ids = self._eviction_policy.select(passable_eps, over_by, context=context)
                sample_ids = evict_ids[:min(3, len(evict_ids))]
                self.emit_event(self.EventType.FLUSH_SCORE, {
                    'mode': 'episode',
                    'policy': getattr(self._eviction_policy, 'name', 'unknown'),
                    'total_passable': total_passable,
                    'over_by': over_by,
                    'sample_episode_ids': sample_ids
                })
            else:
                timestamps = [(ep.timestamp, ep) for ep in passable_eps]
                timestamps.sort(key=lambda x: x[0])
                rank_map = {ep.episode_id: i for i, (ts, ep) in enumerate(timestamps)}
                cx, cy = self.current_pos
                candidates: list[tuple[float, Any]] = []
                for ep in passable_eps:
                    rec_rank = rank_map.get(ep.episode_id, 0)
                    inv_visit = 1.0 / (1.0 + ep.visit_count)
                    ex, ey = ep.position
                    dist = abs(ex - cx) + abs(ey - cy)
                    score = 0.6*rec_rank + 1.2*inv_visit + 0.2*dist
                    candidates.append((score, ep))
                if not candidates:
                    return
                candidates.sort(key=lambda x: x[0])
                to_evict_count = min(over_by, len(candidates))
                evict_slice = candidates[:to_evict_count]
                evict_ids = [ep.episode_id for _, ep in evict_slice]
                sample = candidates[:min(3, len(candidates))]
                self.emit_event(self.EventType.FLUSH_SCORE, {
                    'mode': 'episode',
                    'policy': 'inline_heuristic',
                    'total_passable': total_passable,
                    'over_by': over_by,
                    'sample_evict': [ {'id': ep.episode_id, 'score': float(score), 'visits': ep.visit_count} for score, ep in sample ]
                })
            if not evict_ids:
                return
            for k, ep in list(self.episode_manager.episodes.items()):
                if ep.episode_id in evict_ids:
                    self._record_eviction_metadata(ep)
                    del self.episode_manager.episodes[k]
            if (self._vector_index is not None) and hasattr(self._vector_index, 'remove'):
                try:
                    self._vector_index.remove(evict_ids)
                except Exception:
                    pass
            g = self.graph_manager.graph
            removed_nodes = []
            for node in list(g.nodes()):
                if node in evict_ids:
                    g.remove_node(node)
                    removed_nodes.append(node)
            self._episodes_evicted_total += len(evict_ids)
            self._flush_events += 1
            remaining_passable = len([ep for ep in self.episode_manager.episodes.values() if not getattr(ep,'is_wall',False)])
            self._episode_eviction_events += 1
            self.emit_event(self.EventType.FLUSH_EVICT, {
                'mode': 'episode',
                'total_passable_before': total_passable,
                'over_by': over_by,
                'evicted': evict_ids,
                'remaining_passable': remaining_passable,
                'removed_nodes': removed_nodes,
            })
        except Exception as e:
            self.emit_event(self.EventType.FLUSH_ERROR, str(e))
        finally:
            self._timing.get('flush_ms', []).append((time.perf_counter() - t0) * 1000.0)

    # --- Rehydration Logic (Phase5 T5-2) ---
    def _record_eviction_metadata(self, episode) -> None:  # type: ignore[no-untyped-def]
        try:
            meta = {
                'episode_id': int(episode.episode_id),
                'position': (int(episode.position[0]), int(episode.position[1])),
                'direction': episode.direction,
                'visit_count': int(episode.visit_count),
                'is_wall': bool(getattr(episode, 'is_wall', False)),
                'timestamp': int(getattr(episode, 'timestamp', 0)),
            }
            # Ordered LRU insert/update
            if episode.episode_id in self._evicted_catalog:
                self._evicted_catalog.pop(episode.episode_id, None)
            self._evicted_catalog[episode.episode_id] = meta
            # Trim if over cap
            if self._evicted_catalog_max is not None:
                while len(self._evicted_catalog) > self._evicted_catalog_max:
                    self._evicted_catalog.popitem(last=False)
            # Append to persistence file
            if getattr(self, '_persistence_dir', None):
                try:
                    serializable = dict(meta)
                    # Ensure JSON-friendly types
                    pos = serializable.get('position')
                    if isinstance(pos, tuple):
                        serializable['position'] = list(pos)
                    with open(self._evicted_catalog_path, 'a') as f:  # type: ignore[attr-defined]
                        import json as _json
                        f.write(_json.dumps(serializable, ensure_ascii=False) + '\n')
                except Exception:
                    pass
        except Exception:
            pass

    def _maybe_rehydrate_position(self, position: tuple[int,int]) -> None:
        """If any evicted episodes existed at this position, reconstruct them.

        Strategy: scan catalog for matching position (cheap; catalog usually small after evictions).
        For each direction absent in live episodes create a new Episode with preserved visit_count.
        """
        if not self._evicted_catalog:
            return
        t0 = time.perf_counter()
        from core.episode_manager import Episode  # local import to avoid circular
        existing_dirs = {dir_ for (pos, dir_), ep in self.episode_manager.episodes.items() if pos == position}
        rebuilt = 0
        for eid, meta in list(self._evicted_catalog.items()):
            if meta['position'] != position:
                continue
            direction = meta['direction']
            if direction in existing_dirs:
                continue
            # Recreate vector from base generator consistent with original semantics
            vec = self.vector_processor.create_vector(position, direction, meta['is_wall'], meta['visit_count'])
            from core.episode_manager import Episode as _Ep
            ep = _Ep(position=position, direction=direction, vector=vec, is_wall=meta['is_wall'], visit_count=meta['visit_count'], episode_id=self.episode_manager.episode_counter, timestamp=self.episode_manager.current_step)
            self.episode_manager.episodes[(position, direction)] = ep
            self.episode_manager.episode_counter += 1
            rebuilt += 1
            # Index re-add (best-effort)
            if (self._vector_index is not None) and not meta['is_wall'] and hasattr(ep, 'get_weighted_vector'):
                try:
                    wv = ep.get_weighted_vector(self.decision_engine.weights, self._weights_version, self.vector_processor.apply_weights)
                    import numpy as _np
                    self._vector_index.add([ep.episode_id], _np.asarray(wv, dtype=float).reshape(1,-1))
                except Exception:
                    pass
        if rebuilt:
            self._episodes_rehydrated_total += rebuilt
            self._rehydration_events += 1
            try:
                self._rehydrated_positions.add(position)
            except Exception:
                pass
            self.emit_event(self.EventType.REHYDRATION, {
                'position': position,
                'rebuilt': rebuilt
            })
        # Timing always captured if attempt made and catalog non-empty
        self._timing.get('rehydration_ms', []).append((time.perf_counter() - t0) * 1000.0)

    def rehydrate_position(self, position: tuple[int,int]) -> int:
        """Public helper for tests to force rehydration attempt at a position.
        Returns number of episodes rebuilt."""
        before = self._episodes_rehydrated_total
        self._maybe_rehydrate_position(position)
        return self._episodes_rehydrated_total - before

    def _print_progress(self) -> None:
        """Print concise progress snapshot every 100 steps.

        Shows: step, current position, path length, unique positions,
        recent GeDIG mean (last 20), simple mode metrics if enabled.
        """
        unique_positions = len(set(self.path))
        gedig_recent_mean = None
        if self.gedig_history:
            recent = self.gedig_history[-20:]
            gedig_recent_mean = float(np.mean(recent))
        parts = [
            f"step={self.step_count}",
            f"pos={self.current_pos}",
            f"path_len={len(self.path)}",
            f"unique={unique_positions}",
        ]
        if gedig_recent_mean is not None:
            parts.append(f"gedig_mean20={gedig_recent_mean:.4f}")
        if self.simple_mode and self.step_count >= 1:
            qps = self._query_generated_count / (self.step_count + 1)
            btr = self._backtrack_triggers / (self.step_count + 1)
            parts.append(f"qps={qps:.3f}")
            parts.append(f"bt_rate={btr:.3f}")
        print("[PROGRESS] " + " | ".join(parts))

    def get_statistics(self) -> Dict[str, Any]:
        # Opportunistic late clamp before reporting if still above (safety only)
        if self._enable_flush:
            try:
                if self._max_in_memory_positions is not None or self._max_in_memory:
                    # lightweight check; do not recurse infinitely
                    passable_positions_now = len({ep.position for ep in self.episode_manager.episodes.values() if not getattr(ep,'is_wall',False)})
                    if (self._max_in_memory_positions is not None and passable_positions_now > self._max_in_memory_positions) or \
                       (self._max_in_memory_positions is None and passable_positions_now > self._max_in_memory):
                        self._memory_guard_pass()
            except Exception:
                pass
        stats: Dict[str, Any] = {
            'steps': self.step_count,
            'path_length': len(self.path),
            'unique_positions': len(set(self.path)),
            'goal_reached': self.is_goal_reached,
            'new_cells': getattr(self, '_new_cells', None),
            'revisit_moves': getattr(self, '_revisit_moves', None),
            'wall_selections': getattr(self, '_wall_selections', None),
            'dead_end_events': getattr(self, '_dead_end_events', None),
            'episode_stats': self.episode_manager.get_statistics(),
            'graph_stats': self.graph_manager.get_graph_statistics(),
            'branch_stats': self.backtrack_detector.get_branch_statistics(),
            'stagnation_resets': self._stagnation_resets,
            'wiring_top_k': self.query_wiring_k,
            'global_recall_enabled': self.global_recall_enabled,
            'recall_threshold': self.recall_score_threshold,
            'dense_metric_interval': self._dense_metric_interval,
            'snapshot_skip_idle': self._snapshot_skip_idle,
            'vector_index_size': (len(self._vector_index) if (self._vector_index is not None) else 0),
            # Phase5 placeholders
            'flush_enabled': self._enable_flush,
            'flush_interval': self._flush_interval,
            'max_in_memory': self._max_in_memory,
            'max_in_memory_positions': self._max_in_memory_positions,
            'flush_events': self._flush_events,
            'episodes_evicted_total': self._episodes_evicted_total,
            'episodes_rehydrated_total': self._episodes_rehydrated_total,
            'rehydration_events': getattr(self, '_rehydration_events', 0),
            'rehydrated_unique_positions': len(getattr(self, '_rehydrated_positions', set())),
            'episode_eviction_events': getattr(self, '_episode_eviction_events', 0),
            'position_eviction_events': getattr(self, '_position_eviction_events', 0),
            'evicted_catalog_size': len(getattr(self, '_evicted_catalog', {})),
            'ann_backend': getattr(self, '_ann_backend', None),
            'ann_init_error': getattr(self, '_ann_init_error', None),
            'ann_index_elements': (len(self._vector_index) if (self._vector_index and getattr(self, '_ann_backend', None)) else None),
            'ann_upgrade_threshold': getattr(self, '_ann_upgrade_threshold', None),
            'evicted_catalog_bytes': self._catalog_last_bytes,
        }
        if self._enable_flush:
            passable_positions = {pos for (pos, _d), ep in self.episode_manager.episodes.items() if not getattr(ep,'is_wall',False)}
            stats['passable_positions'] = len(passable_positions)
        # Timing aggregates
        if any(len(v) for v in self._timing.values()):
            def agg_list(lst: List[float]) -> Dict[str, float]:
                if not lst:
                    return {'count':0,'mean_ms':0.0,'p95_ms':0.0,'max_ms':0.0}
                return {
                    'count': len(lst),
                    'mean_ms': float(np.mean(lst)),
                    'p95_ms': float(np.percentile(lst,95)) if len(lst) > 1 else float(lst[-1]),
                    'max_ms': float(np.max(lst))
                }
            stats['timing'] = {k: agg_list(self._timing[k]) for k in self._timing}
        if self.simple_mode and self.step_count >= 0:
            stats['simple_mode'] = {
                'query_generated': self._query_generated_count,
                'queries_per_step': (self._query_generated_count / (self.step_count + 1)) if self.step_count >= 0 else None,
                'backtrack_trigger_rate': (self._backtrack_triggers / (self.step_count + 1)) if self.step_count >= 0 else 0.0,
                'nn_degeneracy_triggers': getattr(self, '_nn_degeneracy_triggers', 0)
            }
        # Frontier exploration bias stats
        try:
            if self._frontier_bias_records:
                recs = self._frontier_bias_records
                total = len(recs)
                with_new = [r for r in recs if r[2]]
                if with_new:
                    mean_prob_new = float(np.mean([r[1] for r in with_new]))
                    choose_rate_when_available = float(np.mean([1.0 if r[3] else 0.0 for r in with_new]))
                else:
                    mean_prob_new = None; choose_rate_when_available = None
                stats['frontier_bias'] = {
                    'samples': total,
                    'steps_with_new_options': len(with_new),
                    'mean_prob_mass_to_new_when_available': mean_prob_new,
                    'choose_new_rate_when_available': choose_rate_when_available
                }
        except Exception:
            pass
        if self.gedig_history:
            vals = self.gedig_history
            stats['gedig_stats'] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'min': float(min(vals)),
                'max': float(max(vals)),
                'recent': vals[-10:] if len(vals) >= 10 else vals
            }
        # Derived exploration metrics
        try:
            if getattr(self, '_open_cells', None):
                stats['unique_coverage'] = len(set(self.path)) / max(1, self._open_cells)
            if self.step_count > 0:
                stats['novelty_ratio'] = self._new_cells / (self.step_count + 1)
                stats['revisit_ratio'] = self._revisit_moves / (self.step_count + 1)
        except Exception:
            pass
        # Transition decomposition Stage0 metrics
        obs = getattr(self, '_transition_observer', None)
        if obs is not None:
            try:
                stats.update(obs.get_metrics())
            except Exception:
                pass
        # Diagnostics for observer activation
        stats['transition_observer'] = {
            'enabled': self._transition_observer_enabled,
            'move_hooks': getattr(self, '_transition_move_hooks', None),
            'on_step_calls': getattr(self, '_transition_on_step_calls', None)
        }
        # Macro target metrics
        mtp = getattr(self, '_macro_target_planner', None)
        if mtp is not None:
            try:
                mt_metrics = mtp.get_metrics()
                stats.update(mt_metrics)
                if getattr(self, '_macro_target_analysis_enabled', True):
                    import time as _time
                    t0 = _time.perf_counter()
                    try:
                        from experiments.maze_transition_decomposition.macro_target_analyzer import MacroTargetAnalyzer  # type: ignore
                        analyzer = getattr(self, '_macro_target_analyzer', None)
                        if analyzer is None:
                            analyzer = MacroTargetAnalyzer()
                            self._macro_target_analyzer = analyzer
                        snap = mtp.get_snapshot()
                        analysis = analyzer.analyze(snap)
                        stats['macro_target_analysis'] = analysis
                    except Exception:
                        pass
                    dt_ms = ( _time.perf_counter() - t0 ) * 1000.0
                    try:
                        if hasattr(analyzer, 'record_timing'):
                            analyzer.record_timing(dt_ms)
                    except Exception:
                        pass
                    self._macro_target_analysis_times.append(dt_ms)
                    if len(self._macro_target_analysis_times) > 200:
                        self._macro_target_analysis_times.pop(0)
                    try:
                        import statistics as _st
                        stats['macro_target_analysis_timing'] = {
                            'last_ms': dt_ms,
                            'mean_ms': float(_st.fmean(self._macro_target_analysis_times)),
                            'samples': len(self._macro_target_analysis_times)
                        }
                    except Exception:
                        pass
            except Exception:
                pass
        return stats

    # Runtime toggle for analyzer
    def set_macro_target_analysis(self, enabled: bool) -> None:
        self._macro_target_analysis_enabled = bool(enabled)

    def _capture_gedig(self) -> None:
        # Prefer within-step 'before' snapshot if available; fallback to last saved snapshot
        prev_graph = getattr(self, '_step_before_graph', None)
        if prev_graph is None:
            if not self.graph_manager.graph_history:
                return
            prev_graph = self.graph_manager.graph_history[-1]
        curr_graph = self.graph_manager.get_graph_snapshot()

        # --- Score / escalation ---
        escalated = False; shortcut_flag = False
        multihop_scores = None; multihop_raw = None; multihop_variation = None
        dynamic_threshold_value: float | None = None
        if self.use_escalation and (self.dynamic_escalation or self.escalation_threshold == 'dynamic'):
            hlen = len(self.gedig_history)
            if self._theta_dynamic_enabled and hlen >= max(self._theta_window, 10):
                # Quantile-based threshold: choose q such that P(F < theta) ≈ target_rate
                recent = np.array(self.gedig_history[-self._theta_window:])
                try:
                    q = float(np.quantile(recent, self._theta_target_rate))
                except Exception:
                    q = float(np.median(recent))
                dynamic_threshold_value = q
            else:
                # Legacy median-offset heuristic as fallback during warmup
                if hlen >= self._dynamic_warmup:
                    recent = np.array(self.gedig_history[-self._dynamic_window:]); med = float(np.median(recent))
                    dynamic_threshold_value = max(-0.02, med - self.dynamic_offset)
                else:
                    dynamic_threshold_value = 0.12 if hlen > 3 else 0.15
        # Optional: precompute NA score from previous step context if enabled (needs ig, streak, K)
        na_score = None; na_theta = None; escalated_by_na = False
        # Primary evaluation
        if self.use_escalation and hasattr(self.gedig_evaluator, 'evaluate_escalating'):
            eval_res = self.gedig_evaluator.evaluate_escalating(
                prev_graph, curr_graph,
                escalation_threshold=(dynamic_threshold_value if dynamic_threshold_value is not None else (None if isinstance(self.escalation_threshold, str) else self.escalation_threshold))
            )  # type: ignore[arg-type]
            value = eval_res['score']; escalated = bool(eval_res.get('escalated'))
            shortcut_flag = bool(eval_res.get('shortcut'))
            multihop_scores = eval_res.get('multihop'); multihop_raw = eval_res.get('multihop_raw')
            if escalated and multihop_scores:
                ms_vals = list(multihop_scores.values())
                if ms_vals:
                    multihop_variation = max(ms_vals) - min(ms_vals)
            # NA gating (0-hop):
            # 1) Optional simple gate: if geDIG > threshold then escalate multi-hop (no new threshold)
            #    When MAZE_NA_USE_QUERY_ONLY=1, skip structural trigger here（構造0-hopは参照しない）
            _na_use_query_only = os.environ.get('MAZE_NA_USE_QUERY_ONLY', '0').strip() not in ("0","false","False","")
            _ge_thr = (self._na_ge_thresh if self._na_ge_thresh is not None else (0.0 if self._na_ge0 else None))
            if not _na_use_query_only:
                if (_ge_thr is not None) and (not escalated) and (g0_current is not None) and (float(g0_current) > float(_ge_thr)):
                    try:
                        eval_res2 = self.gedig_evaluator.evaluate_escalating(prev_graph, curr_graph, escalation_threshold=1.0)
                        multihop_scores = eval_res2.get('multihop') or multihop_scores
                        multihop_raw = eval_res2.get('multihop_raw') or multihop_raw
                        escalated = True; escalated_by_na = True
                    except Exception:
                        pass
            # 2) Quantile-based S_na (disabled by default); compute score and force escalation if above theta
            try:
                if self._na_enable and not escalated:
                    raw_c = eval_res.get('raw_core') if isinstance(eval_res, dict) else None
                    ig_raw = float(raw_c.get('ig_value')) if isinstance(raw_c, dict) and raw_c.get('ig_value') is not None else 0.0
                    # components
                    s_ig = max(0.0, -ig_raw)
                    s_streak = float(getattr(self, '_no_growth_streak', 0)) if self._na_use_struct else 0.0
                    # last K (candidates)
                    try:
                        k_last = (self.graph_manager._l1_candidate_counts[-1] if getattr(self.graph_manager, '_l1_candidate_counts', None) else 0)
                    except Exception:
                        k_last = 0
                    s_k = max(0.0, float(2 - int(k_last)))  # prefer low K (more uncertainty)
                    # combine (weights fixed for now: 1.0/0.5/0.5)
                    na_score = float(s_ig + 0.5 * s_streak + 0.5 * s_k)
                    self._na_scores.append(na_score)
                    if len(self._na_scores) > max(10, self._na_window * 2):
                        del self._na_scores[:-self._na_window]
                    if len(self._na_scores) >= max(10, self._na_window):
                        import numpy as _np
                        na_theta = float(_np.quantile(_np.array(self._na_scores[-self._na_window:]), self._na_target_rate))
                        if na_score > na_theta and not escalated:
                            # Force multi-hop escalation once to fetch multihop details
                            eval_res2 = self.gedig_evaluator.evaluate_escalating(prev_graph, curr_graph, escalation_threshold=1.0)  # base_score<thr likely true
                            multihop_scores = eval_res2.get('multihop') or multihop_scores
                            multihop_raw = eval_res2.get('multihop_raw') or multihop_raw
                            escalated = True; escalated_by_na = True
                            # keep value as 0-hop score
            except Exception:
                pass
            # Emit certificate (analysis) with core raw fields if available
            try:
                raw_c = eval_res.get('raw_core') if isinstance(eval_res, dict) else None
                details = eval_res.get('details') if isinstance(eval_res, dict) else None
                g0_detail = None
                gmin_detail = None
                try:
                    if isinstance(details, dict):
                        if details.get('g0') is not None:
                            g0_detail = float(details.get('g0'))
                        if details.get('gmin') is not None:
                            gmin_detail = float(details.get('gmin'))
                except Exception:
                    g0_detail = None; gmin_detail = None
                if g0_detail is None:
                    g0_detail = g0_current
                if gmin_detail is None and isinstance(multihop_raw, dict) and multihop_raw:
                    try:
                        gmin_detail = float(min(multihop_raw.values()))
                    except Exception:
                        gmin_detail = None
                self._last_g0 = g0_detail
                self._last_gmin = gmin_detail
                cert = {
                    'kind': 'certificate',
                    'step': self.step_count,
                    'gedig': float(value),
                    'theta': (details.get('threshold_used') if isinstance(details, dict) else None),
                    'struct_improvement': (raw_c.get('structural_improvement') if isinstance(raw_c, dict) else None),
                    'ig': (raw_c.get('ig_value') if isinstance(raw_c, dict) else None),
                    'multihop_raw': multihop_raw,
                    'na_score': na_score,
                    'na_theta': na_theta,
                    'na_escalated': escalated_by_na,
                    'na_ge0': bool(self._na_ge0) and (g0_detail is not None) and (g0_detail > 0.0),
                    'na_ge_thresh': (self._na_ge_thresh if self._na_ge_thresh is not None else None),
                    'g0': g0_detail,
                    'gmin': gmin_detail,
                }
                self.emit_event(self.EventType.ANALYSIS, cert)
            except Exception:
                pass
        else:
            value = self.gedig_evaluator.calculate(prev_graph, curr_graph)
            try:
                self._last_g0 = float(value)
            except Exception:
                self._last_g0 = None
            self._last_gmin = None

        # --- Structural ---
        nodes_before = prev_graph.number_of_nodes(); nodes_after = curr_graph.number_of_nodes()
        edges_before = prev_graph.number_of_edges(); edges_after = curr_graph.number_of_edges()
        nodes_added = max(0, nodes_after - nodes_before); edges_added = max(0, edges_after - edges_before)
        density_before = nx.density(prev_graph) if nodes_before > 1 else 0.0
        density_after = nx.density(curr_graph) if nodes_after > 1 else 0.0
        density_change = density_after - density_before
        if (not getattr(self, 'enable_diameter_metrics', True)) or \
           nodes_after > getattr(self, '_diameter_node_cap', 1000) or \
           nodes_before > getattr(self, '_diameter_node_cap', 1000):
            dia_before = None; dia_after = None
        else:
            # Lightweight try; skip on exception or excessive time
            t_d = time.perf_counter()
            try:
                dia_before = (nx.diameter(prev_graph) if nodes_before > 1 and nx.is_connected(prev_graph) else None)
                if (time.perf_counter() - t_d)*1000.0 < getattr(self, '_diameter_time_budget_ms', 50.0):
                    dia_after = (nx.diameter(curr_graph) if nodes_after > 1 and nx.is_connected(curr_graph) else None)
                else:
                    dia_after = None; dia_before = None
            except Exception:
                dia_before = None; dia_after = None
        diameter_change = None if (dia_before is None or dia_after is None) else (dia_after - dia_before)

        # --- Global SP (average shortest path) delta (optional, single compute per step) ---
        sp_global_delta = None
        sp_before = None
        sp_after = None
        if getattr(self, '_sp_global_enabled', False):
            t_sp = time.perf_counter()
            try:
                def _avg_shortest_path(g: nx.Graph) -> float | None:
                    if g.number_of_nodes() <= 1:
                        return None
                    # Optionally contract episodes to position-level nodes
                    if getattr(self, '_sp_global_poslevel', False):
                        try:
                            g_pos = nx.Graph()
                            pos_map = {}
                            for n, data in g.nodes(data=True):
                                pos = data.get('position') if isinstance(data, dict) else None
                                if pos is not None:
                                    pos_t = tuple(pos)
                                    pos_map[n] = pos_t
                                    if not g_pos.has_node(pos_t):
                                        g_pos.add_node(pos_t)
                            for a, b in g.edges():
                                pa = pos_map.get(a); pb = pos_map.get(b)
                                if pa is None or pb is None or pa == pb:
                                    continue
                                g_pos.add_edge(pa, pb)
                            g_work = g_pos
                        except Exception:
                            g_work = g
                    else:
                        g_work = g
                    # Use largest connected component on working graph
                    if not nx.is_connected(g_work):
                        comps = sorted(nx.connected_components(g_work), key=len, reverse=True)
                        if not comps:
                            return None
                        g_use = g_work.subgraph(comps[0]).copy()
                    else:
                        g_use = g_work
                    if g_use.number_of_nodes() <= 1:
                        return None
                    if getattr(self, '_sp_global_full', False):
                        try:
                            return float(nx.average_shortest_path_length(g_use))
                        except Exception:
                            return None
                    # Sampled mode
                    import random as _rnd
                    nodes = list(g_use.nodes())
                    if len(nodes) < 2:
                        return None
                    samples = min(max(1, getattr(self, '_sp_global_samples', 200)), (len(nodes) * (len(nodes) - 1)) // 2)
                    total = 0.0; count = 0
                    # Precompute few sources then sample targets per source
                    srcs = _rnd.sample(nodes, min(len(nodes), max(1, samples // max(1, int(np.log2(len(nodes)+1))))))
                    for s in srcs:
                        try:
                            dists = nx.single_source_shortest_path_length(g_use, s)
                        except Exception:
                            continue
                        t_choices = _rnd.sample(nodes, min(len(nodes), max(1, samples // max(1, len(srcs)))))
                        for t in t_choices:
                            if t == s:
                                continue
                            d = dists.get(t)
                            if d is None:
                                continue
                            total += float(d); count += 1
                            if (time.perf_counter() - t_sp) * 1000.0 > getattr(self, '_sp_global_budget_ms', 40.0):
                                break
                        if (time.perf_counter() - t_sp) * 1000.0 > getattr(self, '_sp_global_budget_ms', 40.0):
                            break
                    if count == 0:
                        return None
                    return total / count

                # NA-active detection (same rule as DA aggregation)
                na_active = False
                try:
                    ge_thr_tmp = (self._na_ge_thresh if getattr(self, '_na_ge_thresh', None) is not None else (0.0 if getattr(self, '_na_ge0', False) else None))
                    if (ge_thr_tmp is not None) and (g0_current is not None) and (float(g0_current) > float(ge_thr_tmp)):
                        na_active = True
                except Exception:
                    na_active = False

                # Fast path: single compute + last_avg delta
                sp_after = _avg_shortest_path(curr_graph)
                sp_before = getattr(self, '_sp_last_avg', None)
                if (sp_before is not None) and (sp_after is not None):
                    sp_global_delta = float(sp_after - sp_before)
                self._sp_last_avg = sp_after if (sp_after is not None) else self._sp_last_avg

                # Forced SP at NA: compute both prev and curr with boosted budget/samples to ensure a reading
                if getattr(self, '_sp_force_on_na', False) and (na_active or escalated):
                    # Temporarily override samples/budget
                    bak_samples = self._sp_global_samples; bak_budget = self._sp_global_budget_ms
                    try:
                        self._sp_global_samples = max(self._sp_global_samples, getattr(self, '_sp_force_samples', self._sp_global_samples))
                        self._sp_global_budget_ms = max(self._sp_global_budget_ms, getattr(self, '_sp_force_budget_ms', self._sp_global_budget_ms))
                        sp_b = _avg_shortest_path(prev_graph)
                        sp_a = _avg_shortest_path(curr_graph)
                        if (sp_b is not None) and (sp_a is not None):
                            sp_global_delta = float(sp_a - sp_b)
                            self._sp_last_avg = sp_a
                    finally:
                        self._sp_global_samples = bak_samples; self._sp_global_budget_ms = bak_budget
            except Exception:
                pass
        drop_ratio = None
        if self.gedig_structural:
            prev_rec = self.gedig_structural[-1]
            if prev_rec['nodes_added'] > 0 and nodes_added == 0:
                prev_value = prev_rec['value']; drop_ratio = (prev_value - value) / max(abs(prev_value), 1e-6)
        growth_ratio = (nodes_added + 0.5 * edges_added) / max(nodes_after, 1.0)

        # --- Dead-end detection (pure structural; require both conditions to reduce false positives) ---
        dead_end_flag = False
        if self.simple_mode and self.step_count > 0:
            r, c = self.current_pos
            dirs = [(-1,0),(1,0),(0,-1),(0,1)]
            visited_set = set(self.path[:-1])
            unvisited_open = 0
            for dx, dy in dirs:
                nr, nc = r+dx, c+dy
                if 0 <= nr < self.h and 0 <= nc < self.w and self.maze[nr, nc] == 0:
                    if (nr, nc) not in visited_set:
                        unvisited_open += 1
            degree = curr_graph.degree(self.current_pos) if curr_graph.has_node(self.current_pos) else 0
            # Require: no unvisited open neighbor AND structural degree <= 1 (position-level)
            if (unvisited_open == 0 and degree <= 1) and self.current_pos not in self._dead_end_positions:
                dead_end_flag = True
                shortcut_flag = False
                self._dead_end_positions.add(self.current_pos)
            # update growth streak
            if nodes_added == 0:
                self._no_growth_streak += 1
            else:
                self._no_growth_streak = 0

                self._dead_end_positions.add(self.current_pos)
            # update growth streak
            if nodes_added == 0:
                self._no_growth_streak += 1
            else:
                self._no_growth_streak = 0

        # Apply instant geDIG drop at cul-de-sac endpoint and during stagnation (optional)
        if dead_end_flag and self._dead_end_penalty:
            try:
                value = float(value) - float(self._dead_end_penalty)
            except Exception:
                pass
        if getattr(self, '_no_growth_streak', 0) >= 2 and self._nogrowth_penalty:
            try:
                # Scale gently with streak up to 3
                scale = min(3, int(self._no_growth_streak)) / 3.0
                value = float(value) - float(self._nogrowth_penalty) * scale
            except Exception:
                pass

        # --- Backtrack evaluation value (DA-aware aggregation) ---
        # Default uses 0-hop value; optionally incorporate multi-hop when NA has fired.
        bt_eval_value = float(value)
        try:
            g0_current = float(value)
        except Exception:
            g0_current = None
        try:
            use_da_if_na = os.environ.get('MAZE_BT_USE_DA_IF_NA', '1').strip() not in ("0","false","False","")
            agg_mode = os.environ.get('MAZE_BT_AGG', 'na_min').strip()  # 'na_min'|'base'|'min'|'h1'
        except Exception:
            use_da_if_na = True; agg_mode = 'na_min'
        da_min = None
        if isinstance(multihop_scores, dict) and multihop_scores:
            try:
                da_min = float(min(float(v) for v in multihop_scores.values() if isinstance(v, (int, float))))
            except Exception:
                da_min = None
        # Aggregation policy
        if agg_mode == 'min':
            if da_min is not None:
                bt_eval_value = min(bt_eval_value, da_min)
        elif agg_mode == 'h1':
            try:
                h1 = multihop_scores.get(1) if isinstance(multihop_scores, dict) else None
                if isinstance(h1, (int, float)):
                    bt_eval_value = float(h1)
            except Exception:
                pass
        else:  # 'na_min' (default) or 'base'
            # Consider NA active if geDIG exceeds NA threshold (or query‑geDIG if MAZE_NA_USE_QUERY_ONLY=1)
            na_active = False
            try:
                ge_thr_tmp = (self._na_ge_thresh if getattr(self, '_na_ge_thresh', None) is not None else (0.0 if getattr(self, '_na_ge0', False) else None))
                if ge_thr_tmp is not None:
                    if os.environ.get('MAZE_NA_USE_QUERY_ONLY', '0').strip() not in ("0","false","False",""):
                        # query-only NA判定
                        if (query_eval_value is not None) and (float(query_eval_value) > float(ge_thr_tmp)):
                            na_active = True
                    else:
                        if (g0_current is not None) and (float(g0_current) > float(ge_thr_tmp)):
                            na_active = True
            except Exception:
                na_active = escalated_by_na
            if agg_mode == 'na_min' and use_da_if_na and (escalated_by_na or na_active) and da_min is not None:
                bt_eval_value = min(bt_eval_value, da_min)
            # 'base' -> keep 0-hop value

        # --- Optional: Query-evaluation (virtual hub) ---
        query_eval_value = None
        query_full_min = None
        try:
            qmode = os.environ.get('MAZE_QUERY_EVAL_MODE', os.environ.get('MAZE_QUERY_EVAL', '')).strip().lower()
        except Exception:
            qmode = ''
        if qmode in ('v','virtual'):
            try:
                # Build a virtual hub-connected graph without mutating current state
                gq = curr_graph.copy()
                # Use numeric hub id to avoid mixed-type ordering in core
                hub_id = int(-2000000 - int(self.step_count))
                if not gq.has_node(hub_id):
                    gq.add_node(hub_id)
                # Connect hub to "current-step episodes" (timestamp==step_count)
                try:
                    for n, data in list(gq.nodes(data=True)):
                        # only episodes with timestamp equal to current step and not walls (if available)
                        try:
                            ts = int(data.get('timestamp', -1)) if isinstance(data, dict) else -1
                            if ts == int(self.step_count) and (not (isinstance(data, dict) and data.get('is_wall', False))):
                                if not gq.has_edge(hub_id, n):
                                    gq.add_edge(hub_id, n)
                        except Exception:
                            continue
                except Exception:
                    pass
                # Candidate recall by simple L2 on decision vector (TopK, within tau)
                try:
                    import numpy as _np
                    qvec = self.decision_engine.create_query(self.current_pos, prefer_unexplored=True)
                    # prepare list of episode vectors
                    eps = list(getattr(self.episode_manager, 'episodes_by_id', {}).values())
                    # limits
                    try:
                        topk = int(os.environ.get('MAZE_QUERY_EVAL_TOPK', '8'))
                    except Exception:
                        topk = 8
                    try:
                        tau = float(os.environ.get('MAZE_QUERY_EVAL_TAU', os.environ.get('MAZE_L1_NORM_TAU', '1.0')))
                    except Exception:
                        tau = 1.0
                    dlist = []
                    for ep in eps:
                        try:
                            ev = _np.asarray(ep.vector, dtype=float)
                            dv = float(_np.linalg.norm(_np.asarray(qvec, dtype=float) - ev))
                            dlist.append((dv, ep))
                        except Exception:
                            continue
                    dlist.sort(key=lambda t: t[0])
                    if topk > 0:
                        dlist = dlist[:topk]
                    for dv, ep in dlist:
                        if dv <= tau:
                            n = int(getattr(ep, 'episode_id', -1))
                            if gq.has_node(n) and not gq.has_edge(hub_id, n):
                                gq.add_edge(hub_id, n)
                except Exception:
                    pass
                # Evaluate geDIG for virtual hub augmentation (0-hop only for cost)
                try:
                    query_eval_value = float(self.gedig_evaluator.calculate(curr_graph, gq))
                except Exception:
                    query_eval_value = None
                # Evaluate multi-hop on virtual hub graph (for observation; may be heavier)
                try:
                    mh_res = self.gedig_evaluator.evaluate_escalating(curr_graph, gq, escalation_threshold=1.0)
                    mh = mh_res.get('multihop') if isinstance(mh_res, dict) else None
                    if isinstance(mh, dict) and mh:
                        try:
                            query_full_min = float(min(float(v) for v in mh.values() if isinstance(v, (int, float))))
                        except Exception:
                            query_full_min = None
                except Exception:
                    query_full_min = None
            except Exception:
                query_eval_value = None

        # --- NA/BT integration with query-eval (optional / unified gating) ---
        try:
            _na_use_query = os.environ.get('MAZE_NA_USE_QUERY', '1').strip() not in ("0","false","False","")
        except Exception:
            _na_use_query = True
        # Unified NA gate: single equation F = ΔGED - λΔIG (query-0hop if available), escalate when F>0
        try:
            _na_unified = os.environ.get('MAZE_NA_UNIFIED', '0').strip() not in ("0","false","False","")
        except Exception:
            _na_unified = False
        if _na_unified and (not escalated):
            try:
                f_eff = query_eval_value if (query_eval_value is not None) else value
                if (f_eff is not None) and (float(f_eff) > 0.0):
                    eval_res2 = self.gedig_evaluator.evaluate_escalating(prev_graph, curr_graph, escalation_threshold=1.0)
                    multihop_scores = eval_res2.get('multihop') or multihop_scores
                    multihop_raw = eval_res2.get('multihop_raw') or multihop_raw
                    escalated = True; escalated_by_na = True
            except Exception:
                pass
        # If NA not yet escalated and query_eval exceeds NA threshold, escalate multi-hop for inspection
        try:
            if (not _na_unified) and _na_use_query and (not escalated):
                # Compute query-NA threshold (dual-mode: frontier vs dead-end). Fallback to global NA threshold.
                ge_thr_local = (self._na_ge_thresh if getattr(self, '_na_ge_thresh', None) is not None else (0.0 if getattr(self, '_na_ge0', False) else None))
                try:
                    use_query_only = os.environ.get('MAZE_NA_USE_QUERY_ONLY', '0').strip() not in ("0","false","False","")
                except Exception:
                    use_query_only = False
                # Derive dual thresholds from env or dynamic percentiles over recent query_eval history
                def _parse_float_env(k: str) -> float | None:
                    try:
                        s = os.environ.get(k, '')
                        return float(s) if s.strip() != '' else None
                    except Exception:
                        return None
                def _quantile(vals: list[float], q: float) -> float | None:
                    try:
                        import numpy as _np
                        if not vals:
                            return None
                        return float(_np.quantile(_np.array(vals, dtype=float), float(q)))
                    except Exception:
                        return None
                # History window for dynamic thresholds
                try:
                    qwin = int(os.environ.get('MAZE_NA_QUERY_WINDOW', '20'))
                except Exception:
                    qwin = 20
                qe_hist: list[float] = []
                try:
                    for rec in (self.gedig_structural[-qwin:] if self.gedig_structural else []):
                        try:
                            vq = rec.get('query_eval') if isinstance(rec, dict) else None
                            if isinstance(vq, (int, float)):
                                qe_hist.append(float(vq))
                        except Exception:
                            continue
                except Exception:
                    qe_hist = []
                # Static thresholds
                th_dead = _parse_float_env('MAZE_NA_QUERY_TH_DEADEND')
                th_front = _parse_float_env('MAZE_NA_QUERY_TH_FRONTIER')
                # Percentile-based thresholds (if provided 0<q<=1)
                thq_dead = _parse_float_env('MAZE_NA_QUERY_P_DEADEND')
                thq_front = _parse_float_env('MAZE_NA_QUERY_P_FRONTIER')
                if (thq_dead is not None) and (0.0 < float(thq_dead) <= 1.0):
                    qt = _quantile(qe_hist, float(thq_dead))
                    if qt is not None:
                        th_dead = qt
                if (thq_front is not None) and (0.0 < float(thq_front) <= 1.0):
                    qt = _quantile(qe_hist, float(thq_front))
                    if qt is not None:
                        th_front = qt
                # Choose threshold by local frontier state
                try:
                    has_unvisited = bool(getattr(self, '_has_unvisited_obs', False))
                except Exception:
                    has_unvisited = False
                ge_thr_query = None
                if has_unvisited:
                    ge_thr_query = th_front if (th_front is not None) else ge_thr_local
                else:
                    ge_thr_query = th_dead if (th_dead is not None) else ge_thr_local
                # If query-only is requested, override structural NA gate entirely
                if use_query_only:
                    ge_thr_effective = ge_thr_query
                else:
                    ge_thr_effective = ge_thr_local
                if (ge_thr_effective is not None) and (query_eval_value is not None) and (float(query_eval_value) > float(ge_thr_effective)):
                    try:
                        eval_res2 = self.gedig_evaluator.evaluate_escalating(prev_graph, curr_graph, escalation_threshold=1.0)
                        multihop_scores = eval_res2.get('multihop') or multihop_scores
                        multihop_raw = eval_res2.get('multihop_raw') or multihop_raw
                        escalated = True; escalated_by_na = True
                    except Exception:
                        pass
        except Exception:
            pass

        # After query-eval/NA integration, optionally fold query into BT eval
        try:
            _bt_use_q = os.environ.get('MAZE_BT_USE_QUERY_MIN', '0').strip() not in ("0","false","False","")
        except Exception:
            _bt_use_q = False
        if _bt_use_q and (query_eval_value is not None):
            try:
                bt_eval_value = float(min(bt_eval_value, float(query_eval_value)))
            except Exception:
                pass

        # --- Record history ---
        self.gedig_history.append(value)
        rec_payload = {
            'step': self.step_count,
            'value': value,
            'nodes_added': nodes_added,
            'edges_added': edges_added,
            'density_change': density_change,
            'diameter_change': diameter_change,
            'sp_global': {
                'avg_before': sp_before,
                'avg_after': sp_after,
                'delta': sp_global_delta,
            } if getattr(self, '_sp_global_enabled', False) else None,
            'escalated': escalated,
            'shortcut': shortcut_flag,
            'dead_end': dead_end_flag,
            'drop_ratio': drop_ratio,
            'growth_ratio': growth_ratio,
            'multihop': multihop_scores,
            'multihop_raw': multihop_raw,
            'multihop_variation': multihop_variation,
            'dynamic_threshold': dynamic_threshold_value,
            'dead_end_penalty': (self._dead_end_penalty if dead_end_flag else 0.0),
            'nogrowth_penalty': (float(self._nogrowth_penalty) * (min(3, int(getattr(self, '_no_growth_streak', 0))) / 3.0) if getattr(self, '_no_growth_streak', 0) >= 2 and self._nogrowth_penalty else 0.0),
            'bt_eval_value': bt_eval_value,
            'query_eval': (float(query_eval_value) if query_eval_value is not None else None),
            'query_full_min': (float(query_full_min) if query_full_min is not None else None),
            'g0': (float(self._last_g0) if self._last_g0 is not None else None),
            'gmin': (float(self._last_gmin) if self._last_gmin is not None else None),
        }
        self.gedig_structural.append(rec_payload)

        # Optional: geDIGfull（multi-hop最小）駆動のBT発火（論文フロー準拠）
        try:
            ge_bt_enable = os.environ.get('MAZE_BT_GE_ENABLE', '1').strip() not in ("0","false","False","")
        except Exception:
            ge_bt_enable = True
        try:
            ge_bt_thr_s = os.environ.get('MAZE_BACKTRACK_THRESHOLD', '')
            ge_bt_thr = float(ge_bt_thr_s) if ge_bt_thr_s.strip() != '' else None
        except Exception:
            ge_bt_thr = None
        # 発火条件: 有効 / しきい値指定あり / NAゲート通過 / BT未実行 / bt_eval_value <= しきい値
        if ge_bt_enable and (ge_bt_thr is not None) and (not self._pending_backtrack_plan):
            # NA判定
            na_thr_val = None
            try:
                _na_s = os.environ.get('MAZE_NA_GE_THRESH', '')
                na_thr_val = float(_na_s) if _na_s.strip() != '' else None
            except Exception:
                na_thr_val = None
            na_ok = True
            if na_thr_val is not None:
                try:
                    g0 = float(self.gedig_history[-1]) if self.gedig_history else None
                    na_ok = (g0 is not None) and (g0 > na_thr_val)
                except Exception:
                    na_ok = False
            if na_ok and (bt_eval_value is not None) and (bt_eval_value <= float(ge_bt_thr)):
                # Select backtrack target and plan
                bt_target = self._select_backtrack_target()
                if bt_target and (tuple(bt_target) != tuple(self.current_pos)):
                    plan = self._plan_path_to(bt_target)
                    if plan and len(plan) > 1:
                        self._bt_target_current = tuple(bt_target)
                        self._pending_backtrack_plan = plan[1:]
                        self._bt_freeze_prev_remaining = len(self._pending_backtrack_plan)
                        self._bt_freeze_stuck_count = 0
                        try:
                            payload = {
                                'target': bt_target,
                                'length': len(plan),
                                'plan': plan,
                                'reason': 'ge_full',
                                'bt_eval': float(bt_eval_value),
                                'bt_threshold': float(ge_bt_thr),
                                'path_mode': getattr(self, '_bt_path_mode', None)
                            }
                            if getattr(self, '_last_bt_ranking', None):
                                payload['bt_target_ranking'] = self._last_bt_ranking
                            self.emit_event(self.EventType.BACKTRACK_PLAN, payload)
                        except Exception:
                            pass

        # Loop-driven BT (ΔSP) AFTER NA gate and geDIG update
        try:
            _bt_sp_enable = os.environ.get('MAZE_BT_SP_ENABLE', '1').strip() not in ("0","false","False","")
            _bt_sp_tau = float(os.environ.get('MAZE_BT_SP_TAU', '1.0'))
            _bt_sp_allow_with_unvisited = os.environ.get('MAZE_BT_SP_ALLOW_WITH_UNVISITED', '0').strip() not in ("0","false","False","")
        except Exception:
            _bt_sp_enable = True; _bt_sp_tau = 1.0; _bt_sp_allow_with_unvisited = False
        if _bt_sp_enable and not self._pending_backtrack_plan:
            # Skip when local unvisited observations exist (観測に未探索がある間は通常選択を優先)
            if not _bt_sp_allow_with_unvisited and bool(getattr(self, '_has_unvisited_obs', False)):
                pass
            else:
                try:
                    gm = getattr(self, 'graph_manager', None)
                    if gm is not None and hasattr(gm, 'edge_creation_log'):
                        # Require NA gate if configured
                        try:
                            require_na = os.environ.get('MAZE_BT_SP_REQUIRE_NA', '1').strip() not in ("0","false","False","")
                        except Exception:
                            require_na = True
                        na_ok = True
                        if require_na:
                            try:
                                na_thr_s = os.environ.get('MAZE_NA_GE_THRESH', '')
                                na_thr_val = float(na_thr_s) if na_thr_s.strip() != '' else None
                            except Exception:
                                na_thr_val = None
                            try:
                                g0 = float(self.gedig_history[-1]) if self.gedig_history else None
                                na_ok = (na_thr_val is None) or (g0 is not None and g0 > na_thr_val)
                            except Exception:
                                na_ok = False
                        if not na_ok:
                            raise RuntimeError('loop_bt(after NA): NA gate not satisfied')
                        step_now = int(self.step_count)
                        cand = []
                        for rec in gm.edge_creation_log[-50:]:
                            try:
                                if not isinstance(rec, dict):
                                    continue
                                if int(rec.get('step', -1)) != step_now:
                                    continue
                                cyc = rec.get('cycle')
                                if not isinstance(cyc, dict):
                                    continue
                                dsp = float(cyc.get('delta_sp', 0.0))
                                if dsp >= _bt_sp_tau and isinstance(cyc.get('position_path'), list):
                                    cand.append((dsp, cyc.get('position_path')))
                            except Exception:
                                continue
                        if cand:
                            cand.sort(key=lambda t: t[0], reverse=True)
                            dsp, ppath = cand[0]
                            plan = None
                            # Minimum plan length (positions), default 2; allow shorter via env
                            try:
                                bt_min_len = max(1, int(os.environ.get('MAZE_BT_MIN_PLAN_LEN', '2')))
                            except Exception:
                                bt_min_len = 2
                            try:
                                if tuple(self.current_pos) == tuple(ppath[0]):
                                    plan = [tuple(p) for p in ppath]
                                elif tuple(self.current_pos) == tuple(ppath[-1]):
                                    plan = [tuple(p) for p in reversed(ppath)]
                            except Exception:
                                plan = None
                            # Fallback: try endpoints as anchors via planner
                            if (not plan) or (len(plan) < bt_min_len):
                                for anchor in (ppath[0], ppath[-1]):
                                    try:
                                        plan2 = self._plan_path_to(tuple(anchor))
                                    except Exception:
                                        plan2 = None
                                    if plan2 and len(plan2) >= bt_min_len:
                                        plan = plan2
                                        break
                            # Fallback: try nearest position on cycle path
                            if (not plan) or (len(plan) < bt_min_len):
                                try:
                                    import math as _math
                                    cx, cy = self.current_pos
                                    nearest = min(ppath, key=lambda p: _math.fabs(p[0]-cx)+_math.fabs(p[1]-cy)) if ppath else None
                                except Exception:
                                    nearest = None
                                if nearest is not None:
                                    try:
                                        plan2 = self._plan_path_to(tuple(nearest))
                                    except Exception:
                                        plan2 = None
                                    if plan2 and len(plan2) >= bt_min_len:
                                        plan = plan2
                            if plan and len(plan) >= bt_min_len:
                                self._pending_backtrack_plan = plan[1:]
                                try:
                                    self._bt_target_current = tuple(plan[-1])
                                    self._bt_freeze_prev_remaining = len(self._pending_backtrack_plan)
                                    self._bt_freeze_stuck_count = 0
                                except Exception:
                                    pass
                                try:
                                    payload = {
                                        'target': plan[-1],
                                        'length': len(plan),
                                        'plan': plan,
                                        'reason': 'loop_sp',
                                        'delta_sp': float(dsp),
                                        'path_mode': getattr(self, '_bt_path_mode', None)
                                    }
                                    if getattr(self, '_last_bt_ranking', None):
                                        payload['bt_target_ranking'] = self._last_bt_ranking
                                    self.emit_event(self.EventType.BACKTRACK_PLAN, payload)
                                except Exception:
                                    pass
                except Exception:
                    pass
        # NA re-eval trigger for next step wiring (geDIG > threshold)
        try:
            if _na_unified:
                self._na_recheck_flag = bool(escalated_by_na)
            else:
                    ge_thr = (self._na_ge_thresh if getattr(self, '_na_ge_thresh', None) is not None else (0.0 if getattr(self, '_na_ge0', False) else None))
                    self._na_recheck_flag = (ge_thr is not None) and (g0_current is not None) and (float(g0_current) > float(ge_thr))
        except Exception:
            self._na_recheck_flag = False

        # --- Emit events ---
        if dead_end_flag:
            self.emit_event(self.EventType.DEAD_END, {
                'step': self.step_count,
                'score': value,
                'stagnation_streak': getattr(self, '_escalated_no_growth_streak', 0),
                'growth_ratio': growth_ratio,
                'escalated': escalated
            })
            try:
                self._dead_end_events += 1
            except Exception:
                pass
            if self._bt_use_deadend_trigger and self.simple_mode and not any(e['type'] in {self.EventType.BACKTRACK_TRIGGER.value, self.EventType.BACKTRACK_PLAN.value} for e in self.event_log):
                self.emit_event(self.EventType.BACKTRACK_TRIGGER, self._augment_backtrack_payload({'score': value, 'reason': 'dead_end_fallback'}))
        elif escalated and shortcut_flag:
            self.emit_event(self.EventType.SHORTCUT_CANDIDATE, {'step': self.step_count, 'score': value})

        # --- Backtrack logic --- (Skip if already executing a backtrack plan)
        if self.simple_mode and not self._pending_backtrack_plan:
            # Memory-based trigger (pure memory signals) before threshold trigger
            if getattr(self, '_bt_use_memory_trigger', False):
                # NA active?
                na_active = False
                try:
                    ge_thr_tmp = (self._na_ge_thresh if getattr(self, '_na_ge_thresh', None) is not None else (0.0 if getattr(self, '_na_ge0', False) else None))
                    if (ge_thr_tmp is not None) and (g0_current is not None) and (float(g0_current) > float(ge_thr_tmp)):
                        na_active = True
                except Exception:
                    na_active = False
                streak = int(getattr(self, '_no_growth_streak', 0) or 0)
                try:
                    k_last = (self.graph_manager._l1_candidate_counts[-1] if getattr(self.graph_manager, '_l1_candidate_counts', None) else 0)
                except Exception:
                    k_last = 0
                votes = 0
                if (not getattr(self, '_bt_mem_require_na', True)) or na_active:
                    if na_active:
                        votes += 1
                    if streak >= getattr(self, '_bt_mem_streak', 3):
                        votes += 1
                    if k_last <= getattr(self, '_bt_mem_cand_max', 2):
                        votes += 1
                if votes >= getattr(self, '_bt_mem_min_votes', 2):
                    self.emit_event(self.EventType.BACKTRACK_TRIGGER, self._augment_backtrack_payload({'score': bt_eval_value, 'base_score': value, 'reason': 'memory_trigger', 'memory_signals': {'na_active': na_active, 'streak': streak, 'l1_cand_last': k_last, 'votes': votes}, 'passive': self.passive_heuristics}))
                    if not self.passive_heuristics:
                        self._last_backtrack_step = self.step_count
                        bt_target = self._select_backtrack_target()
                        if bt_target and bt_target != self.current_pos:
                            plan = self._plan_path_to(bt_target)
                            if plan and len(plan) > 1:
                                self.emit_event(self.EventType.BACKTRACK_PLAN, {'target': bt_target,'length': len(plan),'plan': plan,'reason': 'memory_trigger','path_mode': getattr(self, '_bt_path_mode', None)})
                                self._pending_backtrack_plan = plan[1:]
                    return
            # Require NA activation (0-hop geDIG > NA threshold) before threshold BT
            na_gate = False
            try:
                ge_thr_tmp = (self._na_ge_thresh if getattr(self, '_na_ge_thresh', None) is not None else (0.0 if getattr(self, '_na_ge0', False) else None))
                if ge_thr_tmp is not None and (g0_current is not None) and (float(g0_current) > float(ge_thr_tmp)):
                    na_gate = True
            except Exception:
                na_gate = False
            # Also accept explicit NA-driven escalation as NA-active
            try:
                if not na_gate:
                    na_gate = bool(escalated_by_na)
            except Exception:
                pass

            # Combined ranking assist: if memory dv is significantly better than local obs dv
            combined_trigger = False
            if getattr(self, '_bt_from_combined', False):
                try:
                    cr = getattr(self, '_last_combined_ranking', None)
                    if isinstance(cr, list) and cr:
                        best_obs = None; best_mem = None
                        for r in cr:
                            try:
                                src = r.get('source') if isinstance(r, dict) else None
                                dvv = float(r.get('dv')) if isinstance(r, dict) and (r.get('dv') is not None) else None
                            except Exception:
                                dvv = None; src = None
                            if dvv is None:
                                continue
                            if src == 'obs':
                                if (best_obs is None) or (dvv < best_obs):
                                    best_obs = dvv
                            elif src == 'mem':
                                if (best_mem is None) or (dvv < best_mem):
                                    best_mem = dvv
                        if (best_obs is not None) and (best_mem is not None):
                            if float(best_mem) + float(getattr(self, '_bt_dv_margin', 0.02)) < float(best_obs):
                                combined_trigger = True
                except Exception:
                    combined_trigger = False

            # If combined_trigger suggests BT, allow NA gate override when configured
            if combined_trigger and (na_gate or (not getattr(self, '_bt_require_na_general', True))):
                can_fire = True
                if self.backtrack_debounce and self._static_last_backtrack_step is not None and (self.step_count - self._static_last_backtrack_step) < self.backtrack_cooldown:
                    can_fire = False
                if can_fire:
                    self._backtrack_triggers += 1
                    if not self.passive_heuristics:
                        self._static_last_backtrack_step = self.step_count
                    payload = {'score': bt_eval_value, 'base_score': value, 'reason': 'combined_rank', 'na_active': bool(na_gate), 'margin': float(getattr(self, '_bt_dv_margin', 0.02)), 'passive': self.passive_heuristics}
                    try:
                        payload['combined_best'] = {'obs': float(best_obs) if (best_obs is not None) else None, 'mem': float(best_mem) if (best_mem is not None) else None}  # type: ignore[name-defined]
                    except Exception:
                        pass
                    self.emit_event(self.EventType.BACKTRACK_TRIGGER, self._augment_backtrack_payload(payload))
                    if not self.passive_heuristics:
                        bt_target = self._select_backtrack_target()
                        if bt_target and bt_target != self.current_pos:
                            plan = self._plan_path_to(bt_target)
                            if plan and len(plan) > 1:
                                self.emit_event(self.EventType.BACKTRACK_PLAN, {
                                    'target': bt_target,
                                    'length': len(plan),
                                    'plan': plan,
                                    'reason': 'combined_rank',
                                    'path_mode': getattr(self, '_bt_path_mode', None)
                                })
                                self._bt_target_current = bt_target
                                self._pending_backtrack_plan = plan[1:]
                return

            if na_gate and (bt_eval_value <= self.backtrack_threshold):
                can_fire = True
                if self.backtrack_debounce and self._static_last_backtrack_step is not None and (self.step_count - self._static_last_backtrack_step) < self.backtrack_cooldown:
                    can_fire = False
                if can_fire:
                    # Count trigger regardless of passive mode
                    self._backtrack_triggers += 1
                    if not self.passive_heuristics:
                        self._static_last_backtrack_step = self.step_count
                    self.emit_event(self.EventType.BACKTRACK_TRIGGER, self._augment_backtrack_payload({'score': bt_eval_value, 'base_score': value, 'reason': 'simple_threshold', 'na_active': True, 'passive': self.passive_heuristics}))
                    if not self.passive_heuristics:
                        bt_target = self._select_backtrack_target()
                        if bt_target and bt_target != self.current_pos:
                            plan = self._plan_path_to(bt_target)
                            if plan and len(plan) > 1:
                                self.emit_event(self.EventType.BACKTRACK_PLAN, {
                                    'target': bt_target,
                                    'length': len(plan),
                                    'plan': plan,
                                    'reason': 'simple_threshold',
                                    'path_mode': getattr(self, '_bt_path_mode', None)
                                })
                                self._bt_target_current = bt_target
                                self._pending_backtrack_plan = plan[1:]
            return

        # Skip non-simple mode backtrack if already executing a plan
        if self._pending_backtrack_plan:
            return

        dynamic_bt_threshold: Optional[float] = None
        if self.dynamic_backtrack_enabled and len(self.gedig_history) >= self._backtrack_min_samples:
            recent_hist = np.array(self.gedig_history[-self._backtrack_window:])
            p05 = float(np.percentile(recent_hist, 5)); dynamic_bt_threshold = p05 - self._dynamic_bt_margin
        eff_threshold = self.backtrack_threshold
        if dynamic_bt_threshold is not None:
            eff_threshold = max(eff_threshold, dynamic_bt_threshold)
        # Optional combined-ranking assist (non-simple mode)
        trigger_combined = False
        try:
            if getattr(self, '_bt_from_combined', False):
                # Compute NA gate (same rule as simple branch)
                na_gate2 = False
                try:
                    ge_thr_tmp = (self._na_ge_thresh if getattr(self, '_na_ge_thresh', None) is not None else (0.0 if getattr(self, '_na_ge0', False) else None))
                    if ge_thr_tmp is not None and (g0_current is not None) and (float(g0_current) > float(ge_thr_tmp)):
                        na_gate2 = True
                except Exception:
                    na_gate2 = False
                try:
                    if not na_gate2:
                        na_gate2 = bool(escalated_by_na)
                except Exception:
                    pass
                cr = getattr(self, '_last_combined_ranking', None)
                if isinstance(cr, list) and cr:
                    best_obs = None; best_mem = None
                    for r in cr:
                        try:
                            src = r.get('source') if isinstance(r, dict) else None
                            dvv = float(r.get('dv')) if isinstance(r, dict) and (r.get('dv') is not None) else None
                        except Exception:
                            dvv = None; src = None
                        if dvv is None:
                            continue
                        if src == 'obs':
                            if (best_obs is None) or (dvv < best_obs):
                                best_obs = dvv
                        elif src == 'mem':
                            if (best_mem is None) or (dvv < best_mem):
                                best_mem = dvv
                    if (best_obs is not None) and (best_mem is not None):
                        if float(best_mem) + float(getattr(self, '_bt_dv_margin', 0.02)) < float(best_obs):
                            trigger_combined = (na_gate2 or (not getattr(self, '_bt_require_na_general', True)))
        except Exception:
            trigger_combined = False

        threshold_trigger = (bt_eval_value <= eff_threshold) or trigger_combined
        stagnation_stats = self._detect_backtrack_stagnation(); stagnation_trigger = stagnation_stats is not None
        can_fire = (self._last_backtrack_step is None) or ((self.step_count - self._last_backtrack_step) >= self.backtrack_cooldown)
        if can_fire and (threshold_trigger or stagnation_trigger):
            reason = ('combined_rank' if trigger_combined else ('threshold' if threshold_trigger else 'stagnation'))
            self.emit_event(self.EventType.BACKTRACK_TRIGGER, self._augment_backtrack_payload({'score': bt_eval_value,'base_score': value,'reason': reason,'effective_threshold': eff_threshold,'dynamic_bt_threshold': dynamic_bt_threshold,'stagnation': stagnation_stats,'passive': self.passive_heuristics}))
            if not self.passive_heuristics:
                self._last_backtrack_step = self.step_count
                bt_target = self._select_backtrack_target()
                if bt_target and bt_target != self.current_pos:
                    plan = self._plan_path_to(bt_target)
                    if plan and len(plan) > 1:
                        self.emit_event(self.EventType.BACKTRACK_PLAN, {
                            'target': bt_target,
                            'length': len(plan),
                            'plan': plan,
                            'reason': reason,
                            'path_mode': getattr(self, '_bt_path_mode', None)
                        })
                        self._bt_target_current = bt_target
                        self._pending_backtrack_plan = plan[1:]

    def _handle_stagnation(self, episodes: Dict[str, Any]) -> None:
        self._recent_positions.append(self.current_pos)
        if len(self._recent_positions) > self._stagnation_window:
            self._recent_positions.pop(0)
        if len(self._recent_positions) == self._stagnation_window and len(set(self._recent_positions)) == 1:
            open_dirs = [d for d, ep in episodes.items() if not getattr(ep, 'is_wall', False)]
            if open_dirs:
                import random
                if not self.passive_heuristics:
                    fb = random.choice(open_dirs)
                    if self._move(fb):
                        self.emit_event(self.EventType.FALLBACK_MOVE, fb)
                        self._stagnation_resets += 1
            else:
                self.emit_event(self.EventType.FALLBACK_FAILED, 'no_open_dirs')

    def _move(self, direction: str) -> bool:
        if not self.episode_manager.move(self.current_pos, direction):
            return False
        delta = {'N': (0, -1), 'S': (0, 1), 'E': (1, 0), 'W': (-1, 0)}
        dx, dy = delta[direction]
        nx_, ny_ = self.current_pos[0] + dx, self.current_pos[1] + dy
        if 0 <= nx_ < self.w and 0 <= ny_ < self.h and self.maze[ny_, nx_] == 0:
            self.current_pos = (nx_, ny_)
            is_new = self.current_pos not in self.path
            self.path.append(self.current_pos)
            if is_new:
                self._new_cells += 1
            else:
                self._revisit_moves += 1
            # Stage0 passive transition observer hook
            try:
                self._transition_move_hooks += 1
                obs = getattr(self, '_transition_observer', None)
                if obs is not None:
                    obs.on_step()
                    self._transition_on_step_calls += 1
            except Exception:
                pass
            # Macro target planner passive hook
            try:
                mtp = getattr(self, '_macro_target_planner', None)
                if mtp is not None:
                    mtp.on_step()
            except Exception:
                pass
            return True
        return False

    def _plan_path_to(self, target: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Plan a route to target according to configured path mode.

        Modes:
          - memory_graph: BFS on episode-graph collapsed to position-level
          - visited: BFS on maze restricted to visited cells only
          - maze: BFS on full maze open cells (fallback)
        """
        from collections import deque
        # Strict memory-only planning flag: when enabled, never fall back to visited/full-maze BFS
        try:
            _strict_mem = os.environ.get('MAZE_BT_STRICT_MEMORY', '1').strip() not in ("0","false","False","")
        except Exception:
            _strict_mem = True
        if self.current_pos == target:
            return [self.current_pos]

        def bfs_on_maze(allow_unvisited: bool) -> Optional[List[Tuple[int,int]]]:
            q = deque([self.current_pos])
            parent: Dict[Tuple[int,int], Optional[Tuple[int,int]]] = {self.current_pos: None}
            dirs = [(0,-1),(0,1),(1,0),(-1,0)]
            visited_cells = set(self.path)
            while q:
                p = q.popleft()
                if p == target:
                    break
                x, y = p
                for dx, dy in dirs:
                    nx_, ny_ = x+dx, y+dy
                    np_ = (nx_, ny_)
                    if not (0 <= nx_ < self.w and 0 <= ny_ < self.h):
                        continue
                    if self.maze[ny_, nx_] != 0:
                        continue
                    if not allow_unvisited and np_ not in visited_cells:
                        continue
                    if np_ in parent:
                        continue
                    parent[np_] = p
                    q.append(np_)
            if target not in parent:
                return None
            rev = []
            cur = target
            while cur is not None:
                rev.append(cur)
                cur = parent[cur]
            return list(reversed(rev))

        def bfs_on_memory_positions() -> Optional[List[Tuple[int,int]]]:
            """BFS on remembered trajectory graph only (edges = actually traversed steps).

            Builds an undirected position graph from the recorded path self.path
            by connecting successive positions. This avoids using the global maze
            and ignores non-physical memory edges (e.g., geDIG/query links).
            """
            try:
                if not self.path:
                    return None
                # Build adjacency from actual traversed steps
                g_pos: Dict[Tuple[int,int], set[Tuple[int,int]]] = {}
                def add_edge(pa: Tuple[int,int], pb: Tuple[int,int]) -> None:
                    if pa == pb:
                        return
                    g_pos.setdefault(pa, set()).add(pb)
                    g_pos.setdefault(pb, set()).add(pa)
                for i in range(1, len(self.path)):
                    a = self.path[i-1]; b = self.path[i]
                    add_edge(a, b)
                if self.current_pos not in g_pos or target not in g_pos:
                    return None
                # BFS
                from collections import deque as _dq
                q = _dq([self.current_pos])
                parent: Dict[Tuple[int,int], Optional[Tuple[int,int]]] = {self.current_pos: None}
                while q:
                    p = q.popleft()
                    if p == target:
                        break
                    for nb in g_pos.get(p, ()):  # neighbors are positions from memory trajectory
                        if nb in parent:
                            continue
                        parent[nb] = p
                        q.append(nb)
                if target not in parent:
                    return None
                rev: list[Tuple[int,int]] = []
                cur = target
                while cur is not None:
                    rev.append(cur)
                    cur = parent[cur]
                return list(reversed(rev))
            except Exception:
                return None

        def bfs_on_episode_graph_positions() -> Optional[List[Tuple[int,int]]]:
            """BFS on episode wiring graph collapsed to positions (uses memory edges)."""
            try:
                g = self.graph_manager.graph
                if g.number_of_nodes() == 0:
                    return None
                # Build position-level graph from episode graph
                g_pos: Dict[Tuple[int,int], set[Tuple[int,int]]] = {}
                def add_edge(pa: Tuple[int,int], pb: Tuple[int,int]) -> None:
                    if pa == pb:
                        return
                    g_pos.setdefault(pa, set()).add(pb)
                    g_pos.setdefault(pb, set()).add(pa)
                node_pos: Dict[Any, Tuple[int,int]] = {}
                for n, data in g.nodes(data=True):
                    try:
                        pos = data.get('position') if isinstance(data, dict) else None
                        if pos is not None:
                            node_pos[n] = (int(pos[0]), int(pos[1]))
                    except Exception:
                        continue
                for a, b in g.edges():
                    pa = node_pos.get(a); pb = node_pos.get(b)
                    if pa is None or pb is None:
                        continue
                    add_edge(pa, pb)
                if self.current_pos not in g_pos or target not in g_pos:
                    return None
                from collections import deque as _dq
                q = _dq([self.current_pos])
                parent: Dict[Tuple[int,int], Optional[Tuple[int,int]]] = {self.current_pos: None}
                while q:
                    p = q.popleft()
                    if p == target:
                        break
                    for nb in g_pos.get(p, ()):  # neighbors from memory edges
                        if nb in parent:
                            continue
                        parent[nb] = p
                        q.append(nb)
                if target not in parent:
                    return None
                rev: list[Tuple[int,int]] = []
                cur = target
                while cur is not None:
                    rev.append(cur)
                    cur = parent[cur]
                return list(reversed(rev))
            except Exception:
                return None

        def bfs_on_memory_adjacent() -> Optional[List[Tuple[int,int]]]:
            try:
                eps = list(self.episode_manager.episodes.values())
                if not eps:
                    return None
                dir_map = {'N': (0,-1), 'S': (0,1), 'E': (1,0), 'W': (-1,0)}
                g_adj: Dict[Tuple[int,int], set[Tuple[int,int]]] = {}
                def add_edge(a: Tuple[int,int], b: Tuple[int,int]) -> None:
                    if a == b:
                        return
                    g_adj.setdefault(a, set()).add(b)
                    g_adj.setdefault(b, set()).add(a)
                for ep in eps:
                    try:
                        if getattr(ep, 'is_wall', False):
                            continue
                        dx, dy = dir_map.get(ep.direction, (0,0))
                        a = tuple(ep.position)
                        b = (int(ep.position[0])+dx, int(ep.position[1])+dy)
                        add_edge(a, b)
                    except Exception:
                        continue
                if (self.current_pos not in g_adj) or (target not in g_adj):
                    return None
                from collections import deque as _dq
                q = _dq([self.current_pos])
                parent: Dict[Tuple[int,int], Optional[Tuple[int,int]]] = {self.current_pos: None}
                while q:
                    p = q.popleft()
                    if p == target:
                        break
                    for nb in g_adj.get(p, ()):  # memory-adjacent neighbors only
                        if nb in parent:
                            continue
                        parent[nb] = p
                        q.append(nb)
                if target not in parent:
                    return None
                rev: list[Tuple[int,int]] = []
                cur = target
                while cur is not None:
                    rev.append(cur)
                    cur = parent[cur]
                return list(reversed(rev))
            except Exception:
                return None

        mode = getattr(self, '_bt_path_mode', 'memory_graph')
        plan = None
        if mode == 'memory_adjacent':
            plan = bfs_on_memory_adjacent()
            if plan is None:
                # Memory-only fallbacks
                plan = bfs_on_memory_positions() or bfs_on_episode_graph_positions()
                if not _strict_mem and plan is None:
                    # legacy fallbacks (visited -> maze)
                    plan = bfs_on_maze(allow_unvisited=False) or bfs_on_maze(allow_unvisited=True)
        elif mode == 'memory_graph':
            plan = bfs_on_memory_positions()
            if plan is None:
                # Memory-only secondary
                plan = bfs_on_memory_adjacent() or bfs_on_episode_graph_positions()
                if not _strict_mem and plan is None:
                    plan = bfs_on_maze(allow_unvisited=False) or bfs_on_maze(allow_unvisited=True)
        elif mode == 'episode_graph':
            plan = bfs_on_episode_graph_positions()
            if plan is None:
                plan = bfs_on_memory_positions() or bfs_on_memory_adjacent()
                if not _strict_mem and plan is None:
                    plan = bfs_on_maze(allow_unvisited=False) or bfs_on_maze(allow_unvisited=True)
        elif mode == 'visited':
            # Respect strict memory mode: do not plan via maze when strict
            if not _strict_mem:
                plan = bfs_on_maze(allow_unvisited=False) or bfs_on_maze(allow_unvisited=True)
            else:
                plan = None
        else:  # 'maze'
            if not _strict_mem:
                plan = bfs_on_maze(allow_unvisited=True)
            else:
                plan = None
        return plan

    def _direction_towards(self, next_pos: Tuple[int, int]) -> Optional[str]:
        cx, cy = self.current_pos; nx_, ny_ = next_pos
        dx, dy = nx_ - cx, ny_ - cy
        return {(0,-1):'N',(0,1):'S',(1,0):'E',(-1,0):'W'}.get((dx, dy))

    # --- Backtrack target selection ---
    def _select_backtrack_target(self) -> Optional[Tuple[int,int]]:
        """Choose a backtrack anchor depending on configured strategy/policy.

        - If strategy=='semantic' (deprecated), follow semantic path (kept for compatibility; forced-off in __init__).
        - Otherwise use target policy:
            * 'gedig': rank unfinished branch entries by F = w1 * travel_cost - kT * ig_gain
              - travel_cost: path length from current to entry on memory graph
              - ig_gain: number of unvisited open neighbors at the entry (proxy for information gain)
            * 'heuristic': fall back to nearest unfinished branch entry (Manhattan)
        """
        if self.backtrack_target_strategy != 'semantic':
            if self._bt_target_policy in ('gedig','gedig_l1'):
                try:
                    return self._select_backtrack_target_gedig_l1()
                except Exception:
                    # Fallback to legacy heuristic on any error
                    try:
                        return self.backtrack_detector.suggest_backtrack_target(self.current_pos, set(self.path))
                    except Exception:
                        return None
            elif self._bt_target_policy in ('gedig_branches','gedig_legacy'):
                try:
                    return self._select_backtrack_target_gedig()
                except Exception:
                    try:
                        return self.backtrack_detector.suggest_backtrack_target(self.current_pos, set(self.path))
                    except Exception:
                        return None
            else:
                try:
                    return self.backtrack_detector.suggest_backtrack_target(self.current_pos, set(self.path))
                except Exception:
                    return None
        # Optional: index-only target selection (pure vector ranking, no frontier)
        try:
            _src = os.environ.get('MAZE_BT_SEM_SOURCE', 'frontier').strip().lower()
        except Exception:
            _src = 'frontier'
        if _src in ('index','idx','vector','l1'):
            try:
                if self._vector_index is None:
                    raise RuntimeError('vector_index unavailable')
                # Build query consistent with index mode (weighted vs raw)
                q = self.decision_engine.create_query(self.current_pos, prefer_unexplored=True)
                # Optional: BT-specific weights (override L1 weights for BT distance)
                bt_w = None
                try:
                    _s = os.environ.get('MAZE_BT_SEM_WEIGHTS', '')
                    if _s:
                        parts = [float(x) for x in _s.split(',')]
                        if len(parts) == 8:
                            bt_w = np.array(parts, dtype=float)
                except Exception:
                    bt_w = None
                try:
                    if getattr(self, '_index_use_weighted', True) and (bt_w is None):
                        q_search = self.vector_processor.apply_weights(q, self.decision_engine.weights)
                    else:
                        q_search = q
                except Exception:
                    q_search = q
                try:
                    topk = int(os.environ.get('MAZE_BT_SEM_IDX_TOPK', '64'))
                except Exception:
                    topk = 64
                try:
                    visited_only = os.environ.get('MAZE_BT_SEM_IDX_VISITED_ONLY', '1').strip() not in ("0","false","False","")
                except Exception:
                    visited_only = True
                try:
                    allow_walls = os.environ.get('MAZE_BT_SEM_IDX_ALLOW_WALLS', '0').strip() not in ("0","false","False","")
                except Exception:
                    allow_walls = False
                try:
                    spatial_gate = int(os.environ.get('MAZE_BT_SEM_SPATIAL_GATE', os.environ.get('MAZE_SPATIAL_GATE', '0')))
                except Exception:
                    spatial_gate = 0
                try:
                    require_frontier = os.environ.get('MAZE_BT_SEM_IDX_REQUIRE_FRONTIER', '0').strip() not in ("0","false","False","")
                except Exception:
                    require_frontier = False
                res = self._vector_index.search(q_search, max(1, topk))
                visited_set = set(self.path)
                # Optional: build frontier positions (visited that still have unvisited neighbors)
                frontier_pos = None
                if require_frontier:
                    frontier_pos = set()
                    for ep in self.episode_manager.episodes.values():
                        try:
                            if not getattr(ep, 'is_wall', False) and getattr(ep, 'visit_count', 0) == 0:
                                frontier_pos.add(tuple(ep.position))
                        except Exception:
                            continue
                # Deduplicate by position, keep best (lowest) distance
                best_by_pos: Dict[Tuple[int,int], Tuple[float,int]] = {}
                for eid, dist in (res or []):
                    ep = self.episode_manager.episodes_by_id.get(int(eid))
                    if not ep:
                        continue
                    if getattr(ep, 'is_wall', False) and not allow_walls:
                        continue
                    pos = tuple(ep.position)
                    if visited_only and (pos not in visited_set):
                        continue
                    if require_frontier and (frontier_pos is not None) and (pos not in frontier_pos):
                        continue
                    if spatial_gate and (abs(pos[0]-self.current_pos[0]) + abs(pos[1]-self.current_pos[1]) > spatial_gate):
                        continue
                    d = float(dist) if isinstance(dist, (int, float)) else float('inf')
                    # If BT-specific weights provided, recompute distance in that metric
                    if bt_w is not None:
                        try:
                            q_bt = (np.asarray(q, dtype=float) * bt_w)
                            v_bt = (np.asarray(ep.vector, dtype=float) * bt_w)
                            d = float(np.linalg.norm(q_bt - v_bt))
                        except Exception:
                            pass
                    prev = best_by_pos.get(pos)
                    if (prev is None) or (d < prev[0]):
                        best_by_pos[pos] = (d, int(eid))
                if best_by_pos:
                    # Choose nearest by index distance
                    chosen_pos = min(best_by_pos.items(), key=lambda t: t[1][0])[0]
                    return (int(chosen_pos[0]), int(chosen_pos[1]))
            except Exception:
                # Fallback to frontier-based selection
                pass
        # Build frontier
        try:
            frontier: list[Tuple[int,int]] = []
            if os.environ.get('MAZE_FRONTIER_FROM_MEMORY', '1').strip() not in ("0","false","False",""):
                # Pure memory-based frontier: positions that have at least one non-wall, unvisited episode
                pos_has_unvisited: Dict[Tuple[int,int], bool] = {}
                for ep in self.episode_manager.episodes.values():
                    try:
                        if not getattr(ep,'is_wall',False) and getattr(ep,'visit_count',0) == 0:
                            pos_has_unvisited[ep.position] = True
                    except Exception:
                        continue
                frontier = list(pos_has_unvisited.keys())
            else:
                # Legacy frontier: visited cells that have an open, unvisited neighbor in the maze
                visited_set = set(self.path)
                dirs = [(-1,0),(1,0),(0,-1),(0,1)]
                for cell in visited_set:
                    x,y = cell; has_unvisited = False
                    for dx, dy in dirs:
                        nx_, ny_ = x+dx, y+dy
                        if 0 <= nx_ < self.w and 0 <= ny_ < self.h and self.maze[ny_, nx_] == 0 and (nx_, ny_) not in visited_set:
                            has_unvisited = True; break
                    if has_unvisited:
                        frontier.append(cell)
            if not frontier:
                return self.backtrack_detector.suggest_backtrack_target(self.current_pos, set(self.path))
            # Prepare current query embedding (raw and weighted)
            query_vec = self.decision_engine.create_query(self.current_pos, prefer_unexplored=True)
            q_raw = query_vec
            try:
                q_w = self.vector_processor.apply_weights(query_vec, self.decision_engine.weights)
            except Exception:
                q_w = q_raw
            # Representative vector per frontier cell
            # rep_mode: 'recent' (default legacy) or 'best' (pick episode with minimal distance to current query)
            try:
                rep_mode = os.environ.get('MAZE_BT_SEM_POS_REP', 'recent').strip().lower()
            except Exception:
                rep_mode = 'recent'
            # distance metric mode for semantic selection
            try:
                dist_mode = os.environ.get('MAZE_BT_SEM_DIST', 'pos').strip().lower()
            except Exception:
                dist_mode = 'pos'
            pos_to_vec_raw: Dict[Tuple[int,int], np.ndarray] = {}
            pos_to_vec_w: Dict[Tuple[int,int], np.ndarray] = {}
            pos_to_ts: Dict[Tuple[int,int], int] = {}
            pos_to_best: Dict[Tuple[int,int], float] = {}
            for ep in self.episode_manager.episodes.values():
                try:
                    if ep.position not in frontier or getattr(ep, 'is_wall', False):
                        continue
                    # candidate raw/weighted vectors for this episode
                    v_raw = ep.vector
                    try:
                        v_w = ep.get_weighted_vector(self.decision_engine.weights, self._weights_version, self.vector_processor.apply_weights)
                    except Exception:
                        v_w = v_raw
                    if rep_mode != 'best' or (dist_mode in ('pos','manhattan','pos_l2','pos_euclid')):
                        # Legacy: most recent non-wall episode
                        ts = int(getattr(ep, 'timestamp', 0))
                        if (ep.position not in pos_to_ts) or (ts > pos_to_ts[ep.position]):
                            pos_to_vec_raw[ep.position] = v_raw
                            pos_to_vec_w[ep.position] = v_w
                            pos_to_ts[ep.position] = ts
                    else:
                        # Select episode at this pos minimizing distance to query
                        try:
                            if dist_mode in ('l2','raw_l2'):
                                dval = float(np.linalg.norm(q_raw - v_raw))
                            else:  # 'l2_w' and others default to weighted L2
                                dval = float(np.linalg.norm(q_w - v_w))
                        except Exception:
                            dval = float('inf')
                        prev = pos_to_best.get(ep.position)
                        if (prev is None) or (dval < prev):
                            pos_to_best[ep.position] = dval
                            pos_to_vec_raw[ep.position] = v_raw
                            pos_to_vec_w[ep.position] = v_w
                except Exception:
                    continue
            # Score frontier
            scores: list[tuple[float, Tuple[int,int], float, float, int]] = []
            # recency map (first occurrence index) lower index = older
            first_index: Dict[Tuple[int,int], int] = {}
            for idx, p in enumerate(self.path):
                if p not in first_index:
                    first_index[p] = idx
            cx, cy = self.current_pos
            # Distance policy for nearest target selection (dist_mode obtained above)
            # Optional: BT-specific weights for distance
            bt_w_arr = None
            try:
                _ws = os.environ.get('MAZE_BT_SEM_WEIGHTS', '')
                if _ws:
                    parts = [float(x) for x in _ws.split(',')]
                    if len(parts) == 8:
                        bt_w_arr = np.array(parts, dtype=float)
            except Exception:
                bt_w_arr = None
            # Diagnostics: track nearest by metric and Manhattan
            best_near = (float('inf'), None)  # (metric, pos)
            best_mh = (float('inf'), None)  # (manhattan, pos)
            for cell in frontier:
                dist = abs(cell[0]-cx)+abs(cell[1]-cy)
                rec = first_index.get(cell, 0)
                vec_r = pos_to_vec_raw.get(cell)
                vec_w = pos_to_vec_w.get(cell)
                sim_penalty = 0.0
                if (vec_r is not None) or (vec_w is not None):
                    try:
                        # Similarity for composite score uses weighted by default
                        if dist_mode in ('l2','raw_l2'):
                            base_vec = vec_r
                            sim = float(np.linalg.norm(q_raw - base_vec))
                        elif bt_w_arr is not None and (vec_r is not None):
                            base_vec = vec_r
                            sim = float(np.linalg.norm((np.asarray(q_raw)*bt_w_arr) - (np.asarray(base_vec)*bt_w_arr)))
                        else:
                            base_vec = vec_w if vec_w is not None else vec_r
                            sim = float(np.linalg.norm((q_w if base_vec is vec_w else q_raw) - base_vec))
                        sim_penalty = sim  # encourage dissimilar (possible new region gateway)
                    except Exception:
                        sim_penalty = 0.0
                if dist < best_mh[0]:
                    best_mh = (dist, cell)
                # Nearest metric selection
                try:
                    if dist_mode in ('pos','manhattan'):
                        metric = float(dist)
                    elif dist_mode in ('pos_l2','pos_euclid'):
                        dx = float(cell[0]-cx); dy = float(cell[1]-cy)
                        metric = float((dx*dx + dy*dy) ** 0.5)
                    elif dist_mode in ('l2','raw_l2'):
                        metric = float(np.linalg.norm(q_raw - vec_r)) if vec_r is not None else float('inf')
                    else:  # 'l2_w'
                        if bt_w_arr is not None and (vec_r is not None):
                            metric = float(np.linalg.norm((np.asarray(q_raw)*bt_w_arr) - (np.asarray(vec_r)*bt_w_arr)))
                        else:
                            metric = float(np.linalg.norm(q_w - vec_w)) if vec_w is not None else float('inf')
                except Exception:
                    metric = float(dist)
                if metric < best_near[0]:
                    best_near = (metric, cell)
                # Composite: prioritize distance & dissimilarity, modest boost for older (smaller rec)
                score = 1.2*dist + 1.0*sim_penalty + 0.1*(-rec)
                scores.append((score, cell, dist, sim_penalty, rec))
            if not scores:
                return self.backtrack_detector.suggest_backtrack_target(self.current_pos, set(self.path))
            scores.sort(key=lambda x: x[0], reverse=True)
            chosen = best_near[1] if getattr(self, '_bt_sem_force_nearest_l2', False) and best_near[1] is not None else (scores[0][1] if scores else None)
            # Keep small diagnostics for event payload
            try:
                self._last_semantic_candidates = {
                    'policy': ('force_nearest_metric' if getattr(self, '_bt_sem_force_nearest_l2', False) else 'score_max'),
                    'metric': dist_mode,
                    'chosen': {'pos': chosen, 'score': (float(scores[0][0]) if scores else None), 'manhattan': (float(scores[0][2]) if scores else None), 'l2': (float(scores[0][3]) if scores else None)},
                    'nearest_metric': {'pos': best_near[1], 'value': float(best_near[0]) if best_near[1] is not None else None},
                    'nearest_manhattan': {'pos': best_mh[1], 'manhattan': float(best_mh[0]) if best_mh[1] is not None else None},
                    'k': min(10, len(scores)),
                    'topk': [
                        {'pos': s[1], 'score': float(s[0]), 'manhattan': float(s[2]), 'l2': float(s[3]), 'rec': int(s[4])}
                        for s in scores[:10]
                    ]
                }
            except Exception:
                self._last_semantic_candidates = None
            return chosen
        except Exception:
            return None

    def _select_backtrack_target_gedig(self) -> Optional[Tuple[int,int]]:
        """Rank unfinished branch entries by a geDIG-like proxy and return the best.

        Proxy objective (per candidate entry t):
          F(t) = w1 * travel_cost(t) - kT * ig_gain(t)
          travel_cost = shortest path length from current_pos to t on memory graph
          ig_gain = count of unvisited open neighbors around t (structural frontier exposure)

        Keeps a small ranking snapshot in self._last_bt_ranking for export.
        """
        # Collect unfinished branch entries
        try:
            branch_points = list(getattr(self.backtrack_detector, 'branch_points', set()))
            completed = {b.get('entry') for b in getattr(self.backtrack_detector, 'completed_branches', []) if isinstance(b, dict)}
        except Exception:
            branch_points = []
            completed = set()
        cand = [p for p in branch_points if p not in completed]
        # Limit candidate count for speed (nearest by Manhattan first)
        cx, cy = self.current_pos
        cand.sort(key=lambda p: abs(p[0]-cx)+abs(p[1]-cy))
        if self._bt_cand_max and len(cand) > int(self._bt_cand_max):
            cand = cand[:int(self._bt_cand_max)]
        if not cand:
            self._last_bt_ranking = None
            return None
        visited = set(self.path)
        def open_unvisited_neighbors(pos: Tuple[int,int]) -> int:
            x, y = pos
            cnt = 0
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                nx_, ny_ = x+dx, y+dy
                if 0 <= nx_ < self.w and 0 <= ny_ < self.h and self.maze[ny_, nx_] == 0:
                    if (nx_, ny_) not in visited:
                        cnt += 1
            return cnt
        ranked: list[dict] = []
        best_pos: Optional[Tuple[int,int]] = None
        best_F: Optional[float] = None
        for t in cand:
            plan = self._plan_path_to(t)
            if not plan or len(plan) <= 1:
                continue
            travel = float(len(plan) - 1)
            ig = float(open_unvisited_neighbors(t))
            F = (self._bt_w1 * travel) - (self._bt_kT * ig)
            ranked.append({'pos': (int(t[0]), int(t[1])), 'plan_len': int(len(plan)), 'travel': travel, 'ig_gain': ig, 'F': float(F)})
            if (best_F is None) or (F < best_F):
                best_F = F; best_pos = t
        # Persist ranking for export (top 10)
        try:
            ranked.sort(key=lambda r: r['F'])
            self._last_bt_ranking = {
                'policy': 'gedig',
                'weights': {'w1': float(self._bt_w1), 'kT': float(self._bt_kT)},
                'topk': ranked[:10]
            }
        except Exception:
            self._last_bt_ranking = None
        return best_pos

    def _select_backtrack_target_gedig_l1(self) -> Optional[Tuple[int,int]]:
        """Rank memory candidates (L1近傍) by geDIG-like proxy and return the best.

        F(t) = w1 * travel_cost(t) - kT * ig_gain(t)
          - travel_cost: 記憶グラフ上のBFS距離（_plan_path_toで算出）
          - ig_gain: 入口tの未訪問開放近傍セル数
        候補は現在ステップの観測＋メモリからL1ノルム検索したTopK（τ以内）。
        """
        # L1候補の抽出設定
        try:
            topk = int(os.environ.get('MAZE_BT_L1_TOPK', os.environ.get('MAZE_L1_CAND_TOPK', '16')))
        except Exception:
            topk = 16
        try:
            tau = float(os.environ.get('MAZE_BT_L1_TAU', os.environ.get('MAZE_L1_NORM_TAU', '0.75')))
        except Exception:
            tau = 0.75
        # クエリベクトル（観測優先で未探索を好む）
        try:
            import numpy as _np
            q = self.decision_engine.create_query(self.current_pos, prefer_unexplored=True)
            # 検索対象：すべての既存エピソード
            eps = list(getattr(self.episode_manager, 'episodes_by_id', {}).values())
            ranked = []  # (dv, pos)
            for ep in eps:
                try:
                    ev = _np.asarray(ep.vector, dtype=float)
                    dv = float(_np.linalg.norm(_np.asarray(q, dtype=float) - ev))
                    if dv <= tau:
                        pos = tuple(getattr(ep, 'position'))  # type: ignore[arg-type]
                        ranked.append((dv, pos))
                except Exception:
                    continue
            ranked.sort(key=lambda t: t[0])
            if topk > 0 and len(ranked) > topk:
                ranked = ranked[:topk]
            # 重複位置を統合（最良dv残し）
            best_by_pos: dict[Tuple[int,int], float] = {}
            for dv, pos in ranked:
                if pos not in best_by_pos or dv < best_by_pos[pos]:
                    best_by_pos[pos] = dv
            cand_pos = list(best_by_pos.keys())
        except Exception:
            cand_pos = []
        if not cand_pos:
            self._last_bt_ranking = None
            return None
        visited = set(self.path)
        def open_unvisited_neighbors(pos: Tuple[int,int]) -> int:
            x, y = pos
            cnt = 0
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                nx_, ny_ = x+dx, y+dy
                if 0 <= nx_ < self.w and 0 <= ny_ < self.h and self.maze[ny_, nx_] == 0:
                    if (nx_, ny_) not in visited:
                        cnt += 1
            return cnt
        rankedF: list[dict] = []
        best_pos: Optional[Tuple[int,int]] = None
        best_F: Optional[float] = None
        for pos in cand_pos:
            plan = self._plan_path_to(pos)
            if not plan or len(plan) <= 1:
                continue
            travel = float(len(plan) - 1)
            ig = float(open_unvisited_neighbors(pos))
            F = (self._bt_w1 * travel) - (self._bt_kT * ig)
            rankedF.append({'pos': (int(pos[0]), int(pos[1])), 'plan_len': int(len(plan)), 'travel': travel, 'ig_gain': ig, 'F': float(F)})
            if (best_F is None) or (F < best_F):
                best_F = F; best_pos = pos
        try:
            rankedF.sort(key=lambda r: r['F'])
            self._last_bt_ranking = {'policy':'gedig_l1','weights':{'w1':float(self._bt_w1),'kT':float(self._bt_kT)}, 'topk': rankedF[:10]}
        except Exception:
            self._last_bt_ranking = None
        return best_pos

    def _wire_episodes(self) -> None:
        episodes = list(self.episode_manager.episodes.values())
        if not episodes:
            return
        if self.wiring_strategy == 'query':
            self._wire_episodes_query_based()
        elif self.wiring_strategy == 'gedig':
            self.graph_manager.wire_with_gedig(episodes, self.gedig_threshold)
        elif self.wiring_strategy == 'gedig_optimized':
            # Use optimized wiring for the optimized graph manager
            self.graph_manager._wire_with_gedig_optimized(episodes, self.gedig_threshold)
        elif self.wiring_strategy == 'loop_test':
            recent = [ep for ep in episodes if ep.episode_id >= (max(episodes, key=lambda e: e.episode_id).episode_id - 5)]
            ids = [ep.episode_id for ep in recent]
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    if not self.graph_manager.graph.has_edge(ids[i], ids[j]):
                        self.graph_manager.graph.add_edge(ids[i], ids[j])
                        self.graph_manager._log_edge_creation(ids[i], ids[j], 'loop_test')  # type: ignore[attr-defined]
        else:
            self.graph_manager.wire_edges(episodes, self.wiring_strategy)

    # --- Query-based wiring (non-cheating, similarity-centric) ---
    def _wire_episodes_query_based(self, query_vec: Optional[np.ndarray] = None) -> None:
        """Connect newly added episode(s) to a small set of nearest remembered episodes
        using the decision engine's embedding space, plus an optional forced edge from
        the immediately previous position (trajectory continuity).

        Design:
          1. Identify the newest episode (max episode_id).
          2. Optionally force-connect to previous path position's episode.
          3. Build a query vector at the new episode position.
          4. Rank existing (non-wall unless allowed) episodes by weighted distance.
          5. Connect top-K within a distance cap, logging creation reason 'query_sim'.
        """
        if not self.episode_manager.episodes:
            return
        episodes = list(self.episode_manager.episodes.values())
        newest = max(episodes, key=lambda e: e.episode_id)

        # Dynamic ANN upgrade: if no ANN yet and candidate count exceeds threshold
        if self._ann_backend is None and self._vector_index and type(self._vector_index).__name__ == 'InMemoryIndex':
            threshold = self._ann_upgrade_threshold  # configurable heuristic
            if len(self._vector_index) > threshold:
                try:
                    from indexes.hnsw_index import HNSWLibIndex  # type: ignore
                    ann = HNSWLibIndex(dim=8, max_elements=max(2000, len(self._vector_index)*2))
                    # Reinsert existing vectors into ANN
                    # We don't have direct access to raw vectors from the linear index; skip migration if attribute missing
                    store = getattr(self._vector_index, '_vectors', None)
                    if store:
                        import numpy as _np
                        ids = list(store.keys())
                        vecs = _np.vstack([store[i] for i in ids]).astype('float32')
                        ann.add(ids, vecs)
                    self._vector_index = ann
                    self._ann_backend = 'hnsw'
                    self.emit_event(self.EventType.ANN_UPGRADE, {'threshold': threshold, 'size': len(ids) if store else len(self._vector_index)})
                except Exception as e:
                    self.emit_event(self.EventType.ANN_UPGRADE_FAILED, str(e))

        g = self.graph_manager.graph

        # 1. Ensure node exists (already added earlier in step())
        if newest.episode_id not in g:
            self.graph_manager.add_episode_node(newest)

        # 2. Force-connect to previous path cell episode (trajectory continuity)
        if self.query_wiring_force_previous and len(self.path) > 1:
            prev_pos = self.path[-2]
            prev_ep = None
            # Find episode whose position matches prev_pos and has lowest distance in time (largest id among matches)
            for ep in reversed(sorted(episodes, key=lambda e: e.episode_id)):
                if ep.position == prev_pos:
                    prev_ep = ep
                    break
            if prev_ep and prev_ep.episode_id != newest.episode_id:
                if not g.has_edge(prev_ep.episode_id, newest.episode_id):
                    g.add_edge(prev_ep.episode_id, newest.episode_id)
                    # best-effort logging if helper exists
                    if hasattr(self.graph_manager, '_log_edge_creation'):
                        self.graph_manager._log_edge_creation(prev_ep.episode_id, newest.episode_id, 'trajectory')  # type: ignore[attr-defined]

        # 3. Build / reuse query vector (外部提供が無ければ生成)
        if query_vec is None:
            query_vec = self.decision_engine.create_query(newest.position, prefer_unexplored=True)

        # 4. Candidate selection (index if present else heap-based linear scan)
        # Choose representation for search
        if self._index_use_weighted:
            q_search = self.vector_processor.apply_weights(query_vec, self.decision_engine.weights)
        else:
            q_search = query_vec  # raw search space
        k = self.query_wiring_k
        ranked: List[Tuple[float, Any]] = []
        if self._vector_index is not None:
            try:
                search_res = self._vector_index.search(q_search, k + 5)  # slight oversample; filter below
                id_to_ep = {ep.episode_id: ep for ep in episodes}
                for eid, dist in search_res:
                    if eid == newest.episode_id:
                        continue
                    ep = id_to_ep.get(eid)
                    if not ep:
                        continue
                    if (not self.query_wiring_include_walls) and getattr(ep, 'is_wall', False):
                        continue
                    ranked.append((dist, ep))
                ranked.sort(key=lambda x: x[0])
                if len(ranked) > k:
                    ranked = ranked[:k]
            except Exception:
                ranked = []  # fallback to heap path
        # --- NN 退化検知用: 生距離統計 (index 利用時のみ) ---
        if ranked and self.nn_degeneracy_enabled and (self._vector_index is not None):
            try:
                dists = [float(d) for d,_ in ranked]
                if dists:
                    mean = sum(dists)/len(dists)
                    var = sum((x-mean)**2 for x in dists)/len(dists)
                    rng = max(dists)-min(dists)
                    unvisited = 0
                    visited_set = set(self.path)
                    for _d, ep in ranked:
                        if ep.position not in visited_set:
                            unvisited += 1
                    unvisited_ratio = unvisited / max(1,len(ranked))
                    no_growth_recent = (len(self.gedig_structural) >= self.nn_deg_min_window_no_growth and all(r['nodes_added']==0 for r in self.gedig_structural[-self.nn_deg_min_window_no_growth:]))
                    if getattr(self, 'nn_degeneracy_simple_mode', False):
                        degeneracy = (var < self.nn_deg_var_thresh)
                    else:
                        degeneracy = (var < self.nn_deg_var_thresh and rng < self.nn_deg_range_thresh and unvisited_ratio < self.nn_deg_min_unvisited_ratio and no_growth_recent)
                    self._nn_last_ranked_snapshot = {'step': self.step_count,'var': var,'range': rng,'unvisited_ratio': unvisited_ratio,'degeneracy': degeneracy,'no_growth_recent': no_growth_recent,'k': len(ranked)}
                    if degeneracy:
                        can_fire_bt = (self._last_backtrack_step is None) or ((self.step_count - self._last_backtrack_step) >= self.backtrack_cooldown)
                        if can_fire_bt:
                            if not self.passive_heuristics:
                                self._last_backtrack_step = self.step_count
                            self._nn_degeneracy_triggers += 1
                            self.emit_event(self.EventType.BACKTRACK_TRIGGER, self._augment_backtrack_payload({'score': (self.gedig_history[-1] if self.gedig_history else None), 'reason': 'nn_degeneracy', 'nn_stats': self._nn_last_ranked_snapshot, 'passive': self.passive_heuristics}))
                            if not self.passive_heuristics:
                                bt_target = self._select_backtrack_target()
                                if bt_target and bt_target != self.current_pos:
                                    plan = self._plan_path_to(bt_target)
                                    if plan and len(plan) > 1:
                                        self.emit_event(self.EventType.BACKTRACK_PLAN, {'target': bt_target,'length': len(plan),'plan': plan,'reason': 'nn_degeneracy'})
                                        self._pending_backtrack_plan = plan[1:]
            except Exception:
                pass
        if not ranked:  # fallback linear heap
            heap: list[tuple[float, int, Any]] = []
            counter = 0
            for ep in episodes:
                if ep.episode_id == newest.episode_id:
                    continue
                if (not self.query_wiring_include_walls) and getattr(ep, 'is_wall', False):
                    continue
                if self._index_use_weighted:
                    ep_vec = ep.get_weighted_vector(self.decision_engine.weights, self._weights_version, self.vector_processor.apply_weights)
                    q_vec_for_dist = q_search
                else:
                    ep_vec = ep.vector
                    q_vec_for_dist = q_search
                dist = float(np.linalg.norm(q_vec_for_dist - ep_vec))
                if len(heap) < k:
                    heapq.heappush(heap, (-dist, counter, ep))
                else:
                    if dist < -heap[0][0]:
                        heapq.heapreplace(heap, (-dist, counter, ep))
                counter += 1
            ranked = sorted([(-d, ep) for d, _, ep in heap], key=lambda x: x[0])

        # 5. Connect within distance threshold
        # 履歴用にランキング詳細を収集
        try:
            ranked_details = []
            for dist, ep in (ranked or []):
                try:
                    ranked_details.append({
                        'id': int(getattr(ep, 'episode_id', -1)),
                        'dv': float(dist),
                        'visit': int(getattr(ep, 'visit_count', 0)),
                        'obs': int(getattr(ep, 'observation_count', 0)),
                        'dir': str(getattr(ep, 'direction', None)),
                        'is_wall': bool(getattr(ep, 'is_wall', False)),
                        'pos': [int(ep.position[0]), int(ep.position[1])] if hasattr(ep, 'position') else None,
                    })
                except Exception:
                    continue
            # GraphManager に L1 履歴として残す
            try:
                self.graph_manager.l1_history.append({
                    'step': int(self.step_count),
                    'mode': 'global',
                    'source': int(getattr(newest, 'episode_id', -1)),
                    'candidates': ranked_details,
                    'meta': {
                        'k': int(self.query_wiring_k),
                        'dist_cap': float(self.query_wiring_max_dist),
                        'weighted': bool(self._index_use_weighted),
                        'ann_backend': getattr(self, '_ann_backend', None)
                    }
                })
            except Exception:
                pass
        except Exception:
            pass

        wired = 0; skipped_dist = 0; skipped_samples = []
        for dist, ep in ranked:
            if dist > self.query_wiring_max_dist:
                skipped_dist += 1
                if self.verbosity >= 3 and len(skipped_samples) < 5:
                    skipped_samples.append({'ep_id': ep.episode_id, 'dist': float(dist)})
                continue
            if not g.has_edge(ep.episode_id, newest.episode_id):
                g.add_edge(ep.episode_id, newest.episode_id, weight=dist, reason='query_sim')
                if hasattr(self.graph_manager, '_log_edge_creation'):
                    self.graph_manager._log_edge_creation(ep.episode_id, newest.episode_id, f'query_sim:{dist:.3f}')  # type: ignore[attr-defined]
                wired += 1
        if self.verbosity >= 2:
            self.emit_event(self.EventType.ANALYSIS, {
                'kind': 'wiring_summary',
                'newest_id': newest.episode_id,
                'ranked': len(ranked),
                'wired': wired,
                'skipped_distance': skipped_dist,
                'distance_cap': self.query_wiring_max_dist,
                'skipped_samples': skipped_samples if skipped_samples else None
            })

    def _maybe_save_snapshot(self) -> None:
        """条件付きスナップショット保存: growth 無し連続時はスキップ (オプション)"""
        g = self.graph_manager.graph
        nodes = g.number_of_nodes(); edges = g.number_of_edges()
        growth = False
        if self._last_snapshot_nodes is None:
            growth = True  # 初回
        else:
            if nodes > self._last_snapshot_nodes or edges > self._last_snapshot_edges:
                growth = True
        if self._snapshot_skip_idle and not growth:
            self._idle_growth_streak += 1
            # 閾値: 2連続 growth 無しならスキップ
            if self._idle_growth_streak >= 2:
                return
        else:
            self._idle_growth_streak = 0
        self.graph_manager.save_snapshot()
        self._last_snapshot_nodes = nodes
        self._last_snapshot_edges = edges

    def _analyze_branch_completion(self) -> None:
        if len(self.graph_manager.graph_history) < 2:
            return
        prev_g = self.graph_manager.graph_history[-2]
        curr_g = self.graph_manager.get_graph_snapshot()
        analysis = self.gedig_evaluator.analyze_graph_change(prev_g, curr_g)
        # Emit structured analysis event (Phase4 event schema consolidation)
        self.emit_event(self.EventType.ANALYSIS, analysis)

    def _log_event(self, event_type: Union[str, 'MazeNavigator.EventType'], message: Any) -> None:
        """Low-level append (deprecated direct use). Use emit_event instead."""
        etype = event_type.value if isinstance(event_type, MazeNavigator.EventType) else str(event_type)
        self.event_log.append({'step': self.step_count, 'type': etype, 'message': message, 'position': self.current_pos})

    # Unified event emitter (future: filtering / listeners)
    def emit_event(self, etype: Union['MazeNavigator.EventType', str], payload: Any) -> None:
        """Unified event emission.

        Accepts EventType or raw string (mapped if known). Ensures single append path.
        """
        if isinstance(etype, str):
            # Map to enum if matches value; fallback keep original string
            try:
                etype_enum = next((e for e in self.EventType if e.value == etype), None)
                if etype_enum is not None:
                    etype = etype_enum
            except Exception:
                pass
        self._log_event(etype, payload)
        # Optional live print for analysis events under high verbosity
        try:
            if not self._broken_pipe and isinstance(etype, MazeNavigator.EventType):
                v = getattr(self, 'verbosity', 0)
                # 1) Special-case: graph_growth を --log-graph-growth 指定時は verbosity>=1 でも出す
                if self._log_graph_growth and isinstance(payload, dict) and payload.get('kind') == 'graph_growth' and v >= 1:
                    self._safe_print(f"[ANALYSIS] step={self.step_count} kind=graph_growth data={payload}")
                # 2) 従来の高 verbosity (>=2) での詳細イベント出力
                if v >= 2 and etype in {self.EventType.ANALYSIS, self.EventType.BACKTRACK_PLAN, self.EventType.BACKTRACK_TRIGGER, self.EventType.DEAD_END}:
                    if isinstance(payload, dict) and payload.get('kind') in {'graph_growth','wiring_summary','visit_scale_auto','frontier_skip'}:
                        # graph_growth は既に 1) で出している可能性があるが冗長でも害は小さい; 重複抑止
                        if not (payload.get('kind') == 'graph_growth' and self._log_graph_growth and v == 1):
                            self._safe_print(f"[ANALYSIS] step={self.step_count} kind={payload.get('kind')} data={payload}")
                    elif etype != self.EventType.ANALYSIS:
                        self._safe_print(f"[{etype.value}] step={self.step_count} {payload}")
        except Exception:
            pass

    # --- Enrichment helper for BACKTRACK_TRIGGER payloads ---
    def _augment_backtrack_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Attach last local decision snapshot (4方向オプション確率など)
            ctx = getattr(self, '_last_decision_context', None)
            if ctx and 'local_options' not in payload:
                opts = ctx.get('options', {})
                local = []
                for d, info in opts.items():
                    try:
                        local.append({
                            'dir': d,
                            'prob': info.get('probability'),
                            'is_wall': info.get('is_wall'),
                            'visit_count': info.get('visit_count'),
                            'chosen': (d == ctx.get('dir')),
                            'pos': info.get('position') if isinstance(info, dict) else None
                        })
                    except Exception:
                        pass
                payload['local_options'] = local
            # Attach semantic candidate list if semantic strategy active
            if self.backtrack_target_strategy == 'semantic' and getattr(self, '_last_semantic_candidates', None) and 'semantic_candidates' not in payload:
                payload['semantic_candidates'] = self._last_semantic_candidates
        except Exception:
            pass
        return payload

    def _safe_print(self, *args, **kwargs) -> None:
        """Print wrapper that suppresses BrokenPipeError (e.g. when piped to head).

        Upon first BrokenPipeError it disables further verbose output by lowering verbosity.
        """
        if self._broken_pipe:
            return
        try:
            print(*args, **kwargs)
        except BrokenPipeError:
            # Suppress subsequent prints
            self._broken_pipe = True
            try:
                self.verbosity = 0  # disable future verbose prints
            except Exception:
                pass

    # --- Catalog maintenance ---
    def compact_eviction_catalog(self) -> bool:
        """Rewrite the eviction catalog file with only current in-memory LRU map.

        Returns True if compaction succeeded and file replaced.
        Captures size before/after and updates self._catalog_last_bytes.
        Safe no-op if persistence disabled.
        """
        path = getattr(self, '_evicted_catalog_path', None)
        if not path or not os.path.isfile(path):
            return False
        try:
            import json as _json, tempfile, shutil
            before_bytes = os.path.getsize(path)
            fd, tmp_path = tempfile.mkstemp(prefix='catalog_compact_', suffix='.jsonl')
            import os as _os
            try:
                with _os.fdopen(fd, 'w') as f:
                    for eid, rec in self._evicted_catalog.items():
                        out = dict(rec)
                        pos = out.get('position')
                        if isinstance(pos, tuple):
                            out['position'] = list(pos)
                        f.write(_json.dumps(out, ensure_ascii=False) + '\n')
            except Exception:
                try:
                    _os.unlink(tmp_path)
                except Exception:
                    pass
                raise
            shutil.move(tmp_path, path)
            after_bytes = os.path.getsize(path)
            self._catalog_last_bytes = after_bytes
            self.emit_event(self.EventType.CATALOG_COMPACT, {
                'before_bytes': before_bytes,
                'after_bytes': after_bytes,
                'records': len(self._evicted_catalog)
            })
            return True
        except Exception as e:
            self.emit_event(self.EventType.CATALOG_COMPACT_FAILED, str(e))
            return False

    def _reverse_trace_from_current(self, max_steps: int = 50) -> Dict[str, Any]:
        if not self.path:
            return {}
        reverse_positions: List[Tuple[int, int]] = []
        origin_pos: Optional[Tuple[int, int]] = None
        seen: set[Tuple[int, int]] = set()
        degree_cache: Dict[Tuple[int, int], int] = {}
        def get_degree(pos: Tuple[int, int]) -> int:
            if pos in degree_cache:
                return degree_cache[pos]
            deg = 0
            for nid, data in self.graph_manager.graph.nodes(data=True):
                if data.get('position') == pos:
                    deg = self.graph_manager.graph.degree[nid]
                    break
            degree_cache[pos] = deg
            return deg
        for p in reversed(self.path):
            if p not in seen:
                reverse_positions.append(p)
                seen.add(p)
            deg = get_degree(p)
            if deg >= 3 and len(reverse_positions) > 1:
                origin_pos = p
                break
            if len(reverse_positions) >= max_steps:
                origin_pos = p
                break
            if p == self.start_pos:
                origin_pos = p
                break
        if origin_pos is None and reverse_positions:
            origin_pos = reverse_positions[-1]
        return {
            'origin': origin_pos,
            'trace': reverse_positions,
            'length': len(reverse_positions),
            'origin_degree': get_degree(origin_pos) if origin_pos else None,
            'terminated': (
                (origin_pos == self.start_pos) or
                (len(reverse_positions) >= max_steps) or
                (get_degree(origin_pos) >= 3 if origin_pos else False)
            )
        }

    # ---- Backtrack 強化ヘルパ ----
    def _detect_backtrack_stagnation(self) -> Optional[Dict[str, Any]]:
        """低成長停滞検知: 直近 window でノード/エッジ増加ほぼ停止 + geDIG 微振幅。"""
        if len(self.gedig_structural) < self._backtrack_min_samples:
            return None
        W = self._backtrack_window
        window = self.gedig_structural[-W:]
        if len(window) < max(self._backtrack_min_samples, 6):
            return None
        nodes_added = sum(r['nodes_added'] for r in window)
        edges_added = sum(r['edges_added'] for r in window)
        scores = [r['value'] for r in window]
        if not scores:
            return None
        amplitude = (max(scores) - min(scores)) if scores else 0.0
        mean_score = float(np.mean(scores))
        if nodes_added <= 1 and edges_added <= 1 and amplitude < 0.003 and mean_score < 0.008:
            return {
                'window': W,
                'nodes_added': nodes_added,
                'edges_added': edges_added,
                'score_mean': mean_score,
                'score_amp': amplitude
            }
        return None

    # ---- Frontier Jump (novelty-based) ----
    def _maybe_frontier_jump(self) -> None:
        # Feature flag (default off for non-cheating, geDIG主導の実験整合性)
        try:
            _enable = os.environ.get('MAZE_FRONTIER_JUMP_ENABLE', '0').strip() not in ("0","false","False","")
        except Exception:
            _enable = False
        if not _enable:
            return
        if self._frontier_jump_window <= 10:
            return
        if self._last_frontier_jump_step is not None and (self.step_count - self._last_frontier_jump_step) < self._frontier_cooldown:
            return
        if self.step_count < self._frontier_jump_window:
            return
        recent_path = self.path[-self._frontier_jump_window:]
        recent_new = 0
        seen_local = set()
        # Count new cells within window (first occurrence inside window counts)
        for p in recent_path:
            if p not in seen_local:
                seen_local.add(p)
                recent_new += 1
        novelty = recent_new / max(1, self._frontier_jump_window)
        if novelty >= self._frontier_novelty_threshold:
            # Optional: emit trace for diagnostics at high verbosity
            if self.verbosity >= 2 and (self.step_count % 200 == 0):
                self.emit_event(self.EventType.ANALYSIS, {'kind':'frontier_skip','reason':'novelty_ok','novelty':novelty,'threshold':self._frontier_novelty_threshold})
            return
        # Build frontier: visited cells having at least one unvisited open neighbor
        visited_set = set(self.path)
        dirs = [(-1,0),(1,0),(0,-1),(0,1)]
        frontier: List[tuple[int,int]] = []
        for cell in visited_set:
            x, y = cell
            for dx, dy in dirs:
                nx_, ny_ = x+dx, y+dy
                if 0 <= nx_ < self.w and 0 <= ny_ < self.h and self.maze[ny_, nx_] == 0 and (nx_, ny_) not in visited_set:
                    frontier.append(cell)
                    break
        if not frontier:
            if self.verbosity >= 2:
                self.emit_event(self.EventType.ANALYSIS, {'kind':'frontier_skip','reason':'no_frontier','novelty':novelty})
            return
        # Choose frontier cell farthest (Manhattan) from current to diversify
        cx, cy = self.current_pos
        frontier.sort(key=lambda p: abs(p[0]-cx)+abs(p[1]-cy), reverse=True)
        target_anchor = frontier[0]
        plan = self._plan_path_to(target_anchor)
        if not plan or len(plan) <= 1:
            if self.verbosity >= 2:
                self.emit_event(self.EventType.ANALYSIS, {'kind':'frontier_skip','reason':'no_plan','target':target_anchor,'novelty':novelty})
            return
        # Ensure plan leads to anchor; anchor must still have unvisited neighbor
        self._pending_backtrack_plan = plan[1:]
        self._last_frontier_jump_step = self.step_count
        self.emit_event(self.EventType.BACKTRACK_PLAN, {
            'target': target_anchor,
            'length': len(plan),
            'plan': plan,
            'reason': 'frontier_jump',
            'novelty_recent': novelty,
            'window': self._frontier_jump_window,
            'threshold': self._frontier_novelty_threshold
        })

    # ---- Global Recall (過去分岐想起) ----
    def _maybe_global_recall(self, query_vec: Optional[np.ndarray]) -> None:
        """Recent geDIG 成長が低く backtrack 未発火時に過去分岐ターゲットを想起してプラン生成。

        条件 (最初の実装):
          - 最新 geDIG 値 <= recall_score_threshold
          - バックトラックプラン未保有
          - suggest_backtrack_target が現在位置と異なる座標を返す
        将来: 分岐残タスクスコアリング / 優先度キュー に拡張予定
        """
        if not self.gedig_history:
            return
        latest = self.gedig_history[-1]
        if latest > self.recall_score_threshold:
            return
        bt_target = self.backtrack_detector.suggest_backtrack_target(self.current_pos, set(self.path))
        if not bt_target or bt_target == self.current_pos:
            return
        plan = self._plan_path_to(bt_target)
        if plan and len(plan) > 1:
            self._pending_backtrack_plan = plan[1:]
            self.emit_event(self.EventType.BACKTRACK_PLAN, {
                'target': bt_target,
                'length': len(plan),
                'plan': plan,
                'reason': 'global_recall',
                'gedig_latest': latest,
                'threshold': self.recall_score_threshold
            })
