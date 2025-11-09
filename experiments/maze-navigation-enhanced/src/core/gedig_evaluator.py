"""Adapter: Experimental GeDIGEvaluator -> Core GeDIGCore 移行版

旧 experiment 用の簡易 GeDIGEvaluator を、本体 `GeDIGCore` に接続するアダプタ。

目的:
 1. 既存コード (maze_navigator / graph_manager / branch_detector) のインターフェイス互換性維持
 2. Core 側の構造/情報統合計測を利用しつつ、既存の閾値ロジック ("正: 探索 / 負: 短絡") の方向性を保つ
 3. 将来: 閾値再学習後に簡易スコア -> 本来の `result.structural_improvement` / `result.gedig_value` へ切替え可能

スコア設計 (暫定):
  - Core の `structural_improvement` は (近いほど 0 付近, 乖離で負方向) なので符号反転して拡張で正寄りに
  - `ig_value` (分散低下 = 統合) が大きい場合は情報凝集が進んだとみなし微小ペナルティ
  - 短絡検出は旧ヒューリスティック (密度急増 / 新ノード無しでエッジ大量追加 など) を継続

NOTE: 閾値は再キャリブレーション推奨。既存値 (例: backtrack_threshold=-0.2) はコア導入後に分布確認を。
"""

from __future__ import annotations

import os, sys
from typing import Dict, Any, Optional, Tuple
import networkx as nx
import numpy as np

# ルート src へのパス (実行ディレクトリが experiments 下でも本体を import 可能に)
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..', 'src'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_IMPORT_ERROR = None
# Prefer direct module file import to avoid heavy package __init__ (torch dependency)
try:
    import importlib.util as _ilu, types as _types, sys as _sys, os as _os
    _core_path = _os.path.join(_ROOT, 'insightspike', 'algorithms', 'gedig_core.py')
    if _os.path.exists(_core_path):
        # Prepare namespace package stubs to avoid importing real package __init__ (torch deps)
        if 'insightspike' not in _sys.modules:
            _pkg = _types.ModuleType('insightspike'); _pkg.__path__ = []  # type: ignore[attr-defined]
            _sys.modules['insightspike'] = _pkg
        if 'insightspike.algorithms' not in _sys.modules:
            _pkg2 = _types.ModuleType('insightspike.algorithms'); _pkg2.__path__ = []  # type: ignore[attr-defined]
            _sys.modules['insightspike.algorithms'] = _pkg2
        if 'insightspike.algorithms.core' not in _sys.modules:
            _pkg3 = _types.ModuleType('insightspike.algorithms.core'); _pkg3.__path__ = []  # type: ignore[attr-defined]
            _sys.modules['insightspike.algorithms.core'] = _pkg3
        # Preload metrics module under expected name
        _metrics_path = _os.path.join(_ROOT, 'insightspike', 'algorithms', 'core', 'metrics.py')
        if _os.path.exists(_metrics_path) and 'insightspike.algorithms.core.metrics' not in _sys.modules:
            _mspec = _ilu.spec_from_file_location('insightspike.algorithms.core.metrics', _metrics_path)
            if _mspec and _mspec.loader:
                _mmod = _ilu.module_from_spec(_mspec)
                _sys.modules['insightspike.algorithms.core.metrics'] = _mmod
                _mspec.loader.exec_module(_mmod)  # type: ignore[attr-defined]
        # Load under canonical module name to satisfy dataclass __module__ lookups
        _modname = 'insightspike.algorithms.gedig_core'
        _spec = _ilu.spec_from_file_location(_modname, _core_path)
        if _spec and _spec.loader:
            _mod = _ilu.module_from_spec(_spec)
            _sys.modules[_modname] = _mod
            _spec.loader.exec_module(_mod)  # type: ignore[attr-defined]
            GeDIGCore = getattr(_mod, 'GeDIGCore')  # type: ignore
            GeDIGResult = getattr(_mod, 'GeDIGResult')  # type: ignore
        else:
            raise ImportError('spec loader missing for gedig_core.py')
    else:
        raise ImportError('gedig_core.py not found')
except Exception as e_local:
    try:
        # Fallback: standard package import (may fail if torch is missing)
        from insightspike.algorithms.gedig_core import GeDIGCore, GeDIGResult  # type: ignore
    except Exception as e_pkg:
        _IMPORT_ERROR = e_pkg
        # Final fallback: lightweight shim without multihop
        GeDIGResult = None  # type: ignore
        class GeDIGCore:  # type: ignore
            """Torchなしフォールバック: 最低限の0-hopのみの実装（multihopなし）。"""
            def __init__(self, **_):
                self.decay_factor = 0.7
            def calculate(self, g_prev, g_now, **_):  # type: ignore
                n1, n2 = g_prev.number_of_nodes(), g_now.number_of_nodes()
                e1, e2 = g_prev.number_of_edges(), g_now.number_of_edges()
                dn = max(0, n2 - n1); de = max(0, e2 - e1)
                denom = (n1 + n2 + 1)
                structural_improvement = - (dn + 0.5 * de) / denom if denom > 0 else 0.0
                class _R:
                    def __init__(self, si: float):
                        self.structural_improvement = si
                        self.ig_value = 0.0
                        self.reward = si
                        self.gedig_value = si
                        self.hop_results = {}
                return _R(structural_improvement)


class GeDIGEvaluator:
    """Core ラッパ (旧インターフェイス互換)。

    Methods 保持:
      - calculate(g1,g2) -> float
      - calculate_multihop(g1,g2,max_hop) -> Dict[int,float]
      - detect_shortcut(g1,g2) -> bool
      - analyze_graph_change(g1,g2) -> Dict
    """

    def __init__(self,
                 multihop: bool | None = None,
                 max_hops: int = 3,
                 decay: float = 0.7,
                 use_core_reward: bool = False,
                 always_multihop: bool = False,
                 mode: str = 'core_raw',
                 scale: float = 25.0,
                 ig_weight: float = 0.1):
        if GeDIGCore is None:
            raise ImportError(f"GeDIGCore import failed (and no fallback): {_IMPORT_ERROR}")

        enable_multihop = multihop if multihop is not None else False
        self.always_multihop = always_multihop

        # --- Runtime flags (env) for local normalization and SP gain ---
        use_local_norm_flag = os.environ.get('MAZE_GEDIG_LOCAL_NORM', '0')
        use_sp_gain_flag = os.environ.get('MAZE_GEDIG_SP_GAIN', '0')
        self._sp_norm_mode = os.environ.get('MAZE_GEDIG_SP_MODE', 'relative')
        self._local_norm_mode = os.environ.get('MAZE_GEDIG_LOCAL_MODE', 'layer1')
        self._use_local_normalization = use_local_norm_flag.strip() not in ("0", "false", "False", "")
        self._use_multihop_sp_gain = use_sp_gain_flag.strip() not in ("0", "false", "False", "")
        try:
            self._sp_beta = float(os.environ.get('MAZE_GEDIG_SP_BETA', '0.2'))
        except Exception:
            self._sp_beta = 0.2

        if self.always_multihop:
            # 常時 multi-hop (hop0 + 追加 hop 計測を強制)
            self.core = GeDIGCore(
                enable_multihop=True,
                max_hops=max_hops,
                decay_factor=decay,
                adaptive_hops=False,
                use_refactored_reward=True,
                # pass flags
                use_local_normalization=self._use_local_normalization,
                local_norm_mode=self._local_norm_mode,
                use_multihop_sp_gain=self._use_multihop_sp_gain,
                sp_norm_mode=self._sp_norm_mode,
                sp_beta=self._sp_beta,
            )
            self._multihop_core = self.core  # type: ignore
        else:
            # hop0 軽量モード + 遅延 multihop
            self.core = GeDIGCore(
                enable_multihop=False,
                max_hops=0,
                decay_factor=decay,
                use_refactored_reward=True,
                adaptive_hops=True,
                # pass flags for 0-hop path
                use_local_normalization=self._use_local_normalization,
                local_norm_mode=self._local_norm_mode,
                use_multihop_sp_gain=self._use_multihop_sp_gain,
                sp_norm_mode=self._sp_norm_mode,
                sp_beta=self._sp_beta,
            )
            self._multihop_core: GeDIGCore | None = None  # type: ignore

        self._multihop_conf = {"max_hops": max_hops, "decay_factor": decay}
        self._requested_max_hops = max_hops
        self.use_core_reward = use_core_reward

        # --- モード / パラメタ ---
        if mode == 'legacy':
            raise ValueError("Experimental 'legacy' geDIG mode has been removed. Use mode='core_raw'.")
        self.mode = 'core_raw'
        self.scale = 1.0  # scale no longer applied (legacy removed)
        self.ig_weight = ig_weight

        # エスカレーション関連 (raw threshold: 非スケール空間)
        self.default_escalation_threshold_raw = -5e-4

        # hop1 近似は完全削除 -> フラグは安全ガードとして常に False
        self.allow_hop1_approx = False

    # -------- Threshold normalization --------
    def _resolve_escalation_threshold(self, threshold: Optional[float]) -> Tuple[float, float]:
        """入力閾値を正規化し (raw, scaled) を返す。legacy 廃止により scaled==raw。

        Parameters
        ----------
        threshold : Optional[float]
            呼び出し元指定の閾値 (None ならデフォルト)

        Returns
        -------
        (raw, scaled) : Tuple[float, float]
            現在は legacy 廃止に伴い scaled==raw
        """
        raw = threshold if threshold is not None else self.default_escalation_threshold_raw
        return raw, raw

    # -------- Basic scoring --------
    def _core_result(self, g1: nx.Graph, g2: nx.Graph, *, l1_candidates: Optional[int] = None) -> GeDIGResult:  # type: ignore
        try:
            from insightspike.algorithms.linkset_adapter import build_linkset_info as _build_ls  # type: ignore
        except Exception:
            _build_ls = None  # type: ignore
        ls = _build_ls(s_link=[], candidate_pool=[], decision={}, query_vector=None, base_mode="link") if _build_ls else None
        try:
            if ls is not None:
                return self.core.calculate(g_prev=g1, g_now=g2, l1_candidates=l1_candidates, linkset_info=ls)
            return self.core.calculate(g_prev=g1, g_now=g2, l1_candidates=l1_candidates)
        except TypeError:
            # Older cores without l1_candidates kwarg
            if ls is not None:
                return self.core.calculate(g_prev=g1, g_now=g2, linkset_info=ls)
            return self.core.calculate(g_prev=g1, g_now=g2)

    def calculate(self, g1: nx.Graph, g2: nx.Graph, *, l1_candidates: Optional[int] = None) -> float:
        """旧簡易 evaluator の符号規約に合わせたスコア。

        戻り値 (暫定):
          positive ≈ 探索進展 (構造拡張)
          negative ≈ 短絡/密度過多
        """
        if g1.number_of_nodes() == 0:
            return 0.5  # 初期ノード追加相当
        res = self._core_result(g1, g2, l1_candidates=l1_candidates)
        base = res.gedig_value if hasattr(res, 'gedig_value') else res.structural_improvement
        return float(base)

    # -------- Multihop (擬似互換) --------
    def calculate_multihop(self, g1: nx.Graph, g2: nx.Graph, max_hop: int = 10) -> Dict[int, float]:
        res = self._core_result(g1, g2)
        base = self.calculate(g1, g2)
        out: Dict[int, float] = {0: base}
        for hop in range(1, min(max_hop + 1, 11)):
            decay = self.core.decay_factor ** hop
            hop_res = res.hop_results.get(hop) if (res.hop_results and hop in res.hop_results) else None  # type: ignore[arg-type]
            if hop_res:
                # hop_res.gedig は core 定義 (構造 - 情報 or 乗算) のため簡易スコアへ再投影
                hop_struct = hop_res.ged  # normalized GED (類似性低下で +)
                hop_gain = -hop_struct  # 拡張で + に合わせる
                hop_score = hop_gain - 0.1 * max(0.0, hop_res.ig)
                out[hop] = float(np.clip(hop_score * self.scale * decay, -2.0, 2.0))
            else:
                out[hop] = float(base * decay)
        return out

    # -------- Escalation evaluation --------
    def evaluate_escalating(self,
                            g1: nx.Graph,
                            g2: nx.Graph,
                            escalation_threshold: float | None = None,
                            max_hops: int | None = None,
                            *,
                            l1_candidates: Optional[int] = None) -> Dict[str, Any]:
        """1-hop 基本評価 → 閾値以下なら multi-hop 深掘り + shortcut 拡張検出。

        Returns dict keys:
          score: 1-hop (scaled) スコア
          escalated: bool
          multihop: Optional[Dict[int,float]]
          shortcut: bool
          details: raw metrics
        """
        base_score = self.calculate(g1, g2, l1_candidates=l1_candidates)
        raw_core_payload = None
        try:
            core_res_parallel = self._core_result(g1, g2, l1_candidates=l1_candidates)
            raw_core_payload = {
                'structural_improvement': core_res_parallel.structural_improvement,
                'ig_value': core_res_parallel.ig_value,
                'reward': getattr(core_res_parallel, 'reward', None),
                'gedig_value': getattr(core_res_parallel, 'gedig_value', None)
            }
        except Exception:
            raw_core_payload = None
        raw_thr, scaled_thr = self._resolve_escalation_threshold(escalation_threshold)
        escalated = base_score < raw_thr
        threshold_used = raw_thr
        multihop_scores = None  # clipped projected scores
        multihop_raw = None     # raw (pre-scale / pre-clip) structural gains per hop
        multihop_missing = None
        hop1_approx = False
        shortcut = False
        gradient_flag = False
        density_flag = False
        edgeburst_flag = False
        path_shortening = None  # shortest path length reduction heuristic

        if escalated or self.always_multihop:
            if self._multihop_core is None and GeDIGCore is not None:
                self._multihop_core = GeDIGCore(enable_multihop=True,
                                                max_hops=self._multihop_conf['max_hops'],
                                                decay_factor=self._multihop_conf['decay_factor'],
                                                adaptive_hops=(not self.always_multihop),
                                                use_refactored_reward=True,
                                                # propagate flags to multihop evaluator
                                                use_local_normalization=self._use_local_normalization,
                                                local_norm_mode=self._local_norm_mode,
                                                use_multihop_sp_gain=self._use_multihop_sp_gain,
                                                sp_norm_mode=self._sp_norm_mode,
                                                sp_beta=self._sp_beta)
            # コア multi-hop 実行
            if self._multihop_core is not None:
                core_res = self._multihop_core.calculate(g_prev=g1, g_now=g2, l1_candidates=l1_candidates)
                # hop results を再投影 (raw + clipped)
                multihop_scores = {}
                multihop_raw = {}
                # lambda for IG from env (default 1.0)
                try:
                    _lam = float(os.environ.get('MAZE_GEDIG_LAMBDA', '1.0'))
                except Exception:
                    _lam = 1.0
                prev_val = base_score
                hop_items = sorted((core_res.hop_results or {}).items(), key=lambda kv: kv[0]) if core_res.hop_results else []
                # 近似生成は廃止 (hop_results が無ければ欠損扱い)
                missing = []
                for hop, hop_res in hop_items:
                    # Use core hop geDIG directly to preserve SP gain and lambda/IG mode
                    try:
                        raw_gain = float(hop_res.gedig)
                    except Exception:
                        # Fallback to simple projection if hop_res lacks gedig
                        raw_gain = (-float(getattr(hop_res, 'ged', 0.0))) - _lam * max(0.0, float(getattr(hop_res, 'ig', 0.0)))
                    multihop_raw[hop] = raw_gain
                    cur_val = raw_gain
                    multihop_scores[hop] = cur_val
                    if hop == 1:
                        # hop0->hop1 勾配急変判定
                        if abs(cur_val - base_score) > max(0.2 * abs(base_score) + 1e-3, 0.01):
                            gradient_flag = True
                    prev_val = cur_val
                # 欠損 hop を記録 (1..requested_max)
                if hop_items:
                    observed = {h for h,_ in hop_items}
                    for h in range(1, self._requested_max_hops+1):
                        if h not in observed:
                            missing.append(h)
                if missing:
                    multihop_missing = missing
                # Shortcut heuristics (拡張)
                n1, n2 = g1.number_of_nodes(), g2.number_of_nodes()
                e1, e2 = g1.number_of_edges(), g2.number_of_edges()
                dn = max(0, n2 - n1); de = max(0, e2 - e1)
                if dn <= 1 and de > 3:
                    edgeburst_flag = True
                # 簡易密度変化
                def _dens(g: nx.Graph) -> float:
                    n = g.number_of_nodes(); e = g.number_of_edges()
                    return 0.0 if n <= 1 else e / (n * (n - 1) / 2)
                if _dens(g2) - _dens(g1) > 0.1:
                    density_flag = True
                # 最短経路短縮 (平均全対でなく近傍サンプル) 指標: ランダム最大 50 ペア
                try:
                    import random
                    nodes = list(g2.nodes())
                    if len(nodes) >= 4:
                        sample_pairs = [tuple(random.sample(nodes, 2)) for _ in range(min(50, len(nodes)//2))]
                        shorten = []
                        for a,b in sample_pairs:
                            try:
                                d_before = nx.shortest_path_length(g1, a, b)
                                d_after = nx.shortest_path_length(g2, a, b)
                                if d_after < d_before:
                                    shorten.append(d_before - d_after)
                            except Exception:
                                continue
                        if shorten:
                            path_shortening = float(np.mean(shorten))
                except Exception:
                    pass
                shortcut = edgeburst_flag or (gradient_flag and (density_flag or de > 0)) or (path_shortening is not None and path_shortening >= 1.0)

        g0_value = float(base_score)
        gmin_value = None
        try:
            if multihop_raw:
                gmin_value = float(min(multihop_raw.values()))
        except Exception:
            gmin_value = None

        return {
            'score': base_score,
            'escalated': escalated,
            'multihop': multihop_scores,
            'multihop_raw': multihop_raw,
            'multihop_missing': multihop_missing,
            'hop1_approx_fallback': False,
            'shortcut': shortcut,
            'raw_core': raw_core_payload,
            'details': {
                'g0': g0_value,
                'gmin': gmin_value,
                'gradient_flag': gradient_flag,
                'density_flag': density_flag,
                'edgeburst_flag': edgeburst_flag,
                'threshold_used': threshold_used,
                'path_shortening_mean': path_shortening,
                'escalation_raw_threshold': raw_thr,
                'escalation_scaled_threshold': scaled_thr,
                'mode': self.mode,
                'hop1_approx_deprecated': False
            }
        }

    # -------- Shortcut detection (従来ヒューリスティック継続) --------
    def detect_shortcut(self, g1: nx.Graph, g2: nx.Graph) -> bool:
        n1, n2 = g1.number_of_nodes(), g2.number_of_nodes()
        e1, e2 = g1.number_of_edges(), g2.number_of_edges()
        node_added = max(0, n2 - n1)
        edge_added = max(0, e2 - e1)
        density1 = self._density(g1)
        density2 = self._density(g2)
        density_change = density2 - density1
        return self._detect_shortcut(node_added, edge_added, density_change, e1)

    def should_create_edge(self, gedig_value: float, threshold: float = 0.3) -> bool:
        return gedig_value >= threshold

    def should_backtrack(self, gedig_value: float, threshold: float = -0.2) -> bool:
        return gedig_value < threshold

    # -------- Analysis --------
    def analyze_graph_change(self, g1: nx.Graph, g2: nx.Graph) -> Dict[str, Any]:
        n1, n2 = g1.number_of_nodes(), g2.number_of_nodes()
        e1, e2 = g1.number_of_edges(), g2.number_of_edges()
        core_res = self._core_result(g1, g2)
        analysis: Dict[str, Any] = {
            'nodes_before': n1,
            'nodes_after': n2,
            'nodes_added': max(0, n2 - n1),
            'edges_before': e1,
            'edges_after': e2,
            'edges_added': max(0, e2 - e1),
            'density_before': self._density(g1),
            'density_after': self._density(g2),
            'gedig_1hop': self.calculate(g1, g2),
            'is_shortcut': self.detect_shortcut(g1, g2),
            # Core raw fields for将来分析
            'core_structural_improvement': core_res.structural_improvement,
            'core_ig_value': core_res.ig_value,
            'core_reward': core_res.reward,
            'core_gedig_value': core_res.gedig_value,
        }
        # Connected components
        analysis['components_before'] = nx.number_connected_components(g1) if n1 > 0 else 0
        analysis['components_after'] = nx.number_connected_components(g2) if n2 > 0 else 0
        # Diameter (connected only)
        analysis['diameter_before'] = nx.diameter(g1) if n1 > 1 and nx.is_connected(g1) else -1
        analysis['diameter_after'] = nx.diameter(g2) if n2 > 1 and nx.is_connected(g2) else -1
        return analysis

    # -------- Internals (旧ロジック移植) --------
    def _density(self, g: nx.Graph) -> float:
        n = g.number_of_nodes()
        if n <= 1:
            return 0.0
        e = g.number_of_edges()
        return e / (n * (n - 1) / 2)

    def _detect_shortcut(self, node_added: int, edge_added: int, density_change: float, edges_before: int) -> bool:
        if node_added == 0 and edge_added > 3:
            return True
        if density_change > 0.1:
            return True
        if edges_before > 0 and edge_added / edges_before > 0.5:
            return True
        return False
