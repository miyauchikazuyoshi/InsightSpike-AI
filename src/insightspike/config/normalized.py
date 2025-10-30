"""Normalized configuration facade.

目的:
- Pydantic `InsightSpikeConfig` / dict / legacy 風オブジェクトを単一アクセスAPIに正規化
- MainAgent からの if 多態分岐削減 (config.processing.enable_layer1_bypass 等)
- 将来的な簡易シリアライズ/比較用メタ情報付与

設計方針 (Phase1a スケルトン):
- まだ既存呼び出しを置換しない (導入フラグで opt-in)
- 取得のみ (書き換えは行わない / immutable 的挙動)
- 足りないフィールドはデフォルト値でカバーし `applied_defaults` に記録

後続 (Phase1a 完了条件):
- main_agent 内 if isinstance(config, InsightSpikeConfig)/dict 分岐の ≥70% 除去
- NormalizedConfig 経由アクセスへの段階的移行
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional, Dict

try:
    from .models import InsightSpikeConfig
except Exception:  # pragma: no cover
    InsightSpikeConfig = Any  # type: ignore


def _coerce_optional_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except Exception:
        return None

@dataclass(frozen=True)
class NormalizedConfig:
    # Core extracted scalar fields (拡張は段階的)
    embedding_model: str
    embedding_dim: int
    enable_layer1_bypass: bool
    bypass_uncertainty_threshold: float
    bypass_known_ratio_threshold: float
    enable_learning: bool
    graph_weight_ged: float
    graph_weight_ig: float
    environment: str
    gedig_mode: str = "full"  # 後で Phase2 で実装されるフラグをここで先行定義
    # --- Added Phase1a (拡張フィールド) ---
    max_retrieved_docs: int = 10
    dynamic_doc_adjustment: bool = True
    max_docs_with_insights: int = 5
    insight_relevance_boost: float = 0.2
    enable_insight_search: bool = True
    max_insights_per_query: int = 5
    enable_graph_search: bool = False
    enable_insight_registration: bool = True
    # graph operational parameters / thresholds
    similarity_threshold: float = 0.3
    hop_limit: int = 2
    path_decay: float = 0.7
    spike_ged_threshold: float = -0.5
    spike_ig_threshold: float = 0.2
    episode_merge_threshold: float = 0.8
    episode_split_threshold: float = 0.3
    episode_prune_threshold: float = 0.1
    theta_cand: float = 0.45
    theta_link: float = 0.35
    candidate_cap: int = 32
    top_m: Optional[int] = None
    ig_denominator: str = "fixed_kstar"
    use_local_normalization: bool = False
    # SP engine and NormSpec
    sp_engine: str = "core"  # core | cached | cached_incr
    norm_spec: Optional[Dict[str, Any]] = None

    # メタ
    source_type: str = field(default="unknown")  # pydantic|dict|legacy
    applied_defaults: tuple[str, ...] = field(default_factory=tuple)

    # 元オブジェクト保持 (デバッグ/将来用途)
    _raw: Any = field(repr=False, default=None)

    @staticmethod
    def from_any(cfg: Any, override: Optional[Dict[str, Any]] = None) -> "NormalizedConfig":
        o: Dict[str, Any] = override or {}
        applied: list[str] = []

        def get(path: str, default: Any):
            parts = path.split('.')
            cur = cfg
            for p in parts:
                if cur is None:
                    cur = None
                    break
                # dict
                if isinstance(cur, dict) and p in cur:
                    cur = cur[p]
                    continue
                # object attr
                if hasattr(cur, p):
                    cur = getattr(cur, p)
                    continue
                cur = None
                break
            if cur is None:
                applied.append(path)
                return default
            return cur

        # 判定
        if isinstance(cfg, dict):
            src_type = 'dict'
        elif isinstance(cfg, InsightSpikeConfig):  # type: ignore
            src_type = 'pydantic'
        else:
            src_type = 'legacy'

        # Build provisional NormSpec (prefer WakeSleep SphereSearchConfig when available)
        def _default_norm_spec() -> Dict[str, Any]:
            dim = int(get('embedding.dimension', 384))
            theta_link = float(get('metrics.theta_link', 0.35))
            theta_cand = float(get('metrics.theta_cand', 0.45))
            # Try derive from WakeSleepConfig.search
            method = str(get('wake_sleep.wake.search.method', 'sphere') or 'sphere').lower()
            scope = 'donut' if method == 'donut' else 'sphere'
            # Intuitive radii preferred
            inner_i = get('wake_sleep.wake.search.intuitive_inner_radius', None)
            outer_i = get('wake_sleep.wake.search.intuitive_outer_radius', None)
            if outer_i is None:
                outer_i = get('wake_sleep.wake.search.intuitive_radius', 0.6)
            if inner_i is None:
                # fallback for sphere (no inner): small default filter
                inner_i = 0.2
            try:
                inner_i = float(inner_i)
                outer_i = float(outer_i)
            except Exception:
                inner_i, outer_i = 0.2, 0.6
            return {
                'metric': 'cosine',
                'radius_mode': 'intuitive',
                'intuitive': {'outer': outer_i, 'inner': inner_i},
                'dimension': dim,
                'scope': scope,
                'effective': {
                    'theta_link': theta_link,
                    'theta_cand': theta_cand,
                },
            }

        # If norm_spec missing, derive minimal defaults
        norm_spec_val = (o.get('norm_spec', get('graph.norm_spec', None)) or None)
        if norm_spec_val is None:
            norm_spec_val = _default_norm_spec()

        nc = NormalizedConfig(
            embedding_model = o.get('embedding_model', get('embedding.model_name', 'sentence-transformers/all-MiniLM-L6-v2')),
            embedding_dim = int(o.get('embedding_dim', get('embedding.dimension', 384))),
            enable_layer1_bypass = bool(o.get('enable_layer1_bypass', get('processing.enable_layer1_bypass', False))),
            bypass_uncertainty_threshold = float(o.get('bypass_uncertainty_threshold', get('processing.bypass_uncertainty_threshold', 0.2))),
            bypass_known_ratio_threshold = float(o.get('bypass_known_ratio_threshold', get('processing.bypass_known_ratio_threshold', 0.9))),
            enable_learning = bool(o.get('enable_learning', get('processing.enable_learning', False))),
            graph_weight_ged = float(o.get('graph_weight_ged', get('graph.weight_ged', 0.5))),
            graph_weight_ig = float(o.get('graph_weight_ig', get('graph.weight_ig', 0.5))),
            environment = str(o.get('environment', get('environment', 'development'))),
            gedig_mode = str(o.get('gedig_mode', get('gedig.mode', 'full'))),
            max_retrieved_docs = int(o.get('max_retrieved_docs', get('memory.max_retrieved_docs', 10))),
            dynamic_doc_adjustment = bool(o.get('dynamic_doc_adjustment', get('processing.dynamic_doc_adjustment', True))),
            max_docs_with_insights = int(o.get('max_docs_with_insights', get('processing.max_docs_with_insights', 5))),
            insight_relevance_boost = float(o.get('insight_relevance_boost', get('processing.insight_relevance_boost', 0.2))),
            enable_insight_search = bool(o.get('enable_insight_search', get('processing.enable_insight_search', True))),
            max_insights_per_query = int(o.get('max_insights_per_query', get('processing.max_insights_per_query', 5))),
            enable_graph_search = bool(o.get('enable_graph_search', get('graph.enable_graph_search', False))),
            enable_insight_registration = bool(o.get('enable_insight_registration', get('processing.enable_insight_registration', True))),
            similarity_threshold = float(o.get('similarity_threshold', get('graph.similarity_threshold', 0.3))),
            hop_limit = int(o.get('hop_limit', get('graph.hop_limit', 2))),
            path_decay = float(o.get('path_decay', get('graph.path_decay', 0.7))),
            spike_ged_threshold = float(o.get('spike_ged_threshold', get('graph.spike_ged_threshold', -0.5))),
            spike_ig_threshold = float(o.get('spike_ig_threshold', get('graph.spike_ig_threshold', 0.2))),
            episode_merge_threshold = float(o.get('episode_merge_threshold', get('graph.episode_merge_threshold', 0.8))),
            episode_split_threshold = float(o.get('episode_split_threshold', get('graph.episode_split_threshold', 0.3))),
            episode_prune_threshold = float(o.get('episode_prune_threshold', get('graph.episode_prune_threshold', 0.1))),
            theta_cand = float(o.get('theta_cand', get('metrics.theta_cand', 0.45))),
            theta_link = float(o.get('theta_link', get('metrics.theta_link', 0.35))),
            candidate_cap = int(o.get('candidate_cap', get('metrics.candidate_cap', 32))),
            top_m = _coerce_optional_int(o.get('top_m', get('metrics.top_m', None))),
            ig_denominator = str(o.get('ig_denominator', get('metrics.ig_denominator', 'fixed_kstar'))).lower(),
            use_local_normalization = bool(o.get('use_local_normalization', get('metrics.use_local_normalization', False))),
            sp_engine = str(o.get('sp_engine', get('graph.sp_engine', 'core'))).lower(),
            norm_spec = norm_spec_val,
            source_type = src_type,
            applied_defaults = tuple(applied),
            _raw = cfg,
        )
        return nc

    # 利便メソッド (段階追加予定)
    def as_dict(self) -> Dict[str, Any]:
        return {
            'embedding_model': self.embedding_model,
            'embedding_dim': self.embedding_dim,
            'enable_layer1_bypass': self.enable_layer1_bypass,
            'bypass_uncertainty_threshold': self.bypass_uncertainty_threshold,
            'bypass_known_ratio_threshold': self.bypass_known_ratio_threshold,
            'enable_learning': self.enable_learning,
            'graph_weight_ged': self.graph_weight_ged,
            'graph_weight_ig': self.graph_weight_ig,
            'environment': self.environment,
            'gedig_mode': self.gedig_mode,
            'max_retrieved_docs': self.max_retrieved_docs,
            'dynamic_doc_adjustment': self.dynamic_doc_adjustment,
            'max_docs_with_insights': self.max_docs_with_insights,
            'insight_relevance_boost': self.insight_relevance_boost,
            'enable_insight_search': self.enable_insight_search,
            'max_insights_per_query': self.max_insights_per_query,
            'enable_graph_search': self.enable_graph_search,
            'enable_insight_registration': self.enable_insight_registration,
            'similarity_threshold': self.similarity_threshold,
            'hop_limit': self.hop_limit,
            'path_decay': self.path_decay,
            'spike_ged_threshold': self.spike_ged_threshold,
            'spike_ig_threshold': self.spike_ig_threshold,
            'episode_merge_threshold': self.episode_merge_threshold,
            'episode_split_threshold': self.episode_split_threshold,
            'episode_prune_threshold': self.episode_prune_threshold,
            'theta_cand': self.theta_cand,
            'theta_link': self.theta_link,
            'candidate_cap': self.candidate_cap,
            'top_m': self.top_m,
            'ig_denominator': self.ig_denominator,
            'use_local_normalization': self.use_local_normalization,
            'sp_engine': self.sp_engine,
            'norm_spec': self.norm_spec,
            'source_type': self.source_type,
            'applied_defaults': list(self.applied_defaults),
        }

__all__ = ['NormalizedConfig']
