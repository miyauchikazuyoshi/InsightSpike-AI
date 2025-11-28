"""
L3 Graph Reasoning - Enhanced GNN with ΔGED/ΔIG Analysis
======================================================

Implements graph-based reasoning with spike detection and conflict analysis.

DIAG: Import start/end markers added when INSIGHTSPIKE_DIAG_IMPORT=1 to
pinpoint potential import-time stalls.
"""

import logging
import os
if os.getenv('INSIGHTSPIKE_DIAG_IMPORT') == '1':
    print('[layer3_graph_reasoner] module import start', flush=True)
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
# Lightweight cosine similarity fallback always available
def _cosine_similarity(a: np.ndarray, b: Optional[np.ndarray] = None):
    if b is None:
        b = a
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T
cosine_similarity = _cosine_similarity  # may be overridden in heavy path
from ...core.episode import Episode
from .layer3.conflict import ConflictScore
from .layer3.graph_builder import GraphBuilder

# Helper to detect heavy deps availability
def _have_torch_geometric() -> bool:
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
        return True
    except Exception:
        return False

LITE_MODE_ACTIVE = os.getenv("INSIGHTSPIKE_LITE_MODE") == "1" or os.getenv("INSIGHTSPIKE_MIN_IMPORT") == "1"
# 新設: GNN 強制無効フラグ (PyG 拡張未導入 macOS などで安定運用するため)
DISABLE_GNN = os.getenv("INSIGHTSPIKE_DISABLE_GNN") == "1"

# Prepare logger early
logger = logging.getLogger(__name__)

# Lite/minimal mode early exit: provide no-op placeholder to avoid heavy torch / pyg imports
if LITE_MODE_ACTIVE and not _have_torch_geometric():  # pragma: no cover
    class Data:  # minimal placeholder for type compatibility
        def __init__(self, *args, **kwargs):
            self.x = kwargs.get('x', None)
            self.edge_index = kwargs.get('edge_index', None)
            self.num_nodes = 0

    class L3GraphReasoner:  # lightweight stub
        def __init__(self, config=None):
            self.config = config
            self.enabled = False
            self.current_graph = None

        def initialize(self) -> bool:
            return True

        def analyze(self, *args, **kwargs):  # keep legacy interface
            return {"enabled": False, "reason": "lite_mode"}

    def analyze_documents(self, documents, context=None):
        # Delegate to analyzer_runner; keep stub result as last resort
        try:
            from .layer3.analyzer_runner import run_analysis
            return run_analysis(self, documents, context or {})
        except Exception:
            return {
                "graph": None,
                "metrics": {"delta_ged": 0.0, "delta_ig": 0.0},
                "conflicts": {"total": 0},
                "reward": {"insight_reward": 0.0, "quality_bonus": 0.0},
                "reasoning_quality": 0.5,
                "spike_detected": False,
            }

    logger.warning("Layer3GraphReasoner: lite/min mode -> using lightweight stub (torch/geometric skipped)")
else:  # full imports only when not in lite/min mode
    # --- Heavy dependency import (timed + segmented) ---
    _diag = os.getenv('INSIGHTSPIKE_DIAG_IMPORT') == '1'
    _t_import = None
    if _diag:
        import time as _time
        _t_import = _time.time()
        print('[layer3_graph_reasoner] importing heavy deps: torch, torch_geometric', flush=True)
    try:
        import torch  # type: ignore  # noqa: F401
        import torch.nn.functional as F  # type: ignore  # noqa: F401
        from torch_geometric.data import Data  # type: ignore
        from torch_geometric.nn import GCNConv, global_mean_pool  # type: ignore
        _TORCH_OK = True
    except Exception as _e:  # pragma: no cover
        _TORCH_OK = False
        if _diag:
            print('[layer3_graph_reasoner] WARN torch/pyg import failed -> fallback stubs:', _e, flush=True)
        class Data:  # minimal fallback
            def __init__(self, x=None, edge_index=None, **kw):
                self.x = x
                self.edge_index = edge_index
                self.num_nodes = getattr(x, 'shape', [0])[0] if x is not None else 0
        class GCNConv:  # pragma: no cover - fallback
            def __init__(self, *a, **k):
                raise RuntimeError('GCNConv unavailable (torch import failed)')
        def global_mean_pool(x, batch):  # pragma: no cover
            return None
        F = None  # type: ignore
    if _diag and _t_import is not None:
        print(f"[layer3_graph_reasoner] heavy deps imported ok={_TORCH_OK} elapsed={(_time.time()-_t_import):.2f}s", flush=True)

    # cosine_similarity (sklearn 遅延切替)
    _HAVE_SKLEARN = False
    try:  # optional heavy import
        import importlib
        if os.getenv('INSIGHTSPIKE_USE_SKLEARN', '0') == '1':
            sk_mod = importlib.import_module('sklearn.metrics.pairwise')
            cosine_similarity = sk_mod.cosine_similarity  # type: ignore
            _HAVE_SKLEARN = True
        else:
            cosine_similarity = _cosine_similarity  # type: ignore
    except Exception:  # pragma: no cover
        cosine_similarity = _cosine_similarity  # type: ignore

# Advanced metrics / message passing を遅延ロードするためのフラグ
_ADV_METRICS_LOADED = False
_ADV_METRICS_FAILED = False

def _load_advanced_metrics():  # lazy import
    global advanced_delta_ged, advanced_delta_ig, ADVANCED_METRICS_AVAILABLE
    global _ADV_METRICS_LOADED, _ADV_METRICS_FAILED
    if _ADV_METRICS_LOADED or _ADV_METRICS_FAILED or LITE_MODE_ACTIVE:
        return
    try:
        from ...metrics.advanced_graph_metrics import delta_ged as _adv_ged, delta_ig as _adv_ig
        advanced_delta_ged = _adv_ged  # type: ignore
        advanced_delta_ig = _adv_ig  # type: ignore
        ADVANCED_METRICS_AVAILABLE = True
        _ADV_METRICS_LOADED = True
        if os.getenv('INSIGHTSPIKE_DIAG_IMPORT') == '1':
            print('[layer3_graph_reasoner] advanced metrics loaded lazily', flush=True)
    except Exception as _e:  # pragma: no cover
        _ADV_METRICS_FAILED = True
        ADVANCED_METRICS_AVAILABLE = False
        if os.getenv('INSIGHTSPIKE_DIAG_IMPORT') == '1':
            print('[layer3_graph_reasoner] advanced metrics load failed -> fallback simple', _e, flush=True)

#############################
# Lazy component placeholders
#############################
GraphAnalyzer = None  # type: ignore
RewardCalculator = None  # type: ignore
MessagePassing = None  # type: ignore
EdgeReevaluator = None  # type: ignore

# Import both simple metrics (lightweight)
from ...metrics.graph_metrics import delta_ged as simple_delta_ged
from ...metrics.graph_metrics import delta_ig as simple_delta_ig

_COMPONENTS_LOADED = False
def _load_graph_reasoning_components():  # lazy heavy (non-torch) components
    global GraphAnalyzer, RewardCalculator, _COMPONENTS_LOADED
    if _COMPONENTS_LOADED:
        return
    try:
        if os.getenv('INSIGHTSPIKE_DIAG_IMPORT') == '1':
            print('[layer3_graph_reasoner] loading graph_reasoning components lazily', flush=True)
        from .layer3.analysis import GraphAnalyzer as _GA, RewardCalculator as _RC  # type: ignore
        GraphAnalyzer = _GA  # type: ignore
        RewardCalculator = _RC  # type: ignore
        _COMPONENTS_LOADED = True
    except Exception as _e:  # pragma: no cover
        if os.getenv('INSIGHTSPIKE_DIAG_IMPORT') == '1':
            print('[layer3_graph_reasoner] WARN lazy graph_reasoning load failed:', _e, flush=True)
        _COMPONENTS_LOADED = False

_MSG_PASS_LOADED = False
def _load_message_passing_components():
    global MessagePassing, EdgeReevaluator, _MSG_PASS_LOADED
    if _MSG_PASS_LOADED:
        return
    try:
        if os.getenv('INSIGHTSPIKE_DIAG_IMPORT') == '1':
            print('[layer3_graph_reasoner] loading message_passing components lazily', flush=True)
        from .layer3.message_passing import MessagePassing as _MP, EdgeReevaluator as _ER  # type: ignore
        MessagePassing = _MP  # type: ignore
        EdgeReevaluator = _ER  # type: ignore
        _MSG_PASS_LOADED = True
    except Exception as _e:  # pragma: no cover
        if os.getenv('INSIGHTSPIKE_DIAG_IMPORT') == '1':
            print('[layer3_graph_reasoner] WARN lazy message_passing load failed:', _e, flush=True)
        _MSG_PASS_LOADED = False

ADVANCED_METRICS_AVAILABLE = False
advanced_delta_ged = simple_delta_ged  # initial fallback
advanced_delta_ig = simple_delta_ig  # initial fallback
from ...config import get_config
from ...config.legacy_adapter import LegacyConfigAdapter
from ...core.base import L3GraphReasonerInterface, LayerInput, LayerOutput
from .scalable_graph_builder import ScalableGraphBuilder

if 'logger' not in globals():  # ensure logger defined (when full path)
    logger = logging.getLogger(__name__)

# --- Safety patch: ensure torch symbols not referenced when _TORCH_OK is False ---
# Some test environments intentionally run without torch / PyG installed. Previous
# implementation left methods (_init_gnn, _process_with_gnn) that referenced torch
# unguarded, producing NameError during collection as annotations/bodies were evaluated.
# We defensively replace those methods with no-op fallbacks when torch imports failed.
try:
    _TORCH_OK  # type: ignore  # already set in import block when not lite mode
except NameError:  # if variable not defined (e.g. lite mode stub), define for clarity
    _TORCH_OK = False  # type: ignore

if not LITE_MODE_ACTIVE and not _TORCH_OK:
    # Delay import of typing for minimal overhead
    from typing import Optional as _Opt, Any as _Any, Dict as _Dict

    def _no_torch_init_gnn(self):  # type: ignore
        self.gnn = None
        logger.info("GNN disabled (torch not available)")

    def _no_torch_process_with_gnn(self, graph):  # type: ignore
        return None

    # We patch onto the class only if it already exists (non-lite path). Class
    # definition appears later in file; so we store into a temporary registry and
    # apply after class creation via a small hook.
    _NEED_GNN_METHOD_PATCH = True
else:
    _NEED_GNN_METHOD_PATCH = False

__all__ = ["L3GraphReasoner", "ConflictScore", "GraphBuilder", "ScalableGraphBuilder"]
if os.getenv('INSIGHTSPIKE_DIAG_IMPORT') == '1':
    print('[layer3_graph_reasoner] module import mid (post class defs)', flush=True)

# Apply no-torch GNN method patches if needed (ensures NameError free collection)
if not LITE_MODE_ACTIVE and '_NEED_GNN_METHOD_PATCH' in globals() and _NEED_GNN_METHOD_PATCH:
    try:  # pragma: no cover - patch logic trivial
        if 'L3GraphReasoner' in globals() and hasattr(L3GraphReasoner, '_init_gnn'):
            L3GraphReasoner._init_gnn = _no_torch_init_gnn  # type: ignore
        if 'L3GraphReasoner' in globals() and hasattr(L3GraphReasoner, '_process_with_gnn'):
            L3GraphReasoner._process_with_gnn = _no_torch_process_with_gnn  # type: ignore
        logger.info('L3GraphReasoner: torch not available -> GNN methods patched to no-op')
    except Exception as _patch_e:  # pragma: no cover
        logger.warning(f'L3GraphReasoner: failed to patch GNN methods: {_patch_e}')


class L3GraphReasoner(L3GraphReasonerInterface):
    """
    Enhanced graph reasoning layer with GNN processing and spike detection.

    Features:
    - PyTorch Geometric graph construction
    - ΔGED and ΔIG calculation for insight detection
    - Conflict scoring between reasoning states
    - Reward calculation for memory updates
    """

    def __init__(self, config=None):
        # Set layer_id for LayerInterface
        super().__init__("layer3_graph_reasoner", config)
        # Store original config for message passing settings
        self._original_config = config
        self.config = LegacyConfigAdapter.ensure_pydantic(config or get_config())
        # Apply environment preset overlay if requested (first-class switch)
        try:
            _preset = str(os.getenv('INSIGHTSPIKE_PRESET', '')).strip().lower()
            if _preset == 'paper':
                # Enforce paper-aligned knobs directly on Pydantic config
                if hasattr(self.config, 'graph') and self.config.graph is not None:
                    self.config.graph.sp_scope_mode = 'union'
                    self.config.graph.sp_eval_mode = 'fixed_before_pairs'
                    self.config.graph.ged_norm_scheme = 'candidate_base'
                    self.config.graph.ig_source_mode = 'linkset'
                    # λ/γ tuned to paper runs unless overridden later
                    try:
                        self.config.graph.lambda_weight = float(getattr(self.config.graph, 'lambda_weight', 1.0) or 1.0)
                    except Exception:
                        self.config.graph.lambda_weight = 1.0
                    try:
                        self.config.graph.sp_beta = float(getattr(self.config.graph, 'sp_beta', 1.0) or 1.0)
                    except Exception:
                        self.config.graph.sp_beta = 1.0
                    # Keep query weight unity for linkset entropy unless caller changes
                    self.config.graph.linkset_query_weight = 1.0
                if hasattr(self.config, 'metrics') and self.config.metrics is not None:
                    # Fix IG denominator to log k★ and use local normalization
                    try:
                        self.config.metrics.ig_denominator = 'fixed_kstar'
                    except Exception:
                        pass
                    try:
                        self.config.metrics.use_local_normalization = True
                    except Exception:
                        pass
        except Exception:
            # Preset overlay is best-effort; proceed with existing config
            pass
        # Use ScalableGraphBuilder for better performance
        self.graph_builder = ScalableGraphBuilder(self.config)
        self.conflict_scorer = ConflictScore(self.config)
        self.previous_graph = None
        # Expose latest analyzed graph to callers (e.g., ConfigurableAgent)
        self.current_graph = None

        # Initialize refactored components (delegated to layer3 package)
        _load_graph_reasoning_components()
        if GraphAnalyzer is not None and RewardCalculator is not None:
            self.graph_analyzer = GraphAnalyzer(self.config)  # type: ignore
            self.reward_calculator = RewardCalculator(self.config)  # type: ignore
        else:  # pragma: no cover - fallback minimal stubs
            class _StubAnalyzer:
                def calculate_metrics(self, *a, **k):
                    return {
                        "delta_ged": 0.0,
                        "delta_ig": 0.0,
                        "delta_ged_norm": 0.0,
                        "delta_sp": 0.0,
                        "g0": 0.0,
                        "gmin": 0.0,
                    }
                def detect_spike(self, *a, **k):
                    return False
                def assess_quality(self, metrics, conflicts):
                    return 0.0
            class _StubReward:
                def calculate_reward(self, metrics, conflicts):
                    return {"base":0.0,"structure":0.0,"novelty":0.0,"total":0.0}
            self.graph_analyzer = _StubAnalyzer()
            self.reward_calculator = _StubReward()

        # Initialize metrics selector with configuration
        try:
            from .layer3.metrics_controller import MetricsController

            mc = MetricsController(config)
            self.metrics_selector = getattr(mc, "_selector", mc)
            self.delta_ged = mc.delta_ged
            self.delta_ig = mc.delta_ig
            algo_info = getattr(mc, "info", {})
        except Exception as _mc_exc:  # pragma: no cover
            logger.warning("MetricsController init failed: %s", _mc_exc)
            self.metrics_selector = None
            self.delta_ged = lambda *a, **k: 0.0  # type: ignore
            self.delta_ig = lambda *a, **k: 0.0  # type: ignore
            algo_info = {}

        # Log algorithm selection
        if algo_info:
            logger.info(
                f"Metrics algorithms - GED: {algo_info.get('ged_algorithm')}, IG: {algo_info.get('ig_algorithm')}"
            )
        # Initialize simple GNN if needed
        self.gnn = None
        self._gnn_requested = bool(getattr(self.config, 'graph', None) and getattr(self.config.graph, 'use_gnn', False))
        # 拡張未導入やフラグ指定時は無効化
        if DISABLE_GNN:
            if self._gnn_requested:
                logger.info("GNN disabled via INSIGHTSPIKE_DISABLE_GNN=1")
            self._gnn_requested = False
        elif not globals().get('_TORCH_OK', False):  # torch / pyg stubs状態
            if self._gnn_requested:
                logger.info("GNN disabled (torch/pyg not fully available)")
            self._gnn_requested = False
        # これまでは遅延初期化だったが、ユニットテストの期待に合わせて
        # use_gnn=True の場合はコンストラクタで即時初期化する
        if self._gnn_requested:
            try:
                self._init_gnn()
            except Exception as _e:  # pragma: no cover (best-effort)
                logger.warning(f"GNN eager init failed (will fallback to lazy): {_e}")
                # 遅延初期化へ切り替え
                self.gnn = None
        elif os.getenv('INSIGHTSPIKE_DIAG_IMPORT') == '1':
            print('[layer3_graph_reasoner] GNN not requested', flush=True)

        # Initialize message passing components
        self._init_message_passing()

        # Knowledge counters (M1)
        self._knowledge_counters = {
            "updates": 0,
            "total_facts": 0,  # proxy: documents processed
            "total_relations": 0,  # proxy: edges
        }
        self._knowledge_first_logged = False

    def initialize(self) -> bool:
        """Initialize the layer"""
        try:
            # Any initialization needed
            self._is_initialized = True
            logger.info("L3GraphReasoner initialized successfully")
            return True
        except Exception as e:
            logger.error(f"L3GraphReasoner initialization failed: {e}")
            return False

    def process(self, input_data) -> Any:
        """Process input through this layer"""
        try:
            # Handle LayerInput format if provided
            if hasattr(input_data, "data"):
                documents = input_data.data
                context = input_data.context or {}
            else:
                documents = input_data
                context = {}

            return self.analyze_documents(documents, context)
        except Exception as e:
            logger.error(f"L3GraphReasoner processing failed: {e}")
            return self._fallback_result()

    def cleanup(self):
        """Cleanup resources"""
        try:
            self.previous_graph = None
            self.gnn = None
            self._is_initialized = False
            logger.info("L3GraphReasoner cleaned up successfully")
        except Exception as e:
            logger.error(f"L3GraphReasoner cleanup failed: {e}")

    def analyze_documents(
        self, documents: List[Dict[str, Any]], context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze documents via analyzer_runner (legacy body removed)."""
        context = context or {}

        try:
            from .layer3.analyzer_runner import run_analysis
            return run_analysis(self, documents, context)
        except Exception as exc:  # pragma: no cover
            logger.warning("analyzer_runner failed; returning fallback result: %s", exc)
            return self._fallback_result()

    def _get_spike_thresholds(self) -> Dict[str, float]:
        """Get spike detection thresholds from config."""
        return {
            "ged": self.config.graph.spike_ged_threshold,
            "ig": self.config.graph.spike_ig_threshold,
            "conflict": self.config.graph.conflict_threshold,
        }

    def _init_message_passing(self):
        """Initialize message passing components."""
        try:
            from .layer3.message_passing_controller import MessagePassingController

            controller = MessagePassingController(
                config=self.config, original_config=self._original_config
            )
            controller.initialize()
            self.message_passing_enabled = controller.message_passing_enabled
            self.message_passing = controller.message_passing
            self.edge_reevaluator = controller.edge_reevaluator
            if self.message_passing_enabled:
                logger.info("Message passing initialized successfully (controller)")
            else:
                logger.info("Message passing disabled in config")

        except Exception as e:
            logger.warning(f"Message passing initialization failed: {e}")
            self.message_passing_enabled = False
            self.message_passing = None
            self.edge_reevaluator = None

    def _init_gnn(self):
        """Initialize a simple GNN for graph processing.

        注意: GCNConv は forward(x, edge_index) を要求するため、torch.nn.Sequential
        では引数の形が合わない。ここでは Module を定義して正しいシグネチャを保つ。
        """
        try:
            from .layer3.gnn import build_simple_gnn

            hidden_dim = int(getattr(self.config.graph, 'gnn_hidden_dim', 64) or 64)
            input_dim = int(getattr(self.config.embedding, 'dimension', 384) or 384)
            self.gnn = build_simple_gnn(input_dim, hidden_dim)
            if self.gnn is not None:
                logger.info("Initialized GNN for graph processing")
            else:
                logger.info("GNN unavailable; proceeding without GNN")

        except Exception as e:
            logger.warning(f"GNN initialization failed: {e}")
            self.gnn = None

    def _process_with_gnn(self, graph: Data):  # return type: optional tensor-like
        """Process graph through GNN if available."""
        # Lazy build if requested but not yet initialized
        if self.gnn is None and getattr(self, '_gnn_requested', False):
            try:
                self._init_gnn()
                if os.getenv('INSIGHTSPIKE_DIAG_IMPORT') == '1':
                    print('[layer3_graph_reasoner] GNN lazy-initialized', flush=True)
            except Exception as _e:  # pragma: no cover
                if os.getenv('INSIGHTSPIKE_DIAG_IMPORT') == '1':
                    print('[layer3_graph_reasoner] GNN lazy init failed:', _e, flush=True)
                self._gnn_requested = False
                self.gnn = None
        if self.gnn is None or graph.num_nodes == 0:
            return None

        try:
            with torch.no_grad():
                x = self.gnn(graph.x, graph.edge_index)
                # Global pooling to get graph-level representation
                graph_repr = global_mean_pool(
                    x, torch.zeros(graph.num_nodes, dtype=torch.long)
                )
                return graph_repr

        except Exception as e:
            logger.error(f"GNN processing failed: {e}")
            return None

    def _fallback_result(self) -> Dict[str, Any]:
        """Fallback result for error cases."""
        return {
            "graph": self.graph_builder._empty_graph(),
            "metrics": {"delta_ged": 0.0, "delta_ig": 0.0},
            "conflicts": {
                "structural": 0.0,
                "semantic": 0.0,
                "temporal": 0.0,
                "total": 0.0,
            },
            "reward": {"base": 0.0, "structure": 0.0, "novelty": 0.0, "total": 0.0},
            "spike_detected": False,
            "graph_features": None,
            "reasoning_quality": 0.0,
            "graph_context": {"nodes": 0, "edges": 0, "summary": "Empty graph"},
        }

    def _build_graph_context(self, graph: Data, metrics: Dict[str, Any], conflicts: Dict[str, Any]) -> Dict[str, Any]:
        """Create lightweight context summary for downstream layers/tests."""
        try:
            if graph is None or not hasattr(graph, "num_nodes"):
                return {"nodes": 0, "edges": 0, "summary": "No graph"}
            num_nodes = int(getattr(graph, 'num_nodes', 0))
            num_edges = int(graph.edge_index.size(1)) if getattr(graph, 'edge_index', None) is not None else 0
            ged = metrics.get("delta_ged")
            ig = metrics.get("delta_ig")
            conflict_total = conflicts.get("total") if conflicts else None
            parts = [f"nodes={num_nodes}", f"edges={num_edges}"]
            if isinstance(ged, (int, float)):
                parts.append(f"ΔGED={ged:.3f}")
            if isinstance(ig, (int, float)):
                parts.append(f"ΔIG={ig:.3f}")
            if isinstance(conflict_total, (int, float)):
                parts.append(f"conflict={conflict_total:.3f}")
            summary = ", ".join(parts)
            sample_texts = []
            if hasattr(graph, "documents") and graph.documents:
                for d in graph.documents[:3]:
                    if isinstance(d, dict):
                        txt = d.get("text", "")
                    else:
                        txt = str(d)
                    if len(txt) > 80:
                        txt = txt[:77] + "..."
                    sample_texts.append(txt)
            return {"nodes": num_nodes, "edges": num_edges, "summary": summary, "samples": sample_texts}
        except Exception as e:
            logger.debug(f"Failed to build graph_context: {e}")
            return {"nodes": 0, "edges": 0, "summary": "Unavailable"}

    # --- Public helper: knowledge counters (M1) ---
    def get_knowledge_counters(self) -> Dict[str, int]:
        return dict(self._knowledge_counters)

    def __repr__(self):  # enrich representation with counters
        base = super().__repr__() if hasattr(super(), '__repr__') else f"L3GraphReasoner(id={self.layer_id})"
        try:
            kc = self._knowledge_counters
            return base + f" knowledge(updates={kc['updates']}, facts={kc['total_facts']}, relations={kc['total_relations']})"
        except Exception:
            return base

    # Interface methods implementation
    def build_graph(self, vectors: np.ndarray) -> Any:
        """Build similarity graph.

        Accepts either:
        - numpy.ndarray of shape (n,d)
        - list of document dicts each containing 'embedding' or 'vector'
        This prevents object-dtype arrays reaching the builder (causing ufunc errors).
        """
        try:
            # Case: list of documents
            if isinstance(vectors, list) and (len(vectors) == 0 or isinstance(vectors[0], dict)):
                documents = vectors
                embeddings = []
                for d in documents:
                    emb = d.get("embedding") or d.get("vector")
                    if emb is None:
                        emb = np.random.randn(self.config.embedding.dimension)
                    if isinstance(emb, list):
                        emb = np.array(emb)
                    if isinstance(emb, torch.Tensor):
                        emb = emb.detach().cpu().numpy()
                    embeddings.append(emb)
                embeddings = np.array(embeddings, dtype=np.float32)
                return self.graph_builder.build_graph(documents, embeddings)
            # Case: ndarray
            if isinstance(vectors, np.ndarray):
                documents = [{"embedding": vec, "text": f"doc_{i}"} for i, vec in enumerate(vectors)]
                return self.graph_builder.build_graph(documents, vectors)
            logger.warning("Unsupported input type to build_graph; returning empty graph")
            return self.graph_builder._empty_graph()
        except Exception as e:
            logger.error(f"build_graph failed: {e}")
            return self.graph_builder._empty_graph()

    # Backward compatibility: some tests/agents may call analyze_graph
    def analyze_graph(self, documents_or_vectors, query_vector=None, **kwargs):
        """Compatibility wrapper forwarding to analyze_documents.

        Accepts either list of document dicts or ndarray of embeddings plus optional
        query_vector for message passing. Additional kwargs are merged into context.
        """
        context = kwargs.copy()
        if query_vector is not None:
            context.setdefault("query_vector", query_vector)
            # Also support legacy key name used in tests
            context.setdefault("query_embedding", query_vector)
        if isinstance(documents_or_vectors, np.ndarray):
            arr = documents_or_vectors
            if arr.ndim == 1:  # Single vector (e.g., query only)
                docs = [{"embedding": arr, "text": "query_vector_doc"}]
            else:
                docs = [
                    {"embedding": v, "text": f"doc_{i}"}
                    for i, v in enumerate(arr)
                ]
        else:
            docs = documents_or_vectors
        return self.analyze_documents(docs, context)

    def calculate_ged(self, graph1: Any, graph2: Any) -> float:
        """Calculate graph edit distance (lazy advanced metrics)."""
        _load_advanced_metrics()
        try:
            return advanced_delta_ged(graph1, graph2)
        except Exception as e:
            logger.error(f"GED calculation failed: {e}")
            return 0.0

    def calculate_ig(self, old_state: Any, new_state: Any) -> float:
        """Calculate information gain (lazy advanced metrics)."""
        _load_advanced_metrics()
        try:
            return advanced_delta_ig(old_state, new_state)
        except Exception as e:
            logger.error(f"IG calculation failed: {e}")
            return 0.0

    def detect_eureka_spike(self, delta_ged: float, delta_ig: float) -> bool:
        """Detect if current state constitutes a eureka spike"""
        metrics = {"delta_ged": delta_ged, "delta_ig": delta_ig}
        conflicts = {"total": 0.0}  # No conflicts for direct call
        thresholds = self._get_spike_thresholds()
        spike_raw = self.graph_analyzer.detect_spike(metrics, conflicts, thresholds)
        # Some legacy analyzer implementations may return (bool, score). Normalize to bool.
        if isinstance(spike_raw, tuple):
            spike_bool = bool(spike_raw[0])
        else:
            spike_bool = bool(spike_raw)

        # Feature flag: SPIKE_BOOL_WRAPPER (default ON). When enabled we also attach
        # a lightweight contextual spike score into an internal attrib for optional
        # debugging/inspection while keeping the public signature pure bool.
        # Score = normalized combined magnitude of (|ΔGED| over threshold, ΔIG over threshold)
        try:
            flag_val = os.getenv("SPIKE_BOOL_WRAPPER", "1").strip().lower()
            flag_enabled = flag_val in ("1", "true", "on", "yes")
        except Exception:
            flag_enabled = True

        if flag_enabled:
            ged_thr = abs(float(thresholds.get("ged", -0.5)) or 1.0)
            ig_thr = float(thresholds.get("ig", 0.2) or 1.0)
            # Compute over-threshold ratios (clip at 0..1 for stability)
            ged_component = 0.0
            if ged_thr > 0:
                # thresholds['ged'] is negative (improvement), delta_ged should be < thresh
                ged_improve = max(0.0, abs(delta_ged) - abs(thresholds.get("ged", -0.5)))
                ged_component = min(1.0, ged_improve / ged_thr)
            ig_component = 0.0
            if ig_thr > 0:
                ig_excess = max(0.0, delta_ig - ig_thr)
                ig_component = min(1.0, ig_excess / ig_thr)
            context_spike_score = round((ged_component + ig_component) / 2.0, 4)
            # Store last spike context metrics (non-breaking, purely optional)
            try:
                self._last_spike_context = {
                    "spike": spike_bool,
                    "context_spike_score": context_spike_score,
                    "metrics": metrics,
                    "thresholds": thresholds,
                }
            except Exception:
                pass
        return spike_bool
    
    def update_graph(self, episodes: List[Episode]):
        """
        Update graph with new episodes.
        
        This method is called by MainAgent but was missing from the implementation.
        For now, we store the reference to episodes for future graph building.
        
        Args:
            episodes: List of new episodes to incorporate into the graph
        """
        # Log the update request
        logger.debug(f"Graph update requested with {len(episodes)} episodes")
        
        # In a full implementation, this would:
        # 1. Extract vectors from episodes
        # 2. Update the existing graph structure
        # 3. Recalculate graph metrics
        
        # For now, we just acknowledge the request
        # The actual graph update happens in build_graph when needed
        pass

    # ------------------------------------------------------------------
    # Backward Compatibility Helpers
    # ------------------------------------------------------------------
    def load_graph(self):  # pragma: no cover - simple compatibility shim
        """Return previously stored graph (legacy API shim).

        Older code paths (e.g. MainAgent._legacy_load_state) expect L3GraphReasoner
        to expose a load_graph() method. Modern persistence is handled by the
        DataStore at the agent level; we therefore just surface the in-memory
        previous_graph so legacy loading doesn't raise AttributeError.
        """
        try:
            return getattr(self, "previous_graph", None)
        except Exception:
            return None

    def save_graph(self, graph):  # pragma: no cover - compatibility only
        """Store graph in-memory (legacy stub)."""
        try:
            self.previous_graph = graph
            return True
        except Exception:
            return False

# Final module import end marker
if os.getenv('INSIGHTSPIKE_DIAG_IMPORT') == '1':
    print('[layer3_graph_reasoner] module import end', flush=True)
