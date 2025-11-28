"""Shared pytest fixtures and configuration for the test suite."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock

# numpy は一部テストでのみ必要。収集段階のオーバーヘッドを避けるため遅延インポート。
def _np():  # type: ignore
    import importlib
    return importlib.import_module("numpy")
import pytest

"""NOTE:
重い import (config や core サブパッケージ) がテスト収集段階で実行されると
ハング/時間超過を誘発するケースがあったため、環境フラグを最優先で設定し、
InsightSpike モジュールの import は各フィクスチャ内で遅延評価する。
"""

# 早期にライト/ミニマルモードを強制 (収集段階で有効化)
os.environ.setdefault("INSIGHTSPIKE_LITE_MODE", "1")
os.environ.setdefault("INSIGHTSPIKE_MIN_IMPORT", "1")

# Add src to path for imports (tests/ から実行され rootdir が tests になるケースに対応)
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# --- Optional module availability (maze legacy deps) ---
import importlib.util as _ilu

def _safe_find_spec(name: str):
    try:
        return _ilu.find_spec(name)
    except ModuleNotFoundError:  # pragma: no cover - defensive
        return None

_NAV_AVAILABLE = _safe_find_spec("navigation") is not None
_MAZE_CORE_AVAILABLE = _safe_find_spec("core.eviction_policy") is not None  # legacy path

# 依存欠如時に収集で失敗するテストをデフォルトで無視
collect_ignore_glob = []  # pytest picks this up at collection time
if not _NAV_AVAILABLE:
    collect_ignore_glob.extend([
        "tests/maze/*",
        "tests/maze-query-hub-prototype/*",
        "tests/test_macro_target_adaptive_p*.py",
        "tests/test_macro_target_metrics.py",
        "tests/test_maze_navigator_smoke.py",
        "tests/unit/test_maze_simple_mode.py",
    ])
if not _MAZE_CORE_AVAILABLE:
    collect_ignore_glob.append("tests/maze/test_eviction_policy.py")

# 遅延 import 用ヘルパ (必要になった時点でのみ import)
def _load_config_types():
    from insightspike.config.models import InsightSpikeConfig  # type: ignore
    from insightspike.config.presets import ConfigPresets  # type: ignore
    return InsightSpikeConfig, ConfigPresets

# torch / torch_geometric 利用可否を早期判定 (テストskip用途)
_TORCH_AVAILABLE = _ilu.find_spec("torch") is not None
_PYG_AVAILABLE = _ilu.find_spec("torch_geometric") is not None

# --- Torch stub injection (collection safety) ---
# 多数のテストが import 時点で torch シンボルを参照し NameError になるため
# 本物の torch が無い環境では最小限のスタブを差し込んで回避する。
if not _TORCH_AVAILABLE:
    import types
    try:
        import numpy as _np  # type: ignore
    except Exception:  # noqa: BLE001
        _np = None  # type: ignore

    def _build_fallback_tensor(shape, fill_value=0.0):
        """Create a nested list matching the requested shape."""
        if not shape:
            return fill_value
        return [_build_fallback_tensor(shape[1:], fill_value) for _ in range(shape[0])]

    def _to_array(data, dtype=None):
        if _np is not None:
            return _np.array(data, dtype=_np.float32 if dtype is None else dtype)
        return data

    torch_stub = types.ModuleType("torch")
    # dtypes
    torch_stub.long = int  # type: ignore
    torch_stub.float32 = float  # type: ignore
    # 基本テンソル生成 (numpy 配列を返す)
    def _empty(*shape, dtype=None):
        if _np is not None:
            return _np.empty(shape, dtype=_np.float32 if dtype is None else dtype)
        return _build_fallback_tensor(shape, 0.0)
    def _zeros(*shape, dtype=None):
        if _np is not None:
            return _np.zeros(shape, dtype=_np.float32 if dtype is None else dtype)
        return _build_fallback_tensor(shape, 0.0)
    torch_stub.empty = _empty  # type: ignore
    torch_stub.zeros = _zeros  # type: ignore
    def _tensor(data, dtype=None):
        return _to_array(data, dtype=dtype)
    torch_stub.tensor = _tensor  # type: ignore
    # no_grad コンテキスト (ダミー)
    class _NoGrad:
        def __enter__(self): return None
        def __exit__(self, *a): return False
    def _no_grad(): return _NoGrad()
    torch_stub.no_grad = _no_grad  # type: ignore
    # 簡易 nn サブモジュール
    nn_mod = types.ModuleType("torch.nn")
    class _ReLU:
        def __call__(self, x): return x
    class _Sequential(list):
        def __call__(self, x, *args, **kwargs):
            for m in self:
                try:
                    x = m(x)
                except Exception:
                    return x
            return x
    nn_mod.ReLU = _ReLU  # type: ignore
    nn_mod.Sequential = _Sequential  # type: ignore
    torch_stub.nn = nn_mod  # type: ignore
    torch_stub.__version__ = "0.0.0-stub"
    import sys as _sys
    _sys.modules['torch'] = torch_stub

def pytest_collection_modifyitems(config, items):  # type: ignore
    """torch / torch_geometric が無い環境で該当テストを自動 skip。

    基準:
      - ファイル本文に 'import torch' 文字列
      - あるいは 'from torch' / 'torch_geometric' を含む
    大量I/O避けるため簡易 substring 判定。
    """
    if _TORCH_AVAILABLE and _PYG_AVAILABLE:
        return
    skip_reason = []
    if not _TORCH_AVAILABLE:
        skip_reason.append("torch")
    if not _PYG_AVAILABLE:
        skip_reason.append("torch_geometric")
    reason = f"missing: {','.join(skip_reason)}"
    substrings = ["import torch", "from torch ", "torch_geometric"]
    for item in items:
        try:
            # 小さいテストファイル前提: サイズ > 32KB の場合はスキャンスキップ
            p = item.fspath  # type: ignore[attr-defined]
            if p.size() > 32768:  # pragma: no cover
                continue
            content = p.read()
            if any(s in content for s in substrings):
                item.add_marker(pytest.mark.skip(reason=reason))
        except Exception:  # noqa: BLE001
            continue


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")
    config.addinivalue_line("markers", "requires_gpu: marks tests that require GPU")


# Global fixtures
@pytest.fixture(scope="session")
def test_data_dir():
    """Create a session-scoped temporary directory for test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def config_development():
    """Development configuration preset."""
    _, ConfigPresets = _load_config_types()
    return ConfigPresets.development()


@pytest.fixture
def config_experiment():
    """Experiment configuration preset."""
    _, ConfigPresets = _load_config_types()
    return ConfigPresets.experiment()


@pytest.fixture
def config_custom():
    """Custom test configuration."""
    InsightSpikeConfig, _ = _load_config_types()
    return InsightSpikeConfig(
        environment="test",
        llm={"provider": "mock", "temperature": 0.5, "max_tokens": 256},
        memory={"episodic_memory_capacity": 100, "max_retrieved_docs": 5},
        embedding={"model_name": "test-model", "dimension": 384},
        graph={"spike_ged_threshold": -0.5, "spike_ig_threshold": 0.3},
    )


@pytest.fixture
def mock_datastore():
    """Create a mock datastore for unit tests."""
    datastore = Mock()
    datastore.load_episodes.return_value = []
    datastore.save_episodes.return_value = True
    datastore.load_graph.return_value = None
    datastore.save_graph.return_value = True
    datastore.get_metadata.return_value = {}
    datastore.set_metadata.return_value = True
    return datastore


@pytest.fixture
def filesystem_datastore(tmp_path):
    """Create a real filesystem datastore for integration tests."""
    # Return a simple mock for now, can be expanded later
    datastore = Mock()
    datastore.base_path = tmp_path
    return datastore


@pytest.fixture
def sample_embeddings():
    """Generate sample embeddings for testing."""
    np = _np()
    np.random.seed(42)
    return {
        "doc1": np.random.randn(384).astype(np.float32),
        "doc2": np.random.randn(384).astype(np.float32),
        "doc3": np.random.randn(384).astype(np.float32),
        "doc4": np.random.randn(384).astype(np.float32),
        "doc5": np.random.randn(384).astype(np.float32),
    }


@pytest.fixture
def sample_documents(sample_embeddings):
    """Generate sample documents with embeddings."""
    return [
        {
            "text": "InsightSpike is an AI system for discovering insights.",
            "embedding": sample_embeddings["doc1"],
            "metadata": {"source": "intro", "importance": 0.9},
        },
        {
            "text": "It uses graph-based reasoning to detect patterns.",
            "embedding": sample_embeddings["doc2"],
            "metadata": {"source": "technical", "importance": 0.8},
        },
        {
            "text": "Multiple layers process information hierarchically.",
            "embedding": sample_embeddings["doc3"],
            "metadata": {"source": "architecture", "importance": 0.7},
        },
        {
            "text": "Emergence occurs when systems integrate.",
            "embedding": sample_embeddings["doc4"],
            "metadata": {"source": "theory", "importance": 0.85},
        },
        {
            "text": "The system learns from patterns in data.",
            "embedding": sample_embeddings["doc5"],
            "metadata": {"source": "ml", "importance": 0.75},
        },
    ]


@pytest.fixture
def knowledge_base():
    """Create a knowledge base for testing."""
    return [
        "System A operates independently with its own logic.",
        "System B operates independently with different logic.",
        "When A and B integrate, new properties emerge.",
        "This emergence is more than the sum of parts.",
        "Complex systems exhibit non-linear behavior.",
        "Feedback loops amplify small changes.",
        "Pattern recognition reveals hidden connections.",
        "Graph structures represent relationships effectively.",
        "Information gain measures the value of new knowledge.",
        "Spike detection identifies significant insights.",
    ]


# Test utilities
class TestHelpers:
    """Utility functions for tests."""

    @staticmethod
    def create_mock_episode(text: str, c_value: float = 0.7, dimension: int = 384):
        """Create a mock episode with embedding.

        Uses delayed numpy import for speed during collection.
        """
        from insightspike.core.episode import Episode  # type: ignore
        np = _np()
        vec = np.random.randn(dimension).astype(np.float32)
        # Normalize vector
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return Episode(text=text, vec=vec, c=c_value)

    @staticmethod
    def create_mock_graph_data(num_nodes=5, num_edges=8):
        """Create mock graph data for testing."""
        try:
            import torch
            from torch_geometric.data import Data

            # Create node features
            x = torch.randn(num_nodes, 384)

            # Create random edges
            edge_list = []
            for _ in range(num_edges):
                import numpy as _npy  # local import
                src = _npy.random.randint(0, num_nodes)
                dst = _npy.random.randint(0, num_nodes)
                if src != dst:
                    edge_list.append([src, dst])

            if not edge_list:
                edge_list = [[0, 1], [1, 0]]  # Minimum edges

            edge_index = torch.tensor(edge_list, dtype=torch.long).t()

            return Data(x=x, edge_index=edge_index)
        except ImportError:
            # Return a mock object if PyTorch Geometric not available
            mock_graph = Mock()
            mock_graph.num_nodes = num_nodes
            mock_graph.edge_index = Mock()
            mock_graph.edge_index.size.return_value = (2, num_edges)
            return mock_graph

    @staticmethod
    def assert_config_valid(config):
        """Assert that a configuration is valid."""
        InsightSpikeConfig, _ = _load_config_types()
        assert isinstance(config, InsightSpikeConfig)
        assert config.environment in [
            "development",
            "experiment",
            "production",
            "research",
            "test",
            "custom",
        ]
        assert config.llm.provider in ["local", "openai", "anthropic", "mock", "clean"]
        assert 0.0 <= config.llm.temperature <= 2.0
        assert config.memory.episodic_memory_capacity > 0
        assert config.embedding.dimension > 0


@pytest.fixture
def test_helpers():
    """Provide test helper utilities."""
    return TestHelpers()


# Environment setup
@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment variables."""
    # Disable any external API calls
    monkeypatch.setenv("INSIGHTSPIKE_TEST_MODE", "1")

    # Use mock LLM by default
    monkeypatch.setenv("INSIGHTSPIKE_LLM__PROVIDER", "mock")

    # Disable logging to console during tests
    monkeypatch.setenv("INSIGHTSPIKE_LOGGING__LOG_TO_CONSOLE", "false")

    # Set a test-specific log directory
    monkeypatch.setenv("INSIGHTSPIKE_LOGGING__FILE_PATH", "/tmp/insightspike_test_logs")
    # Force lite/minimal import mode to avoid heavy imports during isolated algorithm tests
    monkeypatch.setenv("INSIGHTSPIKE_LITE_MODE", "1")
    monkeypatch.setenv("INSIGHTSPIKE_MIN_IMPORT", "1")


# Cleanup
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up after each test."""
    yield

    # Clear any cached models or data
    # embedder の heavy import を避ける (ライトモードでは不要)
    try:
        from insightspike.processing.embedder import _model_cache  # type: ignore

        _model_cache.clear()
    except Exception:
        pass


# Performance tracking
@pytest.fixture
def benchmark(request):
    """Simple benchmark fixture for performance testing."""
    import time

    start_time = time.time()

    def get_duration():
        return time.time() - start_time

    request.node.benchmark = get_duration

    yield get_duration

    duration = get_duration()
    if duration > 5.0:  # Warn if test takes more than 5 seconds
        print(f"\nWarning: Test {request.node.name} took {duration:.2f} seconds")
