"""
Pytest configuration for InsightSpike-AI tests
==============================================

This file sets up comprehensive mocking for all dependencies to ensure
tests run without requiring heavy external libraries.
"""
import sys
import types
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# Mock classes for external dependencies
class MockPath:
    def __init__(self, path="mock_path"):
        self.path = path
    
    def replace(self, target):
        return MockPath(target)
    
    def __str__(self):
        return self.path

class DummyModel:
    def encode(self, texts, normalize_embeddings=True):
        if isinstance(texts, str):
            return [np.random.random(384)]
        return [np.random.random(384) for _ in texts]

class MockGraph:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.edge_index = MagicMock()  # Add edge_index attribute for PyG compatibility
        self.x = MagicMock()  # Add x attribute for PyG data object compatibility
    
    def add_nodes_from(self, nodes):
        self.nodes.extend(nodes)
    
    def add_edges_from(self, edges):
        self.edges.extend(edges)
    
    def add_edge(self, u, v):
        self.edges.append((u, v))
    
    def number_of_nodes(self):
        return len(self.nodes)

# Global singleton for embedding model
_mock_model_instance = None

def get_mock_model():
    global _mock_model_instance
    if _mock_model_instance is None:
        _mock_model_instance = DummyModel()
    return _mock_model_instance

# Mock torch completely
dummy_torch = types.SimpleNamespace(
    load=lambda *a, **k: None,
    save=lambda *a, **k: None,
    device=lambda x: "cpu",
    cuda=types.SimpleNamespace(is_available=lambda: False),
    tensor=lambda x: MagicMock(),
    Tensor=MagicMock,  # Add Tensor class for type annotations
    nn=types.SimpleNamespace(
        Module=object,
        Linear=lambda *a, **k: MagicMock(),
        ReLU=lambda: MagicMock(),
        Sequential=lambda *args: MagicMock(),  # Add Sequential for GNN initialization
        functional=types.SimpleNamespace(
            relu=lambda x: x,
            softmax=lambda x, dim=None: x,
            cosine_similarity=lambda x, y, dim=1: MagicMock(),
            normalize=lambda x, p=2, dim=1: x,
        )
    ),
    __version__="2.2.2"  # Add version attribute
)

# Mock networkx
dummy_networkx = types.SimpleNamespace(
    Graph=lambda: MockGraph(),
    is_connected=lambda g: True,
    shortest_path_length=lambda g, s, t: 1
)

# Mock faiss
dummy_faiss = types.SimpleNamespace(
    IndexFlatIP=lambda dim: MagicMock(),
    index_factory=lambda dim, desc: MagicMock(),
    METRIC_INNER_PRODUCT=0
)

# Mock sentence transformers
dummy_sentence_transformers = types.SimpleNamespace(
    SentenceTransformer=lambda model: get_mock_model()
)

# Mock rich
dummy_rich = types.SimpleNamespace(
    print=lambda *a, **k: None,
    console=types.SimpleNamespace(print=lambda *a, **k: None)
)

# Mock transformers
dummy_transformers = types.SimpleNamespace(
    pipeline=lambda task, model=None, **k: MagicMock(return_value="mocked response"),
    AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda x: MagicMock()),
    AutoModel=types.SimpleNamespace(from_pretrained=lambda x: MagicMock()),
    AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=lambda x: MagicMock())
)

# Mock datasets
dummy_datasets = types.SimpleNamespace(
    load_dataset=lambda *a, **k: MagicMock()
)

# Mock scikit-learn
dummy_sklearn = types.SimpleNamespace(
    metrics=types.SimpleNamespace(
        pairwise_distances=lambda x, y=None, metric='cosine': np.random.random((len(x), len(y) if y is not None else len(x))),
        pairwise=types.SimpleNamespace(
            cosine_similarity=lambda x, y=None: np.random.random((len(x), len(y) if y is not None else len(x)))
        )
    )
)

# Mock MainAgent
class MockMainAgent:
    def __init__(self):
        self.initialized = False
    
    def initialize(self):
        self.initialized = True
        return True
    
    def add_document(self, text, c_value=0.5):
        pass
    
    def process_question(self, question, max_cycles=3, verbose=False):
        return {
            'response': 'mocked response',
            'documents': [],
            'graph': MockGraph(),
            'metrics': {},
            'conflicts': {},
            'reasoning_quality': 0.8,
            'success': True
        }

@pytest.fixture(scope="session", autouse=True)
def mock_dependencies():
    """Auto-applied fixture that mocks all external dependencies"""
    
    # System modules that need to be mocked
    sys.modules['torch'] = dummy_torch
    sys.modules['torch.nn'] = dummy_torch.nn
    sys.modules['torch.nn.functional'] = dummy_torch.nn.functional
    sys.modules['networkx'] = dummy_networkx
    sys.modules['faiss'] = dummy_faiss
    sys.modules['faiss-cpu'] = dummy_faiss
    sys.modules['sentence_transformers'] = dummy_sentence_transformers
    sys.modules['rich'] = dummy_rich
    sys.modules['transformers'] = dummy_transformers
    sys.modules['datasets'] = dummy_datasets
    sys.modules['sklearn'] = dummy_sklearn
    sys.modules['sklearn.metrics'] = dummy_sklearn.metrics
    sys.modules['sklearn.metrics.pairwise'] = dummy_sklearn.metrics.pairwise
    
    # Mock torch_geometric modules
    sys.modules['torch_geometric'] = types.SimpleNamespace()
    sys.modules['torch_geometric.data'] = types.SimpleNamespace(
        Data=lambda **kwargs: types.SimpleNamespace(**kwargs)
    )
    sys.modules['torch_geometric.nn'] = types.SimpleNamespace(
        GCNConv=lambda in_channels, out_channels: MagicMock(),
        global_mean_pool=lambda x, batch: MagicMock()
    )
    
    # InsightSpike modules
    sys.modules['insightspike.embedder'] = types.SimpleNamespace(
        get_model=get_mock_model,
        _model=None
    )
    
    sys.modules['insightspike.layer1_error_monitor'] = types.SimpleNamespace(
        uncertainty=lambda x: 0.8 if len(x) > 1 else 0.0
    )
    
    sys.modules['insightspike.layer2_memory_manager'] = types.SimpleNamespace(
        Memory=lambda dim=384: types.SimpleNamespace(
            dim=dim,  # Add the missing dim attribute
            index=MagicMock(),  # Add the missing index attribute
            episodes=[],  # Add episodes attribute
            search=lambda q, k: ([0.8], [0]),
            update_c=lambda idxs, r, eta=0.1: None,
            train_index=lambda: None,
            merge=lambda idxs: None,
            split=lambda idx: None,
            prune=lambda c, i: None,
            add_episode=lambda v, t, c_init=0.2: None,
            save=lambda: MockPath('saved')
        )
    )
    
    def mock_build_graph(vecs, dest=None):
        if dest:
            # Create the file to satisfy the test
            dest.touch()
        return (MockGraph(), None)
    
    sys.modules['insightspike.layer3_graph_pyg'] = types.SimpleNamespace(
        build_graph=mock_build_graph
    )
    
    sys.modules['insightspike.graph_metrics'] = types.SimpleNamespace(
        delta_ged=lambda g1, g2: 0.3,
        delta_ig=lambda v1, v2, k=None: 0.2
    )
    
    # Don't pre-mock layer4_llm - let tests handle it directly
    # sys.modules['insightspike.layer4_llm'] = ...
    
    sys.modules['insightspike.loader'] = types.SimpleNamespace(
        load_corpus=lambda path: ['mocked', 'documents']  # Accept path parameter
    )
    
    # Mock MainAgent in core
    sys.modules['insightspike.core.agents.main_agent'] = types.SimpleNamespace(
        MainAgent=MockMainAgent
    )
    
    yield
    
    # Cleanup is handled automatically by pytest

@pytest.fixture
def mock_memory():
    """Fixture for creating a mock memory object"""
    return types.SimpleNamespace(
        search=lambda q, k: ([0.8], [0]),
        update_c=lambda idxs, r, eta=0.1: None,
        train_index=lambda: None,
        merge=lambda idxs: None,
        split=lambda idx: None,
        prune=lambda c, i: None,
        add_episode=lambda v, t, c_init=0.2: None,
        save=lambda: MockPath('test_path'),
        episodes=[
            types.SimpleNamespace(vec=np.array([0.1, 0.2, 0.3]), text='test document', c=0.5)
        ]
    )

@pytest.fixture
def sample_vectors():
    """Fixture for sample vector data"""
    return np.random.random((5, 384))

@pytest.fixture
def sample_graph():
    """Fixture for sample graph"""
    g = MockGraph()
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    return g
