"""
Test agent_loop module functionality
"""
import pytest
import numpy as np
from insightspike import agent_loop


@pytest.fixture
def mock_memory():
    """Mock memory for testing."""

    class MockMemory:
        def __init__(self):
            self.episodes = []

        def search(self, vec, k):
            return [], []

        def update_c(self, idxs, r, eta=0.1):
            pass

        def train_index(self):
            pass

        def prune(self, c, i):
            pass

        def merge(self, idxs):
            pass

        def split(self, idx):
            pass

    return MockMemory()


def test_cycle(mock_memory):
    """Test basic cycle function"""
    result = agent_loop.cycle(mock_memory, "test question")
    assert isinstance(result, dict)
    assert "answer" in result
    assert "graph" in result


def test_cycle_with_empty_memory():
    """Test cycle with empty memory"""

    class EmptyMemory:
        def __init__(self):
            self.episodes = []

        def search(self, vec, k):
            return [], []

        def update_c(self, idxs, r, eta=0.1):
            pass

        def train_index(self):
            pass

    mem = EmptyMemory()
    result = agent_loop.cycle(mem, "What is quantum physics?")
    assert result is not None
    assert isinstance(result, dict)


def test_cycle_with_single_document():
    """Test cycle with single document in memory"""
    import numpy as np

    class SingleDocMemory:
        def __init__(self):
            self.episodes = [
                type(
                    "obj",
                    (object,),
                    {"vec": np.random.random(384), "text": "Sample document text"},
                )
            ]

        def search(self, vec, k):
            return [0.8], [0]

        def update_c(self, idxs, r, eta=0.1):
            pass

        def train_index(self):
            pass

        def prune(self, c, i):
            pass

        def merge(self, idxs):
            pass

        def split(self, idx):
            pass

        def add_episode(self, vec, text, c_init=0.2):
            pass

        def save(self):
            return "test_path"

    mem = SingleDocMemory()
    result = agent_loop.cycle(mem, "What is a single document?")
    assert result is not None
    assert isinstance(result, dict)


def test_adaptive_loop():
    """Test adaptive loop with multiple questions"""
    import numpy as np

    class AdaptiveMemory:
        def __init__(self):
            self.episodes = [
                type(
                    "obj",
                    (object,),
                    {"vec": np.random.random(384), "text": "Sample document"},
                )
            ]

        def search(self, vec, k):
            return [0.5], [0]

        def update_c(self, idxs, r, eta=0.1):
            return False

        def train_index(self):
            pass

        def prune(self, c, i):
            pass

        def merge(self, idxs):
            pass

        def split(self, idx):
            pass

        def add_episode(self, vec, text, c_init=0.2):
            pass

        def save(self):
            return "adaptive_test"

    mem = AdaptiveMemory()
    questions = ["Test question 1", "Test question 2"]
    results = agent_loop.adaptive_loop(mem, questions)
    assert isinstance(results, list)
    assert len(results) == 2


def test_adaptive_loop_max_iterations():
    """Test adaptive loop with max iterations"""
    import numpy as np

    class MaxIterMemory:
        def __init__(self):
            self.episodes = [
                type(
                    "obj",
                    (object,),
                    {"vec": np.random.random(384), "text": "Sample text"},
                )
            ]

        def search(self, vec, k):
            return [0.3], [0]

        def update_c(self, idxs, r, eta=0.1):
            return False

        def train_index(self):
            pass

        def prune(self, c, i):
            pass

        def merge(self, idxs):
            pass

        def split(self, idx):
            pass

        def add_episode(self, vec, text, c_init=0.2):
            pass

        def save(self):
            return "max_test"

    mem = MaxIterMemory()
    questions = ["Why is quantum physics so strange?"]
    results = agent_loop.adaptive_loop(mem, questions, max_iterations=2)
    assert isinstance(results, list)
    assert len(results) >= 1
