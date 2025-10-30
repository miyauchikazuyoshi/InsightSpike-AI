import sys
import types


def _install_sklearn_stub():
    # Provide a minimal sklearn.metrics.pairwise.cosine_similarity stub
    if 'sklearn' in sys.modules:
        return
    sklearn = types.ModuleType('sklearn')
    metrics = types.ModuleType('sklearn.metrics')
    pairwise = types.ModuleType('sklearn.metrics.pairwise')

    def cosine_similarity(a, b):
        return 0.0

    pairwise.cosine_similarity = cosine_similarity
    metrics.pairwise = pairwise
    sklearn.metrics = metrics
    sys.modules['sklearn'] = sklearn
    sys.modules['sklearn.metrics'] = metrics
    sys.modules['sklearn.metrics.pairwise'] = pairwise


def test_layer4_uses_providerfactory_for_mock(monkeypatch):
    _install_sklearn_stub()

    # Import after installing stub
    from insightspike.implementations.layers.layer4_llm_interface import (
        L4LLMInterface,
        LLMConfig,
        LLMProviderType,
    )

    # Stub ProviderFactory to ensure PF path is exercised
    class _StubPFProvider:
        def generate(self, prompt: str, **kwargs) -> str:
            return "PF_OK"

    def _stub_create_from_config(cfg):
        return _StubPFProvider()

    from insightspike.providers import provider_factory as pf

    monkeypatch.setattr(pf.ProviderFactory, "create_from_config", _stub_create_from_config)

    # Build interface with mock provider and initialize
    cfg = LLMConfig(provider=LLMProviderType.MOCK, model_name="mock-model")
    llm = L4LLMInterface(cfg)
    assert llm.initialize() is True

    # Ensure generation goes through ProviderFactory path
    res = llm.generate_response_detailed({"retrieved_documents": []}, "hello")
    assert res.get("success") is True
    assert res.get("response") == "PF_OK"
