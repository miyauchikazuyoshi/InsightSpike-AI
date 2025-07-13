import os
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest


@pytest.mark.skipif(
    "CI" in os.environ or "GITHUB_ACTIONS" in os.environ,
    reason="Skip in CI due to complex torch_geometric dependencies"
)
def test_generate():
    """Test LLM generation in safe mode."""
    # Force LITE_MODE for CI tests to avoid torch import issues
    os.environ["INSIGHTSPIKE_LITE_MODE"] = "1"
    
    # Skip transformers import in LITE_MODE
    if os.environ.get("INSIGHTSPIKE_LITE_MODE", "0") == "1":
        # In LITE_MODE, just test that we can get a mock provider
        from insightspike.core.layers.layer4_llm_provider import get_llm_provider

        llm = get_llm_provider(safe_mode=True)
        result = llm.generate_response("", "hi")
        # Mock provider should return something
        assert isinstance(result, str)
        assert len(result) > 0
    else:
        # Full test with transformers mocking
        with patch("transformers.pipeline") as mock_pipeline_func:
            with patch("transformers.AutoTokenizer") as mock_tokenizer:
                with patch("transformers.AutoModelForCausalLM") as mock_model:
                    # Set up the mocks
                    mock_tokenizer.from_pretrained.return_value = MagicMock()
                    mock_model.from_pretrained.return_value = MagicMock()

                    # Create a mock pipeline that returns our expected output
                    mock_pipeline = MagicMock()
                    mock_pipeline.return_value = [{"generated_text": "hi answer"}]
                    mock_pipeline_func.return_value = mock_pipeline

                    # Import after setting up mocks
                    from insightspike.core.layers.layer4_llm_provider import (
                        get_llm_provider,
                    )

                    llm = get_llm_provider(safe_mode=True)
                    result = llm.generate_response("", "hi")
                    assert isinstance(result, str)
                    assert len(result) > 0  # Accept any non-empty response
