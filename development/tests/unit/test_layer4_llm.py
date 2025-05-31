from unittest.mock import patch, MagicMock

def test_generate():
    # Mock the transformers components before importing the module
    with patch('transformers.pipeline') as mock_pipeline_func:
        with patch('transformers.AutoTokenizer') as mock_tokenizer:
            with patch('transformers.AutoModelForCausalLM') as mock_model:
                # Set up the mocks
                mock_tokenizer.from_pretrained.return_value = MagicMock()
                mock_model.from_pretrained.return_value = MagicMock()
                
                # Create a mock pipeline that returns our expected output
                mock_pipeline = MagicMock()
                mock_pipeline.return_value = [{'generated_text': 'hi answer'}]
                mock_pipeline_func.return_value = mock_pipeline
                
                # Import after setting up mocks
                from insightspike.layer4_llm import generate
                
                result = generate('hi')
                assert result == 'hi answer'
