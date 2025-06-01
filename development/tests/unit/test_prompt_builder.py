from insightspike.utils.prompt_builder import PromptBuilder


def test_build_prompt():
    pb = PromptBuilder()
    # Test simple prompt building
    simple_p = pb.build_simple_prompt(['d1', 'd2'], 'q')
    assert 'q' in simple_p and 'd1' in simple_p
    
    # Test fallback functionality
    fallback_p = pb._fallback_prompt('q')
    assert 'q' in fallback_p
