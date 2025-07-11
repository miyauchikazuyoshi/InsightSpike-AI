from insightspike.core.layers.layer4_prompt_builder import L4PromptBuilder


def test_build_prompt():
    pb = L4PromptBuilder()
    # Test prompt building with context
    context = {"documents": [{"text": "d1"}, {"text": "d2"}]}
    prompt = pb.build_prompt(context, 'q')
    assert 'q' in prompt
    
    # Test fallback functionality
    context_empty = {"documents": []}
    fb_p = pb.build_prompt(context_empty, 'q')
    assert 'q' in fb_p
