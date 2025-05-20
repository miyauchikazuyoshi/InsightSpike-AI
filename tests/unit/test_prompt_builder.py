from insightspike.prompt_builder import build_prompt


def test_build_prompt():
    p = build_prompt('q', ['d1', 'd2'])
    assert '[Answer]' in p and 'd1' in p and 'q' in p
