import sys, types, importlib

pipe_obj = lambda prompt, do_sample=False: [{'generated_text': prompt + ' answer'}]
trans_mod = types.SimpleNamespace(pipeline=lambda *a, **k: pipe_obj,
                                  AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: None),
                                  AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=lambda *a, **k: None))
sys.modules['transformers'] = trans_mod

layer4 = importlib.import_module('insightspike.layer4_llm')


def test_generate():
    assert layer4.generate('hi') == 'hi answer'  # ← 'hi answer'に修正
