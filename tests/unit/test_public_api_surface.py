import importlib


def test_public_api_create_agent_importable():
    mod = importlib.import_module("insightspike.public")
    assert hasattr(mod, "create_agent"), "public module must expose create_agent"
    assert callable(mod.create_agent)

