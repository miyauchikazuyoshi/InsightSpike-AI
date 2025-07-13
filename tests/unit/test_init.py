import importlib


def test_about_version():
    mod = importlib.import_module("insightspike")
    assert isinstance(mod.About.VERSION, str)
