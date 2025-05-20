import importlib

cli = importlib.import_module('insightspike.cli')

def test_cli_app_exists():
    assert hasattr(cli, 'app')
