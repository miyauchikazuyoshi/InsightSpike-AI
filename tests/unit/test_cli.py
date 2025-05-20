import importlib
import sys, types

cli = importlib.import_module('insightspike.cli')

def dummy_build_graph(*args, **kwargs):
    return None, None

sys.modules['insightspike.layer3_graph_pyg'] = types.SimpleNamespace(build_graph=dummy_build_graph)

def test_cli_app_exists():
    assert hasattr(cli, 'app')
