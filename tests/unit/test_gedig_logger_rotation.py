import os
import tempfile
import networkx as nx
from insightspike.algorithms.gedig_core import GeDIGCore, GeDIGLogger, GeDIGResult


def test_logger_rotation_by_lines():
    g1 = nx.path_graph(3)
    g2 = nx.path_graph(4)
    with tempfile.TemporaryDirectory() as tmp:
        log_path = os.path.join(tmp, 'gedig_log.csv')
        logger = GeDIGLogger(log_path, max_lines=3, max_bytes=10_000_000)
        core = GeDIGCore()
        core.logger = logger
        # Trigger 5 writes -> should create at least 2 files (3 lines + rollover)
        for _ in range(5):
            core.calculate(g_prev=g1, g_now=g2)
        logger.close()
        files = sorted(f for f in os.listdir(tmp) if f.startswith('gedig_log'))
        assert len(files) >= 2, f"Rotation did not occur: files={files}"
        # Validate each file has a single header line
        for f in files:
            with open(os.path.join(tmp, f)) as fh:
                lines = fh.readlines()
            header_count = sum(1 for line in lines if line.startswith('step,'))
            assert header_count == 1, f"File {f} header count {header_count} != 1"


def test_logger_rotation_by_size():
    g1 = nx.complete_graph(10)
    g2 = nx.complete_graph(12)
    with tempfile.TemporaryDirectory() as tmp:
        log_path = os.path.join(tmp, 'gedig_log.csv')
        # Very small byte limit to force rotation quickly
        logger = GeDIGLogger(log_path, max_lines=1000, max_bytes=300)
        core = GeDIGCore()
        core.logger = logger
        for _ in range(20):
            core.calculate(g_prev=g1, g_now=g2)
        logger.close()
        files = sorted(f for f in os.listdir(tmp) if f.startswith('gedig_log'))
        assert len(files) >= 2, f"Size-based rotation did not occur: files={files}"
        for f in files:
            with open(os.path.join(tmp, f)) as fh:
                lines = fh.readlines()
            header_count = sum(1 for line in lines if line.startswith('step,'))
            assert header_count == 1, f"File {f} header count {header_count} != 1"

def test_logger_io_append_and_read(tmp_path):
    """Ensure subsequent writes append and headers not duplicated mid-file."""
    log_path = tmp_path / "gedig_metrics.csv"
    logger = GeDIGLogger(str(log_path), max_lines=50, max_bytes=10_000_000)
    core = GeDIGCore()
    for i in range(10):
        res = GeDIGResult(gedig_value=0.1*i, ged_value=0.0, ig_value=0.0, structural_improvement=0.01*i)
        logger.log(i, res)
    # Append more
    for i in range(10,20):
        res = GeDIGResult(gedig_value=0.1*i, ged_value=0.0, ig_value=0.0, structural_improvement=0.01*i)
        logger.log(i, res)
    with open(str(log_path).replace('.csv','_0.csv')) as fh:
        content = fh.read().strip().splitlines()
    assert content[0].startswith('step')
    # No additional header lines inside
    assert sum(1 for line in content if line.startswith('step')) == 1


def test_logger_rotation_with_compression(tmp_path):
    """Verify that compress_on_rotate produces .gz files after rotation."""
    log_path = tmp_path / "gedig_metrics_comp.csv"
    logger = GeDIGLogger(str(log_path), max_lines=2, max_bytes=10_000_000, compress_on_rotate=True)
    core = GeDIGCore(); core.logger = logger
    import networkx as nx
    g1 = nx.path_graph(3); g2 = nx.path_graph(4)
    # Trigger several rotations (each call may log one row)
    for _ in range(7):
        core.calculate(g_prev=g1, g_now=g2)
    logger.close()
    files = [p.name for p in tmp_path.iterdir()]
    compressed = [f for f in files if f.startswith('gedig_metrics_comp') and f.endswith('.gz')]
    assert compressed, f"No compressed rotated files found: {files}"
