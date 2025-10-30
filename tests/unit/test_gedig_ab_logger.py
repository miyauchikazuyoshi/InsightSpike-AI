import os
import logging

from insightspike.algorithms.gedig_ab_logger import GeDIGABLogger


def _mk_res(val, ged=None, ig=None):
    # Helper to build minimal result dict shape expected
    return {"gedig": float(val), "ged": ged, "ig": ig}


def test_ab_logger_threshold_warning_once(caplog):
    caplog.set_level(logging.WARNING)
    logger = GeDIGABLogger(window=10, threshold=0.9, min_pairs=3, flush_every=100)

    # Provide 3 pairs with perfectly negative correlation (≈ -1.0)
    pure_vals = [1.0, 2.0, 3.0]
    full_vals = [3.0, 2.0, 1.0]
    for i, (p, f) in enumerate(zip(pure_vals, full_vals), start=1):
        logger.record(f"q{i}", _mk_res(p), _mk_res(f))

    metrics = logger.current_metrics()
    assert metrics["count"] == 3
    assert metrics["gedig_corr"] < 0.0  # negative correlation
    # At least one warning emitted
    warns = [r for r in caplog.records if r.levelno == logging.WARNING and "correlation" in r.message]
    assert len(warns) >= 1, "Expected correlation warning after min_pairs reached and below threshold"


def test_ab_logger_auto_flush(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    path = tmp_path / "gedig_ab.csv"
    logger = GeDIGABLogger(window=10, threshold=0.1, min_pairs=2, flush_every=2)
    logger.set_auto_csv_path(str(path))

    # First two records trigger first auto-flush
    logger.record("q1", _mk_res(0.1), _mk_res(0.1))
    logger.record("q2", _mk_res(0.2), _mk_res(0.2))  # should auto-flush here
    assert path.exists(), "CSV should exist after first flush"
    with open(path) as f:
        lines1 = f.readlines()
    assert len(lines1) == 1 + 2  # header + 2 rows

    # Add two more records; second flush overwrites with full history (4 rows)
    logger.record("q3", _mk_res(0.3), _mk_res(0.3))
    logger.record("q4", _mk_res(0.4), _mk_res(0.4))
    with open(path) as f:
        lines2 = f.readlines()
    assert len(lines2) == 1 + 4

    # Info log for auto-flush present at least twice
    flush_logs = [r for r in caplog.records if r.levelno == logging.INFO and "auto-flush" in r.message]
    assert len(flush_logs) >= 2
