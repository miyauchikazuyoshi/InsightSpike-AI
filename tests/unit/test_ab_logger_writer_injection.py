from io import StringIO


def test_ab_logger_writer_injection_and_export():
    from insightspike.algorithms.gedig_ab_logger import GeDIGABLogger

    logger = GeDIGABLogger(window=10, min_pairs=1, threshold=0.85, flush_every=50)

    # Record a couple of pairs (minimal fields)
    pure = {"gedig": 0.5, "ged": 0.2, "ig": 0.6}
    full = {"gedig": 0.4, "ged": 0.2, "ig": 0.5}
    logger.record("q1", pure, full)
    logger.record("q2", pure, full)

    # Inject writer and export
    buf = StringIO()
    logger.set_writer(buf)
    rows = logger.export_csv("ignored_path.csv")
    assert rows >= 2  # header + rows
    content = buf.getvalue()
    assert "query_id" in content and "pure_gedig" in content

