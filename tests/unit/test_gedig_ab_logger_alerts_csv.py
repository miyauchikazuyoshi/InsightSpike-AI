import logging
from insightspike.algorithms.gedig_ab_logger import GeDIGABLogger


def _mk(val):
    return {"gedig": float(val)}


def test_ab_logger_writes_alert_csv(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    base_path = tmp_path / "ab_metric.csv"
    logger = GeDIGABLogger(window=10, threshold=0.9, min_pairs=3, flush_every=50)
    logger.set_auto_csv_path(str(base_path))

    # 3 negatively correlated pairs => correlation ≈ -1 -> threshold(0.9) 下回り alert 発火
    pure = [1.0, 2.0, 3.0]
    full = [3.0, 2.0, 1.0]
    for i, (p, f) in enumerate(zip(pure, full), start=1):
        logger.record(f"q{i}", _mk(p), _mk(f))

    alerts_csv = base_path.with_name(base_path.stem + "_alerts.csv")
    assert alerts_csv.exists(), "alerts CSV が生成されていない"
    lines = alerts_csv.read_text().strip().splitlines()
    assert lines[0].startswith("count,gedig_corr,threshold"), "ヘッダ不整合"
    # 1回だけのアラート行 (現在の仕様: count 違いで再アラート可だがここでは3件のみ)
    assert len(lines) == 2, f"期待: 2 行 (header+1) 実際: {len(lines)}"

    warns = [r for r in caplog.records if r.levelno == logging.WARNING and "correlation" in r.message]
    assert len(warns) >= 1, "WARNING ログが出力されていない"
