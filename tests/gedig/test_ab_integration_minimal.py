import os, tempfile, csv, time
from insightspike.implementations.agents.main_agent import MainAgent

def test_mainagent_ab_minimal_one_record():
    # minimal config enabling ab mode
    cfg = {
        'gedig': {'mode': 'ab'},
        'embedding': {'model_name': None, 'dimension': 64},
        'processing': {'enable_learning': False},
        'memory': {'graph': False},
    }
    os.environ.setdefault('INSIGHTSPIKE_MIN_IMPORT', '1')
    os.environ.setdefault('INSIGHTSPIKE_IMPORT_MAX_LAYER', '2')
    agent = MainAgent(config=cfg)
    logger = getattr(agent, '_gedig_ab_logger', None)
    assert logger is not None, 'A/B logger not initialized'
    # synthetic single pair
    pure = {'gedig': 0.5, 'ged': 0.7, 'ig': 0.4}
    full = {'gedig': 0.502, 'ged': 0.7, 'ig': 0.4}
    logger.record('q0', pure, full)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'ab.csv')
        logger.export_csv(p)
        with open(p) as f:
            r = csv.reader(f)
            header = next(r)
            assert 'k_missing_reason' in header
            row = next(r)
            k_idx = header.index('k_estimate')
            reason_idx = header.index('k_missing_reason')
            assert row[k_idx] not in ('', 'None')
            assert row[reason_idx] == ''
