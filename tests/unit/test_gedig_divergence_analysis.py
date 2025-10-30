import tempfile, os, csv
from insightspike.algorithms.gedig_analysis import analyze_divergence


def test_analyze_divergence_basic():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'divergence.csv')
        with open(path, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['ts','legacy','ref','delta'])
            w.writerow(['1.0','0.10','0.12','0.02'])
            w.writerow(['2.0','0.30','0.65','0.35'])
        stats = analyze_divergence(path, threshold=0.3)
        d = stats.to_dict()
        assert d['count'] == 2
        assert abs(d['mean_delta'] - ((0.02+0.35)/2)) < 1e-9
        assert d['max_delta'] == 0.35
        assert d['pct_over_threshold'] == 0.5  # 1/2 rows over 0.3