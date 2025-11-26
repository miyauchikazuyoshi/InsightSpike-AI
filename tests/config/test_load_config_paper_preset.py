from insightspike.config import load_config


def test_load_config_supports_paper_preset():
    cfg = load_config(preset="paper")
    assert cfg.graph.ged_norm_scheme == "candidate_base"
    assert cfg.graph.ig_source_mode == "linkset"
    assert cfg.graph.sp_beta == 1.0
    assert cfg.graph.lambda_weight == 1.0
