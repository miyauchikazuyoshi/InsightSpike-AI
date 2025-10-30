from insightspike.algorithms.gedig_factory import GeDIGFactory

def test_factory_legacy_preset_applies_threshold_mode():
    core = GeDIGFactory.create({'use_refactored_gedig': False})
    assert core.use_refactored_reward is False
    assert core.use_legacy_formula is True
    assert str(core.spike_detection_mode) in ('threshold','SpikeDetectionMode.THRESHOLD','SpikeDetectionMode.threshold')
