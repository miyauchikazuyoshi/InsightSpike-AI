"""Param fixtures for config pattern tests.

Provides ('name', 'config') for tests that expect parametrized configurations
without declaring @pytest.mark.parametrize in the test module.
"""
from __future__ import annotations

import pytest
from insightspike.config.presets import ConfigPresets


def _config_patterns():
    return [
        (
            "Basic (all disabled)",
            {
                'graph': {
                    'ged_algorithm': 'simple',
                    'ig_algorithm': 'simple',
                    'use_entropy_variance_ig': False,
                    'use_normalized_ged': False,
                    'use_multihop_gedig': False,
                }
            },
        ),
        (
            "Entropy Variance IG",
            {
                'graph': {
                    'ged_algorithm': 'simple',
                    'ig_algorithm': 'simple',
                    'use_entropy_variance_ig': True,
                    'use_normalized_ged': False,
                    'use_multihop_gedig': False,
                }
            },
        ),
        (
            "Normalized GED",
            {
                'graph': {
                    'ged_algorithm': 'simple',
                    'ig_algorithm': 'simple',
                    'use_entropy_variance_ig': False,
                    'use_normalized_ged': True,
                    'use_multihop_gedig': False,
                }
            },
        ),
        (
            "Multi-hop geDIG",
            {
                'graph': {
                    'ged_algorithm': 'simple',
                    'ig_algorithm': 'simple',
                    'use_entropy_variance_ig': False,
                    'use_normalized_ged': False,
                    'use_multihop_gedig': True,
                    'multihop_config': {
                        'max_hops': 2,
                        'decay_factor': 0.7,
                        'adaptive_hops': True,
                    },
                }
            },
        ),
        (
            "All Features",
            {
                'graph': {
                    'ged_algorithm': 'simple',
                    'ig_algorithm': 'simple',
                    'use_entropy_variance_ig': True,
                    'use_normalized_ged': True,
                    'use_multihop_gedig': True,
                    'multihop_config': {
                        'max_hops': 3,
                        'decay_factor': 0.8,
                        'adaptive_hops': True,
                    },
                }
            },
        ),
        ("Preset: Experiment", ConfigPresets.get_preset("experiment")),
        ("Preset: Research", ConfigPresets.get_preset("research")),
        ("Preset: Minimal", ConfigPresets.get_preset("minimal")),
    ]


@pytest.fixture(params=_config_patterns(), ids=lambda p: p[0])
def _config_pair(request):
    return request.param


@pytest.fixture
def name(_config_pair):
    return _config_pair[0]


@pytest.fixture
def config(_config_pair):
    return _config_pair[1]

