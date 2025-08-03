#!/usr/bin/env python3
"""
Create experiment configuration with 2-hop GED/IG
"""

import yaml

# Load current config
with open('experiment_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Add algorithms section at graph level
if 'algorithms' not in config['insightspike']['graph']:
    config['insightspike']['graph']['algorithms'] = {}

config['insightspike']['graph']['algorithms'] = {
    'ged': 'advanced',  # Use gedig_core
    'ig': 'advanced'    # Use gedig_core
}

# Ensure metrics section exists with multihop settings
if 'metrics' not in config['insightspike']['graph']:
    config['insightspike']['graph']['metrics'] = {}

config['insightspike']['graph']['metrics'].update({
    'use_multihop_gedig': True,
    'max_hops': 2,
    'decay_factor': 0.5
})

# Also set at the graph level for L3GraphReasoner
config['insightspike']['graph']['ged_algorithm'] = 'advanced'
config['insightspike']['graph']['ig_algorithm'] = 'advanced'

# Save as new config
with open('experiment_config_2hop.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print("Created experiment_config_2hop.yaml with 2-hop settings")
print("\nKey settings:")
print(f"  - GED algorithm: {config['insightspike']['graph']['ged_algorithm']}")
print(f"  - IG algorithm: {config['insightspike']['graph']['ig_algorithm']}")
print(f"  - Multihop enabled: {config['insightspike']['graph']['metrics']['use_multihop_gedig']}")
print(f"  - Max hops: {config['insightspike']['graph']['metrics']['max_hops']}")
print(f"  - Decay factor: {config['insightspike']['graph']['metrics']['decay_factor']}")