# InsightSpike Configuration Guide

## Overview

InsightSpike uses one canonical, nested Pydantic model:
`insightspike.config.InsightSpikeConfig`. Configuration can come from a
preset, YAML/JSON file, supported environment variables, and explicit
overrides.

When multiple sources are selected, priority is:

1. explicit overrides
2. environment variables
3. configuration file
4. preset
5. model defaults

Each source is migrated independently before merging. This matters when a
higher-priority source still uses an older key name.

## Quick start

```python
from insightspike.config import load_config

config = load_config(
    preset="development",
    overrides={
        "llm": {"temperature": 0.2},
        "datastore": {"type": "memory"},
    },
)
```

An explicit preset is self-contained. A local `config.yaml` is merged only
when no preset is selected, or when a file is explicitly selected through
`config_path` or `INSIGHTSPIKE_CONFIG_PATH`.

```python
config = load_config(config_path="config.yaml")
```

Available presets are `development`, `experiment`, `production`, `research`,
`cloud`/`testing`, `paper`, `production_optimized`, `minimal`,
`graph_enhanced`, and `adaptive_learning`.

## Canonical YAML example

```yaml
environment: development
pre_warm_models: false

llm:
  provider: mock
  model: mock
  temperature: 0.3
  max_tokens: 256

embedding:
  model_name: sentence-transformers/all-MiniLM-L6-v2
  dimension: 384

memory:
  max_retrieved_docs: 10
  episodic_memory_capacity: 60

graph:
  similarity_threshold: 0.3
  spike_ged_threshold: -0.5
  spike_ig_threshold: 0.2
  enable_message_passing: false

datastore:
  type: filesystem
  root_path: ./data/insight_store

output:
  include_reasoning: false
  include_metadata: false

monitoring:
  enabled: true
  performance_tracking: true
```

Unknown root and nested keys are rejected by strict file loading. For example,
`graph.similiarity_threshold` fails validation instead of silently using the
default.

## Datastore settings

| Setting | Default | Description |
|---|---:|---|
| `datastore.type` | `filesystem` | `filesystem`, `memory`, or `sqlite`; `in_memory` remains a legacy alias |
| `datastore.root_path` | `./data/insight_store` | Filesystem base directory and SQLite fallback location |
| `datastore.db_path` | `null` | Explicit SQLite database path |
| `datastore.vector_dim` | `null` | SQLite vector dimension; embedding dimension is used when omitted |

Examples:

```yaml
datastore:
  type: memory
```

```yaml
datastore:
  type: sqlite
  db_path: ./data/insight_store/state.sqlite3
  vector_dim: 384
```

## Supported environment variables

Nested names use a double underscore:

```bash
export INSIGHTSPIKE_LLM__PROVIDER=openai
export INSIGHTSPIKE_LLM__MODEL=gpt-4.1-mini
export INSIGHTSPIKE_LLM__TEMPERATURE=0.2
export INSIGHTSPIKE_MEMORY__MAX_RETRIEVED_DOCS=20
export INSIGHTSPIKE_DATASTORE__TYPE=sqlite
export INSIGHTSPIKE_DATASTORE__DB_PATH=./state.sqlite3
export INSIGHTSPIKE_ENVIRONMENT=production
```

`INSIGHTSPIKE_CONFIG_PATH` selects a configuration file.
`INSIGHTSPIKE_MODEL_NAME`, `INSIGHTSPIKE_DATA_DIR`, and
`INSIGHTSPIKE_LOG_DIR` remain supported compatibility names.

## Save and reload

Use `ConfigLoader` when a configuration must be persisted:

```python
from insightspike.config.loader import ConfigLoader

loader = ConfigLoader()
config = loader.load(preset="development")
loader.save("resolved-config.yaml")

reloaded = ConfigLoader().load_from_file("resolved-config.yaml")
```

Saved YAML is portable and safe-loadable: `Path` values are strings and
Pydantic implementation metadata is not serialized.

## Legacy migration

Older dictionaries remain accepted at compatibility boundaries such as
`MainAgent(config=legacy_dict)`. Migrations and ignored unknown keys emit
structured `UserWarning` instances with their dotted paths.

| Legacy form | Canonical form |
|---|---|
| `l4_config` | `llm` |
| `llm.model_name` | `llm.model` |
| `embedding.model` | `embedding.model_name` |
| `output.show_reasoning` | `output.include_reasoning` |
| `output.show_metadata` | `output.include_metadata` |
| `monitoring.enable_monitoring` | `monitoring.enabled` |
| `monitoring.track_memory_usage` | `monitoring.performance_tracking` |
| `datastore.path` | `datastore.root_path` |
| `datastore.type: in_memory` | `datastore.type: memory` |
| `paths.log_dir` | `paths.logs_dir` |

Deprecated fields without a safe semantic equivalent are removed with a
warning: `output.response_style`, `monitoring.metrics_interval`,
`logging.format`, and `logging.file_enabled`.

When legacy and canonical fields coexist in the same source, the canonical
value wins and a conflict diagnostic is emitted.

## Direct model construction

Use direct construction when strict validation is desired without source
loading:

```python
from insightspike.config import InsightSpikeConfig

config = InsightSpikeConfig(
    llm={"provider": "mock"},
    graph={"similarity_threshold": 0.4},
)
```

Application code should pass the validated object and a configured datastore
to `MainAgent`; public helpers such as `create_agent()` perform this composition
automatically.
