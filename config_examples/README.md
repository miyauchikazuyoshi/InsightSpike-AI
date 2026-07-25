# InsightSpike configuration examples

Every YAML file in this directory is a complete, canonical
`InsightSpikeConfig`. The examples are validated with the same strict loader
used by the application, so unknown root or nested keys are rejected.

## Available examples

- `anthropic_config.yaml`: Anthropic provider with a local embedding model.
- `openai_config.yaml`: OpenAI provider with a local embedding model.
- `config_datastore.yaml`: SQLite-backed episode storage.
- `config_experiment_optimized.yaml`: Larger experiment-oriented capacities
  and runtime tuning.
- `config_message_passing.yaml`: Question-aware graph message passing and edge
  reevaluation.
- `config_normalized.yaml`: Advanced GED/IG with normalized metric settings.
- `ged_ig_algorithms.yaml`: Canonical GED/IG algorithm selection; edit the
  documented enum values or override them at load time.

## Load an example

```python
from insightspike.config import load_config

config = load_config(
    config_path="config_examples/config_message_passing.yaml",
)
```

To use a preset as the lower-priority base, select both sources explicitly:

```python
from insightspike.config import load_config

config = load_config(
    preset="experiment",
    config_path="config_examples/ged_ig_algorithms.yaml",
    overrides={
        "graph": {
            "ged_algorithm": "networkx",
            "ig_algorithm": "entropy",
        },
    },
)
```

The source priority is:

1. explicit `overrides`
2. supported environment variables
3. the selected configuration file
4. the selected preset
5. model defaults

You can also copy one complete example to the project root:

```bash
cp config_examples/config_normalized.yaml config.yaml
```

With no explicit preset or path, `load_config()` discovers `config.yaml`,
`.insightspike.yaml`, then `config.json` in the current directory.

## Canonical structure

Configuration sections match `InsightSpikeConfig`; there is no `core` or
`retrieval` section.

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
  device: cpu

memory:
  max_retrieved_docs: 10
  episodic_memory_capacity: 60

graph:
  ged_algorithm: advanced
  ig_algorithm: advanced
  similarity_threshold: 0.3
  spike_ged_threshold: -0.5
  spike_ig_threshold: 0.2

metrics:
  use_normalized_ged: true

datastore:
  type: memory
```

## Supported environment variables

Configuration overrides are explicit; arbitrary
`INSIGHTSPIKE_<SECTION>_<KEY>` names are not decoded. Nested names use a
double underscore.

```bash
export INSIGHTSPIKE_CONFIG_PATH=config_examples/config_normalized.yaml
export INSIGHTSPIKE_ENVIRONMENT=production

export INSIGHTSPIKE_LLM__PROVIDER=openai
export INSIGHTSPIKE_LLM__MODEL=gpt-4.1-mini
export INSIGHTSPIKE_LLM__TEMPERATURE=0.2
export INSIGHTSPIKE_LLM__MAX_TOKENS=1024

export INSIGHTSPIKE_MEMORY__EPISODIC_MEMORY_CAPACITY=200
export INSIGHTSPIKE_MEMORY__MAX_RETRIEVED_DOCS=20

export INSIGHTSPIKE_DATASTORE__TYPE=sqlite
export INSIGHTSPIKE_DATASTORE__ROOT_PATH=./data/insight_store
export INSIGHTSPIKE_DATASTORE__DB_PATH=./data/insight.db

export INSIGHTSPIKE_LOGGING__LEVEL=INFO
export INSIGHTSPIKE_LOGGING__FILE_PATH=./logs/insightspike.log
```

The compatibility names `INSIGHTSPIKE_MODEL_NAME`,
`INSIGHTSPIKE_DATA_DIR`, and `INSIGHTSPIKE_LOG_DIR` are also accepted.

Provider credentials are separate from configuration overrides. Set
`OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in the process environment when using
those providers. The YAML loader does not interpolate strings such as
`${OPENAI_API_KEY}`, so the examples intentionally do not store credential
placeholders.

## Save a resolved configuration

`ConfigLoader` is available from its loader module:

```python
from insightspike.config.loader import ConfigLoader

loader = ConfigLoader()
config = loader.load(
    config_path="config_examples/config_datastore.yaml",
)
loader.save("resolved-config.yaml")
```

Saved YAML is portable and can be read with `yaml.safe_load`.

## See also

- [Configuration guide](../docs/user-guide/configuration_guide.md)
- [Configuration architecture](../docs/architecture/configuration.md)
