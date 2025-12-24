# Public API Reference (2025-09)

The stable, user‑facing entry points are exposed via `insightspike.public`.

## Imports

```python
from insightspike.public import create_agent, quick_demo, InsightAppWrapper
```

## Classes

### InsightAppWrapper(...)
- High-level wrapper for local knowledge apps (Streamlit-friendly)
- Parameters: provider, model, api_base, api_key, data_dir, temperature, max_cycles
- Methods:
  - ask(question: str, max_cycles: Optional[int] = None) -> str
  - learn(text: str) -> bool
  - get_recent_episodes(limit: int = 5) -> List[Dict[str, Any]]
  - get_graph_networkx(max_nodes: int = 50, min_similarity: float = 0.65, top_k: int = 3, force_connect: bool = True) -> networkx.Graph | None
  - get_stats() -> Dict[str, Any]
  - save() -> bool
  - reset() -> None

Example:
```python
app = InsightAppWrapper(provider="mock")
app.learn("geDIG is a graph distance signal.")
answer = app.ask("What is geDIG?")
```

## Functions

### create_agent(provider: str = "mock", **kwargs) -> MainAgent
- Returns an initialized agent with sensible defaults
- `provider`: `mock | openai | local | clean` (mock by default)
- Additional kwargs (e.g., `model`) are forwarded to the configuration

### quick_demo() -> None
- Runs a lightweight demo showcasing the pipeline

### load_config(...)
- Public wrapper for `insightspike.config.load_config`
- Returns a Pydantic `InsightSpikeConfig`

### get_config_summary(config: Optional[Any] = None) -> Dict
- Returns a lightweight diagnostic summary (memory settings and defaults)
- If `config` is None, loads the default config and summarizes it

### create_datastore(kind: str = "filesystem", **kwargs)
- Create a simple DataStore instance
- Allowed kinds: `filesystem`, `memory` (safe defaults only)
- Examples:
  - `create_datastore("filesystem", root="data")`
  - `create_datastore("memory")`

## Notes & Guardrails
- Top‑level examples MUST import from `insightspike.public` (CI enforced)
- geDIG must be computed via `algorithms.gedig.selector.compute_gedig` (CI enforced; STRICT guard)
- Provider strict mode: set `INSIGHTSPIKE_STRICT_PROVIDER=1` to forbid legacy/fallback initialization
- Lite mode: set `INSIGHTSPIKE_LITE_MODE=1` to minimize heavy imports

## See Also
- Architecture: geDIG selector and core (`docs/architecture/gedig_selector_and_core.md`)
- A/B logging writer injection (`algorithms/gedig_ab_logger.py` and `algorithms/gedig/ab_writer_helper.py`)
