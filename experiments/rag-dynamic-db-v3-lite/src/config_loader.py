"""YAML based configuration loader for RAG v3-lite experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class BaselineConfig:
    name: str
    type: str
    description: str = ""
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeDIGConfig:
    lambda_weight: float = 0.6
    use_multihop: bool = True
    max_hops: int = 3
    decay_factor: float = 0.7
    sp_beta: float = 0.2
    sp_scope_mode: str = "auto"
    sp_hop_expand: int = 0
    sp_eval_mode: str = "connected"
    sp_pair_samples: int = 400
    sp_use_sampling: bool = True
    theta_ag: float = 8.0
    theta_dg: float = 0.6
    ig_mode: str = "raw"
    spike_mode: str = "and"
    entropy_tau: float = 1.0


@dataclass
class ExperimentConfig:
    name: str
    output_dir: Path
    seed: int
    dataset_path: Path
    max_queries: Optional[int]
    embedding_model: Optional[str]
    normalize_embeddings: bool
    embedding_cache: Optional[Path]
    retrieval_top_k: int
    retrieval_bm25_weight: float
    retrieval_embedding_weight: float
    retrieval_expansion_hops: int
    gedig: GeDIGConfig
    baselines: List[BaselineConfig]
    psz_acceptance_threshold: float
    psz_fmr_threshold: float
    psz_latency_p50_ms: float
    log_save_step_logs: bool
    log_save_memory_snapshots: bool
    log_snapshot_interval: int

    @property
    def needs_sentence_transformer(self) -> bool:
        return self.embedding_model is not None


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: Path) -> ExperimentConfig:
    """Load YAML configuration into dataclasses."""

    data = _load_yaml(path)

    exp = data.get("experiment", {})
    dataset = data.get("dataset", {})
    embedding = data.get("embedding", {})
    retrieval = data.get("retrieval", {})
    gedig_section = data.get("gedig", {})
    baselines_section = data.get("baselines", [])
    psz = data.get("psz", {})
    logging_section = data.get("logging", {})

    baselines = [
        BaselineConfig(
            name=item["name"],
            type=item["type"],
            description=item.get("description", ""),
            params={k: v for k, v in item.items() if k not in {"name", "type", "description"}},
        )
        for item in baselines_section
    ]

    gedig = GeDIGConfig(
        lambda_weight=float(gedig_section.get("lambda", 0.6)),
        use_multihop=bool(gedig_section.get("use_multihop", True)),
        max_hops=int(gedig_section.get("max_hops", 3)),
        decay_factor=float(gedig_section.get("decay_factor", 0.7)),
        sp_beta=float(gedig_section.get("sp_beta", 0.2)),
        sp_scope_mode=str(gedig_section.get("sp_scope_mode", "auto")),
        sp_hop_expand=int(gedig_section.get("sp_hop_expand", 0)),
        sp_eval_mode=str(gedig_section.get("sp_eval_mode", "connected")),
        sp_pair_samples=int(gedig_section.get("sp_pair_samples", 400)),
        sp_use_sampling=bool(gedig_section.get("sp_use_sampling", True)),
        theta_ag=float(gedig_section.get("theta_ag", 8.0)),
        theta_dg=float(gedig_section.get("theta_dg", 0.6)),
        ig_mode=str(gedig_section.get("ig_mode", "raw")),
        spike_mode=str(gedig_section.get("spike_mode", "and")),
        entropy_tau=float(gedig_section.get("entropy_tau", 1.0)),
    )

    cfg = ExperimentConfig(
        name=str(exp.get("name", "rag_v3_lite")),
        output_dir=Path(exp.get("output_dir", "results")),
        seed=int(exp.get("seed", 42)),
        dataset_path=Path(dataset.get("path", "data/sample_queries.jsonl")),
        max_queries=dataset.get("max_queries"),
        embedding_model=embedding.get("model"),
        normalize_embeddings=bool(embedding.get("normalize", True)),
        embedding_cache=Path(embedding["cache_dir"]).expanduser() if embedding.get("cache_dir") else None,
        retrieval_top_k=int(retrieval.get("top_k", 4)),
        retrieval_bm25_weight=float(retrieval.get("bm25_weight", 0.5)),
        retrieval_embedding_weight=float(retrieval.get("embedding_weight", 0.5)),
        retrieval_expansion_hops=int(retrieval.get("expansion_hops", 1)),
        gedig=gedig,
        baselines=baselines,
        psz_acceptance_threshold=float(psz.get("acceptance_threshold", 0.95)),
        psz_fmr_threshold=float(psz.get("fmr_threshold", 0.02)),
        psz_latency_p50_ms=float(psz.get("latency_p50_threshold_ms", 200)),
        log_save_step_logs=bool(logging_section.get("save_step_logs", True)),
        log_save_memory_snapshots=bool(logging_section.get("save_memory_snapshots", False)),
        log_snapshot_interval=int(logging_section.get("snapshot_interval", 10)),
    )

    return cfg
