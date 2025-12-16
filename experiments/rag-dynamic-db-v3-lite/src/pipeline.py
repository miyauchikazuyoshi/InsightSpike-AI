"""Experiment orchestration for RAG v3-lite."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime

from .config_loader import BaselineConfig, ExperimentConfig, GeDIGConfig
from .dataset import load_dataset
from .embedder import Embedder
from .gedig_scoring import GeDIGController
from .graph_memory import GraphMemory
from .metrics import MetricsSummary, compute_acceptance, compute_per, simulate_latency_ms
from .retriever import HybridRetriever
from .strategies import StrategyResult, build_strategy


@dataclass
class StrategySummary:
    metrics: MetricsSummary
    gate_logs: List[Dict[str, Any]]
    ag_rate: float
    dg_rate: float
    avg_steps: float
    per_samples: List[Dict[str, Any]]
    qa_pairs: List[Dict[str, str]]


def _init_retriever(cfg: ExperimentConfig, embedder: Embedder) -> HybridRetriever:
    retriever = HybridRetriever(
        embedder=embedder,
        bm25_weight=cfg.retrieval_bm25_weight,
        embedding_weight=cfg.retrieval_embedding_weight,
        seed=cfg.seed,
    )
    return retriever


def _init_gedig_controller(cfg: GeDIGConfig) -> GeDIGController:
    return GeDIGController(
        lambda_weight=cfg.lambda_weight,
        use_multihop=cfg.use_multihop,
        max_hops=cfg.max_hops,
        decay_factor=cfg.decay_factor,
        sp_beta=cfg.sp_beta,
        sp_scope_mode=cfg.sp_scope_mode,
        sp_hop_expand=cfg.sp_hop_expand,
        sp_eval_mode=cfg.sp_eval_mode,
        sp_pair_samples=cfg.sp_pair_samples,
        sp_use_sampling=cfg.sp_use_sampling,
        ig_mode=cfg.ig_mode,
        spike_mode=cfg.spike_mode,
        theta_ag=cfg.theta_ag,
        theta_dg=cfg.theta_dg,
        entropy_tau=cfg.entropy_tau,
    )


def _save_results(output_dir: Path, experiment_name: str, payload: Dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"{experiment_name}_{timestamp}.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return out_path


def run_experiment(cfg: ExperimentConfig) -> Path:
    dataset = load_dataset(cfg.dataset_path, limit=cfg.max_queries)
    embedder = Embedder(cfg.embedding_model, normalize=cfg.normalize_embeddings, cache_dir=cfg.embedding_cache)
    retriever = _init_retriever(cfg, embedder)

    # Build corpus from dataset documents
    corpus: Dict[str, Tuple[str, Dict[str, str]]] = {}
    for sample in dataset:
        for doc in sample.documents:
            corpus[doc.doc_id] = (doc.text, doc.metadata)
    retriever.add_corpus((doc_id, text, metadata) for doc_id, (text, metadata) in corpus.items())

    gedig_controller = _init_gedig_controller(cfg.gedig)

    summaries: Dict[str, StrategySummary] = {}

    for baseline_cfg in cfg.baselines:
        metrics_summary = MetricsSummary()
        gate_logs: List[Dict[str, float]] = []
        per_samples: List[Dict[str, Any]] = []
        qa_pairs: List[Dict[str, str]] = []
        total_ag = 0
        total_dg = 0
        total_steps = 0

        for sample in dataset:
            memory = GraphMemory()
            strategy = build_strategy(
                strategy_type=baseline_cfg.type,
                params=baseline_cfg.params,
                controller=gedig_controller if baseline_cfg.type == "gedig" else None,
                top_k=cfg.retrieval_top_k,
            )
            result: StrategyResult = strategy.run(sample.query, retriever, memory)
            per = compute_per(result.answer, sample.ground_truth)
            accepted = compute_acceptance(result.answer, sample.ground_truth, cfg.psz_acceptance_threshold)
            latency = simulate_latency_ms(result.steps)
            metrics_summary.add(per, accepted, latency)
            total_steps += result.steps

            if result.gate_state:
                log_entry: Dict[str, Any] = {
                    "g0": result.gate_state.g0,
                    "gmin": result.gate_state.gmin,
                    "ag": float(result.gate_state.ag),
                    "dg": float(result.gate_state.dg),
                    "gedig_value": getattr(result.gate_state.result, "gedig_value", 0.0),
                    "backend": getattr(gedig_controller, "backend", "lite"),
                }
                if hasattr(result.gate_state.result, "structural_cost"):
                    log_entry["structural_cost"] = getattr(result.gate_state.result, "structural_cost", 0.0)
                elif hasattr(result.gate_state.result, "structural_improvement"):
                    legacy_value = getattr(result.gate_state.result, "structural_improvement", 0.0)
                    log_entry["structural_cost"] = -float(legacy_value)
                if hasattr(result.gate_state.result, "ig_value"):
                    log_entry["ig_value"] = getattr(result.gate_state.result, "ig_value", 0.0)
                if hasattr(result.gate_state.result, "ged_value"):
                    log_entry["ged_value"] = getattr(result.gate_state.result, "ged_value", 0.0)
                gate_logs.append(log_entry)
                ag_flag = bool(result.gate_state.ag)
                dg_flag = bool(result.gate_state.dg)
                if ag_flag:
                    total_ag += 1
                if dg_flag:
                    total_dg += 1
            else:
                ag_flag = False
                dg_flag = False

            retrieved_ids = [doc.doc_id for doc in result.retrieved_docs]
            per_samples.append(
                {
                    "query": sample.query,
                    "ground_truth": sample.ground_truth,
                    "answer": result.answer,
                    "steps": result.steps,
                    "accepted": bool(accepted),
                    "per": per,
                    "ag": ag_flag,
                    "dg": dg_flag,
                    "retrieved_doc_ids": retrieved_ids,
                    "metadata": result.metadata,
                }
            )
            if baseline_cfg.type == "gedig":
                qa_pairs.append({"question": sample.query, "response": result.answer})

        total_cases = len(dataset) if dataset else 1
        ag_rate = total_ag / total_cases
        dg_rate = total_dg / total_cases
        avg_steps = total_steps / total_cases
        summaries[baseline_cfg.name] = StrategySummary(
            metrics=metrics_summary,
            gate_logs=gate_logs,
            ag_rate=ag_rate,
            dg_rate=dg_rate,
            avg_steps=avg_steps,
            per_samples=per_samples,
            qa_pairs=qa_pairs,
        )

    output_payload = {
        "config": {
            "name": cfg.name,
            "dataset": str(cfg.dataset_path),
            "num_queries": len(dataset),
        },
        "results": {},
    }

    for name, summary in summaries.items():
        stats = summary.metrics.to_dict()
        output_payload["results"][name] = {
            **stats,
            "psz_inside": summary.metrics.inside_psz(
                cfg.psz_acceptance_threshold,
                cfg.psz_fmr_threshold,
                cfg.psz_latency_p50_ms,
            ),
            "ag_rate": summary.ag_rate,
            "dg_rate": summary.dg_rate,
            "avg_steps": summary.avg_steps,
            "gate_logs": summary.gate_logs,
            "per_samples": summary.per_samples,
        }

    result_path = _save_results(cfg.output_dir, cfg.name, output_payload)

    for baseline_name, summary in summaries.items():
        if summary.qa_pairs:
            qa_path = result_path.with_name(f"{result_path.stem}_{baseline_name}_qa_pairs.jsonl")
            with qa_path.open("w", encoding="utf-8") as qa_fh:
                for pair in summary.qa_pairs:
                    qa_fh.write(json.dumps(pair, ensure_ascii=False) + "\n")

    return result_path
