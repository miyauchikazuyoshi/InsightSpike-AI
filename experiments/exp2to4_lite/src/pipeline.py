"""Experiment orchestration for Exp II–III (self-contained)."""

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
import os
import time
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
    zsr: float


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
        ig_mode=cfg.ig_mode,
        spike_mode=cfg.spike_mode,
        theta_ag=cfg.theta_ag,
        theta_dg=cfg.theta_dg,
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
    embedder = Embedder(cfg.embedding_model, normalize=cfg.normalize_embeddings, cache_dir=str(cfg.embedding_cache) if cfg.embedding_cache else None)
    global_retriever: HybridRetriever | None = None

    if cfg.retrieval_scope != "per_query":
        global_retriever = _init_retriever(cfg, embedder)
        # Build corpus
        corpus: Dict[str, Tuple[str, Dict[str, str]]] = {}
        for sample in dataset:
            for doc in sample.documents:
                corpus[doc.doc_id] = (doc.text, doc.metadata)
        if corpus:
            global_retriever.add_corpus((doc_id, text, metadata) for doc_id, (text, metadata) in corpus.items())

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
            if cfg.retrieval_scope == "per_query":
                retriever = _init_retriever(cfg, embedder)
                docs = [(doc.doc_id, doc.text, doc.metadata) for doc in sample.documents]
                if docs:
                    retriever.add_corpus(docs)
            else:
                if global_retriever is None:
                    raise RuntimeError("Global retriever is not initialized")
                retriever = global_retriever
            strategy = build_strategy(
                strategy_type=baseline_cfg.type,
                params=baseline_cfg.params,
                controller=gedig_controller if baseline_cfg.type == "gedig" else None,
                top_k=cfg.retrieval_top_k,
            )
            t0 = time.perf_counter()
            result: StrategyResult = strategy.run(sample.query, retriever, memory)
            t1 = time.perf_counter()
            per = compute_per(result.answer, sample.ground_truth)
            accepted = compute_acceptance(result.answer, sample.ground_truth, cfg.psz_acceptance_threshold)
            if os.environ.get("EXP_LITE_REAL_LATENCY", "0") in ("1", "true", "True"):
                latency = float((t1 - t0) * 1000.0)
            else:
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
                    "ged_min_proxy": getattr(result.gate_state.result, "ged_min_proxy", 0.0),
                    "backend": getattr(gedig_controller, "backend", "lite"),
                }
                gate_logs.append(log_entry)
                if result.gate_state.ag:
                    total_ag += 1
                if result.gate_state.dg:
                    total_dg += 1

            retrieved_ids = [doc.doc_id for doc in result.retrieved_docs]
            per_samples.append(
                {
                    "query": sample.query,
                    "ground_truth": sample.ground_truth,
                    "answer": result.answer,
                    "steps": result.steps,
                    "latency_ms": latency,
                    "accepted": bool(accepted),
                    "per": per,
                    "ag": bool(result.gate_state.ag) if result.gate_state else False,
                    "dg": bool(result.gate_state.dg) if result.gate_state else False,
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
        # Zero-hop success rate (ZSR): fraction of cases with no AG firing across iterations
        def _is_zero_hop(sample: Dict[str, Any]) -> bool:
            meta = sample.get("metadata", {}) or {}
            iter_ag = meta.get("iter_ag", []) or []
            try:
                return sum(int(x) for x in iter_ag) == 0
            except Exception:
                return False

        zsr = float(sum(1 for s in per_samples if _is_zero_hop(s)) / total_cases) if total_cases > 0 else 0.0

        summaries[baseline_cfg.name] = StrategySummary(
            metrics=metrics_summary,
            gate_logs=gate_logs,
            ag_rate=ag_rate,
            dg_rate=dg_rate,
            avg_steps=avg_steps,
            per_samples=per_samples,
            qa_pairs=qa_pairs,
            zsr=zsr,
        )

    output_payload = {
        "config": {
            "name": cfg.name,
            "dataset": str(cfg.dataset_path),
            "num_queries": len(dataset),
            "lambda_weight": float(cfg.gedig.lambda_weight),
            "retrieval_scope": cfg.retrieval_scope,
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
            "zsr": summary.zsr,
            "lambda_weight": float(cfg.gedig.lambda_weight),
        }

    result_path = _save_results(cfg.output_dir, cfg.name, output_payload)

    for baseline_name, summary in summaries.items():
        if summary.qa_pairs:
            qa_path = result_path.with_name(f"{result_path.stem}_{baseline_name}_qa_pairs.jsonl")
            with qa_path.open("w", encoding="utf-8") as qa_fh:
                for pair in summary.qa_pairs:
                    qa_fh.write(json.dumps(pair, ensure_ascii=False) + "\n")

    return result_path
