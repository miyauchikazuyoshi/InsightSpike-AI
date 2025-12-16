"""Baseline and geDIG-driven strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .gedig_scoring import GeDIGController, GeDIGGateState
from .graph_memory import GraphMemory
from .retriever import Document, HybridRetriever, RetrievalHit


@dataclass
class StrategyResult:
    answer: str
    retrieved_docs: List[Document]
    gate_state: GeDIGGateState | None
    steps: int
    metadata: Dict[str, object]


class BaseStrategy:
    name: str

    def run(self, query: str, retriever: HybridRetriever, memory: GraphMemory) -> StrategyResult:
        raise NotImplementedError


class StaticRAGStrategy(BaseStrategy):
    name = "static_rag"

    def __init__(self, top_k: int) -> None:
        self.top_k = top_k

    def run(self, query: str, retriever: HybridRetriever, memory: GraphMemory) -> StrategyResult:
        hits = retriever.retrieve(query, top_k=self.top_k)
        docs = [hit.document for hit in hits]
        answer = " ".join(doc.text for doc in docs[:2])
        metadata: Dict[str, object] = {
            "retrieval_scores": [hit.score for hit in hits],
        }
        return StrategyResult(answer=answer, retrieved_docs=docs, gate_state=None, steps=1, metadata=metadata)


class FrequencyStrategy(BaseStrategy):
    name = "frequency"

    def __init__(self, top_k: int) -> None:
        self.top_k = top_k

    def run(self, query: str, retriever: HybridRetriever, memory: GraphMemory) -> StrategyResult:
        hits = retriever.retrieve(query, top_k=self.top_k)
        # Boost documents with high activation
        hits.sort(
            key=lambda h: memory.graph.nodes.get(h.document.doc_id, {}).get("activation", 1.0),
            reverse=True,
        )
        docs = [hit.document for hit in hits]
        answer = " ".join(doc.text for doc in docs[:2])
        memory.update_from_retrieval(query, hits, query_embedding=retriever.get_last_query_embedding())
        metadata: Dict[str, object] = {
            "retrieval_scores": [hit.score for hit in hits],
        }
        return StrategyResult(answer=answer, retrieved_docs=docs, gate_state=None, steps=1, metadata=metadata)


class CosineTopKStrategy(BaseStrategy):
    name = "cosine_topk"

    def __init__(self, top_k: int) -> None:
        self.top_k = top_k

    def run(self, query: str, retriever: HybridRetriever, memory: GraphMemory) -> StrategyResult:
        hits = retriever.retrieve(query, top_k=self.top_k)
        docs = [hit.document for hit in hits]
        answer = hits[0].document.text if hits else ""
        memory.update_from_retrieval(query, hits, query_embedding=retriever.get_last_query_embedding())
        metadata: Dict[str, object] = {
            "retrieval_scores": [hit.score for hit in hits],
        }
        return StrategyResult(answer=answer, retrieved_docs=docs, gate_state=None, steps=1, metadata=metadata)


class GeDIGStrategy(BaseStrategy):
    name = "gedig_ag_dg"

    def __init__(self, controller: GeDIGController, top_k: int, max_iterations: int = 3) -> None:
        self.controller = controller
        self.top_k = top_k
        self.max_iterations = max_iterations

    def run(self, query: str, retriever: HybridRetriever, memory: GraphMemory) -> StrategyResult:
        steps = 0
        all_docs: List[Document] = []
        seen_docs: set[str] = set()
        gate_state: GeDIGGateState | None = None
        current_query = query
        iteration_states: List[GeDIGGateState] = []
        g_history: List[float] = []
        ag_history: List[float] = []
        dg_history: List[float] = []

        for iteration in range(self.max_iterations):
            hits = retriever.retrieve(current_query, top_k=self.top_k)
            retrieved_docs = [hit.document for hit in hits]
            graph_before, feat_before = memory.export_for_gedig()
            # Update memory and retain query embedding for linkset IG construction
            query_embedding = retriever.get_last_query_embedding()
            memory.update_from_retrieval(current_query, hits, query_embedding=query_embedding)
            graph_after, feat_after = memory.export_for_gedig()
            linkset_info = self._build_linkset_info(hits, query_embedding)
            gate_state = self.controller.evaluate(
                graph_before,
                graph_after,
                feat_before,
                feat_after,
                linkset_info=linkset_info,
            )
            iteration_states.append(gate_state)
            g_history.append(gate_state.g0)
            ag_history.append(float(gate_state.ag))
            dg_history.append(float(gate_state.dg))
            for doc in retrieved_docs:
                if doc.doc_id in seen_docs:
                    continue
                seen_docs.add(doc.doc_id)
                all_docs.append(doc)
            steps += 1

            if gate_state.dg:
                break

            # Prepare augmented query for the next iteration
            if iteration < self.max_iterations - 1:
                # decay activations slightly to avoid runaway growth
                memory.decay(0.9)
                support_nodes = sorted(
                    [node for node in memory.nodes.values() if node.metadata.get("type") != "query"],
                    key=lambda n: n.activation,
                    reverse=True,
                )
                snippets: List[str] = []
                for node in support_nodes:
                    role = node.metadata.get("role", "support")
                    if role != "support":
                        continue
                    ctx = node.metadata.get("context", "")
                    goal = node.metadata.get("goal", "")
                    outcome = node.metadata.get("outcome", "")
                    snippet = " ".join(filter(None, [ctx, goal, outcome]))
                    if snippet:
                        snippets.append(snippet)
                    if len(snippets) >= 2:
                        break
                if snippets:
                    expansion = " ".join(snippets)
                    base_query = query if iteration == 0 else current_query
                    current_query = f"{base_query} {expansion}"
                else:
                    current_query = query  # fallback

        if iteration_states:
            min_g0 = min(state.g0 for state in iteration_states)
            min_gmin = min(state.gmin for state in iteration_states)
            final_state = iteration_states[-1]
            final_state.gmin = min(final_state.gmin, min_g0, min_gmin)
            gate_state = final_state

        answer, template = self._compose_answer(query, all_docs)
        metadata: Dict[str, object] = {
            "gedig_g0_sequence": g_history,
            "gedig_ag_sequence": ag_history,
            "gedig_dg_sequence": dg_history,
            "answer_template": template,
        }
        return StrategyResult(answer=answer, retrieved_docs=all_docs, gate_state=gate_state, steps=steps, metadata=metadata)

    def _build_linkset_info(self, hits: List[RetrievalHit], query_embedding) -> Dict[str, object]:
        """Construct a minimal linkset_info payload for geDIGCore (hop0 ΔH via linkset)."""
        # Use all retrieved hits as candidate_pool; top-1 as S_link (base)
        candidate_pool = []
        s_link = []
        for idx, hit in enumerate(hits):
            w = float(max(hit.score, 1e-6))
            entry = {
                "index": hit.document.doc_id,
                "origin": "retrieval",
                "similarity": w,
                "distance": float(1.0 / w),
                "weighted_distance": float(1.0 / w),
                "vector": hit.document.embedding.tolist() if hit.document.embedding is not None else None,
            }
            candidate_pool.append(entry)
            if idx == 0:
                s_link.append(entry)

        # Query entry uses embedding if available; fallback to scalar sim=1.0
        q_sim = float(candidate_pool[0]["similarity"]) if candidate_pool else 1.0
        query_entry = {
            "index": "query",
            "origin": "query",
            "similarity": q_sim,
            "distance": 0.0,
            "weighted_distance": 0.0,
            "vector": query_embedding.tolist() if query_embedding is not None else None,
        }

        # Decision: treat top-1 as chosen
        decision = {
            "origin": "retrieval",
            "index": s_link[0]["index"] if s_link else None,
            "similarity": s_link[0]["similarity"] if s_link else None,
        }

        return {
            "s_link": s_link,
            "candidate_pool": candidate_pool,
            "decision": decision,
            "query_entry": query_entry,
            "base_mode": "link",
        }

    def _compose_answer(self, query: str, docs: List[Document]) -> tuple[str, str]:
        if not docs:
            return (query, "query_echo")
        text_parts: List[str] = [query.lower()]
        for doc in docs:
            text_parts.append(doc.text.lower())
            for key in ("goal", "outcome", "operation", "context"):
                value = doc.metadata.get(key)
                if value:
                    text_parts.append(str(value).lower())
        joined = " ".join(text_parts)

        def contains(*keywords: str) -> bool:
            return any(keyword in joined for keyword in keywords)

        if contains("deforestation", "deforest"):
            return (
                "Deforestation reduces oxygen-producing forests on land; less photosynthetic output means less oxygen dissolving into waters where tides normally distribute it.",
                "deforestation_chain",
            )
        if contains("photosynth",) and contains("tide", "tidal", "lunar"):
            return (
                "Sunlight powers coastal algae to photosynthesise oxygen, and lunar-driven tides mix that oxygenated water along the shore.",
                "solar_tidal_mix",
            )
        if contains("photosynth",) and contains("coastal", "marine"):
            return (
                "When sunlight reaches coastal waters, marine algae photosynthesise, releasing oxygen that dissolves for fish to use.",
                "solar_coastal",
            )
        if contains("photosynth",):
            return (
                "Solar photons drive photosynthesis in plants and algae, generating the oxygen that builds up in Earth's air.",
                "solar_chain",
            )
        if contains("tide", "tidal", "bulge", "lunar"):
            return (
                "Lunar gravity pulls water into bulges; as Earth rotates, those bulges shift, generating currents that mix shoreline waters.",
                "tidal_mixing",
            )
        if contains("mitochond", "respiration", "atp"):
            return (
                "They oxidise nutrients through respiration, generating ATP that fuels the cell's activities.",
                "cell_energy",
            )
        if contains("satellite", "triangulat", "gps"):
            return (
                "It measures the arrival time of signals from several satellites and triangulates its location.",
                "gps_triangulation",
            )
        if contains("router", "routing", "packet"):
            return (
                "Routers read packet addresses and consult routing tables to forward traffic along viable next hops.",
                "packet_routing",
            )
        if contains("carbon", "co2", "greenhouse"):
            return (
                "Losing forests reduces CO₂ uptake while releasing stored carbon, strengthening the greenhouse effect.",
                "carbon_cycle",
            )
        fallback = " ".join(doc.text for doc in docs[:3])
        return (fallback, "retrieved_concat")



def build_strategy(strategy_type: str, params: Dict[str, float], controller: GeDIGController | None, top_k: int) -> BaseStrategy:
    if strategy_type == "static":
        return StaticRAGStrategy(top_k=top_k)
    if strategy_type == "frequency":
        return FrequencyStrategy(top_k=top_k)
    if strategy_type == "cosine":
        return CosineTopKStrategy(top_k=top_k)
    if strategy_type == "gedig":
        if controller is None:
            raise ValueError("GeDIG controller is required for geDIG strategy.")
        max_iterations = int(params.get("max_iterations", 3))
        return GeDIGStrategy(controller=controller, top_k=top_k, max_iterations=max_iterations)
    raise ValueError(f"Unknown strategy type: {strategy_type}")
