"""
Main Agent with Query Transformation
====================================

Enhanced main agent that supports query transformation through graph exploration.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..query_transformation import (
    QueryState,
    QueryTransformationHistory,
    QueryTransformer,
)
from ..agents.main_agent import MainAgent, CycleResult
from ..layers.layer3_graph_reasoner import L3GraphReasoner
from ...utils.graph_construction import GraphBuilder

logger = logging.getLogger(__name__)


@dataclass
class TransformationCycleResult(CycleResult):
    """Extended cycle result with query transformation data"""

    query_state: Optional[QueryState] = None
    transformation_history: Optional[QueryTransformationHistory] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary including transformation data"""
        result = super().to_dict()
        if self.query_state:
            result["query_state"] = self.query_state.to_dict()
        if self.transformation_history:
            result["transformation_history"] = self.transformation_history.to_dict()
        return result


class MainAgentWithQueryTransform(MainAgent):
    """
    Enhanced main agent with query transformation capability.

    New features:
    - Query is placed on the knowledge graph
    - Query transforms through message passing
    - New insights emerge from query transformation
    """

    def __init__(self, config=None, enable_query_transformation: bool = True):
        super().__init__(config)

        self.enable_query_transformation = enable_query_transformation
        if enable_query_transformation:
            self.query_transformer = QueryTransformer(
                embedding_model_name=getattr(
                    self.config.embedding, "model_name", "all-MiniLM-L6-v2"
                ),
                use_gnn=True,
            )
            # Ensure L3 uses GNN
            if hasattr(self, "l3_graph") and self.l3_graph:
                self.l3_graph.use_gnn = True
                logger.info("Enabled GNN in L3GraphReasoner for query transformation")

        self.transformation_histories: Dict[str, QueryTransformationHistory] = {}
        self.max_cycles = 3  # Default max cycles for transformation

    def process_question(self, question: str) -> Dict[str, Any]:
        """Process question with optional query transformation"""

        if not self.enable_query_transformation:
            # Fallback to standard processing
            return super().process_question(question)

        # Initialize transformation history
        transformation_history = QueryTransformationHistory(initial_query=question)
        self.transformation_histories[question] = transformation_history

        # Enhanced processing with query transformation
        result = self._process_with_transformation(question, transformation_history)

        return result

    def _process_with_transformation(
        self, question: str, transformation_history: QueryTransformationHistory
    ) -> Dict[str, Any]:
        """Process question with query transformation enabled"""

        logger.info(f"Processing with query transformation: {question}")

        # Initialize query state
        initial_state = self.query_transformer.place_query_on_graph(
            question, self._get_current_knowledge_graph()
        )
        transformation_history.add_state(initial_state)

        best_result = None
        max_reasoning_quality = 0.0

        for cycle in range(self.max_cycles):
            logger.info(f"\n{'='*50}\nCycle {cycle + 1}/{self.max_cycles}\n{'='*50}")

            # Get current query state
            current_state = transformation_history.get_current_state()

            # Use transformed query for retrieval
            query_text = self._get_effective_query(current_state)

            # Run one cycle with current query state
            cycle_result = self._run_single_cycle_with_transformation(
                query_text, current_state, transformation_history, cycle
            )

            if cycle_result.reasoning_quality > max_reasoning_quality:
                max_reasoning_quality = cycle_result.reasoning_quality
                best_result = cycle_result

            # Check for convergence
            if current_state.stage == "complete" or current_state.confidence > 0.9:
                logger.info(f"Query transformation complete at cycle {cycle + 1}")
                break

            # Check for insight spike
            if cycle_result.spike_detected:
                logger.info(f"Insight spike detected at cycle {cycle + 1}!")
                if current_state.stage != "insight":
                    current_state.stage = "insight"
                    current_state.add_insight("Spike-triggered insight emergence")

        # Prepare final result
        if best_result:
            final_result = best_result.to_dict()
            final_result["transformation_history"] = transformation_history.to_dict()
            final_result["query_evolution"] = {
                "initial": question,
                "final_state": current_state.to_dict(),
                "insights_discovered": transformation_history.get_total_insights(),
                "confidence_trajectory": transformation_history.get_confidence_trajectory(),
            }
        else:
            final_result = {"error": "No valid results produced"}

        return final_result

    def _run_single_cycle_with_transformation(
        self,
        query_text: str,
        query_state: QueryState,
        transformation_history: QueryTransformationHistory,
        cycle: int,
    ) -> TransformationCycleResult:
        """Run a single reasoning cycle with query transformation"""

        try:
            # L1: Error monitoring
            error_state = self.l1_error_monitor.analyze_uncertainty(query_text)

            # L2: Memory search with transformed query
            retrieved_documents = self.l2_memory.search_episodes(query_text, k=5)

            # L3: Graph reasoning with query as node
            if self.l3_graph and retrieved_documents:
                # Build graph from retrieved documents
                graph_builder = GraphBuilder()
                knowledge_graph = graph_builder.build_from_documents(
                    retrieved_documents
                )

                # Transform query through graph
                new_query_state = self.query_transformer.transform_query(
                    query_state, knowledge_graph, retrieved_documents
                )
                transformation_history.add_state(new_query_state)

                # Standard graph analysis
                graph_analysis = self.l3_graph.analyze_documents(
                    documents=retrieved_documents,
                    context={"cycle": cycle, "query_state": new_query_state},
                )

                # Extract metrics
                spike_detected = graph_analysis.get("spike_detected", False)
                reasoning_quality = graph_analysis.get("metrics", {}).get(
                    "reasoning_quality", 0.5
                )

                # If spike detected, enhance query state
                if spike_detected:
                    new_query_state.add_insight("Graph analysis detected insight spike")
                    new_query_state.confidence = min(
                        1.0, new_query_state.confidence + 0.2
                    )

            else:
                graph_analysis = {}
                spike_detected = False
                reasoning_quality = 0.5
                new_query_state = query_state

            # L4: Language generation with context from transformation
            context = self._build_llm_context(
                retrieved_documents, graph_analysis, new_query_state
            )

            response = self.l4_llm.generate(
                prompt=query_text, context=context, max_tokens=512
            )

            # Create cycle result
            result = TransformationCycleResult(
                question=query_text,
                retrieved_documents=retrieved_documents,
                graph_analysis=graph_analysis,
                response=response,
                reasoning_quality=reasoning_quality,
                spike_detected=spike_detected,
                error_state=error_state,
                cycle_number=cycle,
                query_state=new_query_state,
                transformation_history=transformation_history,
            )

            # Store new episode with transformation data
            self._store_episode_with_transformation(result)

            return result

        except Exception as e:
            logger.error(f"Cycle {cycle} failed: {e}")
            return TransformationCycleResult(
                question=query_text,
                retrieved_documents=[],
                graph_analysis={},
                response=f"Error: {str(e)}",
                reasoning_quality=0.0,
                spike_detected=False,
                error_state={"error": str(e)},
                cycle_number=cycle,
                success=False,
                query_state=query_state,
            )

    def _get_current_knowledge_graph(self):
        """Get current knowledge graph from memory"""
        # For now, build from recent episodes
        recent_episodes = self.l2_memory.search_episodes("", k=20)

        if not recent_episodes:
            # Return empty graph
            import networkx as nx

            return nx.Graph()

        # Build graph from episodes
        graph_builder = GraphBuilder()
        return graph_builder.build_from_documents(recent_episodes)

    def _get_effective_query(self, query_state: QueryState) -> str:
        """Get effective query text based on transformation state"""

        if not query_state.absorbed_concepts:
            return query_state.text

        # Enhance query with absorbed concepts
        concepts = " ".join(query_state.absorbed_concepts[-3:])  # Last 3 concepts
        enhanced_query = f"{query_state.text} [Context: {concepts}]"

        return enhanced_query

    def _build_llm_context(
        self,
        documents: List[Dict[str, Any]],
        graph_analysis: Dict[str, Any],
        query_state: QueryState,
    ) -> str:
        """Build context for LLM including transformation insights"""

        # Standard context
        context_parts = []

        # Add documents
        for i, doc in enumerate(documents):
            context_parts.append(f"Document {i+1}: {doc.get('text', '')[:200]}...")

        # Add graph insights
        if graph_analysis.get("spike_detected"):
            context_parts.append("\n[INSIGHT SPIKE DETECTED]")

        # Add query transformation insights
        if query_state.insights:
            context_parts.append("\n[Query Transformation Insights]:")
            for insight in query_state.insights:
                context_parts.append(f"- {insight}")

        if query_state.absorbed_concepts:
            context_parts.append(
                f"\n[Absorbed Concepts]: {', '.join(query_state.absorbed_concepts)}"
            )

        return "\n".join(context_parts)

    def _store_episode_with_transformation(self, result: TransformationCycleResult):
        """Store episode including transformation data"""

        episode_data = {
            "text": result.response,
            "question": result.question,
            "timestamp": time.time(),
            "metadata": {
                "spike_detected": result.spike_detected,
                "reasoning_quality": result.reasoning_quality,
                "cycle": result.cycle_number,
                "transformation": {
                    "stage": result.query_state.stage if result.query_state else None,
                    "confidence": result.query_state.confidence
                    if result.query_state
                    else 0,
                    "insights": result.query_state.insights
                    if result.query_state
                    else [],
                },
            },
        }

        self.l2_memory.add_episode(**episode_data)
