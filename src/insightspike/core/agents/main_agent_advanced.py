"""
Advanced Main Agent with Enhanced Query Transformation
=====================================================

Phase 2: Multi-hop reasoning, adaptive exploration, and query branching.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
import networkx as nx

from ..query_transformation import QueryState, QueryTransformationHistory
from ..query_transformation.enhanced_query_transformer import (
    EnhancedQueryTransformer, QueryBranch
)
from .main_agent_with_query_transform import MainAgentWithQueryTransform, TransformationCycleResult

logger = logging.getLogger(__name__)


@dataclass
class AdvancedTransformationResult(TransformationCycleResult):
    """Extended result with branching information"""
    
    branches: List[QueryBranch] = field(default_factory=list)
    reasoning_paths: List[List[str]] = field(default_factory=list)
    exploration_strategy: str = "adaptive"
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["branches"] = [
            {
                "branch_id": b.branch_id,
                "direction": b.exploration_direction,
                "priority": b.priority,
                "insights": b.current_state.insights
            }
            for b in self.branches
        ]
        result["reasoning_paths"] = self.reasoning_paths
        result["exploration_strategy"] = self.exploration_strategy
        return result


class MainAgentAdvanced(MainAgentWithQueryTransform):
    """
    Advanced main agent with enhanced query transformation capabilities.
    
    New features:
    - Multi-hop reasoning across knowledge graph
    - Adaptive exploration strategies
    - Query branching for parallel exploration
    - Path-based insight discovery
    """
    
    def __init__(self, 
                 config=None, 
                 enable_multi_hop: bool = True,
                 enable_branching: bool = True,
                 max_branches: int = 3):
        # Initialize parent with basic query transformation
        super().__init__(config, enable_query_transformation=True)
        
        # Replace with enhanced transformer
        self.query_transformer = EnhancedQueryTransformer(
            embedding_model_name=getattr(self.config.embedding, 'model_name', 'all-MiniLM-L6-v2'),
            use_multi_hop=enable_multi_hop,
            enable_branching=enable_branching,
            max_branches=max_branches
        )
        
        # Advanced tracking
        self.branch_histories: Dict[str, QueryTransformationHistory] = {}
        self.discovered_paths: List[List[str]] = []
        self.exploration_temperature = 1.0  # For adaptive exploration
    
    def process_question_advanced(self, question: str) -> Dict[str, Any]:
        """Process question with advanced transformation features"""
        
        logger.info(f"Processing with advanced transformation: {question}")
        
        # Initialize main transformation history
        main_history = QueryTransformationHistory(initial_query=question)
        self.transformation_histories[question] = main_history
        
        # Place query on graph
        initial_state = self.query_transformer.place_query_on_graph(
            question, 
            self._get_current_knowledge_graph()
        )
        main_history.add_state(initial_state)
        
        # Run advanced transformation cycles
        result = self._process_with_advanced_transformation(
            question, main_history
        )
        
        return result
    
    def _process_with_advanced_transformation(self,
                                            question: str,
                                            main_history: QueryTransformationHistory) -> Dict[str, Any]:
        """Process with multi-hop and branching"""
        
        best_result = None
        max_insight_quality = 0.0
        all_branches = []
        
        for cycle in range(self.max_cycles):
            logger.info(f"\n{'='*50}\nAdvanced Cycle {cycle + 1}/{self.max_cycles}\n{'='*50}")
            
            current_state = main_history.get_current_state()
            
            # Run advanced cycle
            cycle_result = self._run_advanced_cycle(
                question,
                current_state,
                main_history,
                cycle
            )
            
            # Track branches
            if cycle_result.branches:
                all_branches.extend(cycle_result.branches)
                logger.info(f"Created {len(cycle_result.branches)} exploration branches")
            
            # Evaluate result quality
            insight_quality = self._evaluate_insight_quality(cycle_result)
            if insight_quality > max_insight_quality:
                max_insight_quality = insight_quality
                best_result = cycle_result
            
            # Check for convergence
            if self._check_advanced_convergence(main_history, all_branches):
                logger.info(f"Convergence achieved at cycle {cycle + 1}")
                break
            
            # Adaptive temperature decay
            self.query_transformer.adaptive_explorer.temperature *= 0.8
        
        # Synthesize final result
        final_result = self._synthesize_advanced_result(
            best_result, main_history, all_branches
        )
        
        return final_result
    
    def _run_advanced_cycle(self,
                          question: str,
                          current_state: QueryState,
                          main_history: QueryTransformationHistory,
                          cycle: int) -> AdvancedTransformationResult:
        """Run single advanced transformation cycle"""
        
        try:
            # L1-L2: Standard retrieval
            error_state = self.l1_error_monitor.analyze_uncertainty(question)
            retrieved_documents = self.l2_memory.search_episodes(
                self._get_effective_query(current_state), k=7  # More documents for multi-hop
            )
            
            if not retrieved_documents:
                logger.warning("No documents retrieved")
                return self._create_empty_advanced_result(current_state)
            
            # Build knowledge graph
            from ...utils.graph_construction import GraphBuilder
            graph_builder = GraphBuilder()
            knowledge_graph = graph_builder.build_from_documents(retrieved_documents)
            
            # Advanced transformation with branching
            new_state, branches = self.query_transformer.transform_query_advanced(
                current_state,
                knowledge_graph,
                retrieved_documents,
                allow_branching=(cycle < self.max_cycles - 1)  # No branching on last cycle
            )
            
            # Update main history
            main_history.add_state(new_state)
            
            # Process branches in parallel (conceptually)
            branch_insights = []
            for branch in branches:
                branch_result = self._explore_branch(
                    branch, knowledge_graph, retrieved_documents
                )
                branch_insights.extend(branch_result)
            
            # Multi-hop graph analysis
            graph_analysis = self._perform_multi_hop_analysis(
                knowledge_graph, new_state, retrieved_documents
            )
            
            # Extract reasoning paths
            reasoning_paths = self._extract_reasoning_paths(
                new_state, knowledge_graph, graph_analysis
            )
            
            # Generate response with all context
            response = self._generate_advanced_response(
                question, new_state, branch_insights, reasoning_paths
            )
            
            return AdvancedTransformationResult(
                cycle=cycle,
                response=response,
                spike_detected=graph_analysis.get("spike_detected", False),
                reasoning_quality=graph_analysis.get("reasoning_quality", 0.5),
                confidence=new_state.confidence,
                query_state=new_state,
                transformation_history=main_history,
                branches=branches,
                reasoning_paths=reasoning_paths,
                exploration_strategy=new_state.metadata.get(
                    "exploration_direction", "adaptive"
                )
            )
            
        except Exception as e:
            logger.error(f"Advanced cycle failed: {e}")
            return self._create_empty_advanced_result(current_state)
    
    def _explore_branch(self,
                       branch: QueryBranch,
                       knowledge_graph: nx.Graph,
                       documents: List[Dict[str, Any]]) -> List[str]:
        """Explore a single branch"""
        
        logger.info(f"Exploring branch: {branch.branch_id} ({branch.exploration_direction})")
        
        # Create branch-specific history
        branch_history = QueryTransformationHistory(branch.current_state.text)
        self.branch_histories[branch.branch_id] = branch_history
        
        # Limited exploration steps for branch
        branch_insights = []
        for step in range(2):  # Limit branch depth
            # Transform within branch context
            new_state = self.query_transformer.transform_query(
                branch.current_state,
                knowledge_graph,
                documents
            )
            
            branch.current_state = new_state
            branch_history.add_state(new_state)
            
            # Collect branch-specific insights
            branch_insights.extend(new_state.insights[-1:])  # Latest insight
            
            # Early termination if insight found
            if new_state.stage == "insight":
                break
        
        # Mark successful paths
        if branch_insights:
            self.query_transformer.adaptive_explorer.update_success_pattern(
                [branch.exploration_direction], 
                branch.current_state.confidence
            )
        
        return branch_insights
    
    def _perform_multi_hop_analysis(self,
                                  knowledge_graph: nx.Graph,
                                  query_state: QueryState,
                                  documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform multi-hop graph analysis"""
        
        if not self.l3_graph:
            return {"spike_detected": False, "reasoning_quality": 0.5}
        
        # Enhanced context for multi-hop
        context = {
            "multi_hop": True,
            "query_state": query_state,
            "hop_limit": 3,
            "exploration_strategy": query_state.metadata.get("exploration_direction", "general")
        }
        
        # Run L3 analysis
        analysis = self.l3_graph.analyze_documents(documents, context)
        
        # Extract multi-hop specific metrics
        if "reasoning_paths" in analysis:
            self.discovered_paths.extend(analysis["reasoning_paths"])
        
        return analysis
    
    def _extract_reasoning_paths(self,
                               query_state: QueryState,
                               knowledge_graph: nx.Graph,
                               graph_analysis: Dict[str, Any]) -> List[List[str]]:
        """Extract human-readable reasoning paths"""
        
        paths = []
        
        # Get paths from graph analysis
        if "reasoning_paths" in graph_analysis:
            for path in graph_analysis["reasoning_paths"]:
                readable_path = []
                for node_id in path:
                    if isinstance(node_id, int) and node_id < len(knowledge_graph.nodes()):
                        node_name = list(knowledge_graph.nodes())[node_id]
                        readable_path.append(node_name)
                    else:
                        readable_path.append(str(node_id))
                
                if readable_path:
                    paths.append(readable_path)
        
        # Add paths from absorbed concepts
        if len(query_state.absorbed_concepts) > 2:
            concept_path = query_state.absorbed_concepts[-3:]
            paths.append(concept_path)
        
        return paths
    
    def _generate_advanced_response(self,
                                  question: str,
                                  query_state: QueryState,
                                  branch_insights: List[str],
                                  reasoning_paths: List[List[str]]) -> str:
        """Generate response incorporating all advanced features"""
        
        # Build comprehensive context
        context_parts = [
            f"Query: {question}",
            f"Confidence: {query_state.confidence:.0%}",
            f"Stage: {query_state.stage}"
        ]
        
        if query_state.absorbed_concepts:
            context_parts.append(
                f"Key concepts: {', '.join(query_state.absorbed_concepts[-3:])}"
            )
        
        if query_state.insights:
            context_parts.append(
                f"Main insight: {query_state.insights[-1]}"
            )
        
        if branch_insights:
            context_parts.append(
                f"Branch discoveries: {'; '.join(branch_insights[:2])}"
            )
        
        if reasoning_paths:
            path_str = " → ".join(reasoning_paths[0]) if reasoning_paths else ""
            context_parts.append(f"Reasoning path: {path_str}")
        
        # For demo, create informative response
        if query_state.insights:
            response = query_state.insights[-1]
            if branch_insights:
                response += f" Additionally: {branch_insights[0]}"
        else:
            response = f"Exploring connections between concepts: {', '.join(query_state.absorbed_concepts[:3])}"
        
        return response
    
    def _evaluate_insight_quality(self, result: AdvancedTransformationResult) -> float:
        """Evaluate quality of insights discovered"""
        
        quality = 0.0
        
        # Base quality from confidence
        quality += result.confidence * 0.3
        
        # Bonus for insights
        if result.query_state and result.query_state.insights:
            quality += min(len(result.query_state.insights) * 0.1, 0.3)
        
        # Bonus for successful branches
        if result.branches:
            successful_branches = sum(1 for b in result.branches 
                                    if b.current_state.confidence > 0.5)
            quality += successful_branches * 0.1
        
        # Bonus for reasoning paths
        if result.reasoning_paths:
            quality += min(len(result.reasoning_paths) * 0.05, 0.2)
        
        # Spike detection bonus
        if result.spike_detected:
            quality += 0.2
        
        return min(quality, 1.0)
    
    def _check_advanced_convergence(self,
                                  main_history: QueryTransformationHistory,
                                  branches: List[QueryBranch]) -> bool:
        """Check for convergence in advanced transformation"""
        
        # Standard convergence check
        if main_history.reached_insight():
            return True
        
        # Check if branches have converged
        if branches:
            converged_branches = sum(1 for b in branches 
                                   if b.current_state.stage == "insight")
            if converged_branches >= len(branches) * 0.5:
                return True
        
        # Check for stability in confidence
        confidence_trajectory = main_history.get_confidence_trajectory()
        if len(confidence_trajectory) >= 3:
            recent_changes = [abs(confidence_trajectory[i] - confidence_trajectory[i-1]) 
                            for i in range(-3, 0)]
            if all(change < 0.05 for change in recent_changes):
                return True
        
        return False
    
    def _synthesize_advanced_result(self,
                                  best_result: AdvancedTransformationResult,
                                  main_history: QueryTransformationHistory,
                                  all_branches: List[QueryBranch]) -> Dict[str, Any]:
        """Synthesize final result from all explorations"""
        
        if not best_result:
            return {"error": "No valid results produced"}
        
        # Collect all insights
        all_insights = set()
        if best_result.query_state:
            all_insights.update(best_result.query_state.insights)
        
        for branch in all_branches:
            all_insights.update(branch.current_state.insights)
        
        # Build comprehensive result
        result = best_result.to_dict()
        
        # Add synthesis information
        result["synthesis"] = {
            "total_insights": list(all_insights),
            "num_branches_explored": len(all_branches),
            "successful_branches": sum(1 for b in all_branches 
                                     if b.current_state.confidence > 0.5),
            "discovered_paths": self.discovered_paths[-5:],  # Last 5 paths
            "exploration_summary": self._create_exploration_summary(all_branches)
        }
        
        # Add advanced metrics
        result["advanced_metrics"] = {
            "multi_hop_coverage": len(self.discovered_paths),
            "branch_diversity": len(set(b.exploration_direction for b in all_branches)),
            "convergence_rate": self._calculate_convergence_rate(main_history),
            "insight_density": len(all_insights) / max(len(all_branches), 1)
        }
        
        return result
    
    def _create_exploration_summary(self, branches: List[QueryBranch]) -> str:
        """Create summary of exploration process"""
        
        if not branches:
            return "Direct exploration without branching"
        
        directions = [b.exploration_direction for b in branches]
        unique_directions = list(set(directions))
        
        summary = f"Explored {len(unique_directions)} directions: {', '.join(unique_directions)}"
        
        successful = [b for b in branches if b.current_state.stage == "insight"]
        if successful:
            summary += f". Found insights in: {', '.join(b.exploration_direction for b in successful)}"
        
        return summary
    
    def _calculate_convergence_rate(self, history: QueryTransformationHistory) -> float:
        """Calculate how quickly the query converged to insight"""
        
        states = history.states
        if not states:
            return 0.0
        
        # Find first insight state
        insight_idx = None
        for i, state in enumerate(states):
            if state.stage == "insight" or state.confidence > 0.8:
                insight_idx = i
                break
        
        if insight_idx is None:
            return 0.0
        
        # Rate = 1 - (steps_to_insight / total_steps)
        rate = 1.0 - (insight_idx / len(states))
        return rate
    
    def _create_empty_advanced_result(self, 
                                    current_state: QueryState) -> AdvancedTransformationResult:
        """Create empty result for error cases"""
        
        return AdvancedTransformationResult(
            cycle=0,
            response="Unable to process query",
            spike_detected=False,
            reasoning_quality=0.0,
            confidence=0.0,
            query_state=current_state,
            branches=[],
            reasoning_paths=[],
            exploration_strategy="failed"
        )
    
    # Override the parent's process_question to use advanced version
    def process_question(self, question: str) -> Dict[str, Any]:
        """Process question with advanced transformation"""
        return self.process_question_advanced(question)