"""
Legacy agent_loop module - Migration wrapper with EurekaSpike integration
========================================================================

This module provides backward compatibility for code that uses the old cycle function.
Now includes integrated EurekaSpike detection for complete insight detection.
"""
from __future__ import annotations
from typing import Dict, Any, Optional
import logging

# Graceful imports for optional dependencies
try:
    import networkx as nx
except ImportError:
    nx = None

try:
    import numpy as np
except ImportError:
    np = None

# Import core components
try:
    from .core.agents.main_agent import MainAgent
    USE_NEW_AGENT = True
except ImportError:
    USE_NEW_AGENT = False

from .graph_metrics import delta_ged, delta_ig
from .eureka_spike import EurekaDetector
from .layer1_error_monitor import analyze_input, KnownUnknownAnalysis
from .adaptive_topk import calculate_adaptive_topk, estimate_chain_reaction_potential
from .unknown_learner import UnknownLearner
from .insight_fact_registry import InsightFactRegistry

logger = logging.getLogger(__name__)

# Initialize global insight fact registry
insight_registry = InsightFactRegistry()

def cycle(memory, question: str, g_old=None, top_k=10, device=None, **kwargs) -> Dict[str, Any]:
    """
    Legacy cycle function - migrated to use new MainAgent.
    
    Args:
        memory: Legacy Memory object (will be converted)
        question: Question to process
        g_old: Previous graph (legacy parameter, ignored)
        top_k: Number of documents to retrieve (used in config)
        device: Device to use (used in config)
        **kwargs: Additional parameters (mostly ignored)
        
    Returns:
        Dictionary with legacy format results
    """
    try:
        logger.info(f"Processing question with legacy cycle function: {question[:50]}...")
        
        # ===== INITIALIZE UNKNOWN LEARNER =====
        unknown_learner = UnknownLearner()
        
        # ===== LAYER 1: Known/Unknown Information Separation =====
        logger.info("🔍 Layer1: Analyzing input for known/unknown information separation")
        
        # Prepare context from existing memory
        context_docs = []
        kb_stats = None
        
        if hasattr(memory, 'episodes') and memory.episodes:
            context_docs = [getattr(ep, 'text', str(ep)) for ep in memory.episodes[:20]]  # Sample for analysis
            
            # Build simple knowledge base statistics
            all_words = []
            for doc in context_docs:
                all_words.extend(doc.lower().split())
            
            from collections import Counter
            word_counts = Counter(all_words)
            kb_stats = {
                'concept_frequencies': dict(word_counts.most_common(100)),
                'total_concepts': len(word_counts)
            }
        
        # Perform Layer1 analysis with unknown learner integration
        l1_analysis = analyze_input(question, context_docs, kb_stats, unknown_learner)
        
        # ===== ADAPTIVE TOPK CALCULATION =====
        # Calculate optimal topK values based on Layer1 analysis
        adaptive_topk_result = calculate_adaptive_topk(l1_analysis.__dict__)
        adaptive_topk = {k: v for k, v in adaptive_topk_result.items() if not k.startswith('adaptation')}
        adaptation_factors = adaptive_topk_result['adaptation_factors']
        
        # Estimate chain reaction potential
        chain_reaction_potential = estimate_chain_reaction_potential(l1_analysis.__dict__, adaptive_topk)
        
        logger.info(f"L1 Analysis - Known: {len(l1_analysis.known_elements)}, "
                   f"Unknown: {len(l1_analysis.unknown_elements)}, "
                   f"Synthesis Required: {l1_analysis.requires_synthesis}, "
                   f"Complexity: {l1_analysis.query_complexity:.3f}")
        
        logger.info(f"Adaptive topK - L1:{adaptive_topk['layer1_k']}, "
                   f"L2:{adaptive_topk['layer2_k']}, L3:{adaptive_topk['layer3_k']}, "
                   f"Chain Potential: {chain_reaction_potential:.3f}")
        
        # ===== MAIN AGENT PROCESSING =====
        # Create new agent
        agent = MainAgent()
        
        # Initialize agent with Layer1 insights
        if not agent.initialize():
            logger.error("Failed to initialize MainAgent")
            return _legacy_error_result(question, "Failed to initialize agent")
        
        # Migrate memory data if provided
        if hasattr(memory, 'episodes') and memory.episodes:
            logger.info(f"Migrating {len(memory.episodes)} episodes to new memory system")
            for episode in memory.episodes:
                try:
                    c_value = getattr(episode, 'c', 0.5)
                    text = getattr(episode, 'text', str(episode))
                    agent.add_document(text, c_value)
                except Exception as e:
                    logger.warning(f"Failed to migrate episode: {e}")
        
        # Process question with adaptive topK and Layer1 insights
        # Adjust processing parameters based on Layer1 analysis and chain reaction potential
        max_cycles = 5 if l1_analysis.requires_synthesis else 3
        if chain_reaction_potential > 0.7:
            max_cycles += 2  # Allow more cycles for high chain reaction potential
            
        enhanced_processing = l1_analysis.query_complexity > 0.6 or chain_reaction_potential > 0.5
        
        logger.info(f"Processing with {max_cycles} max cycles, enhanced={enhanced_processing}, "
                   f"using adaptive topK values")
        
        # Pass adaptive topK to agent (if supported)
        processing_kwargs = {
            'max_cycles': max_cycles, 
            'verbose': enhanced_processing,
            'adaptive_topk': adaptive_topk,
            'chain_reaction_potential': chain_reaction_potential
        }
        
        result = agent.process_question(question, **processing_kwargs)
        
        # ===== INSIGHT FACT EXTRACTION AND REGISTRATION =====
        # Extract and register insights from the agent's response
        try:
            response_text = result.get('response', '')
            reasoning_quality = result.get('reasoning_quality', 0.0)
            
            # Extract graphs for optimization evaluation if available
            graph_before = g_old  # Previous graph from legacy parameter
            graph_after = result.get('graph')  # Current graph from agent result
            
            if response_text and reasoning_quality > 0.3:  # Only process quality responses
                discovered_insights = insight_registry.extract_insights_from_response(
                    question=question,
                    response=response_text,
                    l1_analysis=l1_analysis,
                    reasoning_quality=reasoning_quality,
                    graph_before=graph_before,
                    graph_after=graph_after
                )
                
                # Log insight discovery
                if discovered_insights:
                    logger.info(f"🧠 Discovered {len(discovered_insights)} insights from response")
                    for insight in discovered_insights:
                        logger.info(f"  - {insight.relationship_type}: {insight.text[:80]}...")
                        logger.info(f"    Quality: {insight.quality_score:.3f}, "
                                   f"GED: {insight.ged_optimization:.3f}, "
                                   f"IG: {insight.ig_improvement:.3f}")
                else:
                    logger.debug("No qualifying insights extracted from response")
                    
        except Exception as e:
            logger.warning(f"Insight extraction failed: {e}")
            discovered_insights = []
        
        # ===== ENHANCED RESULT WITH LAYER1 INSIGHTS =====
        # Convert to legacy format with Layer1 insights
        legacy_result = {
            'answer': result.get('response', ''),
            'documents': result.get('documents', []),
            'graph': result.get('graph'),
            'metrics': result.get('metrics', {}),
            'conflicts': result.get('conflicts', {}),
            'reward': result.get('reward', {}),
            'spike_detected': result.get('spike_detected', False),
            'reasoning_quality': result.get('reasoning_quality', 0.0),
            'success': result.get('success', True),
            
            # Layer1 analysis results
            'l1_analysis': {
                'known_elements': l1_analysis.known_elements,
                'unknown_elements': l1_analysis.unknown_elements,
                'certainty_scores': l1_analysis.certainty_scores,
                'query_complexity': l1_analysis.query_complexity,
                'requires_synthesis': l1_analysis.requires_synthesis,
                'error_threshold': l1_analysis.error_threshold,
                'analysis_confidence': l1_analysis.analysis_confidence
            },
            
            # Adaptive topK results
            'adaptive_topk': {
                'layer1_k': adaptive_topk['layer1_k'],
                'layer2_k': adaptive_topk['layer2_k'],
                'layer3_k': adaptive_topk['layer3_k'],
                'adaptation_factors': adaptation_factors,
                'chain_reaction_potential': chain_reaction_potential
            },
            
            # Legacy fields
            'delta_ged': result.get('metrics', {}).get('delta_ged', 0.0),
            'delta_ig': result.get('metrics', {}).get('delta_ig', 0.0),
            'eureka': result.get('spike_detected', False),
            'confidence': result.get('reasoning_quality', 0.5),
            'updated_episodes': result.get('documents', []),
            
            # Insight fact extraction results
            'discovered_insights': discovered_insights,
            'insight_count': len(discovered_insights),
            'insight_quality_avg': sum(i.quality_score for i in discovered_insights) / len(discovered_insights) if discovered_insights else 0.0,
            'insight_optimization_avg': sum(i.ged_optimization for i in discovered_insights) / len(discovered_insights) if discovered_insights else 0.0
        }
        
        logger.info(f"Legacy cycle completed - Quality: {legacy_result['reasoning_quality']:.3f}, "
                   f"L1 Confidence: {l1_analysis.analysis_confidence:.3f}, "
                   f"Chain Potential: {chain_reaction_potential:.3f}, "
                   f"Insights Discovered: {len(discovered_insights)}")
        return legacy_result
        
    except Exception as e:
        logger.error(f"Legacy cycle function failed: {e}")
        return _legacy_error_result(question, str(e))

def _legacy_error_result(question: str, error: str) -> Dict[str, Any]:
    """Generate legacy-compatible error result"""
    return {
        'answer': f"Error processing question: {error}",
        'documents': [],
        'graph': None,
        'metrics': {},
        'conflicts': {},
        'reward': {},
        'spike_detected': False,
        'reasoning_quality': 0.0,
        'success': False,
        'delta_ged': 0.0,
        'delta_ig': 0.0,
        'eureka': False,
        'confidence': 0.0,
        'updated_episodes': [],
        'error': error
    }

# Legacy adaptive_loop function
def adaptive_loop(memory, questions, max_iterations=10, **kwargs):
    """
    Legacy adaptive_loop function - processes multiple questions.
    """
    results = []
    
    for i, question in enumerate(questions):
        try:
            result = cycle(memory, question, **kwargs)
            results.append(result)
            
            # Early stopping if high quality achieved
            if result.get('reasoning_quality', 0) > 0.8:
                logger.info(f"High quality achieved at iteration {i+1}, stopping adaptive loop")
                break
                
        except Exception as e:
            logger.error(f"Adaptive loop iteration {i+1} failed: {e}")
            results.append(_legacy_error_result(question, str(e)))
    
    return results

# Export for backward compatibility
__all__ = ["cycle", "adaptive_loop"]
