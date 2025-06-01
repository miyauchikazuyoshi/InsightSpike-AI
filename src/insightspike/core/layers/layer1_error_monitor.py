"""
Layer 1: Error Monitor - Cerebellum Analog
==========================================

Implements uncertainty calculation and error monitoring for the InsightSpike architecture.
"""

import numpy as np
import re
from typing import Sequence, List, Any, Dict, Optional
from dataclasses import dataclass
from ..interfaces import L1ErrorMonitorInterface, LayerInput, LayerOutput
from ...config import get_config


@dataclass
class KnownUnknownAnalysis:
    """Analysis result separating known and unknown information components"""
    known_elements: List[str]
    unknown_elements: List[str]
    certainty_scores: Dict[str, float]
    query_complexity: float
    requires_synthesis: bool
    error_threshold: float
    analysis_confidence: float


class ErrorMonitor(L1ErrorMonitorInterface):
    """
    Layer 1 implementation for error monitoring and uncertainty calculation.
    
    This layer acts as the cerebellum analog, providing uncertainty measures
    and prediction error calculations to guide the reasoning process.
    """
    
    def __init__(self, layer_id: str = "L1_ErrorMonitor", config: Dict[str, Any] = None):
        super().__init__(layer_id, config)
        self.global_config = get_config()
        self.error_history = []
        self.uncertainty_threshold = self.config.get('uncertainty_threshold', 0.5)
    
    def initialize(self) -> bool:
        """Initialize the error monitor"""
        try:
            self.error_history = []
            self._is_initialized = True
            return True
        except Exception as e:
            print(f"Error initializing L1 ErrorMonitor: {e}")
            return False
    
    def process(self, input_data: LayerInput) -> LayerOutput:
        """Process input through error monitor"""
        if 'scores' in input_data.data:
            uncertainty_value = self.calculate_uncertainty(input_data.data['scores'])
        else:
            uncertainty_value = 0.0
        
        return LayerOutput(
            result={'uncertainty': uncertainty_value},
            confidence=1.0 - uncertainty_value,
            metadata={'layer_id': self.layer_id},
            metrics={'uncertainty': uncertainty_value}
        )
    
    def calculate_error(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        """
        Calculate prediction error between predicted and actual values.
        
        Args:
            predicted: Predicted values
            actual: Actual values
            
        Returns:
            float: Calculated error value
        """
        if len(predicted) != len(actual):
            raise ValueError("Predicted and actual arrays must have same length")
        
        # Mean squared error
        error = float(np.mean((predicted - actual) ** 2))
        self.error_history.append(error)
        
        # Keep only recent history
        if len(self.error_history) > 100:
            self.error_history = self.error_history[-100:]
        
        return error
    
    def get_uncertainty(self, input_sequence: List[Any]) -> float:
        """
        Get uncertainty measure for input sequence.
        
        Args:
            input_sequence: Input sequence to evaluate
            
        Returns:
            float: Uncertainty value (0.0 to 1.0)
        """
        if not input_sequence:
            return 1.0
        
        # Simple entropy-based uncertainty
        if isinstance(input_sequence[0], (int, float)):
            return self.calculate_uncertainty(input_sequence)
        
        # For non-numeric sequences, use length-based heuristic
        return min(1.0, 1.0 / (len(input_sequence) + 1))
    
    def calculate_uncertainty(self, scores: Sequence[float]) -> float:
        """
        Calculate uncertainty using entropy of probability distribution.
        
        Args:
            scores: Sequence of scores to evaluate
            
        Returns:
            float: Uncertainty value
        """
        probs = np.array(scores, dtype=float)
        
        # Normalize to probabilities
        probs = probs / (probs.sum() + 1e-9)
        
        # Calculate entropy
        entropy = float(-np.sum(probs * np.log(probs + 1e-9)))
        
        # Normalize to [0, 1] range
        max_entropy = np.log(len(probs)) if len(probs) > 1 else 1.0
        normalized_uncertainty = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return min(1.0, max(0.0, normalized_uncertainty))
    
    def get_error_trend(self) -> str:
        """Get trend of recent errors"""
        if len(self.error_history) < 2:
            return "insufficient_data"
        
        recent_errors = self.error_history[-5:]
        if len(recent_errors) < 2:
            return "stable"
        
        trend = np.polyfit(range(len(recent_errors)), recent_errors, 1)[0]
        
        if trend > 0.01:
            return "increasing"
        elif trend < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    def is_error_spike(self) -> bool:
        """Check if there's an error spike"""
        if len(self.error_history) < 3:
            return False
        
        current_error = self.error_history[-1]
        recent_mean = np.mean(self.error_history[-10:-1]) if len(self.error_history) > 10 else np.mean(self.error_history[:-1])
        
        return current_error > recent_mean * 2.0
    
    def cleanup(self):
        """Cleanup resources"""
        self.error_history.clear()
        self._is_initialized = False


def analyze_input(query: str, context_documents: List[str] = None, 
                 knowledge_base_stats: Dict[str, Any] = None,
                 unknown_learner: Optional[Any] = None) -> KnownUnknownAnalysis:
    """
    Analyze input query to separate known vs unknown information components.
    
    This function implements the core Layer1 functionality of distinguishing
    what information is already available (known) vs what needs to be discovered
    (unknown) to properly answer a question.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    context_documents = context_documents or []
    knowledge_base_stats = knowledge_base_stats or {}
    
    # Extract concepts from query using simple NLP
    query_concepts = _extract_key_concepts(query)
    
    # Question type indicators for synthesis detection
    question_indicators = _detect_question_indicators(query)
    
    # Separate known vs unknown elements
    known_elements = []
    unknown_elements = []
    certainty_scores = {}
    
    for concept in query_concepts:
        certainty = _calculate_concept_certainty(concept, context_documents, knowledge_base_stats)
        certainty_scores[concept] = certainty
        
        if certainty >= 0.7:  # High certainty threshold
            known_elements.append(concept)
        elif certainty <= 0.3:  # Low certainty threshold
            unknown_elements.append(concept)
        else:
            # Medium certainty - could be either, lean towards unknown for safety
            unknown_elements.append(concept)
    
    # Determine if synthesis is required
    requires_synthesis = _requires_synthesis_analysis(query, question_indicators, 
                                                    known_elements, unknown_elements)
    
    # Calculate query complexity
    query_complexity = _calculate_query_complexity(query, question_indicators, 
                                                  len(known_elements), len(unknown_elements))
    
    # Calculate error threshold based on complexity
    error_threshold = _calculate_error_threshold(query_complexity, requires_synthesis)
    
    # Calculate analysis confidence
    analysis_confidence = _calculate_analysis_confidence(certainty_scores, context_documents)
    
    # Register unknown relationships for learning if learner provided
    if unknown_learner is not None and len(query_concepts) > 1:
        try:
            unknown_learner.register_question_relationships(query, query_concepts)
        except Exception as e:
            # Don't fail analysis if learning fails
            pass
    
    return KnownUnknownAnalysis(
        known_elements=known_elements,
        unknown_elements=unknown_elements,
        certainty_scores=certainty_scores,
        query_complexity=query_complexity,
        requires_synthesis=requires_synthesis,
        error_threshold=error_threshold,
        analysis_confidence=analysis_confidence
    )

def _extract_key_concepts(query: str) -> List[str]:
    """Extract key concepts from query using simple NLP"""
    # Remove common stop words and extract meaningful terms
    stop_words = {'the', 'is', 'at', 'which', 'on', 'and', 'a', 'to', 'are', 'as', 'was', 'will', 'for', 'of', 'with', 'by'}
    
    # Simple concept extraction: words longer than 3 chars, not stop words
    words = re.findall(r'\b\w{4,}\b', query.lower())
    concepts = [word for word in words if word not in stop_words]
    
    # Also extract proper nouns and technical terms
    proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', query)
    technical_terms = re.findall(r'\b\w*[A-Z]\w*\b', query)
    
    all_concepts = list(set(concepts + [term.lower() for term in proper_nouns + technical_terms]))
    return all_concepts[:10]  # Limit to top 10 concepts

def _detect_question_indicators(query: str) -> Dict[str, bool]:
    """Detect question type indicators"""
    query_lower = query.lower()
    
    return {
        'what_is': 'what is' in query_lower or 'what are' in query_lower,
        'how_does': 'how does' in query_lower or 'how do' in query_lower,
        'why_does': 'why does' in query_lower or 'why do' in query_lower,
        'compare': 'compare' in query_lower or 'versus' in query_lower or 'vs' in query_lower,
        'explain': 'explain' in query_lower or 'describe' in query_lower,
        'analyze': 'analyze' in query_lower or 'analysis' in query_lower,
        'relationship': 'relationship' in query_lower or 'relation' in query_lower,
        'difference': 'difference' in query_lower or 'differ' in query_lower,
        'synthesis': 'integrate' in query_lower or 'synthesize' in query_lower or 'combine' in query_lower
    }

def _calculate_concept_certainty(concept: str, context_documents: List[str], 
                               knowledge_base_stats: Dict[str, Any]) -> float:
    """Calculate certainty score for a concept"""
    certainty = 0.0
    
    # Check presence in context documents
    if context_documents:
        doc_mentions = sum(1 for doc in context_documents if concept.lower() in doc.lower())
        doc_ratio = doc_mentions / len(context_documents)
        certainty += doc_ratio * 0.4
    
    # Check knowledge base statistics if available
    if knowledge_base_stats:
        # Assume knowledge_base_stats contains term frequencies or similar metrics
        term_freq = knowledge_base_stats.get('term_frequencies', {}).get(concept.lower(), 0)
        if term_freq > 0:
            # Normalize by max frequency for scaling
            max_freq = max(knowledge_base_stats.get('term_frequencies', {}).values()) or 1
            certainty += (term_freq / max_freq) * 0.4
    
    # Boost certainty for common technical terms (heuristic)
    if len(concept) > 6 and any(char.isupper() for char in concept):
        certainty += 0.1
    
    # Boost certainty for domain-specific terms
    domain_terms = ['learning', 'neural', 'graph', 'memory', 'analysis', 'algorithm']
    if any(term in concept.lower() for term in domain_terms):
        certainty += 0.1
    
    return min(1.0, certainty)

def _requires_synthesis_analysis(query: str, question_indicators: Dict[str, bool], 
                               known_elements: List[str], unknown_elements: List[str]) -> bool:
    """Determine if the query requires synthesis of multiple information sources"""
    
    # Strong synthesis indicators
    if question_indicators['compare'] or question_indicators['relationship'] or question_indicators['synthesis']:
        return True
    
    # Multiple concepts with mixed known/unknown status suggests synthesis
    if len(known_elements) > 1 and len(unknown_elements) > 0:
        return True
    
    # Analysis questions often require synthesis
    if question_indicators['analyze'] and (len(known_elements) + len(unknown_elements)) > 2:
        return True
    
    # Complex "how" or "why" questions
    if (question_indicators['how_does'] or question_indicators['why_does']) and len(unknown_elements) > 1:
        return True
    
    return False

def _calculate_query_complexity(query: str, question_indicators: Dict[str, bool], 
                              num_known: int, num_unknown: int) -> float:
    """Calculate query complexity score (0-1)"""
    complexity = 0.0
    
    # Base complexity from query length and structure
    word_count = len(query.split())
    complexity += min(0.3, word_count / 50)  # Normalize by reasonable max length
    
    # Question type complexity
    complex_questions = ['compare', 'analyze', 'synthesis', 'relationship']
    simple_questions = ['what_is', 'explain']
    
    if any(question_indicators[q] for q in complex_questions):
        complexity += 0.4
    elif any(question_indicators[q] for q in simple_questions):
        complexity += 0.1
    
    # Concept distribution complexity
    total_concepts = num_known + num_unknown
    if total_concepts > 0:
        # More unknown elements increase complexity
        unknown_ratio = num_unknown / total_concepts
        complexity += unknown_ratio * 0.3
        
        # Many concepts increase complexity
        complexity += min(0.2, total_concepts / 10)
    
    return min(1.0, complexity)

def _calculate_error_threshold(query_complexity: float, requires_synthesis: bool) -> float:
    """Calculate appropriate error threshold based on query characteristics"""
    base_threshold = 0.3
    
    # Higher complexity = higher error tolerance
    complexity_factor = query_complexity * 0.2
    
    # Synthesis requires higher error tolerance
    synthesis_factor = 0.15 if requires_synthesis else 0.0
    
    return min(0.8, base_threshold + complexity_factor + synthesis_factor)

def _calculate_analysis_confidence(certainty_scores: Dict[str, float], 
                                 context_documents: List[str]) -> float:
    """Calculate confidence in the analysis itself"""
    if not certainty_scores:
        return 0.5  # Neutral confidence when no concepts
    
    # Average certainty of all concepts
    avg_certainty = sum(certainty_scores.values()) / len(certainty_scores)
    
    # Confidence boost if we have context documents
    context_boost = 0.1 if context_documents else 0.0
    
    # Confidence penalty for very uncertain concepts
    very_uncertain = sum(1 for score in certainty_scores.values() if score < 0.2)
    uncertainty_penalty = (very_uncertain / len(certainty_scores)) * 0.2
    
    confidence = avg_certainty + context_boost - uncertainty_penalty
    return max(0.1, min(1.0, confidence))

# Backward compatibility functions
def uncertainty(scores: Sequence[float]) -> float:
    """Legacy uncertainty function for backward compatibility"""
    monitor = ErrorMonitor()
    monitor.initialize()
    return monitor.calculate_uncertainty(scores)


# Export main symbols
__all__ = ['ErrorMonitor', 'uncertainty', 'analyze_input', 'KnownUnknownAnalysis']
