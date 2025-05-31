"""L1 – Error Monitor with Known/Unknown Information Separation"""
import numpy as np
import re
from typing import Sequence, Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

__all__ = ["uncertainty", "analyze_input", "KnownUnknownAnalysis"]

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

def uncertainty(scores: Sequence[float]) -> float:
    """Calculate uncertainty metric from similarity scores"""
    probs = np.array(scores, dtype=float)
    probs = probs / (probs.sum() + 1e-9)
    return float(-np.sum(probs * np.log(probs + 1e-9)))

def analyze_input(query: str, context_documents: List[str] = None, 
                 knowledge_base_stats: Dict[str, Any] = None,
                 unknown_learner: Optional[Any] = None) -> KnownUnknownAnalysis:
    """
    Analyze input query to separate known vs unknown information components.
    
    This function implements the core Layer1 functionality of distinguishing
    what information is already available (known) vs what needs to be discovered
    or synthesized (unknown) before proceeding to retrieval layers.
    
    Args:
        query: Input question/query to analyze
        context_documents: Optional list of available documents for context
        knowledge_base_stats: Optional statistics about knowledge base coverage
        unknown_learner: Optional UnknownLearner instance for registering discoveries
        
    Returns:
        KnownUnknownAnalysis containing separated information components
    """
    
    # Extract key entities and concepts from query
    query_concepts = _extract_concepts(query)
    question_indicators = _identify_question_type(query)
    
    # Analyze concept familiarity if context available
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

def _extract_concepts(query: str) -> List[str]:
    """Extract key concepts and entities from query text"""
    # Simple concept extraction using patterns
    # In production, this would use NLP models
    
    # Remove question words and articles
    stopwords = {'what', 'how', 'why', 'when', 'where', 'who', 'which', 'is', 'are', 'was', 'were',
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
    
    # Split and clean words
    words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
    concepts = [w for w in words if w not in stopwords and len(w) > 2]
    
    # Look for compound concepts (simple heuristic)
    compound_concepts = []
    for i in range(len(concepts) - 1):
        if len(concepts[i]) > 3 and len(concepts[i+1]) > 3:
            compound = f"{concepts[i]} {concepts[i+1]}"
            compound_concepts.append(compound)
    
    return list(set(concepts + compound_concepts))

def _identify_question_type(query: str) -> Dict[str, bool]:
    """Identify question type indicators"""
    query_lower = query.lower()
    
    return {
        'comparison': any(word in query_lower for word in ['compare', 'versus', 'vs', 'difference', 'similar']),
        'causal': any(word in query_lower for word in ['why', 'because', 'cause', 'reason', 'effect']),
        'synthesis': any(word in query_lower for word in ['combine', 'integrate', 'relationship', 'connection']),
        'definition': any(word in query_lower for word in ['what is', 'define', 'meaning', 'explain']),
        'procedural': any(word in query_lower for word in ['how to', 'steps', 'process', 'method']),
        'temporal': any(word in query_lower for word in ['when', 'before', 'after', 'during', 'timeline']),
        'hypothetical': any(word in query_lower for word in ['if', 'suppose', 'assume', 'imagine', 'would']),
        'paradox': any(word in query_lower for word in ['paradox', 'contradiction', 'impossible', 'puzzle'])
    }

def _calculate_concept_certainty(concept: str, context_documents: List[str] = None,
                                knowledge_base_stats: Dict[str, Any] = None) -> float:
    """Calculate certainty level for a specific concept"""
    
    base_certainty = 0.5  # Default neutral certainty
    
    # If we have context documents, check for concept presence
    if context_documents:
        matches = sum(1 for doc in context_documents if concept.lower() in doc.lower())
        coverage = matches / len(context_documents) if context_documents else 0
        base_certainty = min(0.9, 0.3 + coverage * 0.6)
    
    # If we have knowledge base statistics, adjust based on frequency
    if knowledge_base_stats and 'concept_frequencies' in knowledge_base_stats:
        freq = knowledge_base_stats['concept_frequencies'].get(concept, 0)
        total_concepts = knowledge_base_stats.get('total_concepts', 1000)
        relative_freq = freq / total_concepts
        
        # Higher frequency = higher certainty, but cap at 0.95
        freq_boost = min(0.4, relative_freq * 10)
        base_certainty = min(0.95, base_certainty + freq_boost)
    
    # Adjust for concept complexity (longer concepts often more specific/uncertain)
    if len(concept.split()) > 2:
        base_certainty *= 0.8  # Multi-word concepts are often more uncertain
    
    return max(0.1, min(0.95, base_certainty))

def _requires_synthesis_analysis(query: str, question_indicators: Dict[str, bool],
                               known_elements: List[str], unknown_elements: List[str]) -> bool:
    """Determine if query requires synthesis of multiple concepts"""
    
    # Strong indicators of synthesis requirement
    synthesis_indicators = [
        question_indicators['comparison'],
        question_indicators['causal'],
        question_indicators['synthesis'],
        question_indicators['paradox'],
        question_indicators['hypothetical']
    ]
    
    if any(synthesis_indicators):
        return True
    
    # If many unknown elements relative to known, likely requires synthesis
    if len(unknown_elements) > len(known_elements) and len(unknown_elements) > 2:
        return True
    
    # Complex queries with multiple concepts often require synthesis
    total_concepts = len(known_elements) + len(unknown_elements)
    if total_concepts > 4:
        return True
    
    # Look for connecting words that suggest relationships
    connecting_words = ['relationship', 'connection', 'link', 'relate', 'affect', 'influence']
    if any(word in query.lower() for word in connecting_words):
        return True
    
    return False

def _calculate_query_complexity(query: str, question_indicators: Dict[str, bool],
                              num_known: int, num_unknown: int) -> float:
    """Calculate overall query complexity score"""
    
    # Base complexity from query length and structure
    base_complexity = min(1.0, len(query.split()) / 20.0)
    
    # Add complexity for question type
    complexity_weights = {
        'definition': 0.2,
        'procedural': 0.4, 
        'temporal': 0.5,
        'comparison': 0.7,
        'causal': 0.8,
        'synthesis': 0.9,
        'hypothetical': 0.8,
        'paradox': 1.0
    }
    
    type_complexity = max([complexity_weights.get(q_type, 0.3) 
                          for q_type, present in question_indicators.items() if present] or [0.3])
    
    # Add complexity for unknown elements
    unknown_complexity = min(0.5, num_unknown * 0.1)
    
    # Combine factors
    total_complexity = min(1.0, base_complexity * 0.3 + type_complexity * 0.5 + unknown_complexity * 0.2)
    
    return total_complexity

def _calculate_error_threshold(query_complexity: float, requires_synthesis: bool) -> float:
    """Calculate appropriate error threshold for monitoring"""
    
    # Base threshold inversely related to complexity
    base_threshold = 0.5 - (query_complexity * 0.3)
    
    # Lower threshold (more sensitive) for synthesis tasks
    if requires_synthesis:
        base_threshold *= 0.8
    
    # Ensure reasonable bounds
    return max(0.1, min(0.8, base_threshold))

def _calculate_analysis_confidence(certainty_scores: Dict[str, float], 
                                 context_documents: List[str] = None) -> float:
    """Calculate confidence in the analysis itself"""
    
    if not certainty_scores:
        return 0.3  # Low confidence with no concepts
    
    # Average certainty as base confidence
    avg_certainty = sum(certainty_scores.values()) / len(certainty_scores)
    
    # Adjust based on availability of context
    context_boost = 0.2 if context_documents and len(context_documents) > 5 else 0.0
    
    # Adjust based on certainty variance (high variance = lower confidence)
    certainty_values = list(certainty_scores.values())
    variance = np.var(certainty_values) if len(certainty_values) > 1 else 0
    variance_penalty = min(0.3, variance * 0.5)
    
    confidence = min(0.95, max(0.2, avg_certainty + context_boost - variance_penalty))
    
    return confidence
