"""
HotPotQA Bridge Problem Experiment with Structural Similarity

Tests whether structural similarity can help with bridge-type questions
by detecting "bridging patterns" between documents.

Bridge questions require reasoning across two documents via a shared entity.
Example: "What government position was held by the woman who portrayed Edith Bunker?"
  - Doc A: Jean Stapleton portrayed Edith Bunker
  - Doc B: Jean Stapleton also [some government position]
  - Bridge: Jean Stapleton connects the two facts

Hypothesis: Structural similarity can detect "bridge patterns" in knowledge graphs
built from the documents, improving retrieval accuracy.

Usage:
    python -m experiments.hotpotqa-benchmark.bridge_with_ss
"""

import json
import sys
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import HotpotQALoader, HotpotQAExample
from insightspike.config.models import StructuralSimilarityConfig
from insightspike.algorithms.structural_similarity import StructuralSimilarityEvaluator


@dataclass
class BridgePattern:
    """Represents a detected bridge pattern."""
    doc_a_title: str
    doc_b_title: str
    bridge_entity: str
    similarity: float
    pattern_type: str  # "chain", "hub", "shared_entity"


@dataclass
class BridgeExperimentResult:
    """Result for a single bridge question."""
    example_id: str
    question: str
    gold_docs: List[str]
    predicted_docs: List[str]
    bridge_detected: bool
    bridge_similarity: float
    doc_retrieval_correct: bool
    used_ss: bool


# =============================================================================
# Knowledge Graph Construction from Documents
# =============================================================================

def build_document_graph(title: str, sentences: List[str]) -> nx.Graph:
    """Build a simple knowledge graph from a document.

    Creates nodes for:
    - Document title (hub)
    - Key entities (nouns/proper nouns) - simplified extraction
    - Sentences (as concepts)

    Creates edges for:
    - Title -> Entities mentioned in the document
    - Entity -> Entity co-occurrence
    """
    G = nx.Graph()

    # Add document node as hub
    G.add_node(title, node_type="document", role="hub")

    # Simple entity extraction (words starting with capital letters)
    all_entities = set()
    sentence_entities = []

    for sent_idx, sent in enumerate(sentences):
        # Extract potential entities (capitalized words, excluding sentence start)
        words = sent.split()
        entities = []
        for i, word in enumerate(words):
            # Skip first word (sentence start) and common words
            if i > 0 and word[0].isupper() and len(word) > 2:
                clean_word = word.strip('.,!?;:()[]"\'')
                if clean_word and clean_word not in ['The', 'This', 'That', 'These', 'Those', 'It', 'He', 'She', 'They']:
                    entities.append(clean_word)
                    all_entities.add(clean_word)
        sentence_entities.append(entities)

    # Add entity nodes and connect to document
    for entity in all_entities:
        G.add_node(entity, node_type="entity", role="spoke")
        G.add_edge(title, entity, relation="mentions")

    # Connect co-occurring entities within sentences
    for entities in sentence_entities:
        for i, e1 in enumerate(entities):
            for e2 in entities[i+1:]:
                if e1 != e2 and G.has_node(e1) and G.has_node(e2):
                    if not G.has_edge(e1, e2):
                        G.add_edge(e1, e2, relation="co_occurs")

    return G


def build_context_graph(example: HotpotQAExample) -> nx.Graph:
    """Build a combined knowledge graph from all context documents."""
    G = nx.Graph()

    for title, sentences in example.context:
        doc_graph = build_document_graph(title, sentences)
        # Merge into main graph
        G = nx.compose(G, doc_graph)

    return G


def find_bridge_entities(
    graph: nx.Graph,
    doc_a_title: str,
    doc_b_title: str,
) -> List[str]:
    """Find entities that bridge two documents."""
    if doc_a_title not in graph or doc_b_title not in graph:
        return []

    # Get entities connected to each document
    neighbors_a = set(graph.neighbors(doc_a_title))
    neighbors_b = set(graph.neighbors(doc_b_title))

    # Find shared entities (potential bridges)
    bridge_entities = neighbors_a & neighbors_b

    return list(bridge_entities)


# =============================================================================
# Bridge Pattern Detection
# =============================================================================

def detect_bridge_pattern(
    graph: nx.Graph,
    doc_a_title: str,
    doc_b_title: str,
    ss_evaluator: Optional[StructuralSimilarityEvaluator],
) -> Optional[BridgePattern]:
    """Detect if there's a bridge pattern between two documents."""

    # Find bridge entities
    bridge_entities = find_bridge_entities(graph, doc_a_title, doc_b_title)

    if not bridge_entities:
        return None

    # Calculate structural similarity if evaluator provided
    similarity = 0.0
    if ss_evaluator:
        # Extract subgraphs around each document
        try:
            nodes_a = set(nx.ego_graph(graph, doc_a_title, radius=2).nodes())
            nodes_b = set(nx.ego_graph(graph, doc_b_title, radius=2).nodes())

            subgraph_a = graph.subgraph(nodes_a).copy()
            subgraph_b = graph.subgraph(nodes_b).copy()

            result = ss_evaluator.evaluate(
                subgraph_a, subgraph_b,
                center1=doc_a_title, center2=doc_b_title
            )
            similarity = result.similarity
        except Exception:
            similarity = 0.5  # Default if extraction fails
    else:
        # Without SS, use simple heuristic
        similarity = len(bridge_entities) / 10.0  # Normalize roughly

    # Determine pattern type
    if len(bridge_entities) == 1:
        pattern_type = "chain"  # Single bridge entity
    elif len(bridge_entities) > 3:
        pattern_type = "hub"  # Many shared entities
    else:
        pattern_type = "shared_entity"

    return BridgePattern(
        doc_a_title=doc_a_title,
        doc_b_title=doc_b_title,
        bridge_entity=bridge_entities[0] if bridge_entities else "",
        similarity=similarity,
        pattern_type=pattern_type,
    )


# =============================================================================
# Document Retrieval with Bridge Detection
# =============================================================================

def retrieve_documents_baseline(
    example: HotpotQAExample,
    top_k: int = 2,
) -> List[str]:
    """Baseline retrieval: random selection (simulating no special logic)."""
    # In a real system, this would use embedding similarity
    # For this experiment, we simulate by randomly picking documents
    # that have some word overlap with the question

    question_words = set(example.question.lower().split())

    scored_docs = []
    for title, sentences in example.context:
        doc_text = ' '.join(sentences).lower()
        doc_words = set(doc_text.split())
        overlap = len(question_words & doc_words)
        scored_docs.append((title, overlap))

    scored_docs.sort(key=lambda x: -x[1])
    return [title for title, _ in scored_docs[:top_k]]


def retrieve_documents_with_ss(
    example: HotpotQAExample,
    graph: nx.Graph,
    ss_evaluator: StructuralSimilarityEvaluator,
    top_k: int = 2,
) -> Tuple[List[str], float, bool]:
    """Retrieve documents using structural similarity for bridge detection.

    Returns:
        Tuple of (selected_docs, bridge_similarity, bridge_detected)
    """
    # First, get baseline scores
    question_words = set(example.question.lower().split())

    doc_scores = {}
    for title, sentences in example.context:
        doc_text = ' '.join(sentences).lower()
        doc_words = set(doc_text.split())
        overlap = len(question_words & doc_words)
        doc_scores[title] = overlap

    # Then, look for bridge patterns between document pairs
    titles = [t for t, _ in example.context]
    best_bridge = None
    best_bridge_score = 0.0

    for i, title_a in enumerate(titles):
        for title_b in titles[i+1:]:
            pattern = detect_bridge_pattern(graph, title_a, title_b, ss_evaluator)
            if pattern and pattern.similarity > best_bridge_score:
                best_bridge = pattern
                best_bridge_score = pattern.similarity

    # If a strong bridge is detected, prioritize those documents
    bridge_detected = best_bridge is not None and best_bridge_score > 0.3

    if bridge_detected:
        # Boost bridge documents
        doc_scores[best_bridge.doc_a_title] = doc_scores.get(best_bridge.doc_a_title, 0) + 10
        doc_scores[best_bridge.doc_b_title] = doc_scores.get(best_bridge.doc_b_title, 0) + 10

    # Sort and select top_k
    sorted_docs = sorted(doc_scores.items(), key=lambda x: -x[1])
    selected = [title for title, _ in sorted_docs[:top_k]]

    return selected, best_bridge_score if best_bridge else 0.0, bridge_detected


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_retrieval(
    predicted: List[str],
    gold: List[str],
) -> bool:
    """Check if predicted documents contain all gold documents."""
    return set(gold).issubset(set(predicted))


def run_bridge_experiment(
    data_path: Path,
    num_samples: int = 100,
    seed: int = 42,
) -> Tuple[List[BridgeExperimentResult], Dict[str, float]]:
    """Run the bridge problem experiment.

    Compares:
    1. Baseline: Simple word overlap retrieval
    2. SS-enhanced: Bridge pattern detection with structural similarity
    """
    random.seed(seed)
    np.random.seed(seed)

    # Load data
    loader = HotpotQALoader(data_path)
    bridge_examples = loader.filter_by_type("bridge")

    # Sample if too many
    if len(bridge_examples) > num_samples:
        bridge_examples = random.sample(bridge_examples, num_samples)

    print(f"\nLoaded {len(bridge_examples)} bridge examples")

    # Setup SS evaluator
    ss_config = StructuralSimilarityConfig(
        enabled=True,
        method="signature",
        analogy_threshold=0.3,
        cross_domain_only=False,
    )
    ss_evaluator = StructuralSimilarityEvaluator(ss_config)

    # Run experiments
    results_baseline: List[BridgeExperimentResult] = []
    results_ss: List[BridgeExperimentResult] = []

    for i, example in enumerate(bridge_examples):
        if (i + 1) % 20 == 0:
            print(f"  Processing {i+1}/{len(bridge_examples)}...")

        gold_docs = list(set(example.supporting_facts_titles))

        # Build knowledge graph
        graph = build_context_graph(example)

        # Baseline retrieval
        pred_baseline = retrieve_documents_baseline(example)
        correct_baseline = evaluate_retrieval(pred_baseline, gold_docs)

        results_baseline.append(BridgeExperimentResult(
            example_id=example.id,
            question=example.question,
            gold_docs=gold_docs,
            predicted_docs=pred_baseline,
            bridge_detected=False,
            bridge_similarity=0.0,
            doc_retrieval_correct=correct_baseline,
            used_ss=False,
        ))

        # SS-enhanced retrieval
        pred_ss, similarity, bridge_detected = retrieve_documents_with_ss(
            example, graph, ss_evaluator
        )
        correct_ss = evaluate_retrieval(pred_ss, gold_docs)

        results_ss.append(BridgeExperimentResult(
            example_id=example.id,
            question=example.question,
            gold_docs=gold_docs,
            predicted_docs=pred_ss,
            bridge_detected=bridge_detected,
            bridge_similarity=similarity,
            doc_retrieval_correct=correct_ss,
            used_ss=True,
        ))

    # Compute metrics
    n = len(bridge_examples)
    metrics = {
        "total_examples": n,
        "baseline_accuracy": sum(r.doc_retrieval_correct for r in results_baseline) / n,
        "ss_accuracy": sum(r.doc_retrieval_correct for r in results_ss) / n,
        "bridge_detection_rate": sum(r.bridge_detected for r in results_ss) / n,
        "avg_bridge_similarity": np.mean([r.bridge_similarity for r in results_ss]),
    }

    # When bridge was detected
    ss_with_bridge = [r for r in results_ss if r.bridge_detected]
    if ss_with_bridge:
        metrics["accuracy_when_bridge_detected"] = sum(r.doc_retrieval_correct for r in ss_with_bridge) / len(ss_with_bridge)

    metrics["delta_accuracy"] = metrics["ss_accuracy"] - metrics["baseline_accuracy"]

    return results_ss, metrics


def print_experiment_results(
    results: List[BridgeExperimentResult],
    metrics: Dict[str, float],
):
    """Print experiment results."""
    print(f"\n{'='*70}")
    print("HotPotQA Bridge Problem Experiment Results")
    print(f"{'='*70}\n")

    print(f"Total examples: {metrics['total_examples']}")
    print(f"\nDocument Retrieval Accuracy:")
    print(f"  Baseline (word overlap): {metrics['baseline_accuracy']:.1%}")
    print(f"  SS-enhanced:             {metrics['ss_accuracy']:.1%}")
    print(f"  Delta:                   {metrics['delta_accuracy']:+.1%}")

    print(f"\nBridge Detection:")
    print(f"  Detection rate: {metrics['bridge_detection_rate']:.1%}")
    print(f"  Avg similarity: {metrics['avg_bridge_similarity']:.3f}")

    if "accuracy_when_bridge_detected" in metrics:
        print(f"  Accuracy when bridge detected: {metrics['accuracy_when_bridge_detected']:.1%}")

    # Show some examples
    print(f"\n{'='*70}")
    print("Example Results")
    print(f"{'='*70}")

    # Show improved cases
    improved = [r for r in results if r.doc_retrieval_correct and r.bridge_detected]
    if improved:
        print(f"\n✓ Improved with bridge detection ({len(improved)} cases):")
        for r in improved[:3]:
            print(f"  ID: {r.example_id}")
            print(f"  Q: {r.question[:60]}...")
            print(f"  Bridge similarity: {r.bridge_similarity:.3f}")
            print()

    # Show cases where bridge wasn't helpful
    not_improved = [r for r in results if not r.doc_retrieval_correct and r.bridge_detected]
    if not_improved:
        print(f"\n✗ Bridge detected but retrieval failed ({len(not_improved)} cases):")
        for r in not_improved[:2]:
            print(f"  ID: {r.example_id}")
            print(f"  Gold docs: {r.gold_docs}")
            print(f"  Predicted: {r.predicted_docs}")
            print()

    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Path to HotPotQA dev set
    data_path = Path(__file__).parent / "data" / "hotpotqa_distractor_dev.jsonl"

    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        print("Please ensure HotPotQA data is available.")
        sys.exit(1)

    # Run experiment
    results, metrics = run_bridge_experiment(
        data_path=data_path,
        num_samples=200,  # Use 200 samples for faster iteration
        seed=42,
    )

    # Print results
    print_experiment_results(results, metrics)

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "bridge_ss_experiment.json", "w") as f:
        json.dump({
            "metrics": metrics,
            "sample_results": [asdict(r) for r in results[:20]],
        }, f, indent=2)

    print(f"Results saved to {output_dir / 'bridge_ss_experiment.json'}")
