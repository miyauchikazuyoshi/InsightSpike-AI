#!/usr/bin/env python3
"""
Adaptive topK Algorithm Based on Layer1 Analysis
===============================================

This module implements adaptive topK selection based on Layer1's known/unknown
information separation analysis. The algorithm dynamically adjusts retrieval
parameters to optimize for different query types and complexity levels.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass
class AdaptiveTopKConfig:
    """Configuration for adaptive topK algorithm"""

    # Base topK values for different layers
    base_layer1_k: int = 20
    base_layer2_k: int = 15
    base_layer3_k: int = 12

    # Scaling factors based on analysis
    synthesis_multiplier: float = 1.5  # Increase topK for synthesis tasks
    complexity_multiplier: float = 1.3  # Increase topK for complex queries
    unknown_ratio_multiplier: float = 1.2  # Increase topK for high unknown ratio

    # Bounds to prevent excessive resource usage
    max_layer1_k: int = 50
    max_layer2_k: int = 30
    max_layer3_k: int = 25
    min_k: int = 3

    # Confidence-based adjustments
    low_confidence_multiplier: float = 1.4  # Increase topK for low confidence
    confidence_threshold: float = 0.6


def calculate_adaptive_topk(
    l1_analysis: Dict[str, Any], config: AdaptiveTopKConfig = None
) -> Dict[str, int]:
    """
    Calculate adaptive topK values for each layer based on Layer1 analysis.

    Args:
        l1_analysis: Layer1 analysis results containing:
            - known_elements: List of known concepts
            - unknown_elements: List of unknown concepts
            - requires_synthesis: Boolean indicating synthesis requirement
            - query_complexity: Float 0-1 indicating complexity
            - analysis_confidence: Float 0-1 indicating confidence

        config: Optional configuration object

    Returns:
        Dictionary with topK values for each layer
    """

    if config is None:
        config = AdaptiveTopKConfig()

    # Extract analysis components
    known_count = len(l1_analysis.get("known_elements", []))
    unknown_count = len(l1_analysis.get("unknown_elements", []))
    requires_synthesis = l1_analysis.get("requires_synthesis", False)
    query_complexity = l1_analysis.get("query_complexity", 0.5)
    analysis_confidence = l1_analysis.get("analysis_confidence", 0.5)

    # Calculate unknown ratio
    total_concepts = known_count + unknown_count
    unknown_ratio = unknown_count / total_concepts if total_concepts > 0 else 0.5

    # Start with base values
    layer1_k = config.base_layer1_k
    layer2_k = config.base_layer2_k
    layer3_k = config.base_layer3_k

    # Apply synthesis multiplier
    if requires_synthesis:
        layer1_k = int(layer1_k * config.synthesis_multiplier)
        layer2_k = int(layer2_k * config.synthesis_multiplier)
        layer3_k = int(layer3_k * config.synthesis_multiplier)

    # Apply complexity multiplier
    complexity_factor = 1.0 + (query_complexity * (config.complexity_multiplier - 1.0))
    layer1_k = int(layer1_k * complexity_factor)
    layer2_k = int(layer2_k * complexity_factor)
    layer3_k = int(layer3_k * complexity_factor)

    # Apply unknown ratio multiplier
    unknown_factor = 1.0 + (unknown_ratio * (config.unknown_ratio_multiplier - 1.0))
    layer1_k = int(layer1_k * unknown_factor)
    layer2_k = int(layer2_k * unknown_factor)
    layer3_k = int(layer3_k * unknown_factor)

    # Apply confidence adjustment
    if analysis_confidence < config.confidence_threshold:
        confidence_factor = config.low_confidence_multiplier
        layer1_k = int(layer1_k * confidence_factor)
        layer2_k = int(layer2_k * confidence_factor)
        layer3_k = int(layer3_k * confidence_factor)

    # Apply bounds
    layer1_k = max(config.min_k, min(layer1_k, config.max_layer1_k))
    layer2_k = max(config.min_k, min(layer2_k, config.max_layer2_k))
    layer3_k = max(config.min_k, min(layer3_k, config.max_layer3_k))

    return {
        "layer1_k": layer1_k,
        "layer2_k": layer2_k,
        "layer3_k": layer3_k,
        "adaptation_factors": {
            "synthesis_applied": requires_synthesis,
            "complexity_factor": complexity_factor,
            "unknown_factor": unknown_factor,
            "confidence_factor": confidence_factor
            if analysis_confidence < config.confidence_threshold
            else 1.0,
            "scaling": {
                "layer1": layer1_k / config.base_layer1_k,
                "layer2": layer2_k / config.base_layer2_k,
                "layer3": layer3_k / config.base_layer3_k,
            },
        },
    }


def estimate_chain_reaction_potential(
    l1_analysis: Dict[str, Any], adaptive_topk: Dict[str, int]
) -> float:
    """
    Estimate the potential for "連鎖反応的洞察向上" (chain reaction insight improvement)
    based on Layer1 analysis and adaptive topK settings.

    Args:
        l1_analysis: Layer1 analysis results
        adaptive_topk: Adaptive topK configuration

    Returns:
        Float 0-1 indicating chain reaction potential
    """

    # Factors that increase chain reaction potential
    synthesis_required = l1_analysis.get("requires_synthesis", False)
    query_complexity = l1_analysis.get("query_complexity", 0.5)
    unknown_count = len(l1_analysis.get("unknown_elements", []))

    # topK density factor
    avg_topk = (
        adaptive_topk["layer1_k"]
        + adaptive_topk["layer2_k"]
        + adaptive_topk["layer3_k"]
    ) / 3
    density_factor = min(1.0, avg_topk / 20.0)  # Normalize against moderate density

    # Synthesis factor
    synthesis_factor = 0.8 if synthesis_required else 0.3

    # Complexity factor (higher complexity = higher potential)
    complexity_factor = query_complexity

    # Unknown element factor (more unknowns = higher potential for connections)
    unknown_factor = min(1.0, unknown_count / 10.0)

    # Combine factors
    chain_reaction_potential = (
        synthesis_factor * 0.4
        + complexity_factor * 0.3
        + density_factor * 0.2
        + unknown_factor * 0.1
    )

    return min(1.0, chain_reaction_potential)


def test_adaptive_topk():
    """Test the adaptive topK algorithm with various scenarios"""

    print("🔬 Testing Adaptive topK Algorithm")
    print("=" * 60)

    # Test scenarios
    test_scenarios = [
        {
            "name": "Simple Question",
            "analysis": {
                "known_elements": ["quantum"],
                "unknown_elements": ["mechanics"],
                "requires_synthesis": False,
                "query_complexity": 0.2,
                "analysis_confidence": 0.8,
            },
        },
        {
            "name": "Medium Synthesis",
            "analysis": {
                "known_elements": ["probability"],
                "unknown_elements": ["theory", "uncertainty", "relationship"],
                "requires_synthesis": True,
                "query_complexity": 0.5,
                "analysis_confidence": 0.6,
            },
        },
        {
            "name": "Complex Cross-Domain",
            "analysis": {
                "known_elements": ["probability"],
                "unknown_elements": [
                    "monty",
                    "hall",
                    "problem",
                    "demonstrate",
                    "relationship",
                    "information",
                    "theory",
                ],
                "requires_synthesis": True,
                "query_complexity": 0.8,
                "analysis_confidence": 0.5,
            },
        },
        {
            "name": "High Complexity Low Confidence",
            "analysis": {
                "known_elements": [],
                "unknown_elements": [
                    "ship",
                    "theseus",
                    "quantum",
                    "particles",
                    "measurement",
                    "observer",
                    "effect",
                ],
                "requires_synthesis": True,
                "query_complexity": 0.9,
                "analysis_confidence": 0.3,
            },
        },
    ]

    for scenario in test_scenarios:
        name = scenario["name"]
        analysis = scenario["analysis"]

        print(f"\n📋 Scenario: {name}")
        print(
            f'   Known: {len(analysis["known_elements"])}, Unknown: {len(analysis["unknown_elements"])}'
        )
        print(
            f'   Synthesis: {analysis["requires_synthesis"]}, Complexity: {analysis["query_complexity"]:.2f}'
        )
        print(f'   Confidence: {analysis["analysis_confidence"]:.2f}')
        print("-" * 50)

        # Calculate adaptive topK
        adaptive_result = calculate_adaptive_topk(analysis)
        topk_values = {
            k: v for k, v in adaptive_result.items() if not k.startswith("adaptation")
        }
        adaptation_factors = adaptive_result["adaptation_factors"]

        # Calculate chain reaction potential
        chain_potential = estimate_chain_reaction_potential(analysis, topk_values)

        print(f"📈 Adaptive topK Results:")
        print(
            f'   Layer1: {topk_values["layer1_k"]} (scale: {adaptation_factors["scaling"]["layer1"]:.2f}x)'
        )
        print(
            f'   Layer2: {topk_values["layer2_k"]} (scale: {adaptation_factors["scaling"]["layer2"]:.2f}x)'
        )
        print(
            f'   Layer3: {topk_values["layer3_k"]} (scale: {adaptation_factors["scaling"]["layer3"]:.2f}x)'
        )

        print(f"⚙️ Applied Factors:")
        print(f'   Synthesis: {adaptation_factors["synthesis_applied"]}')
        print(f'   Complexity: {adaptation_factors["complexity_factor"]:.2f}x')
        print(f'   Unknown Ratio: {adaptation_factors["unknown_factor"]:.2f}x')
        print(f'   Confidence: {adaptation_factors["confidence_factor"]:.2f}x')

        print(f"🔗 Chain Reaction Potential: {chain_potential:.2f}")

        # Interpretation
        if chain_potential > 0.7:
            potential_desc = "Very High - Likely to produce insights"
        elif chain_potential > 0.5:
            potential_desc = "High - Good synthesis potential"
        elif chain_potential > 0.3:
            potential_desc = "Medium - Some insight potential"
        else:
            potential_desc = "Low - Primarily retrieval-based"

        print(f"   Interpretation: {potential_desc}")

    print("\n" + "=" * 60)
    print("✅ Adaptive topK Algorithm Test Complete")

    print("\n🎯 Algorithm Benefits:")
    print("   • Dynamic scaling based on query complexity")
    print("   • Synthesis-aware topK adjustment")
    print("   • Confidence-based uncertainty handling")
    print("   • Chain reaction potential estimation")
    print("   • Resource-conscious with maximum bounds")


if __name__ == "__main__":
    test_adaptive_topk()
