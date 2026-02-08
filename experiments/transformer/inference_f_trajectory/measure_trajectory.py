"""
Measure F-trajectory during Transformer inference.

For each input, measures how F value changes as the input passes through
each layer. Tests the hypothesis that F decreases monotonically during
inference (structure becomes resolved).

Reference: SPEC.md Section 3
"""

import json
import os
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from gedig_hidden import compute_trajectory, GedigResult


# Test sentences from SPEC.md
TEST_SENTENCES = [
    # Short
    "Hello world.",
    "The cat sat on the mat.",
    # Medium
    "Machine learning is transforming how we interact with technology.",
    # Long
    "Natural language processing enables computers to understand, interpret, and generate human language in ways that are both meaningful and useful.",
    # Questions (for generative task)
    "What is the capital of France?",
    "How does photosynthesis work?",
]

# Models to test
MODELS = {
    "bert-base-uncased": {"type": "encoder", "anchor_idx": 0},
    "distilbert-base-uncased": {"type": "encoder", "anchor_idx": 0},
    "gpt2": {"type": "decoder", "anchor_idx": -1},
}


def get_hidden_states(model, tokenizer, text: str, device: str = "cpu") -> list[torch.Tensor]:
    """
    Extract hidden states from all layers for a given input.

    Returns:
        List of hidden states, one per layer (including embedding layer)
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states: tuple of (batch, seq_len, hidden_dim) for each layer
    # First element is embedding output, then each transformer layer
    hidden_states = outputs.hidden_states

    # Remove batch dimension and convert to list
    hidden_states_list = [hs.squeeze(0).cpu() for hs in hidden_states]

    return hidden_states_list


def measure_single_sample(
    model,
    tokenizer,
    text: str,
    model_config: dict,
    device: str = "cpu",
    lambda_: float = 1.0,
    gamma: float = 0.5,
    epc_method: str = "vector"
) -> dict:
    """
    Measure F-trajectory for a single input sample.
    """
    # Get hidden states
    hidden_states = get_hidden_states(model, tokenizer, text, device)
    num_layers = len(hidden_states)

    # Compute trajectory
    anchor_idx = model_config["anchor_idx"]
    trajectory = compute_trajectory(
        hidden_states,
        anchor_idx=anchor_idx,
        lambda_=lambda_,
        gamma=gamma,
        entropy_sign=-1,  # Inference: concentration is good
        epc_method=epc_method
    )

    # Extract values
    f_values = [r.f_value for r in trajectory]
    epc_values = [r.epc for r in trajectory]
    delta_h_values = [r.delta_h for r in trajectory]
    delta_sp_values = [r.delta_sp for r in trajectory]
    h_values = [trajectory[0].h_before] + [r.h_after for r in trajectory]
    sp_values = [trajectory[0].sp_before] + [r.sp_after for r in trajectory]

    # Check monotonicity
    is_monotonic_f = all(f_values[i] >= f_values[i+1] for i in range(len(f_values)-1))

    # Cumulative F
    cumulative_f = np.cumsum(f_values).tolist()

    return {
        "text": text,
        "num_tokens": len(tokenizer.encode(text)),
        "num_layers": num_layers,
        "trajectory": {
            "F": f_values,
            "cumulative_F": cumulative_f,
            "EPC": epc_values,
            "delta_H": delta_h_values,
            "delta_SP": delta_sp_values,
            "H": h_values,
            "SP": sp_values,
        },
        "is_monotonic_F": is_monotonic_f,
        "total_F": sum(f_values),
        "mean_F": np.mean(f_values),
        "h_decrease": h_values[0] - h_values[-1],
        "sp_increase": sp_values[-1] - sp_values[0],
    }


def run_experiment(
    model_name: str,
    sentences: list[str] = None,
    output_dir: str = "results",
    device: str = None,
    lambda_: float = 1.0,
    gamma: float = 0.5,
    epc_method: str = "vector"
) -> dict:
    """
    Run F-trajectory measurement for a model across all test sentences.
    """
    if sentences is None:
        sentences = TEST_SENTENCES

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_config = MODELS.get(model_name, {"type": "encoder", "anchor_idx": 0})

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Type: {model_config['type']}, Anchor: {model_config['anchor_idx']}")
    print(f"Device: {device}")
    print(f"EPC method: {epc_method}")
    print(f"{'='*60}")

    # Load model and tokenizer
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # Add pad token for GPT-2
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Run on each sentence
    results = []
    for text in tqdm(sentences, desc="Processing"):
        result = measure_single_sample(
            model, tokenizer, text, model_config, device, lambda_, gamma, epc_method
        )
        results.append(result)

        # Print summary
        tqdm.write(f"  '{text[:40]}...' → F_total={result['total_F']:.4f}, "
                   f"monotonic={result['is_monotonic_F']}")

    # Aggregate statistics
    monotonic_rate = sum(r["is_monotonic_F"] for r in results) / len(results)
    mean_total_f = np.mean([r["total_F"] for r in results])
    mean_h_decrease = np.mean([r["h_decrease"] for r in results])
    mean_sp_increase = np.mean([r["sp_increase"] for r in results])

    summary = {
        "model": model_name,
        "model_type": model_config["type"],
        "anchor_idx": model_config["anchor_idx"],
        "num_samples": len(results),
        "lambda": lambda_,
        "gamma": gamma,
        "epc_method": epc_method,
        "monotonic_rate": monotonic_rate,
        "mean_total_F": mean_total_f,
        "mean_h_decrease": mean_h_decrease,
        "mean_sp_increase": mean_sp_increase,
        "timestamp": datetime.now().isoformat(),
    }

    print(f"\n--- Summary ---")
    print(f"Monotonic F rate: {monotonic_rate:.1%}")
    print(f"Mean total F: {mean_total_f:.4f}")
    print(f"Mean H decrease: {mean_h_decrease:.4f}")
    print(f"Mean SP increase: {mean_sp_increase:.4f}")

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    safe_name = model_name.replace("/", "_")
    result_file = output_path / f"trajectory_{safe_name}.json"

    output_data = {
        "summary": summary,
        "samples": results
    }

    with open(result_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {result_file}")

    return output_data


def main():
    """Run experiment on all models."""
    import argparse

    parser = argparse.ArgumentParser(description="Measure F-trajectory during inference")
    parser.add_argument("--model", type=str, default=None,
                        help="Specific model to test (default: all)")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory")
    parser.add_argument("--lambda", type=float, default=1.0, dest="lambda_",
                        help="Lambda parameter for F computation")
    parser.add_argument("--gamma", type=float, default=0.5,
                        help="Gamma parameter for SP weight")
    parser.add_argument("--epc-method", type=str, default="vector",
                        choices=["vector", "similarity"],
                        help="EPC calculation method: vector (L2距離) or similarity (類似度行列変化)")

    args = parser.parse_args()

    # Select models
    if args.model:
        models_to_test = [args.model]
    else:
        models_to_test = list(MODELS.keys())

    all_results = {}

    for model_name in models_to_test:
        try:
            results = run_experiment(
                model_name,
                output_dir=args.output,
                lambda_=args.lambda_,
                gamma=args.gamma,
                epc_method=args.epc_method
            )
            all_results[model_name] = results["summary"]
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            continue

    # Save combined summary
    if all_results:
        summary_file = Path(args.output) / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nCombined summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
