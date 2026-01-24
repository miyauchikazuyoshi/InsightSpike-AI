"""
Flash-geDIG Validation
======================

Checks device residency, gradient flow, optional gradcheck, and optional timing.
"""

import argparse
import os
import sys
import time

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from insightspike.gedig import compute_f_score


def _device_from_arg(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _run_forward(device: str, batch: int, heads: int, seq: int, temperature: float, percentile: float, seed: int):
    if seed is not None:
        torch.manual_seed(seed)
    attention_raw = torch.rand(batch, heads, seq, seq, device=device, requires_grad=True)
    attention = torch.softmax(attention_raw, dim=-1)
    attention.retain_grad()

    f_val, metrics = compute_f_score(
        attention,
        temperature=temperature,
        percentile=percentile,
    )

    devices_ok = f_val.device == attention.device
    for value in metrics.values():
        devices_ok = devices_ok and (value.device == attention.device)

    loss = -f_val.mean()
    loss.backward()

    grad_ok = attention_raw.grad is not None
    grad_norm = attention_raw.grad.norm().item() if grad_ok else 0.0

    return f_val.mean().item(), devices_ok, grad_ok, grad_norm


def _run_gradcheck(batch: int, heads: int, seq: int, temperature: float, percentile: float, seed: int):
    def fn(attention_raw):
        attention = torch.softmax(attention_raw, dim=-1)
        f_val, _ = compute_f_score(
            attention,
            temperature=temperature,
            percentile=percentile,
        )
        return f_val.sum()

    if seed is not None:
        torch.manual_seed(seed)
    attention_raw = torch.rand(
        batch, heads, seq, seq, dtype=torch.double, requires_grad=True
    )
    return torch.autograd.gradcheck(fn, (attention_raw,), eps=1e-6, atol=1e-4, rtol=1e-3)


def _profile(device: str, batch: int, heads: int, seq: int, temperature: float, percentile: float, seed: int):
    if seed is not None:
        torch.manual_seed(seed)
    attention = torch.rand(batch, heads, seq, seq, device=device)
    attention = torch.softmax(attention, dim=-1)

    def step():
        compute_f_score(attention, temperature=temperature, percentile=percentile)

    if device.startswith("cuda") and torch.cuda.is_available():
        warmup = 5
        iters = 20
        for _ in range(warmup):
            step()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            step()
        end.record()
        torch.cuda.synchronize()
        return (start.elapsed_time(end) / iters), "cuda"

    warmup = 3
    iters = 10
    for _ in range(warmup):
        step()
    start = time.perf_counter()
    for _ in range(iters):
        step()
    end = time.perf_counter()
    return ((end - start) * 1000.0 / iters), "cpu"


def main():
    parser = argparse.ArgumentParser(description="Flash-geDIG validation checks")
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--seq", type=int, default=16)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--percentile", type=float, default=0.9)
    parser.add_argument("--gradcheck", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    device = _device_from_arg(args.device)
    print("=== Flash-geDIG Validation ===")
    print(f"device: {device}")

    f_mean, devices_ok, grad_ok, grad_norm = _run_forward(
        device,
        args.batch,
        args.heads,
        args.seq,
        args.temperature,
        args.percentile,
        args.seed,
    )
    print(f"f_mean: {f_mean:.6f}")
    print(f"device_resident: {devices_ok}")
    print(f"grad_flow: {grad_ok} (norm={grad_norm:.6f})")

    if args.gradcheck:
        try:
            passed = _run_gradcheck(
                args.batch,
                args.heads,
                args.seq,
                args.temperature,
                args.percentile,
                args.seed,
            )
            print(f"gradcheck: {passed}")
        except Exception as exc:
            print(f"gradcheck: failed ({exc})")

    if args.profile:
        avg_ms, mode = _profile(
            device,
            args.batch,
            args.heads,
            args.seq,
            args.temperature,
            args.percentile,
            args.seed,
        )
        print(f"profile_avg_ms({mode}): {avg_ms:.3f}")


if __name__ == "__main__":
    main()
