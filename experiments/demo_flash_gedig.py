
import torch
import sys
import os

# Ensure package is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src")))

from insightspike.gedig import compute_f_score, FlashGeDIGLoss

def test_flash_gedig():
    print("=== Flash-geDIG Verification Demo ===")
    
    # Setup inputs
    batch, heads, seq = 2, 4, 32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # 1. Functional API Test
    print("\n[1] Testing Functional API...")
    attention_raw = torch.rand(batch, heads, seq, seq, requires_grad=True, device=device)
    # Make it a probability distribution
    attention = torch.softmax(attention_raw, dim=-1)
    # Retain grad for the softmax output if we really want to check it, 
    # but usually we check if gradients flow back to the input parameters (attention_raw)
    attention.retain_grad()
    
    f_val, metrics = compute_f_score(attention)
    print(f"F-Mean: {f_val.mean().item():.4f}")
    print(f"EPC: {metrics['delta_epc'].mean().item():.4f}")
    print(f"Entropy: {metrics['delta_h'].mean().item():.4f}")
    print(f"SP: {metrics['delta_sp'].mean().item():.4f}")
    
    # 2. Backprop Test (Differentiability)
    print("\n[2] Testing Differentiability...")
    loss = -f_val.mean() # Maximize F
    loss.backward()
    
    if attention_raw.grad is not None:
        print("SUCCESS: Gradients propagated to input tensor.")
        print(f"Input Grad Norm: {attention_raw.grad.norm().item():.4f}")
    else:
        print("FAILURE: No gradients on input!")

    # 3. Module API Test
    print("\n[3] Testing Module API (FlashGeDIGLoss)...")
    model_sim_attentions = (
        torch.rand(batch, heads, seq, seq, requires_grad=True, device=device),
        torch.rand(batch, heads, seq, seq, requires_grad=True, device=device)
    )
    criterion = FlashGeDIGLoss()
    loss_module = criterion(model_sim_attentions)
    
    print(f"Loss Output: {loss_module.item():.4f}")
    loss_module.backward()
    print("Backprop complete.")

if __name__ == "__main__":
    test_flash_gedig()
