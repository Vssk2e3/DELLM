"""Quick check (run first, on any GPU or even CPU with the 1B model):
does the LLM really see the whole sequence once the causal mask is removed?

Usage:  python scripts/00_check_mask.py --backbone meta-llama/Llama-3.2-1B
"""
import argparse

import torch

import _bootstrap  # noqa: F401
from dellm.cloud.model import DELLMCloud, check_bidirectional


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--dtype", default="float32", help="float32 on CPU, bfloat16 on GPU")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DELLMCloud(backbone=args.backbone, use_lora=True, dtype=args.dtype,
                       gradient_checkpointing=False).to(device)
    ok = check_bidirectional(model, device=device)
    n = sum(p.numel() for p in model.trainable_parameters())
    print(f"Trainable parameters (projection + LoRA + head): {n:,}")
    print("PASS: flow #1 changes when flow #64 changes." if ok else
          "FAIL: causal mask still active. Check your transformers version.")


if __name__ == "__main__":
    main()
