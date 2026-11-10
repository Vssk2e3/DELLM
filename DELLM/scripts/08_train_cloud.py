"""Step 8: train the cloud model (projection + LoRA + linear head).

Runs the mask test first and stops if the causal mask is still active.
Saves the best checkpoint (by validation macro-F1) to artifacts/cloud/.

Usage:
    python scripts/08_train_cloud.py                      # backbone from the config
    python scripts/08_train_cloud.py --backbone meta-llama/Llama-3.2-1B   # quick development run
    python scripts/08_train_cloud.py --no-llm             # ablation without the LLM
"""
import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import _bootstrap  # noqa: F401
from dellm import NUM_CLASSES
from dellm.cloud.model import DELLMCloud, check_bidirectional, save_cloud
from dellm.config import load_config
from dellm.metrics import macro_f1
from dellm.pipeline import artifacts, global_labels, load_cluster


class SeqData(Dataset):
    """Sequences from several (tokens, labels, seq, copy) sources."""

    def __init__(self, sources):
        self.items = []
        self.sources = sources
        for k, s in enumerate(sources):
            self.items += [(k, i) for i in range(len(s["seq"]))]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, j):
        k, i = self.items[j]
        s = self.sources[k]
        pos = s["seq"][i]
        x = torch.from_numpy(s["tokens"][pos].astype(np.float32))
        y = torch.from_numpy(s["labels"][pos].copy())
        y[torch.from_numpy(s["copy"][i])] = -100          # copies: no label, no loss
        return x, y


def load_sources(cfg, split):
    out = []
    for c in cfg["clusters"]:
        df = load_cluster(cfg, c)
        part = df[df["split"] == split].reset_index(drop=True)
        z = np.load(artifacts(cfg, "sequences", f"{c}_{split}.npz"))
        if len(z["seq"]) == 0:
            continue
        out.append({"tokens": np.load(artifacts(cfg, "tokens", f"{c}_{split}.npy"), mmap_mode="r"),
                    "labels": global_labels(part), "seq": z["seq"], "copy": z["copy"]})
    return out


def class_weights(sources):
    ys = np.concatenate([s["labels"][s["seq"][~s["copy"]]] for s in sources])
    counts = np.bincount(ys, minlength=NUM_CLASSES).astype(float)
    w = np.where(counts > 0, 1.0 / np.sqrt(np.maximum(counts, 1)), 0.0)
    w = w / w[w > 0].mean()
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    yt, yp = [], []
    for x, y in loader:
        logits = model(x.to(device))
        m = y != -100
        yt.append(y[m].numpy())
        yp.append(logits.argmax(-1).cpu()[m].numpy())
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    return macro_f1(yt, yp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--no-lora", action="store_true", help="ablation: frozen LLM")
    ap.add_argument("--out", default="cloud")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cc = cfg["cloud"]
    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_src, val_src = load_sources(cfg, "train"), load_sources(cfg, "val")
    tr = DataLoader(SeqData(train_src), batch_size=cc["batch_size"], shuffle=True, num_workers=2)
    va = DataLoader(SeqData(val_src), batch_size=cc["batch_size"] * 2, shuffle=False, num_workers=2)
    token_dim = train_src[0]["tokens"].shape[1]

    model = DELLMCloud(backbone=args.backbone or cc["backbone"], token_dim=token_dim,
                       n_classes=NUM_CLASSES, use_lora=cc["use_lora"] and not args.no_lora,
                       lora_r=cc["lora_r"], lora_alpha=cc["lora_alpha"], lora_dropout=cc["lora_dropout"],
                       dtype=cc["dtype"], load_in_4bit=cc["load_in_4bit"],
                       no_llm=args.no_llm or cc["no_llm"],
                       gradient_checkpointing=cc["gradient_checkpointing"])
    if not cc["load_in_4bit"]:
        model.to(device)
    else:
        model.proj.to(device)
        model.head.to(device)

    if not check_bidirectional(model, token_dim, cc["seq_len"], device):
        raise SystemExit("MASK TEST FAILED: flow #1 does not see flow #64. "
                         "Your transformers version still applies a causal mask.")
    print("Mask test passed: attention is bidirectional.")

    params = model.trainable_parameters()
    print(f"Trainable parameters: {sum(p.numel() for p in params):,}")
    opt = torch.optim.AdamW(params, lr=cc["lr"], weight_decay=cc["weight_decay"])
    steps = math.ceil(len(tr) / cc["grad_accum"]) * cc["epochs"]
    warm = max(1, int(cc["warmup_frac"] * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(train_src).to(device), ignore_index=-100)

    best, bad, log = -1.0, 0, []
    for epoch in range(cc["epochs"]):
        model.train()
        t0, total = time.time(), 0.0
        for step, (x, y) in enumerate(tr):
            logits = model(x.to(device))
            loss = loss_fn(logits.reshape(-1, NUM_CLASSES), y.to(device).reshape(-1)) / cc["grad_accum"]
            loss.backward()
            total += loss.item()
            if (step + 1) % cc["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
        f1 = evaluate(model, va, device)
        log.append({"epoch": epoch, "train_loss": total / max(1, len(tr)), "val_macro_f1": f1,
                    "seconds": time.time() - t0})
        print(log[-1])
        if f1 > best + 1e-4:
            best, bad = f1, 0
            save_cloud(model, artifacts(cfg, args.out))
        else:
            bad += 1
            if bad >= cc["patience"]:
                print("Early stopping.")
                break
    with open(artifacts(cfg, args.out, "train_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(f"Best validation macro-F1: {best:.4f}")


if __name__ == "__main__":
    main()
