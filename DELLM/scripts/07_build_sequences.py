"""Step 7: order generation -> sequences of 64 flows.

train : a mix of 'full' sequences (all training flows) and 'filtered' sequences
        (only flows the edge would upload), share set by cloud.full_vs_filtered.
val/test and test-only : only the flows the edge uploads (what the cloud really sees).

Saves artifacts/sequences/<key>.npz with seq (S, 64) positions and copy (S, 64) mask.

Usage:  python scripts/07_build_sequences.py
"""
import argparse

import numpy as np

import _bootstrap  # noqa: F401
from dellm.cloud.order import order_sequences
from dellm.config import load_config
from dellm.pipeline import artifacts, load_cluster, load_df, processed_path


def seqs_for(part, mask, cc):
    sub = part[mask].reset_index(drop=True)
    pos = np.flatnonzero(mask)
    seq, cp = order_sequences(sub, L=cc["seq_len"], min_group=cc["min_group"],
                              max_flows=cc["max_flows"], max_wait=cc["max_wait"])
    return pos[seq] if len(seq) else seq, cp  # positions into `part`


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cc = cfg["cloud"]
    rng = np.random.default_rng(cfg["seed"])

    for c in cfg["clusters"]:
        df = load_cluster(cfg, c)
        for split in ("train", "val", "test"):
            part = df[df["split"] == split].reset_index(drop=True)
            dec = load_df(processed_path(cfg, f"decisions_{c}_{split}"))
            up = dec["upload"].to_numpy()
            s_f, c_f = seqs_for(part, up, cc)
            if split == "train":
                s_a, c_a = seqs_for(part, np.ones(len(part), bool), cc)
                share = cc["full_vs_filtered"]
                n_full = int(round(len(s_f) * share / max(1e-9, 1 - share))) if len(s_f) else len(s_a)
                if len(s_a) > n_full:
                    keep = rng.choice(len(s_a), size=n_full, replace=False)
                    s_a, c_a = s_a[keep], c_a[keep]
                seq = np.concatenate([s_a, s_f]) if len(s_f) else s_a
                cp = np.concatenate([c_a, c_f]) if len(s_f) else c_a
            else:
                seq, cp = s_f, c_f
            np.savez_compressed(artifacts(cfg, "sequences", f"{c}_{split}.npz"), seq=seq, copy=cp)
            print(f"{c} {split}: {len(seq)} sequences")

    for name in (cfg.get("test_only") or {}):
        try:
            part = load_df(processed_path(cfg, f"eval_test_only_{name}"))
            dec = load_df(processed_path(cfg, f"decisions_test_only_{name}"))
        except FileNotFoundError:
            continue
        seq, cp = seqs_for(part, dec["upload"].to_numpy(), cc)
        np.savez_compressed(artifacts(cfg, "sequences", f"test_only_{name}.npz"), seq=seq, copy=cp)
        print(f"{name}: {len(seq)} sequences")


if __name__ == "__main__":
    main()
