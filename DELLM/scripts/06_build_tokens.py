"""Step 6: all-expert tokens (768 numbers per flow) for every split, saved as .npy.

The edge models are frozen, so tokens are computed once and reused in training.

Usage:  python scripts/06_build_tokens.py
"""
import argparse

import numpy as np

import _bootstrap  # noqa: F401
from dellm.cloud.tokens import build_tokens
from dellm.config import load_config
from dellm.pipeline import artifacts, load_cluster, load_df, load_edge, load_features, processed_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    feats = load_features(cfg)
    clusters = list(cfg["clusters"])
    package = {c: load_edge(cfg, c) for c in clusters}

    jobs = []
    for c in clusters:
        df = load_cluster(cfg, c)
        for split in ("train", "val", "test"):
            jobs.append((f"{c}_{split}", df[df["split"] == split]))
    for name in (cfg.get("test_only") or {}):
        try:
            jobs.append((f"test_only_{name}", load_df(processed_path(cfg, f"eval_test_only_{name}"))))
        except FileNotFoundError:
            pass

    for key, part in jobs:
        X = part[feats].to_numpy(np.float32)
        tok = build_tokens(X, package, cluster_order=clusters)
        np.save(artifacts(cfg, "tokens", f"{key}.npy"), tok)
        print(f"{key}: tokens {tok.shape}")


if __name__ == "__main__":
    main()
