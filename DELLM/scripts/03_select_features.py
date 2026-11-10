"""Step 3: pick the 12 shared features (Boruta per cluster + consensus).

Usage:  python scripts/03_select_features.py
"""
import argparse
import json

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from dellm.config import load_config
from dellm.data.features import boruta_select, consensus_features, drop_constant_and_correlated
from dellm.data.load import feature_columns
from dellm.pipeline import artifacts, load_cluster


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    fc = cfg["features"]

    train = {}
    for c in cfg["clusters"]:
        df = load_cluster(cfg, c)
        tr = df[df["split"] == "train"]
        n = min(fc["boruta_sample"], len(tr))
        # stratified sample so rare classes are present
        frac = n / len(tr)
        rng = np.random.default_rng(cfg["seed"])
        idx = []
        for _, g in tr.groupby("y").groups.items():
            g = np.asarray(g)
            k = max(1, int(round(len(g) * frac)))
            idx.append(rng.choice(g, size=min(k, len(g)), replace=False))
        train[c] = tr.loc[np.concatenate(idx)]

    common = set.intersection(*[set(feature_columns(d)) for d in train.values()])
    common = sorted(common)
    print(f"{len(common)} numeric features shared by all clusters")
    pooled = pd.concat([d[common] for d in train.values()], ignore_index=True)
    cand = drop_constant_and_correlated(pooled, common, fc["corr_threshold"])
    print(f"{len(cand)} after removing constant / highly correlated features")

    per_cluster = {}
    for c, d in train.items():
        acc, imp = boruta_select(d[cand].to_numpy(), d["y"].to_numpy(),
                                 max_iter=fc["boruta_max_iter"], seed=cfg["seed"])
        per_cluster[c] = pd.DataFrame({"accepted": acc, "importance": imp}, index=cand)
        print(f"{c}: {int(acc.sum())} accepted")

    chosen, summary = consensus_features(per_cluster, fc["n_features"], fc["min_clusters"])
    print("Selected features:", chosen)
    summary.to_csv(artifacts(cfg, "feature_votes.csv"))
    with open(artifacts(cfg, "features.json"), "w") as f:
        json.dump({"features": chosen, "candidates": cand}, f, indent=2)


if __name__ == "__main__":
    main()
