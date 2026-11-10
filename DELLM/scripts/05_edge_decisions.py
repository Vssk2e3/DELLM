"""Step 5: run the edge on every split and record which flows go to the cloud.

For test-only datasets, the scaler and trigger thresholds are re-fitted on the first
10% of their BENIGN flows (by time); those flows are then excluded from evaluation.
No attack label of a test-only dataset is used.

Usage:  python scripts/05_edge_decisions.py
"""
import argparse

import numpy as np

import _bootstrap  # noqa: F401
from dellm.config import load_config, save_df
from dellm.edge.decide import edge_decisions
from dellm.edge.mlp import ensemble_logits
from dellm.edge.trigger import fit_trigger, has_required_columns
from dellm.pipeline import (artifacts, load_cluster, load_edge, load_features, load_test_only,
                            local_to_global, processed_path)


def run(df, meta, feats, cfg, seed):
    X = (df[feats].to_numpy(np.float32) - meta["mean"]) / meta["std"]
    logits = ensemble_logits(meta["models"], X)
    dec = edge_decisions(df, logits, meta["T"], meta["tau"], meta["trigger"],
                         audit_rate=cfg["edge"]["audit_rate"], seed=seed)
    dec["edge_pred"] = local_to_global(meta)[dec["edge_pred_local"].to_numpy()]
    return dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    feats = load_features(cfg)

    for c in cfg["clusters"]:
        meta = load_edge(cfg, c)
        df = load_cluster(cfg, c)
        for split in ("train", "val", "test"):
            part = df[df["split"] == split]
            dec = run(part, meta, feats, cfg, cfg["seed"])
            save_df(dec.reset_index(drop=True), processed_path(cfg, f"decisions_{c}_{split}"))
            print(f"{c} {split}: upload rate {dec['upload'].mean():.3f}  "
                  f"{dec['reason'].value_counts().to_dict()}")

    for name, spec in (cfg.get("test_only") or {}).items():
        try:
            df = load_test_only(cfg, name)
        except FileNotFoundError:
            continue
        meta = load_edge(cfg, spec["cluster"])
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp", kind="stable")
        benign = df[df["y"] == "Benign"]
        calib = benign.iloc[: max(1, len(benign) // 10)]
        if len(calib) > 10:
            meta["mean"] = calib[feats].mean().to_numpy(np.float32)
            meta["std"] = calib[feats].std(ddof=0).replace(0, 1).to_numpy(np.float32)
            if meta["trigger"] is not None and has_required_columns(calib):
                t = cfg["edge"]["trigger"]
                meta["trigger"] = fit_trigger(calib, t["window"], t["percentile"],
                                              t["spike_ratio"], t["history"])
        ev = df.drop(index=calib.index)
        dec = run(ev, meta, feats, cfg, cfg["seed"])
        save_df(ev.reset_index(drop=True), processed_path(cfg, f"eval_test_only_{name}"))
        save_df(dec.reset_index(drop=True), processed_path(cfg, f"decisions_test_only_{name}"))
        print(f"{name}: upload rate {dec['upload'].mean():.3f}")


if __name__ == "__main__":
    main()
