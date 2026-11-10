"""Step 4: train the edge of every cluster.

For each cluster: scaler (train mean/std) -> 3 MLPs on random thirds -> temperature T
on validation -> tau from risk-coverage (eps, budget) -> group-trigger thresholds on
benign training flows. Saves everything to artifacts/edge/<cluster>/.

Usage:  python scripts/04_train_edge.py
"""
import argparse
import json

import numpy as np

import _bootstrap  # noqa: F401
from dellm import CLASSES
from dellm.config import load_config
from dellm.edge.calibration import expected_calibration_error, fit_temperature, normalised_entropy, softmax
from dellm.edge.mlp import ensemble_logits, train_ensemble
from dellm.edge.threshold import select_tau
from dellm.edge.trigger import fit_trigger, has_required_columns
from dellm.pipeline import artifacts, load_cluster, load_features, save_edge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ec = cfg["edge"]
    feats = load_features(cfg)
    summary = {}

    for c in cfg["clusters"]:
        df = load_cluster(cfg, c)
        classes = [k for k in CLASSES if k in set(df.loc[df["split"] == "train", "y"])]
        cid = {k: i for i, k in enumerate(classes)}
        tr = df[(df["split"] == "train") & df["y"].isin(classes)]
        va = df[(df["split"] == "val") & df["y"].isin(classes)]
        mean = tr[feats].mean().to_numpy(np.float32)
        std = tr[feats].std(ddof=0).replace(0, 1).to_numpy(np.float32)
        Xtr = ((tr[feats].to_numpy(np.float32) - mean) / std)
        Xva = ((va[feats].to_numpy(np.float32) - mean) / std)
        ytr = tr["y"].map(cid).to_numpy()
        yva = va["y"].map(cid).to_numpy()

        print(f"\n=== {c}: classes {classes}, train {len(tr):,}, val {len(va):,}")
        models = train_ensemble(Xtr, ytr, Xva, yva, len(classes), n_models=ec["n_models"],
                                seed=cfg["seed"], epochs=ec["epochs"], batch_size=ec["batch_size"],
                                lr=ec["lr"], patience=ec["patience"], device=args.device)
        logits = ensemble_logits(models, Xva)
        T = fit_temperature(logits, yva)
        p = softmax(logits, T)
        h = normalised_entropy(p)
        correct = p.argmax(1) == yva
        tau, info, table = select_tau(h, correct, eps=ec["eps"], budget=ec["budget"])
        table.to_csv(artifacts(cfg, "edge", c, "risk_coverage.csv"), index=False)

        trig = None
        benign = tr[tr["y"] == "Benign"]
        if has_required_columns(tr) and len(benign) > 0:
            t = ec["trigger"]
            trig = fit_trigger(benign, window=t["window"], q=t["percentile"],
                               spike_ratio=t["spike_ratio"], history=t["history"])
        else:
            print(f"{c}: no timestamp/IP columns or no benign flows -> group trigger disabled")

        meta = {"classes": classes, "mean": mean.tolist(), "std": std.tolist(), "features": feats,
                "n_models": ec["n_models"], "hidden": ec["hidden"], "T": T, "tau": tau,
                "tau_info": info, "trigger": trig,
                "val_accuracy": float(correct.mean()),
                "val_ece_before": expected_calibration_error(softmax(logits, 1.0), yva),
                "val_ece_after": expected_calibration_error(p, yva)}
        save_edge(cfg, c, models, meta)
        summary[c] = {k: meta[k] for k in ("classes", "T", "tau", "tau_info", "val_accuracy",
                                           "val_ece_before", "val_ece_after")}
        print(json.dumps(summary[c], indent=1))

    with open(artifacts(cfg, "edge", "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
