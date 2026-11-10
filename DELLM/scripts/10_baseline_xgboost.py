"""Baseline: one gradient-boosted tree model on the 12 features, all clusters pooled.

Uses xgboost if installed, otherwise scikit-learn's HistGradientBoostingClassifier.

Usage:  python scripts/10_baseline_xgboost.py
"""
import argparse
import json

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from dellm.config import ensure_dir, load_config
from dellm.metrics import report
from dellm.pipeline import global_labels, load_cluster, load_df, load_features, processed_path


def make_model(seed):
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.1, tree_method="hist",
                             random_state=seed, n_jobs=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=400, class_weight="balanced", random_state=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    feats = load_features(cfg)
    dfs = {c: load_cluster(cfg, c) for c in cfg["clusters"]}
    tr = pd.concat([d[d["split"] == "train"] for d in dfs.values()], ignore_index=True)
    ytr = global_labels(tr)
    classes = np.unique(ytr)
    remap = {c: i for i, c in enumerate(classes)}
    model = make_model(cfg["seed"])
    model.fit(tr[feats].to_numpy(np.float32), np.vectorize(remap.get)(ytr))

    results = []
    tests = [(f"{c}_test", d[d["split"] == "test"]) for c, d in dfs.items()]
    for name in (cfg.get("test_only") or {}):
        try:
            tests.append((f"test_only_{name}", load_df(processed_path(cfg, f"eval_test_only_{name}"))))
        except FileNotFoundError:
            pass
    for name, te in tests:
        pred = classes[model.predict(te[feats].to_numpy(np.float32)).astype(int)]
        r = report(global_labels(te), pred)
        results.append({"dataset": name, **r})
        print(f"{name}: macro-F1 {r['macro_f1']:.4f}")
    out = ensure_dir(cfg["paths"]["results"]) / "results_baseline_xgboost.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
