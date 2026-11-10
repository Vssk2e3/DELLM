"""Helpers shared by the numbered scripts: paths, saving/loading edge packages, etc."""
import json
from pathlib import Path

import numpy as np

from . import CLASS_TO_ID
from .config import ensure_dir, load_df


def processed_path(cfg, name):
    return Path(cfg["paths"]["processed"]) / name


def artifacts(cfg, *parts):
    p = Path(cfg["paths"]["artifacts"]).joinpath(*parts)
    ensure_dir(p.parent if p.suffix else p)
    return p


def load_cluster(cfg, cluster):
    return load_df(processed_path(cfg, f"{cluster}"))


def load_test_only(cfg, name):
    return load_df(processed_path(cfg, f"test_only_{name}"))


def load_features(cfg):
    with open(artifacts(cfg, "features.json")) as f:
        return json.load(f)["features"]


def global_labels(df):
    return df["y"].map(CLASS_TO_ID).to_numpy().astype(np.int64)


def save_edge(cfg, cluster, models, meta):
    import torch
    d = artifacts(cfg, "edge", cluster)
    for m, model in enumerate(models):
        torch.save(model.state_dict(), d / f"mlp_{m}.pt")
    with open(d / "edge.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_edge(cfg, cluster):
    import torch
    from .edge.mlp import EdgeMLP
    d = artifacts(cfg, "edge", cluster)
    with open(d / "edge.json") as f:
        meta = json.load(f)
    models = []
    for m in range(meta["n_models"]):
        model = EdgeMLP(len(meta["mean"]), len(meta["classes"]), meta["hidden"])
        model.load_state_dict(torch.load(d / f"mlp_{m}.pt", map_location="cpu"))
        models.append(model.eval())
    meta["mean"] = np.asarray(meta["mean"], dtype=np.float32)
    meta["std"] = np.asarray(meta["std"], dtype=np.float32)
    meta["models"] = models
    return meta


def local_to_global(meta):
    """Array mapping cluster-local class index -> global class id."""
    return np.array([CLASS_TO_ID[c] for c in meta["classes"]], dtype=np.int64)
