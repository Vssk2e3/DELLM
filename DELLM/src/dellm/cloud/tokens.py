"""All-expert flow tokens: every flow through all 12 frozen edge MLPs -> 768 numbers."""
import numpy as np

from ..edge.mlp import ensemble_hidden


def build_tokens(X_raw, edge_package, cluster_order=None, dtype=np.float16):
    """X_raw: (N, 12) unscaled features in the selected-feature order.

    edge_package: {cluster: {"mean": (12,), "std": (12,), "models": [3 x EdgeMLP]}}
    Each cluster's own scaler is applied before its models. Returns (N, 64*3*n_clusters).
    """
    cluster_order = cluster_order or sorted(edge_package)
    parts = []
    for c in cluster_order:
        pkg = edge_package[c]
        Xs = (X_raw - pkg["mean"]) / pkg["std"]
        parts.append(ensemble_hidden(pkg["models"], Xs.astype(np.float32)))
    return np.concatenate(parts, axis=1).astype(dtype)
