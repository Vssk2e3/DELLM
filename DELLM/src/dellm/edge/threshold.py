"""Choose the edge threshold tau from a target error and an upload budget."""
import numpy as np
import pandas as pd


def risk_coverage(h_norm, correct, n_points=200):
    """Table of (tau, upload_rate, edge_error) over candidate thresholds."""
    h = np.asarray(h_norm, dtype=float)
    ok = np.asarray(correct, dtype=bool)
    taus = np.unique(np.quantile(h, np.linspace(0, 1, n_points)))
    rows = []
    for t in taus:
        keep = h <= t
        upload = 1.0 - keep.mean()
        err = 1.0 - ok[keep].mean() if keep.any() else 0.0
        rows.append((float(t), float(upload), float(err)))
    return pd.DataFrame(rows, columns=["tau", "upload_rate", "edge_error"])


def select_tau(h_norm, correct, eps=0.01, budget=0.10, n_points=200):
    """Largest tau with edge error <= eps; report whether the upload budget is met.

    Larger tau keeps more flows at the edge (fewer uploads, more errors). If even the
    smallest tau breaks eps, the smallest tau is returned (accuracy is kept, the
    budget is relaxed). Returns (tau, info dict, risk-coverage table).
    """
    table = risk_coverage(h_norm, correct, n_points)
    ok = table[table["edge_error"] <= eps]
    if len(ok):
        row = ok.iloc[-1]
    else:
        row = table.iloc[0]
    info = {
        "tau": float(row["tau"]),
        "upload_rate": float(row["upload_rate"]),
        "edge_error": float(row["edge_error"]),
        "eps": eps, "budget": budget,
        "eps_met": bool(row["edge_error"] <= eps),
        "budget_met": bool(row["upload_rate"] <= budget),
    }
    return info["tau"], info, table
