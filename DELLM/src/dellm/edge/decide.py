"""Edge decision rule: trigger -> uncertain -> edge-final (+ random audit)."""
import numpy as np
import pandas as pd

from .calibration import normalised_entropy, softmax
from .trigger import apply_trigger, has_required_columns


def edge_decisions(df, logits, T, tau, trigger_params=None, audit_rate=0.01, seed=0):
    """Return a DataFrame (same index as df) with edge_pred, h_norm, reason, upload.

    reason is one of: 'trigger', 'uncertain', 'audit', 'edge'.
    """
    p = softmax(logits, T)
    h = normalised_entropy(p)
    out = pd.DataFrame({"edge_pred_local": p.argmax(1), "edge_conf": p.max(1), "h_norm": h},
                       index=df.index)
    if trigger_params is not None and has_required_columns(df):
        trig = apply_trigger(df, trigger_params).to_numpy()
    else:
        trig = np.zeros(len(df), dtype=bool)
    unc = h > tau
    rng = np.random.default_rng(seed)
    audit = (~trig) & (~unc) & (rng.random(len(df)) < audit_rate)
    reason = np.where(trig, "trigger", np.where(unc, "uncertain", np.where(audit, "audit", "edge")))
    out["reason"] = reason
    out["upload"] = reason != "edge"
    return out
