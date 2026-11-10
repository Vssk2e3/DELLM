"""Victim detection and source grouping (pandas only).

Per window: a destination is a victim if its number of attack flows exceeds a threshold
(by default the trigger's benign 99th percentile of flows per destination).
Sources of attacks on a victim:
  DrDoS family                                   -> abused reflectors
  > 100 sources and > 90% send only 1-2 flows    -> likely spoofed
  otherwise                                      -> compromised devices
"""
import numpy as np
import pandas as pd

from . import CLASSES


def find_incidents(df, pred, theta_victim, window=2.0, min_sources_spoof=100, spoof_frac=0.9):
    """df: flows with timestamp, src_ip, dst_ip. pred: global class ids (len(df)).

    Returns (incidents DataFrame, sources DataFrame).
    """
    d = pd.DataFrame({"t": df["timestamp"].astype(float).to_numpy(),
                      "src": df["src_ip"].astype(str).to_numpy(),
                      "dst": df["dst_ip"].astype(str).to_numpy(),
                      "y": np.asarray(pred)})
    d = d[d["y"] != CLASSES.index("Benign")]
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()
    d["w"] = np.floor((d["t"] - d["t"].min()) / window).astype(int)
    per = d.groupby(["w", "dst"])
    counts = per.size()
    victims = counts[counts > theta_victim].index
    incidents, sources = [], []
    for (w, dst) in victims:
        g = d[(d["w"] == w) & (d["dst"] == dst)]
        fam = CLASSES[int(g["y"].mode().iloc[0])]
        per_src = g.groupby("src").size()
        if fam == "DrDoS":
            group = "abused_reflectors"
        elif len(per_src) > min_sources_spoof and (per_src <= 2).mean() > spoof_frac:
            group = "likely_spoofed"
        else:
            group = "compromised_devices"
        incidents.append({"window": w, "victim": dst, "family": fam, "n_flows": len(g),
                          "n_sources": len(per_src), "source_group": group,
                          "start_time": g["t"].min()})
        for s, n in per_src.items():
            sources.append({"window": w, "victim": dst, "source": s, "n_flows": int(n),
                            "source_group": group})
    return pd.DataFrame(incidents), pd.DataFrame(sources)


def compromised_device_scores(sources, true_attackers):
    """Precision / recall of IPs labelled 'compromised_devices' vs a known attacker list."""
    if sources is None or len(sources) == 0:
        return {"precision": 0.0, "recall": 0.0, "n_predicted": 0}
    pred = set(sources.loc[sources["source_group"] == "compromised_devices", "source"])
    true = set(map(str, true_attackers))
    tp = len(pred & true)
    return {"precision": tp / len(pred) if pred else 0.0,
            "recall": tp / len(true) if true else 0.0,
            "n_predicted": len(pred)}
