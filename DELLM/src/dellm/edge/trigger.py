"""Group trigger: catches distributed/low-rate attacks whose single flows look normal.

Counts over tumbling windows of `window` seconds:
  (1) flows per destination, (2) distinct sources per destination,
  (3) distinct destinations per destination /24 subnet,
  (4) spike: flows in the window / mean of the previous `history` windows.
Thresholds (1)-(3) = the q-th percentile of the same counts on benign training traffic.
When a rule fires, the matching flows of the window and of the previous window
(the cache) are uploaded, each flow once.
"""
import numpy as np
import pandas as pd


def _subnet24(ip):
    s = str(ip)
    parts = s.split(".")
    return ".".join(parts[:3]) if len(parts) == 4 else s


def _windowed(df, window):
    t = df["timestamp"].astype(float).to_numpy()
    w = np.floor((t - np.nanmin(t)) / window).astype(np.int64)
    out = pd.DataFrame({"w": w, "src": df["src_ip"].astype(str).to_numpy(),
                        "dst": df["dst_ip"].astype(str).to_numpy()}, index=df.index)
    out["net"] = out["dst"].map(_subnet24)
    return out


def _counts(win):
    per_dst = win.groupby(["w", "dst"]).agg(n_flows=("src", "size"), n_src=("src", "nunique"))
    per_net = win.groupby(["w", "net"]).agg(n_dst=("dst", "nunique"))
    per_win = win.groupby("w").size()
    return per_dst, per_net, per_win


def has_required_columns(df):
    return all(c in df.columns for c in ("timestamp", "src_ip", "dst_ip"))


def fit_trigger(benign_df, window=2.0, q=99.0, spike_ratio=3.0, history=30):
    """Fit thresholds on benign training flows. Returns a dict."""
    win = _windowed(benign_df, window)
    per_dst, per_net, _ = _counts(win)
    return {
        "window": float(window), "q": float(q), "spike_ratio": float(spike_ratio),
        "history": int(history),
        "theta_flows": float(np.percentile(per_dst["n_flows"], q)),
        "theta_src": float(np.percentile(per_dst["n_src"], q)),
        "theta_dst_per_net": float(np.percentile(per_net["n_dst"], q)),
    }


def apply_trigger(df, params, spike_top_k=5):
    """Return a boolean Series: True for flows uploaded because a trigger fired."""
    win = _windowed(df, params["window"])
    per_dst, per_net, per_win = _counts(win)

    hot_dst = set(per_dst.index[(per_dst["n_flows"] > params["theta_flows"]) |
                                (per_dst["n_src"] > params["theta_src"])])
    hot_net = set(per_net.index[per_net["n_dst"] > params["theta_dst_per_net"]])

    full = per_win.reindex(range(int(per_win.index.max()) + 1), fill_value=0)
    past = full.shift(1).rolling(params["history"], min_periods=5).mean()
    spike_w = set(full.index[(past > 0) & (full > params["spike_ratio"] * past)])
    for w in spike_w:  # on a spike, flag the busiest destinations of that window
        if w in per_dst.index.get_level_values(0):
            top = per_dst.loc[w]["n_flows"].nlargest(spike_top_k).index
            hot_dst.update((w, d) for d in top)

    # current window + cache: a target that fired stays "hot" for the next window too
    dst_keys = hot_dst | {(w + 1, d) for (w, d) in hot_dst}
    net_keys = hot_net | {(w + 1, n) for (w, n) in hot_net}
    key_dst = list(zip(win["w"], win["dst"]))
    key_net = list(zip(win["w"], win["net"]))
    fired = [(a in dst_keys) or (b in net_keys) for a, b in zip(key_dst, key_net)]
    return pd.Series(fired, index=df.index, name="trigger")
