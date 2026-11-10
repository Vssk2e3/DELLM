"""Cloud queue batching and order generation (NumPy/pandas only).

1. Batching: a batch closes after `max_wait` seconds or `max_flows` flows.
2. Grouping: within a batch, flows are grouped by (cluster, destination IP); groups
   smaller than `min_group` are merged per cluster so sequences are not mostly copies.
3. Blocks: flows between the same two addresses (either direction) form a block;
   blocks are sorted by protocol, then by total size (largest first).
4. Grid: the sorted flows fill a grid of L rows row by row; each column is read as one
   sequence, so every sequence mixes flows from all related blocks.
5. Reverse filling: empty cells get copies of the group's flows (taken from the end);
   copies are marked and get no label and no loss.
"""
import numpy as np
import pandas as pd


def make_batches(df, max_flows=1024, max_wait=1.0, time_col="timestamp"):
    """Yield arrays of row positions (into df) for each queue batch, in time order."""
    n = len(df)
    if time_col in df.columns and df[time_col].notna().all():
        order = np.argsort(df[time_col].to_numpy(dtype=float), kind="stable")
        t = df[time_col].to_numpy(dtype=float)[order]
        i = 0
        while i < n:
            j_time = np.searchsorted(t, t[i] + max_wait, side="right")
            j = min(i + max_flows, j_time)
            yield order[i:j]
            i = j
    else:
        for i in range(0, n, max_flows):
            yield np.arange(i, min(i + max_flows, n))


def _size(df):
    if {"totlen_fwd_pkts", "totlen_bwd_pkts"} <= set(df.columns):
        return (df["totlen_fwd_pkts"].astype(float) + df["totlen_bwd_pkts"].astype(float)).to_numpy()
    for c in ("flow_byts_s", "tot_fwd_pkts"):
        if c in df.columns:
            return df[c].astype(float).to_numpy()
    return np.zeros(len(df))


def _order_group(g):
    """Return positions of group g sorted into blocks (protocol, then size)."""
    if {"src_ip", "dst_ip"} <= set(g.columns):
        a = g["src_ip"].astype(str).to_numpy()
        b = g["dst_ip"].astype(str).to_numpy()
        pair = np.where(a < b, a + "|" + b, b + "|" + a)
    else:
        pair = np.full(len(g), "all")
    proto = g["protocol"].to_numpy() if "protocol" in g.columns else np.zeros(len(g))
    tmp = pd.DataFrame({"pos": g["_pos"].to_numpy(), "pair": pair, "proto": proto,
                        "size": g["_size"].to_numpy(),
                        "t": g["timestamp"].to_numpy() if "timestamp" in g.columns else np.arange(len(g))})
    blk = tmp.groupby("pair").agg(proto=("proto", lambda s: s.mode().iloc[0]), size=("size", "sum"))
    blk = blk.sort_values(["proto", "size"], ascending=[True, False])
    rank = {p: i for i, p in enumerate(blk.index)}
    tmp["blk"] = tmp["pair"].map(rank)
    tmp = tmp.sort_values(["blk", "t"], kind="stable")
    return tmp["pos"].to_numpy()


def grid_sequences(ordered, L=64):
    """Row-major fill of an L x C grid, read by column, with reverse-filled copies."""
    n = len(ordered)
    C = int(np.ceil(n / L))
    cells = L * C
    fill = np.empty(cells, dtype=np.int64)
    fill[:n] = ordered
    is_copy = np.zeros(cells, dtype=bool)
    if cells > n:
        rev = ordered[::-1]
        k = cells - n
        fill[n:] = np.resize(rev, k)
        is_copy[n:] = True
    grid = fill.reshape(L, C)
    copy_grid = is_copy.reshape(L, C)
    return grid.T.copy(), copy_grid.T.copy()  # (C, L)


def order_sequences(df, L=64, min_group=16, group_col="dst_ip", max_flows=1024, max_wait=1.0):
    """Build sequences for all flows in df (positions are 0..len(df)-1).

    Returns seq (S, L) int64 positions into df and copy (S, L) bool.
    """
    df = df.reset_index(drop=True).copy()
    df["_pos"] = np.arange(len(df))
    df["_size"] = _size(df)
    if "cluster" not in df.columns:
        df["cluster"] = "all"
    seqs, copies = [], []
    for batch in make_batches(df, max_flows=max_flows, max_wait=max_wait):
        b = df.iloc[batch]
        for _, cdf in b.groupby("cluster", sort=False):
            if group_col in cdf.columns:
                sizes = cdf.groupby(group_col)[group_col].transform("size")
                big = cdf[sizes >= min_group]
                small = cdf[sizes < min_group]
                groups = [g for _, g in big.groupby(group_col, sort=False)]
                if len(small):
                    groups.append(small)
            else:
                groups = [cdf]
            for g in groups:
                s, c = grid_sequences(_order_group(g), L)
                seqs.append(s)
                copies.append(c)
    if not seqs:
        return np.zeros((0, L), dtype=np.int64), np.zeros((0, L), dtype=bool)
    return np.concatenate(seqs), np.concatenate(copies)
