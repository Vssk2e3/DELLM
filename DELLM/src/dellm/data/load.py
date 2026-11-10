"""Load flow CSVs, normalise column names, map labels and split by time.

All datasets must be in CICFlowMeter-style flow format (see scripts/01_extract_flows.py)
so the four clusters share one feature space.
"""
import glob
import os
import re

import numpy as np
import pandas as pd

from .labels import map_label

# Columns that identify hosts/flows/time: kept as metadata, never used as features.
META_ALIASES = {
    "src_ip": ["src_ip", "source_ip", "srcip", "ip_src", "saddr"],
    "dst_ip": ["dst_ip", "destination_ip", "dstip", "ip_dst", "daddr"],
    "src_port": ["src_port", "source_port", "sport", "srcport"],
    "dst_port": ["dst_port", "destination_port", "dport", "dstport"],
    "protocol": ["protocol", "proto", "protocol_type"],
    "timestamp": ["timestamp", "time", "ts", "flow_start_time", "start_time"],
    "flow_id": ["flow_id", "flowid", "unnamed:_0", "unnamed_0"],
}

# Common spelling differences between CICFlowMeter versions -> one name.
FEATURE_ALIASES = {
    "total_fwd_packets": "tot_fwd_pkts", "total_backward_packets": "tot_bwd_pkts",
    "total_bwd_packets": "tot_bwd_pkts",
    "total_length_of_fwd_packets": "totlen_fwd_pkts", "fwd_packets_length_total": "totlen_fwd_pkts",
    "total_length_of_bwd_packets": "totlen_bwd_pkts", "bwd_packets_length_total": "totlen_bwd_pkts",
    "fwd_packet_length_max": "fwd_pkt_len_max", "fwd_packet_length_min": "fwd_pkt_len_min",
    "fwd_packet_length_mean": "fwd_pkt_len_mean", "fwd_packet_length_std": "fwd_pkt_len_std",
    "bwd_packet_length_max": "bwd_pkt_len_max", "bwd_packet_length_min": "bwd_pkt_len_min",
    "bwd_packet_length_mean": "bwd_pkt_len_mean", "bwd_packet_length_std": "bwd_pkt_len_std",
    "flow_bytes/s": "flow_byts_s", "flow_bytes_s": "flow_byts_s",
    "flow_packets/s": "flow_pkts_s", "flow_packets_s": "flow_pkts_s",
    "fwd_packets/s": "fwd_pkts_s", "bwd_packets/s": "bwd_pkts_s",
    "min_packet_length": "pkt_len_min", "max_packet_length": "pkt_len_max",
    "packet_length_min": "pkt_len_min", "packet_length_max": "pkt_len_max",
    "packet_length_mean": "pkt_len_mean", "packet_length_std": "pkt_len_std",
    "packet_length_variance": "pkt_len_var",
    "fin_flag_count": "fin_flag_cnt", "syn_flag_count": "syn_flag_cnt",
    "rst_flag_count": "rst_flag_cnt", "psh_flag_count": "psh_flag_cnt",
    "ack_flag_count": "ack_flag_cnt", "urg_flag_count": "urg_flag_cnt",
    "ece_flag_count": "ece_flag_cnt", "cwe_flag_count": "cwr_flag_count",
    "average_packet_size": "pkt_size_avg", "avg_packet_size": "pkt_size_avg",
    "init_win_bytes_forward": "init_fwd_win_byts", "init_win_bytes_backward": "init_bwd_win_byts",
    "fwd_init_win_bytes": "init_fwd_win_byts", "bwd_init_win_bytes": "init_bwd_win_byts",
    "flow_duration": "flow_duration",
}

# Testbed artefacts removed before feature selection (they identify the lab, not the attack).
ARTEFACT_FEATURES = {"init_fwd_win_byts", "init_bwd_win_byts", "simillarhttp", "similarhttp",
                     "inbound", "fwd_header_length.1"}

LABEL_CANDIDATES = ["label", "attack_label", "attack_type", "class", "category", "attack"]


def _norm(name):
    s = str(name).strip().lower()
    s = re.sub(r"\s+", "_", s)
    return s


def normalise_columns(df):
    df = df.rename(columns={c: _norm(c) for c in df.columns})
    rename = {}
    for canon, aliases in META_ALIASES.items():
        for a in aliases:
            if a in df.columns and canon not in df.columns and canon not in rename.values():
                rename[a] = canon
    for c in df.columns:
        if c in FEATURE_ALIASES and FEATURE_ALIASES[c] != c:
            rename.setdefault(c, FEATURE_ALIASES[c])
    df = df.rename(columns=rename)
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def find_label_column(df, preferred=None):
    if preferred and _norm(preferred) in df.columns:
        return _norm(preferred)
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    raise KeyError(f"No label column found. Columns: {list(df.columns)[:20]} ...")


def load_dataset(patterns, label_column=None, label_from_folder=False, extra_rules=None,
                 max_rows_per_file=None, cluster=None, dataset=None):
    """Read all CSVs matching `patterns` and return one DataFrame with:
    numeric features + meta columns + 'raw_label' + 'y' (class name) + 'cluster' + 'dataset'.
    """
    if isinstance(patterns, str):
        patterns = [patterns]
    files = sorted({f for p in patterns for f in glob.glob(p, recursive=True)})
    if not files:
        raise FileNotFoundError(f"No CSV files match {patterns}")
    parts = []
    for f in files:
        df = pd.read_csv(f, low_memory=False, nrows=max_rows_per_file)
        df = normalise_columns(df)
        if label_from_folder:
            df["raw_label"] = os.path.basename(os.path.dirname(f))
        else:
            df["raw_label"] = df[find_label_column(df, label_column)].astype(str)
        df["source_file"] = os.path.basename(f)
        parts.append(df)
    df = pd.concat(parts, ignore_index=True, sort=False)
    df["y"] = df["raw_label"].map(lambda s: map_label(s, extra_rules))
    df = df[df["y"].notna()].reset_index(drop=True)
    if cluster is not None:
        df["cluster"] = cluster
    if dataset is not None:
        df["dataset"] = dataset
    if "timestamp" in df.columns:
        df["timestamp"] = parse_time(df["timestamp"])
    return df


def parse_time(s):
    """Convert a timestamp column to float seconds."""
    if np.issubdtype(s.dtype, np.number):
        return s.astype(float)
    t = pd.to_datetime(s, errors="coerce", dayfirst=False)
    return (t - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)


def feature_columns(df):
    """Numeric columns that may be used as features (no meta, no labels, no artefacts)."""
    excluded = set(META_ALIASES) - {"protocol"} | {"raw_label", "y", "cluster", "dataset",
                                                     "source_file", "split", "label"}
    excluded |= ARTEFACT_FEATURES
    cols = []
    for c in df.columns:
        if c in excluded or c in LABEL_CANDIDATES:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def clean_numeric(df, cols):
    """Replace inf with NaN, then NaN with the column median."""
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)
    df[cols] = df[cols].fillna(df[cols].median(numeric_only=True)).fillna(0.0)
    return df


def split_by_time_per_label(df, fractions=(0.7, 0.1, 0.2), time_col="timestamp", seed=0):
    """Split each label chronologically into train/val/test.

    Splitting per label keeps every class in every split even when each attack was
    captured at a different time (as in CICIoT2023's per-attack files). If there is no
    timestamp, the original row order is used as time order.
    """
    rng = np.random.default_rng(seed)
    split = np.empty(len(df), dtype=object)
    for _, idx in df.groupby("y").groups.items():
        idx = np.asarray(idx)
        if time_col in df.columns and df[time_col].notna().any():
            order = idx[np.argsort(df.loc[idx, time_col].values, kind="stable")]
        else:
            order = idx
        n = len(order)
        a = int(round(n * fractions[0]))
        b = int(round(n * (fractions[0] + fractions[1])))
        split[order[:a]] = "train"
        split[order[a:b]] = "val"
        split[order[b:]] = "test"
    df = df.copy()
    df["split"] = split
    _ = rng  # kept for API stability
    return df
