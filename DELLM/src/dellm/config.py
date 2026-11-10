"""Load the YAML configuration used by every script."""
from pathlib import Path

import yaml


def load_config(path="configs/default.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg.get("paths", {}).get("root", "."))
    for key, val in cfg.get("paths", {}).items():
        if key != "root" and isinstance(val, str):
            cfg["paths"][key] = str(root / val)
    return cfg


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def save_df(df, path):
    """Save as Parquet (needs pyarrow); falls back to pickle."""
    path = Path(path)
    ensure_dir(path.parent)
    try:
        df.to_parquet(path.with_suffix(".parquet"), index=False)
    except ImportError:
        df.to_pickle(path.with_suffix(".pkl"))


def load_df(path):
    import pandas as pd
    path = Path(path)
    if path.with_suffix(".parquet").exists():
        return pd.read_parquet(path.with_suffix(".parquet"))
    return pd.read_pickle(path.with_suffix(".pkl"))
