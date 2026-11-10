"""Step 2: load every dataset, map labels to the 8 classes, clean, split by time, save.

Prints a label-mapping report for each dataset: CHECK IT before continuing.

Usage:  python scripts/02_prepare_data.py
"""
import argparse
import json

import _bootstrap  # noqa: F401
from dellm.config import ensure_dir, load_config, save_df
from dellm.data.labels import mapping_report
from dellm.data.load import clean_numeric, feature_columns, load_dataset, split_by_time_per_label
from dellm.pipeline import processed_path


def prepare(spec, cfg, name, cluster, test_only=False):
    df = load_dataset(spec["files"], label_column=spec.get("label_column"),
                      label_from_folder=spec.get("label_from_folder", False),
                      extra_rules=spec.get("extra_rules"),
                      max_rows_per_file=cfg.get("max_rows_per_file"),
                      cluster=cluster, dataset=name)
    print(f"\n=== {name} ({cluster}) : {len(df):,} flows kept")
    print(json.dumps(mapping_report(df["raw_label"], spec.get("extra_rules")), indent=1))
    print(df["y"].value_counts().to_string())
    cols = feature_columns(df)
    df = clean_numeric(df, cols)
    if test_only:
        df["split"] = "test"
    else:
        df = split_by_time_per_label(df, tuple(cfg["split"]), seed=cfg["seed"])
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ensure_dir(cfg["paths"]["processed"])
    for cluster, spec in cfg["clusters"].items():
        df = prepare(spec, cfg, spec["dataset"], cluster)
        save_df(df, processed_path(cfg, cluster))
    for name, spec in (cfg.get("test_only") or {}).items():
        try:
            df = prepare(spec, cfg, name, spec["cluster"], test_only=True)
        except FileNotFoundError as e:
            print(f"skip {name}: {e}")
            continue
        save_df(df, processed_path(cfg, f"test_only_{name}"))


if __name__ == "__main__":
    main()
