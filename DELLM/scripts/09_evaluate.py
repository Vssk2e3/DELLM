"""Step 9: final labels (edge + cloud), metrics, victims/sources, timing.

Final label per flow:
  reason 'edge'               -> edge label
  reason 'uncertain'/'trigger'-> cloud label
  reason 'audit'              -> edge label (cloud label used only to measure disagreement)

Usage:
    python scripts/09_evaluate.py                 # test split of C1-C4 + test-only datasets
    python scripts/09_evaluate.py --split val
"""
import argparse
import json
import time

import numpy as np
import torch

import _bootstrap  # noqa: F401
from dellm import CLASSES, NUM_CLASSES
from dellm.attribution import compromised_device_scores, find_incidents
from dellm.cloud.model import load_cloud
from dellm.config import ensure_dir, load_config
from dellm.edge.trigger import has_required_columns
from dellm.metrics import report
from dellm.pipeline import artifacts, global_labels, load_cluster, load_df, load_edge, processed_path


@torch.no_grad()
def cloud_predict(model, tokens, seq, copy, device, batch=16):
    """Return per-flow cloud predictions (-1 where the flow was not uploaded) and timing."""
    pred = np.full(len(tokens), -1, dtype=np.int64)
    times = []
    model.eval()
    for i in range(0, len(seq), batch):
        s, c = seq[i:i + batch], copy[i:i + batch]
        x = torch.from_numpy(tokens[s].astype(np.float32)).to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        p = model(x).argmax(-1).cpu().numpy()
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / max(1, (~c).sum()))
        pred[s[~c]] = p[~c]
    return pred, (float(np.mean(times)) if times else 0.0)


def evaluate_part(name, part, dec, tokens, seqz, model, device, cfg, cluster):
    y = global_labels(part)
    cloud, t_flow = cloud_predict(model, tokens, seqz["seq"], seqz["copy"], device)
    reason = dec["reason"].to_numpy()
    edge = dec["edge_pred"].to_numpy()
    final = edge.copy()
    use_cloud = np.isin(reason, ["uncertain", "trigger"]) & (cloud >= 0)
    final[use_cloud] = cloud[use_cloud]
    audit = (reason == "audit") & (cloud >= 0)
    res = {
        "dataset": name, "n_flows": int(len(part)),
        "upload_rate": float(dec["upload"].mean()),
        "reasons": dec["reason"].value_counts().to_dict(),
        "edge_only": report(y, edge),
        "final": report(y, final),
        "audit_disagreement": float((edge[audit] != cloud[audit]).mean()) if audit.any() else None,
        "cloud_seconds_per_flow": t_flow,
    }
    meta = load_edge(cfg, cluster)
    if has_required_columns(part) and meta["trigger"] is not None:
        inc, src = find_incidents(part, final, theta_victim=meta["trigger"]["theta_flows"],
                                  window=meta["trigger"]["window"],
                                  min_sources_spoof=cfg["attribution"]["min_sources_spoof"],
                                  spoof_frac=cfg["attribution"]["spoof_frac"])
        res["n_incidents"] = int(len(inc))
        if len(inc):
            res["incidents_by_family"] = inc["family"].value_counts().to_dict()
            res["incidents_by_source_group"] = inc["source_group"].value_counts().to_dict()
        truth = (cfg["attribution"].get("attacker_ips") or {}).get(part["dataset"].iloc[0])
        if truth:
            res["compromised_devices"] = compromised_device_scores(src, truth)
    print(f"{name}: macro-F1 edge-only {res['edge_only']['macro_f1']:.4f} | "
          f"final {res['final']['macro_f1']:.4f} | upload {res['upload_rate']:.3f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--model-dir", default="cloud")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--no-merge", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cc = cfg["cloud"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok_dim = 64 * cfg["edge"]["n_models"] * len(cfg["clusters"])
    model = load_cloud(str(artifacts(cfg, args.model_dir)), backbone=args.backbone or cc["backbone"],
                       token_dim=tok_dim, n_classes=NUM_CLASSES, use_lora=cc["use_lora"],
                       dtype=cc["dtype"], load_in_4bit=cc["load_in_4bit"],
                       no_llm=args.no_llm or cc["no_llm"], gradient_checkpointing=False)
    if not args.no_merge and not cc["load_in_4bit"]:
        model.merge_lora()
    if not cc["load_in_4bit"]:
        model.to(device)

    results = []
    for c in cfg["clusters"]:
        df = load_cluster(cfg, c)
        part = df[df["split"] == args.split].reset_index(drop=True)
        dec = load_df(processed_path(cfg, f"decisions_{c}_{args.split}"))
        tokens = np.load(artifacts(cfg, "tokens", f"{c}_{args.split}.npy"), mmap_mode="r")
        seqz = np.load(artifacts(cfg, "sequences", f"{c}_{args.split}.npz"))
        results.append(evaluate_part(f"{c}_{args.split}", part, dec, tokens, seqz, model,
                                     device, cfg, c))
    if args.split == "test":
        for name, spec in (cfg.get("test_only") or {}).items():
            try:
                part = load_df(processed_path(cfg, f"eval_test_only_{name}"))
                dec = load_df(processed_path(cfg, f"decisions_test_only_{name}"))
            except FileNotFoundError:
                continue
            tokens = np.load(artifacts(cfg, "tokens", f"test_only_{name}.npy"), mmap_mode="r")
            seqz = np.load(artifacts(cfg, "sequences", f"test_only_{name}.npz"))
            results.append(evaluate_part(f"test_only_{name}", part, dec, tokens, seqz, model,
                                         device, cfg, spec["cluster"]))

    out = ensure_dir(cfg["paths"]["results"]) / f"results_{args.model_dir}_{args.split}.json"
    with open(out, "w") as f:
        json.dump({"classes": CLASSES, "results": results}, f, indent=2, default=str)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
