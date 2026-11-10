"""Step 1: convert PCAPs to CICFlowMeter flow CSVs, keeping the attack folder as the label.

Expected layout (one folder per label):
    data/raw/<Dataset>/<label>/*.pcap      ->  data/flows/<Dataset>/<label>/*.csv

Uses the Python `cicflowmeter` tool (pip install cicflowmeter). For .pcapng files,
convert first:  editcap -F pcap in.pcapng out.pcap   (editcap ships with Wireshark).

Usage:
    python scripts/01_extract_flows.py --dataset CICIoT2023
    python scripts/01_extract_flows.py --dataset Edge-IIoTset --tool "cicflowmeter"
"""
import argparse
import shutil
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401
from dellm.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tool", default="cicflowmeter")
    args = ap.parse_args()
    cfg = load_config(args.config)
    src = Path(cfg["paths"]["raw"]) / args.dataset
    dst = Path(cfg["paths"]["flows"]) / args.dataset
    if shutil.which(args.tool) is None:
        raise SystemExit(f"'{args.tool}' not found. Install it with: pip install cicflowmeter")
    pcaps = sorted(list(src.rglob("*.pcap")) + list(src.rglob("*.pcapng")))
    if not pcaps:
        raise SystemExit(f"No PCAP files under {src}")
    for p in pcaps:
        if p.suffix == ".pcapng":
            print(f"skip (convert to .pcap first): {p}")
            continue
        out = dst / p.parent.relative_to(src) / (p.stem + ".csv")
        if out.exists():
            print(f"exists: {out}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"{p}  ->  {out}")
        subprocess.run([args.tool, "-f", str(p), "-c", str(out)], check=True)


if __name__ == "__main__":
    main()
