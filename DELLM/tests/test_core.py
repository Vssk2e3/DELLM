"""Tests that run without a GPU, torch or any dataset (synthetic data only).

Run:  python -m pytest -q
"""
import numpy as np
import pandas as pd

from dellm import CLASSES
from dellm.attribution import find_incidents
from dellm.cloud.order import order_sequences
from dellm.data.labels import map_label
from dellm.edge.calibration import fit_temperature, normalised_entropy, softmax
from dellm.edge.decide import edge_decisions
from dellm.edge.threshold import select_tau
from dellm.edge.trigger import apply_trigger, fit_trigger


def test_label_mapping():
    assert map_label("BENIGN") == "Benign"
    assert map_label("DDoS-SYN_Flood") == "NetDDoS"
    assert map_label("DrDoS_DNS") == "DrDoS"
    assert map_label("DDoS_HTTP") == "AppDDoS"
    assert map_label("Mirai-greeth_flood") == "Botnet"
    assert map_label("Slowloris") == "LDDoS"
    assert map_label("DoS-SYN_Flood") is None          # single-source DoS is dropped
    assert map_label("Recon-PortScan") is None
    assert map_label("Brute_Force") is None


def test_temperature_and_entropy():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 5000)
    logits = rng.normal(size=(5000, 4))
    logits[np.arange(5000), y] += 2.0
    over = logits * 5                                    # an over-confident model
    T = fit_temperature(over, y)
    assert T > 2.0                                       # scaling softens it
    h = normalised_entropy(softmax(np.zeros((3, 4))))
    assert np.allclose(h, 1.0)                           # uniform -> 1
    h = normalised_entropy(softmax(np.array([[50.0, 0, 0, 0]])))
    assert h[0] < 1e-6                                   # certain -> 0


def test_select_tau_respects_error():
    rng = np.random.default_rng(1)
    h = rng.random(10000)
    correct = rng.random(10000) > h * 0.3               # higher entropy -> more errors
    tau, info, _ = select_tau(h, correct, eps=0.01, budget=0.10)
    kept = h <= tau
    assert (~correct[kept]).mean() <= 0.01 + 1e-9


def _traffic(n_benign=4000, attack=True, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 200, n_benign))
    df = pd.DataFrame({"timestamp": t,
                       "src_ip": [f"10.0.0.{i}" for i in rng.integers(1, 50, n_benign)],
                       "dst_ip": [f"10.0.1.{i}" for i in rng.integers(1, 20, n_benign)],
                       "y": 0})
    if attack:
        m = 3000
        a = pd.DataFrame({"timestamp": rng.uniform(100, 104, m),
                          "src_ip": [f"172.16.{i // 250}.{i % 250}" for i in rng.integers(0, 2000, m)],
                          "dst_ip": "10.0.1.5", "y": CLASSES.index("NetDDoS")})
        df = pd.concat([df, a]).sort_values("timestamp").reset_index(drop=True)
    return df


def test_trigger_fires_on_flood_only():
    params = fit_trigger(_traffic(attack=False), window=2.0, q=99.0)
    df = _traffic(attack=True, seed=1)
    fired = apply_trigger(df, params).to_numpy()
    atk = df["y"].to_numpy() != 0
    assert fired[atk].mean() > 0.9
    assert fired[~atk].mean() < 0.2


def test_edge_decisions_reasons():
    df = _traffic(attack=False)
    logits = np.zeros((len(df), 3))
    logits[: len(df) // 2, 0] = 10.0                     # half confident, half uniform
    dec = edge_decisions(df, logits, T=1.0, tau=0.5, trigger_params=None, audit_rate=0.0)
    assert (dec["reason"].iloc[: len(df) // 2] == "edge").all()
    assert (dec["reason"].iloc[len(df) // 2:] == "uncertain").all()


def test_order_sequences_cover_every_flow_once():
    df = _traffic(attack=True, seed=2).iloc[:1500].copy()
    df["protocol"] = 6
    df["cluster"] = "C1"
    seq, copy = order_sequences(df, L=64, min_group=16)
    assert seq.shape[1] == 64 and copy.shape == seq.shape
    real = seq[~copy]
    assert sorted(real.tolist()) == list(range(len(df)))  # each flow labelled exactly once


def test_attribution_finds_victim():
    df = _traffic(attack=True, seed=3)
    inc, src = find_incidents(df, df["y"].to_numpy(), theta_victim=50, window=2.0)
    assert len(inc) > 0
    assert set(inc["victim"]) == {"10.0.1.5"}
    assert set(inc["source_group"]) == {"likely_spoofed"}
