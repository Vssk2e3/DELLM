"""Temperature scaling and normalised entropy (NumPy only)."""
import numpy as np


def softmax(z, T=1.0):
    z = np.asarray(z, dtype=np.float64) / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits, y, T):
    p = softmax(logits, T)
    return -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))


def fit_temperature(logits, y, lo=0.05, hi=20.0, iters=60):
    """Find T minimising validation NLL (golden-section search on log T)."""
    a, b = np.log(lo), np.log(hi)
    g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll(logits, y, np.exp(c)), nll(logits, y, np.exp(d))
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = nll(logits, y, np.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = nll(logits, y, np.exp(d))
    return float(np.exp((a + b) / 2))


def normalised_entropy(p):
    """H / log K, in [0, 1]."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0)
    k = p.shape[1]
    if k < 2:
        return np.zeros(len(p))
    return -(p * np.log(p)).sum(axis=1) / np.log(k)


def expected_calibration_error(p, y, n_bins=15):
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(ece)
