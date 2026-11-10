"""Edge MLP ensemble (3 models per cluster) and its training loop. Needs PyTorch."""
import numpy as np
import torch
import torch.nn as nn


class EdgeMLP(nn.Module):
    """12 -> 64 -> 64 -> K. The 64-dim output of `hidden` is the TGG token part."""

    def __init__(self, n_features=12, n_classes=3, hidden=64, dropout=0.1):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.out = nn.Linear(hidden, n_classes)

    def forward(self, x):
        return self.out(self.hidden(x))

    @torch.no_grad()
    def embed(self, x):
        self.eval()
        return self.hidden(x)


def _class_weights(y, n_classes):
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def train_mlp(X_tr, y_tr, X_val, y_val, n_classes, epochs=30, batch_size=1024, lr=1e-3,
              patience=5, seed=0, device="cpu"):
    """Train one EdgeMLP with early stopping on validation loss. Returns the best model."""
    torch.manual_seed(seed)
    model = EdgeMLP(X_tr.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=_class_weights(y_tr, n_classes).to(device))
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.long)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.long, device=device)
    best, best_state, bad = float("inf"), None, 0
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xt[idx].to(device), yt[idx].to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = nn.functional.cross_entropy(model(Xv), yv).item()
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model.cpu().eval()


def train_ensemble(X_tr, y_tr, X_val, y_val, n_classes, n_models=3, seed=0, **kw):
    """Train n_models MLPs, each on a different random third of the training data."""
    rng = np.random.default_rng(seed)
    parts = np.array_split(rng.permutation(len(X_tr)), n_models)
    return [train_mlp(X_tr[p], y_tr[p], X_val, y_val, n_classes, seed=seed + m, **kw)
            for m, p in enumerate(parts)]


@torch.no_grad()
def ensemble_logits(models, X, batch_size=65536):
    """Average logits of the ensemble -> numpy array (N, K)."""
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32)
        out.append(torch.stack([m(xb) for m in models]).mean(0).numpy())
    return np.concatenate(out)


@torch.no_grad()
def ensemble_hidden(models, X, batch_size=65536):
    """Concatenate the 64-dim hidden layers of the ensemble -> (N, 64 * n_models)."""
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32)
        out.append(torch.cat([m.embed(xb) for m in models], dim=1).numpy())
    return np.concatenate(out)
