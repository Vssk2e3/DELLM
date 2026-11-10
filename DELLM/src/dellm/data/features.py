"""Feature selection: cleaning + Boruta per cluster + consensus top-B features."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def drop_constant_and_correlated(df, cols, corr_threshold=0.95):
    """Remove constant columns, then one of each pair with |correlation| > threshold."""
    cols = [c for c in cols if df[c].nunique(dropna=True) > 1]
    if len(cols) < 2:
        return cols
    corr = df[cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = {c for c in upper.columns if (upper[c] > corr_threshold).any()}
    return [c for c in cols if c not in drop]


def boruta_select(X, y, max_iter=50, seed=0, n_estimators=200):
    """Run Boruta (BorutaPy) and return (accepted_mask, importances).

    Falls back to a simple shadow-feature test if the `boruta` package is missing.
    """
    try:
        from boruta import BorutaPy
        rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=8,
                                    n_estimators=n_estimators, random_state=seed)
        sel = BorutaPy(rf, n_estimators="auto", max_iter=max_iter, random_state=seed, verbose=0)
        sel.fit(np.asarray(X, dtype=float), np.asarray(y))
        rf.fit(np.asarray(X, dtype=float), np.asarray(y))
        return sel.support_ | sel.support_weak_, rf.feature_importances_
    except ImportError:
        return _shadow_select(X, y, rounds=max(5, max_iter // 10), seed=seed,
                              n_estimators=n_estimators)


def _shadow_select(X, y, rounds=5, seed=0, n_estimators=200):
    """Simplified Boruta: accept a feature if it beats the best shuffled copy in most rounds."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    wins = np.zeros(X.shape[1])
    imp_sum = np.zeros(X.shape[1])
    for r in range(rounds):
        shadow = X.copy()
        for j in range(shadow.shape[1]):
            rng.shuffle(shadow[:, j])
        rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=8,
                                    n_estimators=n_estimators, random_state=seed + r)
        rf.fit(np.hstack([X, shadow]), y)
        imp = rf.feature_importances_
        real, best_shadow = imp[: X.shape[1]], imp[X.shape[1]:].max()
        wins += real > best_shadow
        imp_sum += real
    return wins >= np.ceil(0.6 * rounds), imp_sum / rounds


def consensus_features(per_cluster, n_features=12, min_clusters=3):
    """Combine per-cluster Boruta results.

    per_cluster: {cluster: DataFrame(index=feature, columns=['accepted', 'importance'])}
    Keeps features accepted in >= min_clusters clusters, ranks them by mean standardised
    importance, and returns the top n_features (topping up from the best remaining
    features if fewer pass).
    """
    table = pd.concat({c: d for c, d in per_cluster.items()}, axis=1)
    acc = table.xs("accepted", axis=1, level=1).fillna(False).astype(bool)
    imp = table.xs("importance", axis=1, level=1).fillna(0.0)
    z = (imp - imp.mean()) / (imp.std(ddof=0) + 1e-12)
    score = z.mean(axis=1)
    votes = acc.sum(axis=1)
    passed = score[votes >= min_clusters].sort_values(ascending=False)
    chosen = list(passed.index[:n_features])
    if len(chosen) < n_features:
        rest = score.drop(index=chosen).sort_values(ascending=False)
        chosen += list(rest.index[: n_features - len(chosen)])
    summary = pd.DataFrame({"votes": votes, "mean_z_importance": score}).sort_values(
        ["votes", "mean_z_importance"], ascending=False)
    return chosen, summary
