"""Churn prediction: customers with no purchase in the next N days."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from .config import CHURN_WINDOW_DAYS, SEED


def build_features(tx: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    t = tx[tx.date <= asof]
    g = t.groupby("customer_id")
    f = pd.DataFrame({
        "recency": (asof - g.date.max()).dt.days, "frequency": g.size(),
        "monetary": g.amount.sum(), "avg_order": g.amount.mean(),
        "tenure": (asof - g.date.min()).dt.days})
    f["avg_gap"] = f.tenure / f.frequency.clip(lower=1)
    f["recency_ratio"] = f.recency / f.avg_gap.clip(lower=1)
    for d in (30, 60):
        f[f"freq_{d}d"] = t[t.date > asof - pd.Timedelta(days=d)].groupby("customer_id").size() \
            .reindex(f.index).fillna(0)
    return f


def run_churn(tx: pd.DataFrame, window=CHURN_WINDOW_DAYS):
    last = tx.date.max()
    cutoff = last - pd.Timedelta(days=window)
    X = build_features(tx, cutoff)
    future_buyers = set(tx[(tx.date > cutoff)].customer_id)
    y = (~X.index.isin(future_buyers)).astype(int)   # 1 = churned

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=SEED)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    metrics = {"auc": float(roc_auc_score(yte, p)),
               "avg_precision": float(average_precision_score(yte, p)),
               "churn_rate": float(y.mean())}

    now = build_features(tx, last)
    scores = pd.DataFrame({"customer_id": now.index,
                           "churn_prob": clf.predict_proba(now)[:, 1]})
    scores["risk"] = pd.cut(scores.churn_prob, [-0.01, 0.3, 0.6, 1.0],
                            labels=["Low", "Medium", "High"])
    return scores.merge(now.reset_index(), on="customer_id"), metrics

