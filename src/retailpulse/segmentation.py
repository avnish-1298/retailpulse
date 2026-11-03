"""RFM customer segmentation with K-Means."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

LABELS = ["Hibernating", "At Risk", "Loyal", "Champions"]


def rfm(tx: pd.DataFrame, asof=None) -> pd.DataFrame:
    asof = (asof or tx.date.max()) + pd.Timedelta(days=1)
    g = tx[tx.date < asof].groupby("customer_id")
    return pd.DataFrame({"recency": (asof - g.date.max()).dt.days,
                         "frequency": g.size(), "monetary": g.amount.sum()})


def segment_customers(tx: pd.DataFrame, k=4, seed=42) -> pd.DataFrame:
    f = rfm(tx)
    X = StandardScaler().fit_transform(np.log1p(f))
    f["cluster"] = KMeans(k, n_init=10, random_state=seed).fit_predict(X)
    # rank clusters by value score: low recency (good), high frequency & monetary
    c = f.groupby("cluster")[["recency", "frequency", "monetary"]].mean()
    score = (-c.recency.rank() + c.frequency.rank() + c.monetary.rank()).sort_values()
    names = LABELS if k == len(LABELS) else [f"Segment {i+1}" for i in range(k)]
    f["segment"] = f.cluster.map({cl: names[i] for i, cl in enumerate(score.index)})
    return f.reset_index()
