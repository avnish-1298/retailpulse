"""Data-drift monitoring via Population Stability Index (PSI)."""
import numpy as np
import pandas as pd


def psi(ref, cur, bins=10) -> float:
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.clip(np.histogram(ref, edges)[0] / len(ref), 1e-4, None)
    c = np.clip(np.histogram(cur, edges)[0] / len(cur), 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def status(v: float) -> str:
    return "Stable" if v < 0.1 else "Moderate drift" if v < 0.25 else "Significant drift"


def drift_report(sales: pd.DataFrame, window=28) -> pd.DataFrame:
    cut = sales.date.max() - pd.Timedelta(days=window)
    ref, cur = sales[sales.date <= cut], sales[sales.date > cut]
    rows = []
    for col in ("units", "price", "promo"):
        v = psi(ref[col], cur[col])
        rows.append({"feature": col, "psi": round(v, 4), "status": status(v)})
    return pd.DataFrame(rows)


def needs_retraining(report: pd.DataFrame, threshold=0.25) -> bool:
    return bool((report.psi >= threshold).any())
