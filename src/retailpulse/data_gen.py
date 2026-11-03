"""Synthetic retail data so the whole platform runs with zero external datasets."""
import numpy as np
import pandas as pd
from .config import RAW, SEED

CATEGORIES = ["Grocery", "Beverages", "Household", "Personal Care", "Snacks"]


def generate(n_sku=20, n_cust=1500, days=730, seed=SEED, end=None):
    rng = np.random.default_rng(seed)
    end = pd.Timestamp(end) if end else pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    dates = pd.date_range(end=end, periods=days, freq="D")

    # --- SKUs & daily demand -------------------------------------------------
    skus = [f"SKU{i:03d}" for i in range(1, n_sku + 1)]
    base = rng.uniform(20, 120, n_sku)
    price = rng.uniform(5, 80, n_sku).round(2)
    phase = rng.uniform(0, 2 * np.pi, n_sku)
    rows = []
    t = np.arange(days)
    for i, s in enumerate(skus):
        season = 1 + 0.25 * np.sin(2 * np.pi * t / 365.25 + phase[i])
        weekend = np.where(dates.dayofweek >= 5, 1.25, 1.0)
        trend = 1 + 0.0004 * t
        promo = (rng.random(days) < 0.08).astype(int)
        lam = base[i] * season * weekend * trend * np.where(promo == 1, 1.4, 1.0)
        rows.append(pd.DataFrame({
            "date": dates, "sku": s, "units": rng.poisson(lam),
            "promo": promo, "price": np.where(promo == 1, price[i] * 0.85, price[i]).round(2),
            "category": CATEGORIES[i % len(CATEGORIES)],
        }))
    sales = pd.concat(rows, ignore_index=True)

    # --- Inventory snapshot --------------------------------------------------
    mean_d = sales.groupby("sku").units.mean().reindex(skus).values
    inv = pd.DataFrame({
        "sku": skus,
        "on_hand": (mean_d * rng.uniform(2, 45, n_sku)).astype(int),
        "lead_time_days": rng.integers(3, 15, n_sku),
        "unit_cost": (price * 0.6).round(2),
        "order_cost": 50.0,
        "holding_rate": 0.20,
    })

    # --- Customer transactions (some customers churn) -------------------------
    recs = []
    for c in range(n_cust):
        signup = int(rng.integers(0, int(days * 0.6)))
        p_buy = rng.uniform(0.01, 0.08)
        churn_at = signup + int(rng.integers(60, days)) if rng.random() < 0.35 else days
        active = np.arange(signup, min(churn_at, days))
        hit = active[rng.random(len(active)) < p_buy]
        if len(hit) == 0:
            continue
        recs.append(pd.DataFrame({
            "customer_id": f"C{c:05d}", "date": dates[hit],
            "sku": rng.choice(skus, len(hit)),
            "amount": rng.lognormal(3.5, 0.6, len(hit)).round(2),
        }))
    tx = pd.concat(recs, ignore_index=True)
    return sales, tx, inv


def save_raw(sales, tx, inv):
    sales.to_csv(RAW / "sales.csv", index=False)
    tx.to_csv(RAW / "transactions.csv", index=False)
    inv.to_csv(RAW / "inventory.csv", index=False)
