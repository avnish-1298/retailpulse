"""Automated ETL: validate, clean, and standardise raw retail data."""
import numpy as np
import pandas as pd
from .config import RAW, PROCESSED


def clean_sales(sales: pd.DataFrame) -> pd.DataFrame:
    s = sales.copy()
    s["date"] = pd.to_datetime(s["date"])
    s = s.drop_duplicates(["date", "sku"]).sort_values(["sku", "date"])
    s["units"] = pd.to_numeric(s["units"], errors="coerce").clip(lower=0)
    # complete date x sku grid so lag/seasonal logic is safe
    full = pd.MultiIndex.from_product(
        [pd.date_range(s.date.min(), s.date.max()), s.sku.unique()], names=["date", "sku"])
    s = s.set_index(["date", "sku"]).reindex(full).reset_index().sort_values(["sku", "date"])
    s["units"] = s.groupby("sku").units.transform(lambda x: x.interpolate().bfill().ffill())
    s["promo"] = s["promo"].fillna(0).astype(int)
    s["price"] = s.groupby("sku").price.transform(lambda x: x.ffill().bfill())
    s["category"] = s.groupby("sku").category.transform(lambda x: x.ffill().bfill())
    return s.reset_index(drop=True)


def clean_transactions(tx: pd.DataFrame) -> pd.DataFrame:
    t = tx.copy()
    t["date"] = pd.to_datetime(t["date"])
    t = t.dropna(subset=["customer_id", "date", "amount"])
    return t[t.amount > 0].drop_duplicates().reset_index(drop=True)


def run_etl(sales=None, tx=None, inv=None, save=True):
    sales = pd.read_csv(RAW / "sales.csv") if sales is None else sales
    tx = pd.read_csv(RAW / "transactions.csv") if tx is None else tx
    inv = pd.read_csv(RAW / "inventory.csv") if inv is None else inv
    sales, tx = clean_sales(sales), clean_transactions(tx)
    if save:
        sales.to_csv(PROCESSED / "sales.csv", index=False)
        tx.to_csv(PROCESSED / "transactions.csv", index=False)
        inv.to_csv(PROCESSED / "inventory.csv", index=False)
    return sales, tx, inv
