"""Inventory optimisation: safety stock, reorder point, EOQ, stock status."""
import numpy as np
import pandas as pd
from .config import SERVICE_Z


def optimise_inventory(forecast, sales, inv, z=SERVICE_Z) -> pd.DataFrame:
    recent = sales[sales.date > sales.date.max() - pd.Timedelta(days=90)]
    sigma = recent.groupby("sku").units.std().rename("sigma")
    fc = forecast.groupby("sku").yhat.mean().rename("fc_daily")
    d = inv.set_index("sku").join(fc).join(sigma)
    d["safety_stock"] = z * d.sigma * np.sqrt(d.lead_time_days)
    d["lead_time_demand"] = d.fc_daily * d.lead_time_days
    d["reorder_point"] = d.lead_time_demand + d.safety_stock
    d["eoq"] = np.sqrt(2 * d.fc_daily * 365 * d.order_cost / (d.holding_rate * d.unit_cost))
    d["days_cover"] = d.on_hand / d.fc_daily.clip(lower=1e-9)
    d["status"] = np.select(
        [d.on_hand < d.lead_time_demand, d.on_hand < d.reorder_point, d.days_cover > 60],
        ["Stockout risk", "Reorder now", "Overstock"], default="Healthy")
    need = d.reorder_point + d.fc_daily * 14 - d.on_hand
    d["suggested_order_qty"] = np.where(
        d.status.isin(["Stockout risk", "Reorder now"]), np.ceil(np.maximum(d.eoq, need)), 0).astype(int)
    return d.round(2).reset_index()
