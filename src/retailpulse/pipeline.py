"""End-to-end pipeline:  generate/ingest -> ETL -> models -> outputs.
Run:  python -m retailpulse.pipeline [--small] [--no-heavy]"""
import argparse, json
from . import data_gen, etl
from .config import OUTPUTS, RAW
from .forecasting import run_forecasting
from .segmentation import segment_customers
from .churn import run_churn
from .inventory import optimise_inventory
from .monitoring import drift_report, needs_retraining


def run_all(small=False, include_heavy=True):
    if not (RAW / "sales.csv").exists() or small:
        kw = dict(n_sku=4, n_cust=200, days=420) if small else {}
        data_gen.save_raw(*data_gen.generate(**kw))
    sales, tx, inv = etl.run_etl()

    forecast, board = run_forecasting(sales, include_heavy=include_heavy)
    seg = segment_customers(tx)
    churn, churn_m = run_churn(tx)
    inv_plan = optimise_inventory(forecast, sales, inv)
    drift = drift_report(sales)

    customers = churn.merge(seg[["customer_id", "segment"]], on="customer_id")
    forecast.to_csv(OUTPUTS / "forecast.csv", index=False)
    board.to_csv(OUTPUTS / "model_leaderboard.csv", index=False)
    customers.to_csv(OUTPUTS / "customers.csv", index=False)
    inv_plan.to_csv(OUTPUTS / "inventory_plan.csv", index=False)
    drift.to_csv(OUTPUTS / "drift.csv", index=False)

    summary = {
        "best_model": board.iloc[0].model, "forecast_wmape": float(board.iloc[0].wmape),
        "churn_auc": churn_m["auc"], "churn_rate": churn_m["churn_rate"],
        "high_risk_customers": int((customers.risk == "High").sum()),
        "skus_at_risk": int(inv_plan.status.isin(["Stockout risk", "Reorder now"]).sum()),
        "retrain_recommended": needs_retraining(drift),
    }
    (OUTPUTS / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--no-heavy", action="store_true", help="skip Prophet/LSTM")
    a = ap.parse_args()
    print(json.dumps(run_all(a.small, not a.no_heavy), indent=2))
