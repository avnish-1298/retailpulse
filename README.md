# RetailPulse — AI-Powered Customer Analytics & Demand Forecasting

Predictive analytics for retail: **demand forecasting, customer segmentation, churn prediction,
inventory optimisation**, with production-style MLOps (MLflow, drift monitoring, Docker, Kubernetes, CI/CD).

## Architecture
```
 raw CSVs / synthetic data ──► ETL (validate, clean, date×SKU grid)
                                   │
        ┌──────────────┬───────────┼──────────────┬───────────────┐
   Forecasting      Segmentation  Churn       Inventory       Drift
 GBM·Prophet·LSTM  RFM + K-Means  GBM         SS·ROP·EOQ      PSI
        └──────────────┴───────────┼──────────────┴───────────────┘
                     outputs/*.csv + MLflow runs
                                   ▼
                        Streamlit dashboard
```

## Quick start
```bash
pip install -r requirements.txt && pip install -e .
python -m retailpulse.pipeline          # builds data, trains, writes outputs/
streamlit run app/dashboard.py          # http://localhost:8501
mlflow ui --backend-store-uri file:./mlruns
pytest -q
```
Optional deep-learning/Prophet models: `pip install -r requirements-extra.txt` (auto-detected; the
best model by 28-day backtest WMAPE is chosen automatically).

## Modules (`src/retailpulse/`)
| Module | What it does |
|---|---|
| `data_gen.py` | Synthetic sales (seasonality, trend, promos), transactions (with churners), inventory |
| `etl.py` | Dedup, type coercion, complete date×SKU grid, interpolation of gaps |
| `forecasting.py` | Global GBM, per-SKU Prophet, LSTM; backtest, leaderboard, MLflow logging |
| `segmentation.py` | RFM → log/scale → K-Means → named segments (Champions … Hibernating) |
| `churn.py` | Time-split labelling (no purchase in next 90d), GBM classifier, AUC / PR metrics |
| `inventory.py` | Safety stock `z·σ·√LT`, reorder point, EOQ, days of cover, suggested orders |
| `monitoring.py` | PSI drift on units/price/promo; retrain recommendation at PSI ≥ 0.25 |
| `pipeline.py` | Orchestrates everything; writes `outputs/summary.json` |

## Using your own data
Place CSVs in `data/raw/` and re-run the pipeline:
- `sales.csv`: `date, sku, units, promo, price, category`
- `transactions.csv`: `customer_id, date, sku, amount`
- `inventory.csv`: `sku, on_hand, lead_time_days, unit_cost, order_cost, holding_rate`

## Deployment
- **Docker:** `docker compose up --build` → dashboard :8501, MLflow :5000
- **Kubernetes:** `kubectl apply -f k8s/deployment.yaml` (Deployment, Service, PVC, nightly retrain CronJob)
- **CI/CD:** `.github/workflows/ci.yml` runs tests, a smoke pipeline, and a Docker build

## Notes
Default data is synthetic, so metrics demonstrate the pipeline, not real-world performance.
For production, add a model registry stage, scheduled drift alerts, and a feature store.
