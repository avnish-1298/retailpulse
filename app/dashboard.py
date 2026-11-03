import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import streamlit as st
from retailpulse.config import OUTPUTS, PROCESSED
from retailpulse.pipeline import run_all

st.set_page_config(
    page_title="RetailPulse | Retail intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1440px;}
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(20, 42, 62, .72), rgba(22, 31, 48, .72));
        border: 1px solid rgba(148, 163, 184, .16);
        padding: 1rem 1.15rem;
        border-radius: 12px;
    }
    [data-testid="stMetricLabel"] p {font-size: .82rem; opacity: .8;}
    [data-testid="stMetricValue"] {font-size: 1.65rem;}
    div[data-testid="stTabs"] button {font-weight: 600;}
    @media (max-width: 900px) {
        [data-testid="stHorizontalBlock"] {flex-wrap: wrap; gap: .75rem;}
        [data-testid="stHorizontalBlock"] > [data-testid="column"] {
            flex: 1 1 calc(50% - .75rem) !important;
            min-width: 260px;
        }
    }
    @media (max-width: 560px) {
        [data-testid="stHorizontalBlock"] > [data-testid="column"] {
            flex-basis: 100% !important;
            min-width: 0;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load():
    read_output = lambda filename, **kwargs: pd.read_csv(OUTPUTS / filename, **kwargs)
    return (
        json.loads((OUTPUTS / "summary.json").read_text()),
        pd.read_csv(PROCESSED / "sales.csv", parse_dates=["date"]),
        read_output("forecast.csv", parse_dates=["date"]),
        read_output("model_leaderboard.csv"),
        read_output("customers.csv"),
        read_output("inventory_plan.csv"),
        read_output("drift.csv"),
    )


with st.sidebar:
    st.markdown("## 📈 RetailPulse")
    st.caption("Retail intelligence workspace")
    st.divider()
    st.subheader("Data & pipeline")
    if not (OUTPUTS / "summary.json").exists():
        with st.spinner("Preparing your first dashboard run..."):
            run_all()

    if st.button("↻  Refresh analytics", type="primary", width="stretch"):
        with st.spinner("Running ETL and refreshing models..."):
            run_all()
        st.cache_data.clear()
        st.success("Analytics refreshed.")

    st.caption("Add CSV files to `data/raw/` to use your own data.")

S, sales, forecast, leaderboard, customers, inventory, drift = load()
last_run = datetime.fromtimestamp(OUTPUTS.joinpath("summary.json").stat().st_mtime)
sales_start = sales.date.min().strftime("%b %d, %Y")
sales_end = sales.date.max().strftime("%b %d, %Y")

st.title("Retail performance, at a glance")
st.markdown(
    f"Forecast demand, protect customer value, and keep inventory in balance. "
    f"<span style='opacity:.72'>Sales history: {sales_start} – {sales_end} · "
    f"Last refreshed: {last_run.strftime('%b %d, %Y at %I:%M %p')}</span>",
    unsafe_allow_html=True,
)

metric_columns = st.columns(4)
metric_values = [
    ("Forecast WMAPE", f"{S['forecast_wmape']:.1%}", f"{S['best_model'].upper()} · best 28-day backtest"),
    ("Churn model AUC", f"{S['churn_auc']:.3f}", "90-day churn prediction"),
    (
        "Customers at high risk",
        f"{S['high_risk_customers']:,}",
        f"{S['high_risk_customers'] / max(len(customers), 1):.1%} of customers",
    ),
    ("SKUs needing attention", f"{S['skus_at_risk']:,}", f"of {len(inventory):,} SKUs monitored"),
]
for column, (label, value, caption) in zip(metric_columns, metric_values):
    column.metric(label, value)
    column.caption(caption)

forecast_tab, customer_tab, inventory_tab, monitoring_tab = st.tabs(
    ["🔮  Demand forecast", "👥  Customers", "📦  Inventory", "🛠  Model health"]
)

with forecast_tab:
    st.subheader("Demand forecast")
    sku_options = sorted(sales.sku.unique())
    selected_sku = st.selectbox("Product SKU", sku_options, label_visibility="collapsed")
    sku_history = sales[sales.sku == selected_sku].set_index("date").units
    sku_forecast = forecast[forecast.sku == selected_sku].set_index("date").yhat
    history_window = sku_history.tail(90).rename("Actual demand")
    future_window = sku_forecast.rename("Forecast demand")
    forecast_series = pd.concat([history_window, future_window], axis=1)

    recent_actual = sku_history.tail(28).sum()
    expected_demand = sku_forecast.sum()
    change = (expected_demand / recent_actual - 1) if recent_actual else 0
    col_chart, col_summary = st.columns([2.4, 1], gap="large")
    with col_chart:
        st.caption("Daily units · last 90 days and next 28 days")
        st.line_chart(forecast_series, color=["#60a5fa", "#34d399"], height=350)
    with col_summary:
        st.markdown("#### Next 28 days")
        st.metric("Expected units", f"{expected_demand:,.0f}", f"{change:+.1%} vs prior 28 days")
        st.metric("Daily average", f"{sku_forecast.mean():,.1f} units")
        st.caption("Forecast uses the best-performing model from the 28-day backtest.")

    with st.expander("Compare forecasting models"):
        display_board = leaderboard.copy()
        display_board["wmape"] = display_board["wmape"].map(lambda value: f"{value:.1%}")
        display_board["rmse"] = display_board["rmse"].map(lambda value: f"{value:,.2f}")
        display_board["bias"] = display_board["bias"].map(lambda value: f"{value:+,.2f}")
        st.dataframe(display_board, hide_index=True, width="stretch")
        st.caption("Lower WMAPE and RMSE indicate better accuracy; bias shows average forecast error.")

with customer_tab:
    st.subheader("Customer value & retention")
    segment_counts = customers.segment.value_counts().rename("Customers")
    risk_counts = customers.risk.value_counts().reindex(["High", "Medium", "Low"]).dropna()
    segment_chart, risk_chart = st.columns(2)
    with segment_chart:
        st.markdown("#### Customer segments")
        st.bar_chart(segment_counts, color="#60a5fa", height=260)
    with risk_chart:
        st.markdown("#### Churn risk")
        st.bar_chart(risk_counts, color="#fb7185", height=260)

    high_risk = customers[customers.risk == "High"].sort_values(
        ["churn_prob", "monetary"], ascending=[False, False]
    )
    st.markdown("#### Priority win-back list")
    st.caption("Highest predicted churn risk first, with customer value as a tie-breaker.")
    display_customers = high_risk[
        ["customer_id", "segment", "churn_prob", "recency", "frequency", "monetary"]
    ].head(50).copy()
    display_customers["churn_prob"] = display_customers["churn_prob"].map(
        lambda value: f"{value:.1%}"
    )
    display_customers["monetary"] = display_customers["monetary"].map(
        lambda value: f"${value:,.2f}"
    )
    st.dataframe(display_customers, hide_index=True, width="stretch")

with inventory_tab:
    st.subheader("Inventory action center")
    status_counts = inventory.status.value_counts()
    status_columns = st.columns(max(len(status_counts), 1))
    for column, (status, count) in zip(status_columns, status_counts.items()):
        column.metric(status, f"{count:,}")

    action_statuses = ["Stockout risk", "Reorder now"]
    available_statuses = inventory.status.drop_duplicates().tolist()
    selected_statuses = st.multiselect(
        "Show inventory status",
        available_statuses,
        default=[status for status in action_statuses if status in available_statuses],
    )
    inventory_view = inventory[inventory.status.isin(selected_statuses)].sort_values(
        ["days_cover", "sku"]
    )
    if inventory_view.empty:
        st.info("No SKUs match the selected statuses.")
    else:
        inventory_columns = [
            "sku", "status", "on_hand", "days_cover", "fc_daily",
            "reorder_point", "suggested_order_qty",
        ]
        st.dataframe(
            inventory_view[inventory_columns],
            hide_index=True,
            width="stretch",
            column_config={
                "on_hand": st.column_config.NumberColumn("On hand", format="%,.0f"),
                "days_cover": st.column_config.NumberColumn("Days of cover", format="%.1f"),
                "fc_daily": st.column_config.NumberColumn("Daily forecast", format="%.1f"),
                "reorder_point": st.column_config.NumberColumn("Reorder point", format="%,.0f"),
                "suggested_order_qty": st.column_config.NumberColumn("Suggested order", format="%,.0f"),
            },
        )
        st.caption("Suggested order quantity accounts for reorder needs and economic order quantity.")

with monitoring_tab:
    st.subheader("Model health & data drift")
    if S["retrain_recommended"]:
        st.warning("Significant drift detected. Review incoming data and retrain models.")
    else:
        st.success("No significant drift detected. Models are within the configured monitoring threshold.")

    drift_view = drift.copy()
    drift_view["psi"] = drift_view["psi"].map(lambda value: f"{value:.4f}")
    st.dataframe(drift_view, hide_index=True, width="stretch")
    st.caption("Population Stability Index (PSI): below 0.10 is generally stable; 0.10–0.25 indicates moderate drift; 0.25 or higher recommends retraining.")

    st.markdown("#### Forecast model backtest")
    st.dataframe(leaderboard, hide_index=True, width="stretch")
    st.caption("Experiments are tracked in MLflow. Start the experiment viewer with `make mlflow-ui`.")
