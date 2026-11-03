import numpy as np
from retailpulse import data_gen, etl
from retailpulse.forecasting import run_forecasting, evaluate
from retailpulse.segmentation import segment_customers
from retailpulse.churn import run_churn
from retailpulse.inventory import optimise_inventory
from retailpulse.monitoring import psi, drift_report

SALES, TX, INV = data_gen.generate(n_sku=3, n_cust=250, days=400, seed=1, end="2025-12-31")


def test_psi_identical_is_zero():
    x = np.random.default_rng(0).normal(size=2000)
    assert psi(x, x) < 1e-6


def test_psi_detects_shift():
    r = np.random.default_rng(0)
    assert psi(r.normal(0, 1, 3000), r.normal(2, 1, 3000)) > 0.25


def test_etl_fills_grid():
    s = SALES.drop(SALES.sample(20, random_state=0).index)
    out = etl.clean_sales(s)
    assert len(out) == SALES.shape[0] and out.units.notna().all()


def test_pipeline_components():
    sales, tx = etl.clean_sales(SALES), etl.clean_transactions(TX)
    fc, board = run_forecasting(sales, include_heavy=False)
    assert len(fc) == 28 * 3 and (fc.yhat >= 0).all()
    assert board.iloc[0].wmape < 0.5
    seg = segment_customers(tx)
    assert seg.segment.nunique() == 4
    churn, m = run_churn(tx)
    assert 0.5 < m["auc"] <= 1
    plan = optimise_inventory(fc, sales, INV)
    assert (plan.eoq > 0).all() and (plan.reorder_point >= plan.lead_time_demand).all()
    assert len(drift_report(sales)) == 3


def test_metrics():
    assert evaluate([10, 10], [10, 10])["wmape"] == 0
