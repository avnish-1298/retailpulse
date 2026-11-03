"""Demand forecasting: GBM baseline + optional Prophet & LSTM, tracked in MLflow."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from .config import MLRUNS, HORIZON_DAYS

EPOCH = pd.Timestamp("2020-01-01")


def evaluate(y, yhat) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    err = y - yhat
    return {"wmape": float(np.abs(err).sum() / max(y.sum(), 1e-9)),
            "rmse": float(np.sqrt((err ** 2).mean())),
            "bias": float(err.mean())}


class GBMModel:
    name = "gbm"

    def _X(self, df):
        doy = df.date.dt.dayofyear
        return pd.DataFrame({
            "sku_code": df.sku.map(self.codes).fillna(-1).astype(int),
            "dow": df.date.dt.dayofweek, "month": df.date.dt.month,
            "sin": np.sin(2 * np.pi * doy / 365.25), "cos": np.cos(2 * np.pi * doy / 365.25),
            "promo": df.promo, "price": df.price, "t": (df.date - EPOCH).dt.days,
        })

    def fit(self, df):
        self.codes = {s: i for i, s in enumerate(sorted(df.sku.unique()))}
        self.m = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06, categorical_features=[0], random_state=42)
        self.m.fit(self._X(df), np.log1p(df.units))
        return self

    def predict(self, future):
        out = future[["date", "sku"]].copy()
        out["yhat"] = np.clip(np.expm1(self.m.predict(self._X(future))), 0, None)
        return out


class ProphetModel:
    name = "prophet"

    def fit(self, df):
        from prophet import Prophet
        self.m = {}
        for sku, g in df.groupby("sku"):
            p = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
            p.add_regressor("promo")
            p.fit(g.rename(columns={"date": "ds", "units": "y"})[["ds", "y", "promo"]])
            self.m[sku] = p
        return self

    def predict(self, future):
        parts = []
        for sku, g in future.groupby("sku", sort=False):
            f = self.m[sku].predict(g.rename(columns={"date": "ds"})[["ds", "promo"]])
            parts.append(pd.DataFrame({"date": g.date.values, "sku": sku,
                                       "yhat": f.yhat.clip(lower=0).values}))
        res = pd.concat(parts)
        return future[["date", "sku"]].merge(res, on=["date", "sku"], how="left")


class LSTMModel:
    name = "lstm"

    def __init__(self, window=28, epochs=12):
        self.window, self.epochs = window, epochs

    def fit(self, df):
        import tensorflow as tf
        piv = df.pivot(index="date", columns="sku", values="units").sort_index()
        self.cols, self.mu, self.sd = list(piv.columns), piv.mean(), piv.std().replace(0, 1)
        z = ((piv - self.mu) / self.sd).values
        X, y = [], []
        for s in range(z.shape[1]):
            for i in range(self.window, len(z)):
                X.append(z[i - self.window:i, s]); y.append(z[i, s])
        X, y = np.array(X)[..., None], np.array(y)
        tf.random.set_seed(42)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input((self.window, 1)),
            tf.keras.layers.LSTM(32), tf.keras.layers.Dense(1)])
        self.model.compile("adam", "mse")
        self.model.fit(X, y, epochs=self.epochs, batch_size=256, verbose=0)
        self.last = z[-self.window:]
        return self

    def predict(self, future):
        dates = sorted(future.date.unique())
        hist, out = self.last.copy(), []
        for _ in dates:  # recursive multi-step forecast
            p = self.model.predict(hist.T[..., None], verbose=0)[:, 0]
            out.append(p); hist = np.vstack([hist[1:], p])
        arr = np.array(out) * self.sd.values + self.mu.values
        res = pd.DataFrame(arr, index=dates, columns=self.cols).clip(lower=0).stack().reset_index()
        res.columns = ["date", "sku", "yhat"]
        return future[["date", "sku"]].merge(res, on=["date", "sku"], how="left")


def available_models(include_heavy=True):
    models = {"gbm": GBMModel}
    if include_heavy:
        try:
            import prophet  # noqa: F401
            models["prophet"] = ProphetModel
        except Exception:
            pass
        try:
            import tensorflow  # noqa: F401
            models["lstm"] = LSTMModel
        except Exception:
            pass
    return models


def _log_mlflow(name, metrics, horizon):
    try:
        import mlflow
        mlflow.set_tracking_uri(f"file:{MLRUNS}")
        mlflow.set_experiment("retailpulse-demand-forecasting")
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"model": name, "horizon_days": horizon})
            mlflow.log_metrics(metrics)
    except Exception:
        pass


def run_forecasting(sales, horizon=HORIZON_DAYS, include_heavy=True):
    """Backtest every available model on the last `horizon` days, then refit the
    winner on all data and forecast the next `horizon` days."""
    last = sales.date.max()
    train, test = sales[sales.date <= last - pd.Timedelta(days=horizon)], \
        sales[sales.date > last - pd.Timedelta(days=horizon)]
    board = []
    for name, cls in available_models(include_heavy).items():
        try:
            pred = cls().fit(train).predict(test[["date", "sku", "promo", "price"]])
            m = evaluate(test.units.values, pred.yhat.values)
            _log_mlflow(name, m, horizon)
            board.append({"model": name, **m})
        except Exception as e:  # a failing optional model must not break the pipeline
            print(f"[forecast] {name} skipped: {e}")
    board = pd.DataFrame(board).sort_values("wmape").reset_index(drop=True)
    best = board.iloc[0].model

    model = available_models(include_heavy)[best]().fit(sales)
    last_price = sales.sort_values("date").groupby("sku").price.last()
    future = pd.MultiIndex.from_product(
        [pd.date_range(last + pd.Timedelta(days=1), periods=horizon), sorted(sales.sku.unique())],
        names=["date", "sku"]).to_frame(index=False)
    future["promo"], future["price"] = 0, future.sku.map(last_price)
    forecast = model.predict(future)
    forecast["model"] = best
    return forecast, board
