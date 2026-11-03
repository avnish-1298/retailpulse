import os
from pathlib import Path

ROOT = Path(os.getenv("RETAILPULSE_HOME", Path(__file__).resolve().parents[2]))
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
MLRUNS = ROOT / "mlruns"

HORIZON_DAYS = 28
SERVICE_Z = 1.65          # ~95% service level
CHURN_WINDOW_DAYS = 90    # no purchase in this window == churned
SEED = 42

for p in (RAW, PROCESSED, OUTPUTS):
    p.mkdir(parents=True, exist_ok=True)
