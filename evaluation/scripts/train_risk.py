"""Train the Phase 5 risk model (thin wrapper around the backend CLI).

Usage (from the project root):

    python evaluation/scripts/train_risk.py
    python evaluation/scripts/train_risk.py --limit 20000 --no-refit

Writes: models/risk/risk_pipeline.joblib
        models/risk/risk_meta.json
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.training.risk_train import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
