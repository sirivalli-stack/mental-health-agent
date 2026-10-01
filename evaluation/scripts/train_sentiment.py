"""Train the Phase 3 sentiment model (thin wrapper around the backend CLI).

Usage (from the project root):

    python evaluation/scripts/train_sentiment.py
    python evaluation/scripts/train_sentiment.py --limit 2000 --no-refit

Writes: models/sentiment/sentiment_pipeline.joblib
        models/sentiment/sentiment_meta.json
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.training.sentiment_train import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
