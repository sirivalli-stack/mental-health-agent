"""Train the Phase 4 emotion model (thin wrapper around the backend CLI).

Usage (from the project root):

    python evaluation/scripts/train_emotion.py
    python evaluation/scripts/train_emotion.py --limit 3000 --no-refit

Writes: models/emotion/emotion_pipeline.joblib
        models/emotion/emotion_meta.json
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.training.emotion_train import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
