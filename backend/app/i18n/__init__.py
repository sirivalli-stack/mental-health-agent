"""Multilingual support (Phase 5). [OURS]

EN / TE / HI / TA / KN first-class, with the other Phase 20 preference
codes carried along:

* :mod:`app.i18n.scripts`  - Unicode script detection (pure, testable);
* :mod:`app.i18n.translate` - best-effort translation **for analysis
  only**, plus the language names used in reply-language prompts.

Design commitments, verified by ``tests/test_phase22.py``:

1. **Originals are preserved.** The transcript stores what the user
   wrote; the model answers the user's original words. Translation feeds
   only the English-trained analysis stages.
2. **Fail-open.** No translation call happens for Latin-script input, and
   any failure falls back to the original text - the safety gates then
   behave exactly as they did before Phase 5.
3. **No weakened safety.** Crisis rules evaluate the English rendering
   when one exists (strictly more coverage for Indic input), and blocked
   turns still never reach the model.
4. **Honesty.** Emotion/crisis accuracy on translated text is
   approximate. Nothing in this package claims parity with the English
   path.
"""

from __future__ import annotations

from app.i18n.scripts import (
    detect_script,
    needs_analysis_translation,
    tracked_char_counts,
)
from app.i18n.translate import LANGUAGE_NAMES, translate_for_analysis

__all__ = [
    "LANGUAGE_NAMES",
    "detect_script",
    "needs_analysis_translation",
    "tracked_char_counts",
    "translate_for_analysis",
]
