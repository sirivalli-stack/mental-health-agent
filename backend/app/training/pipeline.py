"""The shared scikit-learn classifier used by Phases 3-5.

TF-IDF over 1-2 grams plus multinomial logistic regression: a transparent,
fast-to-train baseline that runs on CPU and can be re-fit from scratch in
seconds, which keeps every ablation reproducible on this machine.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_pipeline(
    C: float = 1.0,
    class_weight: str | None = None,
    seed: int = 42,
    max_features: int = 120_000,
) -> Pipeline:
    """TF-IDF (1-2 grams) + multinomial logistic regression."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=3,
                    max_df=0.98,
                    max_features=max_features,
                    sublinear_tf=True,
                    token_pattern=r"(?u)\b\w\w+\b",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    class_weight=class_weight,
                    solver="lbfgs",
                    max_iter=200,
                    random_state=seed,
                ),
            ),
        ]
    )
