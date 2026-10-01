"""Split helpers shared by the training modules (Phases 3-5)."""

from __future__ import annotations


def stratified_head(
    texts: list[str], labels: list[str], limit: int | None
) -> tuple[list[str], list[str]]:
    """Take at most `limit` items, round-robin per class, keeping input order.

    Used by the `--limit` smoke path so a small run keeps the class mix instead
    of collapsing to whatever happens to be first in the file.
    """
    if limit is None or limit >= len(texts):
        return texts, labels

    by_class: dict[str, list[int]] = {}
    for i, lab in enumerate(labels):
        by_class.setdefault(lab, []).append(i)

    chosen: list[int] = []
    round_robin = 0
    while len(chosen) < limit:
        added = False
        for lab in sorted(by_class):
            bucket = by_class[lab]
            if round_robin < len(bucket):
                chosen.append(bucket[round_robin])
                added = True
                if len(chosen) >= limit:
                    break
        if not added:
            break
        round_robin += 1
    chosen.sort()
    return [texts[i] for i in chosen], [labels[i] for i in chosen]
