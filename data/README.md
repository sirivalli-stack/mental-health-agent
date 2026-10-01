# Data directory

**No dataset here originates from the two project papers.**

- **Paper 1** (systematic review / meta-analysis) provides **no dataset** — it is
  evidence synthesis only. Its own data availability statement is
  "available from the corresponding author on reasonable request".
- **Paper 2** (MHAI) publishes **evaluation-only** data (D4): 816 rated
  interaction pairs. It is not a training corpus.

## Layout

| Path | Contents |
|---|---|
| `raw/` | Hugging Face cache, figshare download. Git-ignored. |
| `raw/hf_cache/` | `datasets` library cache |
| `raw/mhai_figshare/` | `0. Database_overall.xlsx` + `Analysis.py` (Paper 2, CC BY 4.0) |
| `processed/dataset_manifest.json` | Row counts, label distributions, licenses (regenerable) |

Regenerate everything:

```bash
python evaluation/scripts/download_datasets.py
```

## Registered datasets

All counts below were **measured locally** during Phase 2, not copied from
any paper.

| ID | HF / DOI id | Task | Labels | Splits (rows) | License | Role |
|---|---|---|---|---|---|---|
| **D1** | `cardiffnlp/tweet_eval` (`sentiment`) | 3-class sentiment | negative / neutral / positive | train **45,615** · validation **2,000** · test **12,284** (total 59,899) | HF metadata: **unknown** — cite Barbieri et al., arXiv:2010.12421; do not redistribute raw data | train/eval |
| **D2** | `google-research-datasets/go_emotions` (`simplified`) | emotion classification, single-label | 28 classes (27 emotions + `neutral`) | train **36,308** · validation **4,548** · test **4,590** (total 45,446) | **Apache-2.0** | train/eval |
| **D3** | `vibhorag101/suicide_prediction_dataset_phr` | binary risk detection | suicide / non-suicide | train **185,574** · test **46,394** (total 231,968) | **MIT** | train/eval |
| **D4** | `10.6084/m9.figshare.29606618.v1` | none — rated interaction pairs | 8 Likert criteria (1–5) | **816** rows × 22 columns | **CC BY 4.0** (attribution required) | evaluation only |

### D1 — sentiment
- Load: `load_dataset("cardiffnlp/tweet_eval", "sentiment")`.
- **Must use the namespaced id.** The bare id `tweet_eval` raises
  `HfUriError` under `datasets` 3.x.
- Class distribution (train): neutral 20,673 · positive 17,849 · negative 7,093.
  **Negative is only 15.5 %** → the split is imbalanced; report macro-F1, not
  accuracy alone.
- Same corpus backs `cardiffnlp/twitter-roberta-base-sentiment-latest`, so the
  pretrained model's label space matches ours exactly.

### D2 — emotion
- Load: `load_dataset("google-research-datasets/go_emotions", "simplified")`.
- The `simplified` config stores `labels` as `Sequence(ClassLabel)` of integer
  ids. **8,817 rows (16.4 %) carry more than one label** and are dropped so
  Phase 4 trains a clean single-label classifier:

  | split | before | after | dropped |
  |---|---|---|---|
  | train | 43,410 | 36,308 | 7,102 |
  | validation | 5,426 | 4,548 | 878 |
  | test | 5,427 | 4,590 | 837 |

  The drop is recorded in `dataset_manifest.json` under
  `datasets.D2.extra.splits`, never silently discarded.
- Strongly dominated by `neutral` (12,823 / 36,308 = 35 % in train).
- **Label space → 7-class Ekman grouping.** D2 ships 28 fine labels (27
  emotions + neutral), while `schemas.EmotionLabel` and the configured
  pretrained model both use Ekman's 6 + neutral = 7. We do not invent that
  grouping: it is copied verbatim from the dataset authors' own
  `goemotions/data/ekman_mapping.json`
  (Demszky et al., ACL 2020, arXiv:2005.00547), with `neutral` kept as its
  own class. The table lives in `backend/app/training/emotion_labels.py`
  and is validated against the registry at import time. This is a
  coarse-graining of a research dataset for a non-clinical prototype, not a
  clinical taxonomy.

### D3 — risk / safety
- Load: `load_dataset("vibhorag101/suicide_prediction_dataset_phr")`.
- Balanced binary: train 92,889 suicide / 92,685 non-suicide.
- **Binary labels only.** Our `low / moderate / high / critical` scale is
  **our own design**, produced by combining classifier probability,
  deterministic rule hits and conversation trajectory (Phase 5). It is not
  taken from this dataset and must never be presented as a validated
  clinical scale.
- The text is **pre-cleaned by the dataset authors** (lowercased, URLs/emoji
  removed, contractions expanded, lemmatised, stopwords dropped except
  "not"). Our `normalise_tweet` step is therefore close to a no-op here; it
  is kept only so train-time and serve-time preprocessing stay identical
  across Phases 3–5.
- **Provenance caveat for the pretrained baseline.**
  `vibhorag101/roberta-base-suicide-prediction-phr` was fine-tuned on *this
  same dataset* (model card: 80:10:10 train/test/val, reported F1 ≈ 0.965).
  We cannot verify whether its evaluation rows coincide with D3's test split,
  so its measured score may be optimistic; this is recorded in
  `evaluation/results/risk_metrics.json` under `decision.notes`.

### D4 — Paper 2 evaluation reference
- Downloaded from figshare article `29606618`; the file is named
  **`0. Database_overall.xlsx`** (numeric prefix) — match by extension.
- 816 rows × 22 columns, which independently reproduces Paper 2's reported
  sample of 816 interaction pairs.
- Columns: `ID_S1`, `ID_S2`, `Question (User)`, `Answer (Bot)`, `Researcher`,
  `Model`, `Response Length`, `Lexical Diversity`, `Prompt Tokens`,
  `Completion Tokens`, `Input Cost ($)`, `Output Cost ($)`, `User persona`,
  `English/Spanish`, and the eight criteria
  `Tone`, `Clarity`, `Domain Accuracy (Correctness)`, `Robustness`,
  `Completeness`, `Boundaries`, `Target Language`, `Safety`.
- Used to adopt Paper 2's rubric and to sanity-check our own scoring
  pipeline. **Never a training set, never our results.**

## Rejected candidates

| Candidate | Why rejected |
|---|---|
| `Amod/mental_health_counseling_conversations` | **Gated** — HTTP 401 without Hugging Face authentication, fails the "directly downloadable" requirement; also reference-only |
| `alexandreteles/mental-health-conversational-data` | Ungated but 74–105 MB for a reference-only role |
| DAIC-WOZ, CLPsych (Dreaddit, SMHD) | Require application + data-use agreements; not downloadable programmatically |

## Rules

1. Never attribute any of these datasets to Paper 1 or Paper 2.
2. Record license + citation alongside any artifact derived from them.
3. Do not commit `raw/` (git-ignored); `processed/dataset_manifest.json` is
   regenerable and safe to commit.
4. Synthetic test/dialogue material must contain **no real individuals'
   mental-health information**.
