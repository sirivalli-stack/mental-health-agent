# mental-health-agent

**Working title:** *Effectiveness of AI-Driven Conversational Agents in Improving
Mental Health Among Young People: Systematic Review and Meta-Analysis* —
implemented component: an AI-driven, state-aware conversational support
prototype.

> **Disclaimer.** This is a research prototype for an academic project. It is
> **not** a medical device, **not** a diagnostic system, and **not** a
> replacement for a mental-health professional. It must not be used to claim
> clinical efficacy.

## What it is

A modular backend that analyses each user message for **sentiment, emotion and
risk/safety signals**, fuses them into an explicit **user state**, tracks the
**emotional and risk trajectory** across a conversation, applies **memory and
personalization**, and conditions an **LLM** to respond — with **safety checks
before and after generation**.

The central research question is whether explicit user-state information
improves response relevance, emotional appropriateness, context awareness,
personalization and safety **compared with an LLM-only baseline**, measured by
ablation across four system profiles.

## System profiles

| Profile | Pipeline |
|---|---|
| A | LLM only |
| B | + sentiment + emotion |
| C | + risk, post-generation safety |
| D | full state-aware system (default) |

## Stack

| Layer | Choice | Purpose |
|---|---|---|
| API | Python 3.12 + FastAPI | REST service, validation, docs |
| ML | PyTorch + Hugging Face Transformers | sentiment / emotion / risk classifiers |
| LLM | Llama 3.1 8B Instruct (local, Ollama) | response generation |
| DB | SQLite + SQLAlchemy | sessions, messages, state history |
| Frontend | React | chat interface |

## Layout

```
mental-health-agent/
├── backend/          FastAPI app (app/api, models, services, pipelines,
│                     safety, memory, personalization, state_engine, config)
├── frontend/         React chat UI
├── data/             downloads + processed artifacts (see data/README.md)
├── models/           saved model weights
├── evaluation/       datasets, scripts, results, reports
├── notebooks/
├── docs/             architecture and design decisions
├── .env.example
└── requirements.txt
```

## Quick start

```bash
# 1. configure
copy .env.example .env          # Windows
# cp .env.example .env          # macOS/Linux

# 2. dependencies
pip install -r backend/requirements.txt

# 3. tests (from the repository root)
python -m pytest backend/tests -q
# frontend unit tests
cd frontend && npm install && npm test && cd ..

# line coverage - must run from backend/ (json report lands in evaluation/results/)
cd backend
python -m pytest tests/ -q --cov=app --cov=main --cov-report=term-missing --cov-report=json:../evaluation/results/coverage.json
cd ..

# 4. start the API
cd backend
uvicorn main:app --reload --host 127.0.0.1 --port 8000
# open http://127.0.0.1:8000/docs  (frontend dev server: cd frontend && npm run dev)

# 5. reproduce the Phase 18 experiment (needs Ollama + llama3.1:8b-instruct-q4_K_M)
#    dry run first - outputs are marked "NOT RESULTS" (disposable):
python evaluation/scripts/run_ablation.py --fake-llm --run-id smoke --judge --out-dir evaluation/results/ablation_runs_smoke
#    real run (generation, then re-score the archived run if needed):
python evaluation/scripts/run_ablation.py --run-id phase18 --profiles A,B,C,D
python evaluation/scripts/run_ablation.py --run-id phase18 --judge-only
```

## Status

Phase 1 complete: project skeleton, configuration, structured logging,
`UserState` schema draft, `/health` endpoint, test suite.

Phase 2 complete: dataset registry and loaders (D1–D4), download script,
verified manifest, rejected-candidate log.

Phase 3 complete: sentiment module — a TF-IDF + logistic-regression model
trained on D1 and the pretrained RoBERTa backend, benchmarked on the *same*
held-out test rows; `SentimentService`, `/health` registration, 73 tests.
Results: `evaluation/results/sentiment_metrics.json`.

Phase 4 complete: emotion module — D2's 28 fine labels grouped into the
7-class Ekman space with the **GoEmotions authors' own published mapping**
(Demszky et al., ACL 2020), then trained and benchmarked against the
pretrained `j-hartmann` model on the same test rows. `EmotionService`,
`/health` registration, 127 tests. Results:
`evaluation/results/emotion_metrics.json`.

Phase 5 complete: risk/safety module — D3's binary labels, our own four-level
`low / moderate / high / critical` scale built from probability thresholds +
a deterministic rule layer (`backend/app/safety/rules.py`) + conversation
trajectory, plus `RiskService` and `/health` registration. Benchmarked against
the pretrained `vibhorag101` model and a rules-only baseline on the same test
rows. 190 tests. Results: `evaluation/results/risk_metrics.json`.
The four-level scale is a prototype heuristic, **not** a clinical instrument.

Phase 6 complete: user state engine — `backend/app/state_engine/` assembles the
`UserState` the LLM receives: turn-local component outputs, `previous_*` labels,
a `short_term_window` of recent turns, and two documented trends (a least-squares
direction over valence/risk series, `unknown` below three turns) whose pre-turn
value feeds Phase 5's trajectory fusion. Sessions live in a capacity-bounded
in-memory store (`MAX_SESSIONS`), `/health` reports it, 223 tests.

Phase 7 complete: conversation memory — `backend/app/memory/` persists every
turn as an atomic per-session JSON snapshot under `data/memory`
(`MEMORY_BACKEND=json|none`), pruned to `MAX_SESSIONS`, with a six-operation
`MemoryStore` interface (its SQLite implementation landed in Phase 14). The state
engine auto-recalls a session after a restart, so turn numbering, `previous_*`
labels and trends continue instead of resetting. 243 tests.

Phase 8 complete: personalization — `backend/app/personalization/` learns only
what the user explicitly states (name, language, communication style,
preferences, topics to avoid) through a conservative rule extractor;
ambiguity yields *no* update rather than a guess. Lists are de-duplicated and
capped by `PROFILE_MAX_ITEMS` (0 disables them — an ablation knob), and
`render_profile` produces the factual block the Phase 9 prompt carries.
The extractor sits behind a `ProfileExtractor` protocol, so an LLM extractor
can replace it. 304 tests.

Phase 9 complete: LLM generation — `backend/app/services/llm_service.py` is a
thin OpenAI-compatible client (Ollama, `llama3.1:8b-instruct-q4_K_M`, no
fine-tuning and no RAG, matching Paper 2's baseline) with typed failures
(`LLMUnavailableError` / `LLMProtocolError`) instead of half-answers, plus the
state-aware prompt: one system prompt whose state block is strictly
profile-dependent (A none, B +sentiment/emotion, C +risk/rules, D +trends,
previous labels, recent messages, profile) and always labelled
machine-generated. The disclaimer and the "your local emergency number —
never a specific number" rule are asserted in tests. `/health` now reports an
`llm` component via a 2 s probe. 333 tests.

Phase 10 complete: pre-generation safety — `backend/app/safety/pre_generation.py`
inspects each message *before* the LLM call and returns
`allow / flag / block` with stable reason ids. Crisis (Phase 5's rule set plus
the fused risk level) and prompt-injection attempts are **flagged, never
blocked**: they append a `CRISIS CONTEXT` / `INSTRUCTION INTEGRITY` note to the
system prompt instead. Blocking is reserved for empty/over-length messages and
for requests for emergency phone numbers, which are answered by a deterministic,
locale-free reply without calling the model (prohibited claim #11). Decisions
log only reason ids and message length — never the message text. The content
gates ablate with `PRE_SAFETY_ENABLED=false` (validation stays on);
`/health` reports `safety_pre` (and, from Phase 11 on, `safety_post`).
365 tests.

Phase 11 complete: post-generation safety — `backend/app/safety/post_generation.py`
inspects each *drafted reply* in two layers. Layer 1 is deterministic and runs
without any model: non-empty, no phone number or emergency short code
(prohibited claim #11), no diagnosis claim, no medication/dosage advice, no
guilt, no self-harm encouragement — each violation swaps in a deterministic,
number-free fallback reply. Layer 2 is `mila-ai4h/Mila-Suicide-Prevention-Output-Guardrail`
(BERT, Apache-2.0): its card states input = **assistant replies**, `0` = safe,
`1` = suicide/self-harm violation; the config ships only `LABEL_0/LABEL_1`, so
the mapping was confirmed empirically (supportive → 0 at p ≥ 0.88,
endorsement → 1 at p = 1.00). The model is deliberately narrow (non-SH insults
score safe — verified), which is why both layers exist; unavailability
degrades to layer 1 only, with a logged `guardrail_unavailable` reason.
Ablations: `POST_SAFETY_ENABLED` / `POST_SAFETY_GUARDRAIL`,
threshold `POST_SAFETY_THRESHOLD=0.5` (the card's default operating point).
`/health` reports `safety_post`. 389 tests.

Phase 12 complete: API + pipeline — `backend/app/pipelines/chat.py` is the one
request path (`state -> pre-gate -> LLM -> post-gate`), and
`backend/app/api/chat.py` exposes it: `POST /api/chat` (new session when
`session_id` is omitted, optional per-request `profile` for ablations),
`GET/DELETE /api/sessions/{id}`. Responses carry the full `UserState`,
both safety decisions (`SafetyDecision` now includes guardrail label/score),
a `source` field (`llm` / `pre_blocked` / `post_fallback` /
`llm_unavailable`) and the disclaimer. Blocked turns skip the model and the
guardrail; LLM failures degrade to a deterministic reply instead of a 500;
decisions log reason ids and lengths only. Tests also surfaced and fixed a
Phase 1 bug: pydantic validator errors put raw exceptions in `ctx`, which
broke the JSON 422 handler. 412 tests.

Phase 13 complete: frontend — `frontend/` is a Vite 7 + React 18 chat UI
(JSX, deliberately no TypeScript: the app is one screen and the logic worth
testing lives in pure modules). `src/format.js` (source badges, risk tones,
trend wording, payload builder, `UserState` → panel facts) and `src/api.js`
(a thin `/api` + `/health` client with typed error surfacing) are covered by
17 vitest tests; components render only what the backend sends — the UI never
computes risk or sentiment client-side. The screen shows the transcript with
per-turn source badges/risk chips/latency, a composer, the live `UserState`
and learned profile side panel, an A–D profile selector, session id +
New Session (`DELETE /api/sessions/{id}`), component chips and the disclaimer
fetched from `/health`. Dev and preview share a same-origin proxy to
`127.0.0.1:8000`, so no CORS exists anywhere. `npm test` → 17 passed,
`npm run build` → 49 kB gzip. Live smoke: `GET /` 200, `GET /health` 200 via
proxy, `POST /api/chat` 200 (cold Ollama once hit the 60 s timeout and
degraded to `llm_unavailable` as designed; a warm retry confirmed
`source=llm`, guardrail `safe`, 68 s cold start), whitespace → 422.

Phase 14 complete: database — `backend/app/models/db.py` adds
`SQLiteMemoryStore`, a drop-in third `MemoryStore` backend
(`MEMORY_BACKEND=sqlite`; file `DATABASE_PATH`, default `data/sessions.db`).
It keeps the six Phase 7 operations, the exact list ordering (newest first,
session id descending on ties), the `MAX_SESSIONS` pruning and the
corrupt/stale-payload degradation of the JSON backend — a parametrised
conformance suite runs every contract test against **both** backends. Rows
store the validated snapshot JSON plus denormalised
`created_at/updated_at/turn_count/risk_level` columns so Phases 17-18 can
query sessions without parsing JSON. One locked connection (FastAPI's
threadpool), default journal mode — no WAL (nothing to gain with a single
serialized connection, and no `-wal`/`-shm` sidecars in a synced folder);
`PRAGMA user_version` guards the schema and a non-database file fails fast
with a clear error at startup. `.env`/`.env.example` document
`DATABASE_PATH`; `.gitignore` covers `data/*.db*`. One Phase 7 test changed
(`sqlite` is now valid, `postgres` stays the invalid value). 441 tests.
Live proof: fresh process → one turn → **restart** → `/health` still reports
`backend=sqlite sessions=1` → resumed chat turn returns `turn_index=1` with
`previous_risk` intact → `DELETE` → `sessions=0`.

Phase 15 complete: integration — every chat turn now leaves one
**metadata-only audit row** (`backend/app/pipelines/turn_log.py`, table
`turns` in the same SQLite file): profile, source, risk/sentiment/emotion
labels, safety reason ids, character counts, latency, timestamp — never the
message or reply text (the same "ids + lengths only" posture as the
structured logs since Phase 10). Rows are bounded by `TURN_LOG_MAX_ROWS`,
switched off wholesale with `TURN_LOG_ENABLED=false` (no file is even
opened), written best-effort (a failed row never breaks a turn), and purged
by `DELETE /api/sessions/{id}` so "forget the session" covers the audit
trail too. `run_conversation()` (`pipelines/conversation.py`) drives a
scripted message list through the identical pipeline without HTTP — the
surface Phases 17-18 will reuse for ablations. Lifespan shutdown closes the
SQLite handles (`close_state_engine()`, `reset_turn_log()`); `/health`
gained a `turn_log` component (config-only probe that never creates the
file); the schema moved to `user_version=2` with an additive v1→v2
migration. 460 tests. Live: 3 HTTP turns → `rows=3` → DELETE → `rows=2` →
direct file read shows the 17 metadata columns and **no text** →
`run_conversation` with the real LLM appends 2 more rows (guardrail `safe`).

Phase 16 complete: testing — `backend/tests/fakes.py` centralises the shared
test doubles (`FakeLLM`, `FakeGuardrail`, `fast_post_gate`) and environment
helpers (`audit_env`, `api_client`) reused by the pipeline/API suites, and
`test_phase16.py` adds the cross-module gap suite: an end-to-end journey
(conversation → process restart → snapshot recall → audit rows →
`sessions.turn_count`), 8 threads sharing one engine and one audit database,
hostile input through the API (prompt injection flagged, never refused;
SQL-injection / path-traversal / padded session ids survive and leave both
tables intact), 422 responses and structured logs asserted to never echo
message or reply text, and the `/health` contract (all 14 components).
`pytest-cov` (now in `backend/requirements.txt`) measured **469 tests, 93%
line coverage** of `backend/app` + `main.py` — 23 of 45 modules at 100%,
residual gaps are health/loader/training error branches — written to
`evaluation/results/coverage.json` and `coverage.txt`.

Phase 17 complete: evaluation harness — `evaluation/` now holds the fixed
input set (`datasets/scenarios.json`: 6 fictional English personas, 19 turns
covering crisis, emergency-number request, subclinical, protective, off-topic
and prompt-injection paths, no phone numbers), the runner/scorer
(`scripts/harness.py`: per-profile JSONL of each (user, reply) pair plus a
manifest with the scenario hash; deterministic metrics for safety
activation on `high_risk` turns, benign false flags, layer-1 reply
violations, latency percentiles and system-prompt size per profile;
arithmetic deltas vs the profile-A baseline; a paired Wilcoxon helper), the
8-criterion LLM-as-judge (`scripts/judge.py`: criteria and the 1–5 anchors
transcribed verbatim from Paper 2 Table 1, single-judge limitation
documented — no inter-rater reliability, not Paper 2's mixed-effects), and
the CLI (`scripts/run_ablation.py`, `--fake-llm` dry run whose outputs are
marked `NOT RESULTS`; a warm-up turn keeps latency comparisons fair).
Before experimentation this read **"Results will be generated after
experimentation."** — executed in Phase 18; 486 tests at this point.

Phase 18 complete: experiment — one real run (`run_id=phase18`, 76 records =
19 turns × A–D, `llama3.1:8b-instruct-q4_K_M`, generation temperature 0.4,
judge 0.0, warm-up turn) archived under
`evaluation/results/ablation_runs/phase18/` (per-profile JSONL, manifest with
scenario sha256, `scores.json`, `judge_scores.json`, `judge_summary.json`,
`analysis.json` from `scripts/analyze_ablation.py`, paired Wilcoxon vs the A
baseline). **Simulated evaluation only** (prohibited claim #10): one
non-human judge, 6 fictional scenarios, 74/76 turns rated (1 transient
Ollama error, 1 unparseable judge reply), no inter-rater reliability and no
mixed-effects model — nothing here is a clinical finding. Measured: mean
overall judge score A 4.43, B 4.45, C 4.79, D 4.51; paired scenario-level
deltas vs A = B +0.05 (p=0.56), C +0.43 (p=0.06), D +0.11 (p=0.31) — none
significant at n=6, C largest and closest; `target_language`/`safety`
≈5.0 everywhere. Deterministic metrics were identical across profiles *by
design* (the ablation varies state-in-prompt, not the shared analysis or
gates): high-risk activation 3/3, emergency-number block 1/profile, benign
flag rate 7/9 — the risk model over-flags benign text, a measured
limitation, not a claim of system harm — 1 post-guardrail fallback (B,
`persistent-low-mood` turn 2), 0 layer-1 violations in 76 replies.
Overhead (RQ3): system prompt A 1019 → B 1150 → C 1271 → D 1720 chars
(+701 full state); latency means (A 8.98 s, B 8.04 s, C 6.80 s, D 7.71 s)
were noisy single-pass measurements — no latency effect is claimed. 489 tests.

Phase 19 complete: documentation — final consistency pass. Quick start now
mirrors the verified commands (root-level pytest, the coverage run from
`backend/` whose JSON report lands in `evaluation/results/`, frontend
`npm test`, and the Phase 18 reproduction commands including the
disposable `--fake-llm` dry run); the two requirement manifests are back in
sync (`pytest-cov` and `SQLAlchemy` were each missing from one of them; the
stale deferred-matplotlib note was dropped — Phase 18 needed no plotting
library); `.gitignore` now keeps the measured evidence (metric/coverage
`*.json`/`*.txt` and the archived `evaluation/results/ablation_runs/`) while
still ignoring stray dumps there, and covers `.coverage`; the pre-experiment
sentence quoted in the Phase 17 entry is marked as executed in Phase 18.
Hand-off verification: **489 backend tests, 93% line coverage of
`backend/app` + `main.py` (3134 statements — `coverage.json`/`coverage.txt`
refreshed to this final state), 17 frontend tests**, `--judge-only` re-scored
the archived run, and the documented dry run was executed and cleaned up.
Reproduce coverage: the Quick start `--cov-report=json:` command regenerates
`coverage.json`; `coverage.txt` is the same data as a plain
`python -m coverage report` dump.

See `docs/architecture.md` for the full design, the research question and the
list of claims this project must not make.

## Sources

- **Paper A** — Hang et al., *Effectiveness of AI-driven Conversational Agents
  in Improving Mental Health Among Young People: A Systematic Review and
  Meta-analysis*, JMIR Preprint 69639 (preprint, under review).
- **Paper B** — Villarreal-Zegarra et al., *Development, system design, safety,
  and performance metrics of a conversational agent … The MHAI study*,
  PLOS One 2026;21(3):e0344939 (open access, CC BY).

Dataset provenance and licenses: `data/README.md`.
