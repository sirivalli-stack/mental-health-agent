# Architecture

**System:** AI-driven, state-aware conversational support research prototype.

> Research prototype only. Not a medical device, not a diagnostic system, and
> not a replacement for a mental-health professional.

---

## 1. Target pipeline

```
USER
  ↓
CHAT INTERFACE (React)
  ↓
REST API (FastAPI)  ←── System Profile: A | B | C | D      ← ablation switch
  ↓
TEXT PREPROCESSING (clean/normalize, preserve affect)
  ↓
┌──── USER STATE ANALYSIS ────┐
│ Sentiment   (Phase 3)       │
│ Emotion     (Phase 4)       │
│ Risk/Safety (Phase 5)       │  ML classifier + rule-based layer
└─────────────┬───────────────┘
              ↓
        USER STATE ENGINE (Phase 6)  → structured UserState
              ↓
        ┌── CONTEXT + MEMORY ──┐
        │ short-term  (Phase 7)│
        │ long-term profile    │
        │ state history        │
        └──────────┬───────────┘
              ↓
        PERSONALIZATION LAYER (Phase 8)   state → prompt blocks
              ↓
        LLM GENERATION (Phase 9)          Llama 3.1 8B Instruct
              ↓
        PRE-GENERATION SAFETY (Phase 10)  input gate + prompt constraints
              ↓
        POST-GENERATION SAFETY (Phase 11) guardrail + rules
              ├─ SAFE ──────────→ RESPONSE → USER
              └─ UNSAFE → REVISE → re-check → SAFE FALLBACK
```

Cross-cutting: `config/` (.env), structured logging, offline evaluation harness
(`evaluation/`), SQLite persistence (Phase 14), metadata-only turn audit
(Phase 15), shared test doubles + cross-module gap suite with line-coverage
reporting (Phase 16).

## 2. Ablation profiles

| Profile | Pipeline | State in prompt | Safety |
|---|---|---|---|
| **A** | msg → LLM → resp | none | none |
| **B** | msg → sentiment+emotion → LLM → resp | sentiment, emotion | none |
| **C** | msg → sentiment+emotion+risk → LLM → resp | + risk | post-generation only |
| **D** | msg → analysis → state engine → memory → personalization → LLM → pre/post safety → resp | full state + trajectory + profile | pre + post |

Selected at runtime via `SYSTEM_PROFILE` in `.env`. All other variables
(model, temperature, max tokens, input set) are held fixed across profiles.

## 3. Research question

> Does supplying explicit, structured user-state information (sentiment,
> emotion, risk level, emotional/risk trajectory, conversation context and
> personalization profile) to a large language model improve the relevance,
> emotional appropriateness, context awareness, personalization and safety of
> its responses, compared with an LLM-only baseline — under identical model,
> prompt budget and evaluation conditions?

**Status: first simulated investigation run (Phase 18: 1 non-human judge, 6
fictional scenarios, paired deltas vs baseline not significant at n=6 —
`evaluation/results/ablation_runs/phase18/`). Not an established finding;
real users and multiple human raters remain out of scope.**

Sub-questions: (RQ1) which state components contribute most; (RQ2) does state
tracking change safety-protocol activation under high-risk input; (RQ3) what
latency/token overhead does the full pipeline add.

## 4. Prohibited claims

Do **not** write any of the following in reports, slides or documentation:

1. "First system to combine context + memory + emotion + personalization + safety" — Paper 2 already implements personalization, prior-session storage, therapeutic modules and emergency safety.
2. Novelty for the sentiment + emotion + risk combination itself.
3. Novelty for using Llama 3.1 — Paper 2 already used it.
4. Any "dataset from the base paper" — Paper 1 has no dataset.
5. That the system treats, cures or diagnoses any condition.
6. That safety improves with stronger LLMs — Paper 2 found no safety difference (both means = 4.80).
7. That our system is "better" or "safer" before experiments exist.
8. That we conducted a meta-analysis or clinical trial.
9. Any number, effect size, participant count or accuracy not measured by us.
10. Real-patient or clinical evaluation — ours is **simulated evaluation**.
11. Emergency phone numbers unless the deployment locale is explicitly defined.
12. That Paper 1's final published values match the preprint (preprint status + internal inconsistencies → verify).

Before any experiment: **"Results will be generated after experimentation."**

## 5. Provenance of design decisions

| Decision | Source |
|---|---|
| Focus on youth, depression, subclinical early intervention | **A** — Paper 1 |
| Personalization / adaptivity as technical motivation | **A** — Paper 1 (earlier digital interventions "lack personalization") |
| Adjunct-not-standalone framing, no clinical claims | **A** — Paper 1; **B** — Paper 2 |
| frontend/backend/database layering | **B** — Paper 2 |
| Prior-session memory, preferences, progress tracking | **B** — Paper 2 ("personalized context window") |
| 8 Likert evaluation criteria + numeric metrics | **B** — Paper 2, Table 1 |
| Safety constraint set (no diagnosis, no medication/dosage, no guilt, no treatment-discontinuation, no self-harm encouragement) | **B** — Paper 2, Table 1 "Safety" |
| Simulated personas + offline evaluation of responses | **B** — Paper 2 |
| Llama 3.1-8B, no fine-tuning, no RAG as baseline | **B** — Paper 2 (validated open-source choice, >140× cheaper than GPT-4o) |
| Mixed-effects analysis adjusting for rater | **B** — Paper 2 |
| Explicit `UserState` object + trajectory + ablation A/B/C/D | **C** — ours |
| 7-class Ekman label space, grouped from GoEmotions' 28 fine labels | Demszky et al., ACL 2020 (`goemotions/data/ekman_mapping.json`) — the dataset authors' own mapping, not ours; matches the pretrained model's output space |
| Four-level risk scale | **C** — ours (public data is binary) |
| Risk fusion policy: threshold levels, rules only raise, no silent de-escalation without an `improving` trend | **C** — ours (documented in `backend/app/services/risk_service.py`) |
| Deterministic keyword rule layer | **C** — ours (hand-written patterns in `backend/app/safety/rules.py`; deliberately contains **no emergency phone numbers**, deployment locale is undefined) |
| Pre- and post-generation safety as separate ablatable stages | **C** — ours |
| Pre-generation gate: three actions (`allow` / `flag` / `block`) with stable reason ids; crisis (Phase 5 rules + fused risk level) and prompt injection are *flagged* (prompt note appended, model still runs — a person in crisis is never refused), while empty/over-length messages and emergency-number requests are *blocked* by a deterministic, phone-number-free reply; decisions log ids + length, never message text; `PRE_SAFETY_ENABLED=false` ablates the content gates (validation stays on) | **C** — ours (`backend/app/safety/pre_generation.py`; supports **B** Paper 2's Safety criteria) |
| Post-generation gate in two layers: deterministic reply checks (non-empty, no phone number, no diagnosis, no medication/dosage, no guilt, no self-harm encouragement → fallback) **plus** an output guardrail model as a second layer; guardrail failure degrades to layer 1 with a logged reason instead of failing the request | **C** — ours (`backend/app/safety/post_generation.py`) |
| Output guardrail choice: `mila-ai4h/Mila-Suicide-Prevention-Output-Guardrail` 0.1.0-beta.2 (BERT, Apache-2.0) — card specifies input = **assistant replies**, label `0` = safe / `1` = SH violation, policy aligned with MLCommons AILuminate "Suicide and Self-Harm"; card's own metrics (P 0.908 / R 0.802 / F1 0.852 at τ = 0.5, internal n = 494) are **quoted, not re-claimed**; label mapping verified empirically here; card's guidance to stack guardrails and not use it standalone is why layer 1 exists | **external, Apache-2.0** (model card, arXiv:2503.05731; constitution approach arXiv:2501.18837) + **C** — ours (verification in `backend/tests/test_phase11.py`) |
| Pipeline order: state update **first**, then the pre-gate (so it sees the fused risk level), then the LLM, then the post-gate; blocked turns still count as recorded turns but skip both the model and the guardrail; expected failures (LLM unreachable / malformed reply) become deterministic replies with a `source` field instead of HTTP 5xx | **C** — ours (`backend/app/pipelines/chat.py`) |
| Turn audit: one SQLite row per chat turn holding **metadata only** (profile, source, risk/sentiment/emotion labels, pre/post reason ids, guardrail label, character counts, latency, timestamp) — never message or reply text; bounded by `TURN_LOG_MAX_ROWS`, ablatable via `TURN_LOG_ENABLED` (disabled = no file opened), written best-effort so an audit failure can never fail a turn, and deleted together with its session by `DELETE /api/sessions/{id}` | **C** — ours (`backend/app/pipelines/turn_log.py`; extends the "ids + lengths only" logging posture of Phases 10-12; **supports [B]** Paper 2's prior-session storage / progress tracking) |
| Conversation runner without HTTP: `run_conversation()` feeds a scripted message list through the *same* `run_chat_turn` pipeline (one injected engine/LLM/gates/log, one outcome per message) — the integration surface for tests, demos and the Phase 17-18 ablation harness, guaranteeing experiments and the API cannot drift apart | **C** — ours (`backend/app/pipelines/conversation.py`) |
| Trend estimation over the session (least-squares slope + fixed tolerance; `unknown` below 3 turns; `mixed` when the newest step contradicts the window; sentiment/emotion → valence in [-1,1]) | **C** — ours (documented in `backend/app/state_engine/trends.py`; tolerances pre-set, never fitted) |
| Trajectory evidence for turn *N* is the trend of turns 1..N−1; the stored `risk_trend` summarises turns 1..N | **C** — ours (documented in `backend/app/state_engine/engine.py`) |
| Session store bounds: `MAX_SESSIONS` (LRU eviction), `MAX_STATE_HISTORY` (trend series), `SHORT_TERM_WINDOW` (context shown to the LLM) | **C** — ours (config in `.env`) |
| Conversation memory: pluggable `MemoryStore` (`none` / `json` / `sqlite`), one atomic JSON snapshot per session under `data/memory` **or** one SQLite row per session in `data/sessions.db` (validated payload JSON + denormalised `created_at`/`updated_at`/`turn_count`/`risk_level` columns so experiments can query sessions without parsing JSON), both pruned to `MAX_SESSIONS`, auto-recalled after a restart; identical ordering and corrupt/stale-payload behaviour across backends (parametrised conformance tests), `PRAGMA user_version` schema guard, fail-fast on a non-database file, single locked connection, no WAL (no sidecar files in a synced folder) | **C** — ours (`backend/app/memory/` + `backend/app/models/db.py`; the six operations are the contract every backend implements; no cross-user data, no embeddings) |
| Profile extraction: conservative English phrasings (`my name is …`, `keep it brief`, `don't mention …`, `i prefer …`); unmapped language names are dropped, fragments are length-capped, lists de-duplicated and capped by `PROFILE_MAX_ITEMS` (newest survive, `0` disables) | **C** — ours (`backend/app/personalization/extractor.py`; **supports [B]** Paper 2's "personalized context window", but no personal claim is inferred that the user did not make) |
| `render_profile` → compact factual prompt block; `ProfileExtractor` protocol so an LLM extractor can replace the rules (Phase 9 ships the rule extractor as default), `NullPersonalizer` to disable it | **C** — ours |
| System prompt = Paper 2's safety constraint set (verbatim boundaries) + a profile-dependent state block (A: none, B: +affect, C: +risk/rules, D: +trends, previous labels, recent messages, profile), each line labelled machine-generated; disclaimer appended; **no emergency phone number is ever emitted** (prohibited claim #11) | **B** — Paper 2, Table 1 "Safety"; **C** — ours (`backend/app/services/llm_service.py`, asserted in `backend/tests/test_phase9.py`) |
| LLM serving: Llama 3.1 8B Instruct, no fine-tuning, no RAG, called over an OpenAI-compatible endpoint (Ollama); failures are typed (`LLMUnavailableError` / `LLMProtocolError`) so the API layer can fall back rather than receive a half-answer | **B** — Paper 2 (validated open-source choice); **C** — ours (client + error taxonomy in `llm_service.py`) |
| FastAPI, PyTorch, HF Transformers, SQLite, React | **D** — general engineering |
| Chat UI: Vite 7 + React 18 (JSX, no TypeScript — one screen, and the testable logic is framework-free), dev/preview proxy `/api` + `/health` → `127.0.0.1:8000` so the browser sees a single origin (no CORS anywhere); the UI renders backend state verbatim (`UserState`, `source`, safety decisions) and derives nothing client-side | **D** — general engineering (`frontend/`; pure helpers in `src/format.js`, `src/api.js` covered by 17 vitest tests) |
| Test strategy: shared deterministic doubles for the LLM and the output guardrail (`backend/tests/fakes.py`) so the default pytest suite never touches the network, Ollama or the guardrail weights; cross-module gap suite (`test_phase16.py`: restart/recall journey, 8-thread concurrency, hostile input, log/422 privacy, `/health` contract, sessions↔turns invariant); harness/judge/analysis tests (`test_phase17.py`, `test_phase18.py`); line coverage measured with pytest-cov | **C** — ours (489 tests, 93% of `backend/app` + `main.py`; `evaluation/results/coverage.json`) |
| Offline evaluation: fixed fictional scenario set + deterministic metrics (safety activation, benign false flags, layer-1 reply violations, latency/prompt-size overhead) + an LLM judge rating each pair on **Paper 2's 8 Table 1 criteria, 1–5**; paired Wilcoxon for exploratory comparisons — a single non-human judge means no inter-rater reliability and no mixed-effects-for-rater (Paper 2 had four expert raters; stated as a limitation in every report) | **B** — Paper 2, Table 1 (DOI 10.1371/journal.pone.0344939); **C** — ours (`evaluation/scripts/`, Phase 17) |

## 6. Module map (phase → directory)

| Phase | Directory |
|---|---|
| 1 setup | `backend/app/config/`, `backend/app/models/` |
| 2 datasets | `data/`, `evaluation/datasets/` |
| 3 sentiment | `backend/app/services/sentiment_service.py`, `backend/app/training/` |
| 4 emotion | `backend/app/services/emotion_service.py`, `backend/app/training/emotion_*` |
| 5 risk | `backend/app/services/risk_service.py`, `backend/app/safety/rules.py` |
| 6 state engine | `backend/app/state_engine/` |
| 7 memory | `backend/app/memory/` |
| 8 personalization | `backend/app/personalization/` |
| 9 LLM | `backend/app/services/llm_service.py` |
| 10 pre-safety | `backend/app/safety/pre_generation.py` |
| 11 post-safety | `backend/app/safety/post_generation.py` |
| 12 API | `backend/app/api/`, `backend/app/pipelines/` |
| 13 frontend | `frontend/` |
| 14 database | `backend/app/models/db.py` |
| 15 integration | `backend/app/pipelines/` |
| 16 testing | `backend/tests/` |
| 17–18 experiments | `evaluation/` |
| 19 documentation | `docs/` |
