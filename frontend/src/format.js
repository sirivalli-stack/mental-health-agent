/**
 * Pure presentation helpers - no React, no network, fully unit-testable.
 *
 * Every mapping here mirrors a backend contract from Phase 12:
 *   ChatResponse.source, RiskLevel, Trend, UserState.
 */

export const SOURCE_LABELS = {
  llm: 'Model reply',
  pre_blocked: 'Blocked by input gate',
  post_fallback: 'Replaced by output gate',
  llm_unavailable: 'Model unavailable',
}

export const SOURCE_TONES = {
  llm: 'tone-llm',
  pre_blocked: 'tone-blocked',
  post_fallback: 'tone-fallback',
  llm_unavailable: 'tone-unavailable',
}

export const RISK_TONES = {
  low: 'risk-low',
  moderate: 'risk-moderate',
  high: 'risk-high',
  critical: 'risk-critical',
}

const TREND_TEXT = {
  improving: 'improving',
  stable: 'stable',
  worsening: 'worsening',
  mixed: 'mixed',
  unknown: 'not enough turns',
}

export function sourceLabel(source) {
  return SOURCE_LABELS[source] ?? String(source ?? 'unknown')
}

export function sourceTone(source) {
  return SOURCE_TONES[source] ?? 'tone-unknown'
}

export function riskTone(level) {
  return RISK_TONES[level] ?? 'risk-unknown'
}

export function trendText(trend) {
  return TREND_TEXT[trend] ?? String(trend ?? 'unknown')
}

export function sessionShort(sessionId) {
  if (!sessionId) return 'new session'
  return String(sessionId).slice(0, 8)
}

/**
 * Payload for POST /api/chat. The backend starts a *new* session when
 * `session_id` is absent, so the key must be omitted - not null.
 */
export function buildChatPayload({ message, sessionId, profile }) {
  const payload = { message }
  if (sessionId) payload.session_id = sessionId
  if (profile) payload.profile = profile
  return payload
}

/** ChatResponse -> the assistant bubble the UI renders. */
export function toExchange(response) {
  const state = response.state ?? null
  return {
    id: globalThis.crypto?.randomUUID?.() ?? `id-${Date.now()}-${Math.random()}`,
    role: 'assistant',
    text: response.reply,
    source: response.source,
    risk: state?.risk?.level ?? null,
    turnIndex: response.turn_index,
    latencyMs: response.latency_ms ?? null,
    guardrail: response.post_safety?.guardrail_label ?? null,
  }
}

/** UserState -> flat facts for the state panel (Phase 6's artefact). */
export function stateFacts(state) {
  if (!state) return null
  const name = state.profile?.name
  const style = state.profile?.communication_style
  const avoid = state.profile?.topics_to_avoid ?? []
  return {
    turn: state.turn_index,
    sentiment: state.sentiment?.label ?? '?',
    sentimentConfidence: state.sentiment?.confidence ?? null,
    emotion: state.emotion?.label ?? '?',
    emotionConfidence: state.emotion?.confidence ?? null,
    risk: state.risk?.level ?? '?',
    ruleHits: state.risk?.rule_hits ?? [],
    riskTrend: trendText(state.risk_trend),
    emotionalTrend: trendText(state.emotional_trend),
    previousRisk: state.previous_risk ?? null,
    learnedName: name ?? null,
    style: style ?? null,
    topicsToAvoid: avoid,
  }
}

/** /health -> compact chips for the header. */
export function healthChips(components = []) {
  const wanted = ['llm', 'safety_pre', 'safety_post']
  return components
    .filter((c) => wanted.includes(c.name))
    .map((c) => ({
      name: c.name,
      ok: Boolean(c.loaded),
      text: `${c.name.replace('_', ' ')}: ${c.loaded ? 'on' : 'off'}`,
    }))
}
