import { describe, expect, it } from 'vitest'
import {
  buildChatPayload,
  healthChips,
  riskTone,
  sessionShort,
  sourceLabel,
  sourceTone,
  stateFacts,
  toExchange,
  trendText,
} from '../src/format.js'

describe('sourceLabel / sourceTone', () => {
  it('maps every Phase 12 source', () => {
    expect(sourceLabel('llm')).toBe('Model reply')
    expect(sourceLabel('pre_blocked')).toBe('Blocked by input gate')
    expect(sourceLabel('post_fallback')).toBe('Replaced by output gate')
    expect(sourceLabel('llm_unavailable')).toBe('Model unavailable')
  })

  it('falls back for unknown values', () => {
    expect(sourceLabel(undefined)).toBe('unknown')
    expect(sourceLabel('weird')).toBe('weird')
    expect(sourceTone('weird')).toBe('tone-unknown')
  })
})

describe('risk + trend presentation', () => {
  it('maps the four-level scale to tones', () => {
    expect(riskTone('low')).toBe('risk-low')
    expect(riskTone('critical')).toBe('risk-critical')
    expect(riskTone('nonsense')).toBe('risk-unknown')
  })

  it('never renders the raw enum for unknown trends', () => {
    expect(trendText('unknown')).toBe('not enough turns')
    expect(trendText('worsening')).toBe('worsening')
    expect(trendText(undefined)).toBe('unknown')
  })
})

describe('buildChatPayload', () => {
  it('omits session_id entirely so the backend starts a new session', () => {
    const payload = buildChatPayload({ message: 'hi', sessionId: null })
    expect(payload).toEqual({ message: 'hi' })
    expect('session_id' in payload).toBe(false)
  })

  it('includes session_id and profile only when provided', () => {
    const payload = buildChatPayload({
      message: 'hi',
      sessionId: 'abc123',
      profile: 'B',
    })
    expect(payload).toEqual({ message: 'hi', session_id: 'abc123', profile: 'B' })
  })
})

describe('toExchange', () => {
  const response = {
    reply: 'I hear you.',
    source: 'llm',
    turn_index: 2,
    latency_ms: 1234,
    state: { risk: { level: 'high' } },
    post_safety: { guardrail_label: 'safe' },
  }

  it('projects a ChatResponse onto an assistant bubble', () => {
    const bubble = toExchange(response)
    expect(bubble.role).toBe('assistant')
    expect(bubble.text).toBe('I hear you.')
    expect(bubble.source).toBe('llm')
    expect(bubble.risk).toBe('high')
    expect(bubble.turnIndex).toBe(2)
    expect(bubble.guardrail).toBe('safe')
    expect(bubble.id).toBeTruthy()
  })

  it('tolerates a response without state (pre_blocked still has state, but be safe)', () => {
    const bubble = toExchange({ reply: 'fixed', source: 'llm_unavailable', turn_index: 0 })
    expect(bubble.risk).toBeNull()
    expect(bubble.latencyMs).toBeNull()
  })
})

describe('stateFacts', () => {
  it('returns null without state', () => {
    expect(stateFacts(null)).toBeNull()
  })

  it('flattens the UserState the backend sends', () => {
    const facts = stateFacts({
      turn_index: 3,
      sentiment: { label: 'negative', confidence: 0.8 },
      emotion: { label: 'sadness', confidence: 0.7 },
      risk: { level: 'critical', rule_hits: ['hopelessness'] },
      risk_trend: 'worsening',
      emotional_trend: 'improving',
      previous_risk: 'high',
      profile: { name: 'Priya', communication_style: 'brief', topics_to_avoid: ['exams'] },
    })
    expect(facts.turn).toBe(3)
    expect(facts.sentiment).toBe('negative')
    expect(facts.emotion).toBe('sadness')
    expect(facts.risk).toBe('critical')
    expect(facts.ruleHits).toEqual(['hopelessness'])
    expect(facts.riskTrend).toBe('worsening')
    expect(facts.emotionalTrend).toBe('improving')
    expect(facts.learnedName).toBe('Priya')
    expect(facts.style).toBe('brief')
    expect(facts.topicsToAvoid).toEqual(['exams'])
  })
})

describe('sessionShort + healthChips', () => {
  it('shows a short id and a placeholder for new sessions', () => {
    expect(sessionShort('21a3e875a04b4352')).toBe('21a3e875')
    expect(sessionShort(null)).toBe('new session')
  })

  it('keeps only the components the header displays', () => {
    const chips = healthChips([
      { name: 'config', loaded: true },
      { name: 'llm', loaded: true },
      { name: 'safety_pre', loaded: true },
      { name: 'safety_post', loaded: false },
    ])
    expect(chips.map((c) => c.name)).toEqual(['llm', 'safety_pre', 'safety_post'])
    expect(chips[2].ok).toBe(false)
    expect(chips[2].text).toBe('safety post: off')
  })
})
