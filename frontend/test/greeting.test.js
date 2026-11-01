import { describe, expect, it } from 'vitest'
import {
  firstName,
  formatDate,
  formatClock,
  greetingFor,
  greetingMessage,
  initials,
  relativeTime,
} from '../src/greeting.js'

const at = (h, m = 0) => new Date(2026, 2, 10, h, m) // fixed local date

describe('greetingFor', () => {
  it('uses the local hour of day', () => {
    expect(greetingFor(at(7))).toBe('Good morning')
    expect(greetingFor(at(12))).toBe('Good afternoon')
    expect(greetingFor(at(19))).toBe('Good evening')
    expect(greetingFor(at(3))).toBe('Good evening')
  })
})

describe('greetingMessage', () => {
  it('greets by first name, never hard-coded', () => {
    expect(greetingMessage('Alex Rivera', at(9))).toBe('Good morning, Alex.')
    expect(greetingMessage('  sam ', at(9))).toBe('Good morning, sam.')
    expect(greetingMessage('', at(9))).toBe('Good morning.')
    expect(greetingMessage(null, at(15))).toBe('Good afternoon.')
  })

  it('firstName takes only the first token', () => {
    expect(firstName('Alex Rivera')).toBe('Alex')
    expect(firstName('')).toBe('')
  })
})

describe('relativeTime', () => {
  const now = at(12)
  it('labels recency in growing buckets', () => {
    expect(relativeTime(new Date(at(12, 0) - 10_000).toISOString(), now)).toBe('just now')
    expect(relativeTime(new Date(at(12, 0) - 12 * 60_000).toISOString(), now)).toBe('12m ago')
    expect(relativeTime(new Date(at(9).getTime()).toISOString(), now)).toBe('3h ago')
    expect(relativeTime(new Date(at(12, 0) - 26 * 3600_000).toISOString(), now)).toBe('Yesterday')
    expect(relativeTime(new Date(at(12, 0) - 3 * 86_400_000).toISOString(), now)).toBe('3d ago')
  })

  it('falls back to an absolute date beyond a week', () => {
    const old = new Date(2026, 0, 2, 10).toISOString()
    expect(relativeTime(old, now)).toBe('2 Jan')
  })

  it('returns empty string for invalid input', () => {
    expect(relativeTime('not-a-date', now)).toBe('')
  })
})

describe('formatClock / formatDate / initials', () => {
  it('formats a clock time and a date', () => {
    expect(formatClock(at(14, 5).toISOString())).toMatch(/14:05|02:05/)
    expect(formatDate(new Date(2026, 2, 10).toISOString())).toMatch(/10/)
    expect(formatClock('bad')).toBe('')
  })

  it('builds avatar initials', () => {
    expect(initials('Alex Rivera')).toBe('AR')
    expect(initials('sam')).toBe('S')
    expect(initials('')).toBe('?')
  })
})
