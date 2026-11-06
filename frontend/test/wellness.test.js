/**
 * Phase 7: wellness helper tests (mood + journal display logic).
 *
 * Pure functions only - no network, no React. These pin the contracts the
 * UI relies on: the seven spec moods, date arithmetic that never drifts
 * across DST/month boundaries, the weekly/monthly arrangements, and the
 * API row converters (server snake_case -> UI camelCase).
 */
import { describe, expect, it } from 'vitest'
import { toJournalEntry, toMoodEntry } from '../src/api.js'
import { parseRoute, routes } from '../src/router.js'
import {
  MOODS,
  MOOD_KEYS,
  addDays,
  entriesForDate,
  isMoodKey,
  lastNDays,
  moodsByDate,
  moodMeta,
  monthGrid,
  parseISO,
  todayISO,
  toISO,
} from '../src/wellness.js'

const SPEC_MOODS = [
  'happy',
  'calm',
  'sad',
  'anxious',
  'stressed',
  'angry',
  'overwhelmed',
]

describe('the seven moods', () => {
  it('are exactly the spec list, in order', () => {
    expect(MOOD_KEYS).toEqual(SPEC_MOODS)
    expect(MOODS).toHaveLength(7)
  })

  it('each has a label and a unique colour', () => {
    const dots = new Set()
    for (const m of MOODS) {
      expect(m.label).toBeTruthy()
      expect(m.dot).toMatch(/^#[0-9a-f]{6}$/i)
      expect(dots.has(m.dot)).toBe(false)
      dots.add(m.dot)
    }
  })

  it('lookups validate keys defensively', () => {
    expect(isMoodKey('calm')).toBe(true)
    expect(isMoodKey('depressed')).toBe(false) // never invent moods
    expect(isMoodKey(undefined)).toBe(false)
    expect(moodMeta('anxious')?.label).toBe('Anxious')
    expect(moodMeta('nope')).toBeNull()
  })
})

describe('date helpers', () => {
  it('formats today as YYYY-MM-DD', () => {
    expect(todayISO(new Date(2026, 10, 5))).toBe('2026-11-05')
    expect(todayISO(new Date(2026, 0, 3))).toBe('2026-01-03')
  })

  it('round-trips parse/format', () => {
    expect(toISO(parseISO('2026-11-05'))).toBe('2026-11-05')
    expect(parseISO('2026-11-05').getDate()).toBe(5)
  })

  it('adds days across month and year boundaries', () => {
    expect(addDays('2026-11-30', 1)).toBe('2026-12-01')
    expect(addDays('2026-01-01', -1)).toBe('2025-12-31')
    expect(addDays('2026-03-01', -1)).toBe('2026-02-28')
  })
})

describe('lastNDays', () => {
  it('returns 7 days, oldest first, ending today', () => {
    const week = lastNDays(7, new Date(2026, 10, 5)) // Thursday 2026-11-05
    expect(week).toHaveLength(7)
    expect(week[0].date).toBe('2026-10-30')
    expect(week[6].date).toBe('2026-11-05')
    for (let i = 1; i < week.length; i += 1) {
      expect(week[i].date > week[i - 1].date).toBe(true)
    }
    expect(week[6].weekday).toBe('Thu') // 2026-11-05 is a Thursday
    expect(week[0].weekday).toBe('Fri')
  })

  it('handles a month boundary', () => {
    const week = lastNDays(7, new Date(2026, 11, 2)) // 2026-12-02
    expect(week[0].date).toBe('2026-11-26')
    expect(week[6].date).toBe('2026-12-02')
  })
})

describe('monthGrid', () => {
  it('starts on the right weekday and has the right length', () => {
    // November 2026 starts on a Sunday -> no leading blanks
    const nov = monthGrid(2026, 10)
    expect(nov[0]).not.toBeNull()
    expect(nov[0].date).toBe('2026-11-01')
    expect(nov).toHaveLength(30)

    // October 2026 starts on a Thursday -> 4 leading blanks
    const oct = monthGrid(2026, 9)
    expect(oct.slice(0, 4).every((c) => c === null)).toBe(true)
    expect(oct[4].date).toBe('2026-10-01')
    expect(oct).toHaveLength(35)
  })

  it('handles February in a non-leap year', () => {
    const feb = monthGrid(2026, 1)
    const days = feb.filter(Boolean)
    expect(days).toHaveLength(28)
    expect(days[27].date).toBe('2026-02-28')
  })

  it('marks which days have a check-in', () => {
    const cells = monthGrid(2026, 10, { '2026-11-05': 'calm' })
    const day5 = cells.find((c) => c && c.date === '2026-11-05')
    const day6 = cells.find((c) => c && c.date === '2026-11-06')
    expect(day5.mood).toBe('calm')
    expect(day6.mood).toBeNull()
  })
})

describe('entry grouping', () => {
  const entries = [
    { id: 1, date: '2026-11-05', mood: 'calm' },
    { id: 2, date: '2026-11-04', mood: 'stressed' },
    { id: 3, date: '2026-11-05', mood: 'happy' }, // newest wins
    { id: 4, date: '2026-11-03', mood: 'invented' }, // invalid -> ignored
  ]

  it('maps date -> mood, ignoring unknown moods', () => {
    const map = moodsByDate(entries)
    expect(map['2026-11-05']).toBe('happy')
    expect(map['2026-11-04']).toBe('stressed')
    expect(map['2026-11-03']).toBeUndefined()
  })

  it('filters entries for one date', () => {
    expect(entriesForDate(entries, '2026-11-05').map((e) => e.id)).toEqual([1, 3])
    expect(entriesForDate(entries, '2026-11-01')).toEqual([])
    expect(entriesForDate(null, '2026-11-05')).toEqual([])
  })
})

describe('API row converters', () => {
  it('maps a mood row to the UI shape', () => {
    const ui = toMoodEntry({
      id: 7,
      date: '2026-11-05',
      mood: 'anxious',
      note: 'deadline day',
      created_at: '2026-11-05T10:00:00',
      updated_at: '2026-11-05T10:00:00',
    })
    expect(ui).toEqual({
      id: 7,
      date: '2026-11-05',
      mood: 'anxious',
      note: 'deadline day',
      createdAt: '2026-11-05T10:00:00',
      updatedAt: '2026-11-05T10:00:00',
    })
    expect(toMoodEntry(null)).toBeNull()
  })

  it('maps a journal row to the UI shape', () => {
    const ui = toJournalEntry({
      id: 3,
      title: 'Monday',
      body: 'Today was okay.',
      created_at: 'a',
      updated_at: 'b',
    })
    expect(ui).toEqual({
      id: 3,
      title: 'Monday',
      body: 'Today was okay.',
      createdAt: 'a',
      updatedAt: 'b',
    })
    expect(toJournalEntry(null)).toBeNull()
  })
})

describe('wellness routes', () => {
  it('parses the three new routes', () => {
    expect(parseRoute('#/mood').name).toBe('mood')
    expect(parseRoute('#/journal').name).toBe('journal')
    expect(parseRoute('#/wellness').name).toBe('wellness')
  })

  it('builds their hashes', () => {
    expect(routes.mood()).toBe('#/mood')
    expect(routes.journal()).toBe('#/journal')
    expect(routes.wellness()).toBe('#/wellness')
  })

  it('still requires a single segment (deep links fall back home)', () => {
    expect(parseRoute('#/mood/extra').name).toBe('home')
  })
})
