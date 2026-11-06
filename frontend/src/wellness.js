/**
 * Wellness helpers (Phase 7) - pure functions, no React, no network.
 *
 * The seven check-in moods come straight from the product spec. A check-in
 * is a self-reported feeling for one calendar day: these helpers only
 * arrange what the user chose into weeks/months for display. Nothing here
 * (or in the routes that store it) ever scores, trends-diagnoses or shares
 * a mood - the UI copy says so too.
 */

/** The seven moods, in spec order. `dot` is the chart/list colour token. */
export const MOODS = [
  { key: 'happy', label: 'Happy', dot: '#f2b84b' },
  { key: 'calm', label: 'Calm', dot: '#7fb8a4' },
  { key: 'sad', label: 'Sad', dot: '#6f8fd0' },
  { key: 'anxious', label: 'Anxious', dot: '#a78bd4' },
  { key: 'stressed', label: 'Stressed', dot: '#e08a6b' },
  { key: 'angry', label: 'Angry', dot: '#d4695f' },
  { key: 'overwhelmed', label: 'Overwhelmed', dot: '#9b7fc4' },
]

export const MOOD_KEYS = MOODS.map((m) => m.key)

/** Look up one mood's display data (label + colour) by key. */
export function moodMeta(key) {
  return MOODS.find((m) => m.key === key) ?? null
}

/** True when the key is one of the seven (guards stale/foreign data). */
export function isMoodKey(key) {
  return MOOD_KEYS.includes(key)
}

// ---------------------------------------------------------------------------
// Dates (plain YYYY-MM-DD strings; no timezone surprises in display code)
// ---------------------------------------------------------------------------

/** Today's date as YYYY-MM-DD in the *browser's* local day. */
export function todayISO(now = new Date()) {
  const y = now.getFullYear()
  const m = String(now.getMonth() + 1).padStart(2, '0')
  const d = String(now.getDate()).padStart(2, '0')
  return `${y}-${m}-${d}`
}

/** Parse a YYYY-MM-DD string into a local Date at midnight. */
export function parseISO(iso) {
  const [y, m, d] = String(iso).split('-').map(Number)
  return new Date(y, (m ?? 1) - 1, d ?? 1)
}

/** Format a Date as YYYY-MM-DD (local). */
export function toISO(date) {
  const y = date.getFullYear()
  const m = String(date.getMonth() + 1).padStart(2, '0')
  const d = String(date.getDate()).padStart(2, '0')
  return `${y}-${m}-${d}`
}

/** Add whole days to a YYYY-MM-DD string. */
export function addDays(iso, days) {
  const date = parseISO(iso)
  date.setDate(date.getDate() + days)
  return toISO(date)
}

/**
 * The last `count` days ending today, oldest first:
 * [{ date: 'YYYY-MM-DD', weekday: 'Mon' }, ...]
 */
export function lastNDays(count = 7, now = new Date()) {
  const today = todayISO(now)
  const weekdays = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
  const out = []
  for (let i = count - 1; i >= 0; i -= 1) {
    const iso = addDays(today, -i)
    out.push({ date: iso, weekday: weekdays[parseISO(iso).getDay()] })
  }
  return out
}

/**
 * Monthly grid for `year`/`month` (0-based month): leading blanks + days,
 * each carrying { date, day, weekday } so a cell can be coloured by mood.
 * `moodsByDate` (Map or plain object) marks which days have a check-in.
 */
export function monthGrid(year, month, moodsByDate = {}) {
  const first = new Date(year, month, 1)
  const daysInMonth = new Date(year, month + 1, 0).getDate()
  const lead = first.getDay() // 0 = Sunday first column
  const cells = []
  for (let i = 0; i < lead; i += 1) cells.push(null)
  for (let day = 1; day <= daysInMonth; day += 1) {
    const date = new Date(year, month, day)
    const iso = toISO(date)
    cells.push({
      date: iso,
      day,
      weekday: date.getDay(),
      mood: moodsByDate[iso] ?? null,
    })
  }
  return cells
}

/**
 * Join entries ( [{date, mood, ...}] ) into a { 'YYYY-MM-DD': mood } map.
 * The newest entry for a date wins (the API already upserts one per day,
 * this just stays correct if older rows ever coexist).
 */
export function moodsByDate(entries) {
  const map = {}
  for (const entry of entries ?? []) {
    if (entry?.date && isMoodKey(entry.mood)) map[entry.date] = entry.mood
  }
  return map
}

/** Entries for one date, newest first. */
export function entriesForDate(entries, iso) {
  return (entries ?? []).filter((e) => e.date === iso)
}
