import { useEffect, useState } from 'react'
import { fetchSettings, listMoods, saveMood } from '../api.js'
import { MOODS, entriesForDate, todayISO } from '../wellness.js'
import { Icon } from './ui.jsx'

/**
 * Daily mood check-in (Phase 7): pick one of the seven moods, optionally
 * add a note, save. One check-in per day - re-checking updates today's
 * entry in place. Loads the existing entry so the card reflects what was
 * already recorded (no silent overwrite of an unseen record).
 *
 * Also consumes the user's `daily_reminders` setting: when it is on and
 * today has no check-in yet, a gentle nudge line appears. Self-reported
 * and private - never analysed, never a diagnosis.
 */
export function MoodCheckIn({ onSaved }) {
  const [entries, setEntries] = useState(null)
  const [mood, setMood] = useState(null)
  const [note, setNote] = useState('')
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [error, setError] = useState(null)
  const [remindersOn, setRemindersOn] = useState(false)

  const today = todayISO()

  useEffect(() => {
    let cancelled = false
    listMoods({ limit: 14 })
      .then((loaded) => {
        if (cancelled) return
        setEntries(loaded)
        const existing = entriesForDate(loaded, today)[0]
        if (existing) {
          setMood(existing.mood)
          setNote(existing.note ?? '')
        }
      })
      .catch((e) => {
        if (!cancelled) setError(e.message)
      })
    fetchSettings()
      .then((s) => {
        if (!cancelled) setRemindersOn(Boolean(s.dailyReminders))
      })
      .catch(() => {
        // reminder nudge is cosmetic - a failed settings read hides it
      })
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [today])

  async function submit() {
    if (!mood || saving) return
    setSaving(true)
    setError(null)
    setSaved(false)
    try {
      const stored = await saveMood({ mood, note })
      setEntries((prev) => {
        const rest = (prev ?? []).filter((e) => e.date !== stored.date)
        return [stored, ...rest]
      })
      setSaved(true)
      onSaved?.(stored)
    } catch (e) {
      setError(e.message)
    } finally {
      setSaving(false)
    }
  }

  const alreadyCheckedIn = entries ? entriesForDate(entries, today).length > 0 : false
  const showReminder = remindersOn && !alreadyCheckedIn && !mood

  return (
    <div className="mood-checkin">
      <p className="mood-checkin-lead">
        {alreadyCheckedIn
          ? 'You already checked in today - update it any time.'
          : 'How are you feeling right now? There are no wrong answers.'}
      </p>
      <div className="mood-row" role="radiogroup" aria-label="Today's mood">
        {MOODS.map((m) => (
          <button
            key={m.key}
            type="button"
            role="radio"
            aria-checked={mood === m.key}
            className={`mood-chip${mood === m.key ? ' selected' : ''}`}
            style={mood === m.key ? { '--mood-dot': m.dot } : undefined}
            onClick={() => {
              setMood(m.key)
              setSaved(false)
            }}
          >
            <span className="mood-dot" style={{ background: m.dot }} aria-hidden="true" />
            {m.label}
          </button>
        ))}
      </div>
      <label className="mood-note-label" htmlFor="mood-note">
        Add a note <span className="faint">(optional, private)</span>
      </label>
      <textarea
        id="mood-note"
        className="mood-note"
        rows={2}
        maxLength={2000}
        placeholder="What's behind this feeling?"
        value={note}
        onChange={(e) => setNote(e.target.value)}
      />
      <div className="mood-actions">
        <button
          type="button"
          className="btn btn-primary"
          disabled={!mood || saving}
          onClick={submit}
        >
          {saving ? 'Saving…' : alreadyCheckedIn ? 'Update check-in' : 'Save check-in'}
        </button>
        {saved && (
          <span className="mood-saved" role="status">
            <Icon name="check" size={14} /> Saved - just for you.
          </span>
        )}
      </div>
      {showReminder && (
        <p className="mood-reminder" role="status">
          Daily check-in reminder is on - this is a good moment to pause.
        </p>
      )}
      {error && (
        <div className="notice notice-error" role="alert">
          {error}
        </div>
      )}
      <p className="mood-disclaimer">
        Check-ins are private, self-reported notes - never a diagnosis.
      </p>
    </div>
  )
}
