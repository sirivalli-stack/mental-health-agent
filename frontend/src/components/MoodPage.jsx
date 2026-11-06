import { useEffect, useMemo, useState } from 'react'
import { deleteMood, listMoods } from '../api.js'
import { formatDate } from '../greeting.js'
import {
  MOODS,
  lastNDays,
  moodsByDate,
  moodMeta,
  monthGrid,
  todayISO,
} from '../wellness.js'
import { ConfirmDialog, Icon, Spinner } from './ui.jsx'

const MONTH_NAMES = [
  'January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December',
]

/** Weekly viz: one column per day for the last 7 days. */
function WeekChart({ entries, byDate }) {
  const week = lastNDays(7)
  return (
    <div className="week-chart" role="img" aria-label="Your last seven days of check-ins">
      {week.map((day) => {
        const key = byDate[day.date]
        const meta = key ? moodMeta(key) : null
        return (
          <div key={day.date} className="week-col">
            <div className="week-bar-slot">
              <span
                className={`week-bar${meta ? '' : ' empty'}`}
                style={meta ? { background: meta.dot } : undefined}
                title={meta ? `${day.date}: ${meta.label}` : `${day.date}: no check-in`}
              />
            </div>
            <span className="week-day">{day.weekday}</span>
            <span className="week-date">{day.date.slice(8)}</span>
          </div>
        )
      })}
    </div>
  )
}

/** Monthly overview: every day of the month, coloured by its check-in. */
function MonthGrid({ byDate, monthOffset, onMonthChange }) {
  const base = new Date()
  const target = new Date(base.getFullYear(), base.getMonth() + monthOffset, 1)
  const cells = monthGrid(target.getFullYear(), target.getMonth(), byDate)
  const today = todayISO()

  return (
    <div className="month-block">
      <div className="month-nav">
        <button
          type="button"
          className="btn btn-sm"
          onClick={() => onMonthChange(monthOffset - 1)}
          aria-label="Previous month"
        >
          <Icon name="chevron" size={14} style={{ transform: 'rotate(180deg)' }} />
        </button>
        <span className="month-title">
          {MONTH_NAMES[target.getMonth()]} {target.getFullYear()}
        </span>
        <button
          type="button"
          className="btn btn-sm"
          onClick={() => onMonthChange(monthOffset + 1)}
          disabled={monthOffset >= 0}
          aria-label="Next month"
        >
          <Icon name="chevron" size={14} />
        </button>
      </div>
      <div className="month-grid" role="grid" aria-label="Monthly check-in overview">
        {['S', 'M', 'T', 'W', 'T', 'F', 'S'].map((d, i) => (
          <span key={`h${i}`} className="month-head" aria-hidden="true">
            {d}
          </span>
        ))}
        {cells.map((cell, i) =>
          cell === null ? (
            <span key={`b${i}`} className="month-cell blank" aria-hidden="true" />
          ) : (
            <span
              key={cell.date}
              className={`month-cell${cell.mood ? ' has-mood' : ''}${cell.date === today ? ' is-today' : ''}`}
              role="gridcell"
              title={
                cell.mood
                  ? `${cell.date}: ${moodMeta(cell.mood)?.label ?? cell.mood}`
                  : `${cell.date}: no check-in`
              }
            >
              <span className="month-num">{cell.day}</span>
              <span
                className="month-dot"
                style={cell.mood ? { background: moodMeta(cell.mood)?.dot } : undefined}
              />
            </span>
          ),
        )}
      </div>
      <div className="mood-legend">
        {MOODS.map((m) => (
          <span key={m.key} className="legend-item">
            <span className="mood-dot" style={{ background: m.dot }} aria-hidden="true" />
            {m.label}
          </span>
        ))}
      </div>
    </div>
  )
}

/**
 * Mood history (Phase 7): today's entry editor is on the dashboard; this
 * page shows the record back - last-7-days chart, a monthly overview and
 * the full list with delete. Everything shown is what the user saved.
 */
export function MoodPage({ onOpenMenu }) {
  const [entries, setEntries] = useState(null)
  const [error, setError] = useState(null)
  const [monthOffset, setMonthOffset] = useState(0)
  const [pendingDelete, setPendingDelete] = useState(null)
  const [deleting, setDeleting] = useState(false)

  useEffect(() => {
    let cancelled = false
    listMoods({ limit: 366 })
      .then((loaded) => {
        if (!cancelled) setEntries(loaded)
      })
      .catch((e) => {
        if (!cancelled) setError(e.message)
      })
    return () => {
      cancelled = true
    }
  }, [])

  const byDate = useMemo(() => moodsByDate(entries ?? []), [entries])

  async function confirmDelete() {
    if (!pendingDelete) return
    setDeleting(true)
    try {
      await deleteMood(pendingDelete.id)
      setEntries((prev) => (prev ?? []).filter((e) => e.id !== pendingDelete.id))
    } catch (e) {
      setError(e.message)
    } finally {
      setDeleting(false)
      setPendingDelete(null)
    }
  }

  if (entries === null && !error) {
    return (
      <div className="page-scroll">
        <div className="page-inner">
          <Spinner />
        </div>
      </div>
    )
  }

  return (
    <div className="page-scroll">
      <div className="page-inner">
        <header className="page-head">
          <button
            type="button"
            className="btn-icon mobile-only page-menu-btn"
            onClick={onOpenMenu}
            aria-label="Open navigation"
          >
            <Icon name="menu" size={20} />
          </button>
          <h1>Mood check-ins</h1>
          <p className="page-sub">
            A private record of how you have been feeling - one entry per day,
            yours alone. This is a journal of your own words, never a diagnosis.
          </p>
        </header>

        {error && (
          <div className="notice notice-error" role="alert">
            {error}
          </div>
        )}

        <section className="card dash-card" aria-label="Last seven days">
          <div className="section-title">This week</div>
          {entries && entries.length === 0 ? (
            <p className="mood-empty">
              No check-ins yet. Start from the dashboard - one tap a day is enough.
            </p>
          ) : (
            entries && <WeekChart entries={entries} byDate={byDate} />
          )}
        </section>

        <section className="card dash-card" aria-label="Monthly overview">
          <div className="section-title">Monthly overview</div>
          <MonthGrid byDate={byDate} monthOffset={monthOffset} onMonthChange={setMonthOffset} />
        </section>

        <section className="card dash-card" aria-label="Check-in history">
          <div className="section-title">History</div>
          {entries && entries.length === 0 && (
            <p className="mood-empty">Nothing recorded yet.</p>
          )}
          <div className="mood-history">
            {(entries ?? []).map((entry) => {
              const meta = moodMeta(entry.mood)
              return (
                <div key={entry.id} className="mood-history-row">
                  <span className="mood-dot" style={{ background: meta?.dot }} aria-hidden="true" />
                  <span className="mood-history-date">{formatDate(entry.date)}</span>
                  <span className="mood-history-mood">{meta?.label ?? entry.mood}</span>
                  <span className="mood-history-note">{entry.note}</span>
                  <button
                    type="button"
                    className="btn-icon"
                    aria-label={`Delete check-in for ${entry.date}`}
                    title="Delete this check-in"
                    onClick={() => setPendingDelete(entry)}
                  >
                    <Icon name="trash" size={15} />
                  </button>
                </div>
              )
            })}
          </div>
        </section>
      </div>

      <ConfirmDialog
        open={Boolean(pendingDelete)}
        title="Delete this check-in?"
        body={`${pendingDelete?.date ?? ''} (${moodMeta(pendingDelete?.mood)?.label ?? ''}) will be removed. This cannot be undone.`}
        confirmLabel="Delete"
        danger
        busy={deleting}
        onConfirm={confirmDelete}
        onCancel={() => setPendingDelete(null)}
      />
    </div>
  )
}
