import { useEffect, useState } from 'react'
import { createJournal, deleteJournal, listJournal, updateJournal } from '../api.js'
import { formatDate, relativeTime } from '../greeting.js'
import { ConfirmDialog, Icon, Spinner } from './ui.jsx'

/**
 * Private journal (Phase 7): list, create, edit, delete. Entries are the
 * user's own words - stored as written, never analysed, never shared, and
 * removable at any time. A blank body is never accepted (the API rejects
 * it too); title is optional.
 */
export function JournalPage({ onOpenMenu }) {
  const [entries, setEntries] = useState(null)
  const [error, setError] = useState(null)
  const [editing, setEditing] = useState(null) // {id?, title, body} draft
  const [saving, setSaving] = useState(false)
  const [pendingDelete, setPendingDelete] = useState(null)
  const [deleting, setDeleting] = useState(false)

  useEffect(() => {
    let cancelled = false
    listJournal()
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

  const bodyEmpty = !(editing?.body ?? '').trim()

  async function save() {
    if (!editing || bodyEmpty || saving) return
    setSaving(true)
    setError(null)
    try {
      if (editing.id) {
        const updated = await updateJournal(editing.id, {
          title: editing.title.trim(),
          body: editing.body,
        })
        setEntries((prev) =>
          (prev ?? []).map((e) => (e.id === updated.id ? updated : e)),
        )
      } else {
        const created = await createJournal({
          title: editing.title.trim(),
          body: editing.body,
        })
        setEntries((prev) => [created, ...(prev ?? [])])
      }
      setEditing(null)
    } catch (e) {
      setError(e.message)
    } finally {
      setSaving(false)
    }
  }

  async function confirmDelete() {
    if (!pendingDelete) return
    setDeleting(true)
    try {
      await deleteJournal(pendingDelete.id)
      setEntries((prev) => (prev ?? []).filter((e) => e.id !== pendingDelete.id))
      if (editing?.id === pendingDelete.id) setEditing(null)
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
          <h1>Journal</h1>
          <p className="page-sub">
            Your private writing space. Entries stay yours - never analysed,
            never shared, deletable whenever you want.
          </p>
        </header>

        {error && !editing && (
          <div className="notice notice-error" role="alert">
            {error}
          </div>
        )}

        <section className="card dash-card" aria-label="Write an entry">
          <div className="section-title">{editing?.id ? 'Edit entry' : 'New entry'}</div>
          <input
            className="journal-title-input"
            type="text"
            maxLength={120}
            placeholder="Title (optional)"
            aria-label="Entry title"
            value={editing?.title ?? ''}
            onChange={(e) => setEditing((prev) => ({ ...(prev ?? {}), title: e.target.value, body: prev?.body ?? '' }))}
          />
          <textarea
            className="journal-body-input"
            rows={7}
            maxLength={20000}
            placeholder="What's on your mind?"
            aria-label="Entry text"
            value={editing?.body ?? ''}
            onChange={(e) => setEditing((prev) => ({ ...(prev ?? {}), title: prev?.title ?? '', body: e.target.value }))}
          />
          <div className="mood-actions">
            <button
              type="button"
              className="btn btn-primary"
              disabled={bodyEmpty || saving}
              onClick={save}
            >
              {saving ? 'Saving…' : editing?.id ? 'Save changes' : 'Save entry'}
            </button>
            {editing && (
              <button type="button" className="btn" onClick={() => setEditing(null)}>
                Cancel
              </button>
            )}
            {editing && (
              <span className="faint" style={{ alignSelf: 'center' }}>
                {editing.body.length}/20000
              </span>
            )}
          </div>
          {error && editing && (
            <div className="notice notice-error" role="alert">
              {error}
            </div>
          )}
        </section>

        <section className="card dash-card" aria-label="Your entries">
          <div className="section-title">Your entries</div>
          {entries && entries.length === 0 && (
            <p className="mood-empty">Nothing written yet - the box above is ready when you are.</p>
          )}
          <div className="journal-list">
            {(entries ?? []).map((entry) => (
              <article key={entry.id} className="journal-item">
                <div className="journal-item-head">
                  <button
                    type="button"
                    className="journal-item-title"
                    onClick={() =>
                      setEditing({ id: entry.id, title: entry.title, body: entry.body })
                    }
                  >
                    {entry.title || 'Untitled entry'}
                  </button>
                  <span className="when">
                    {formatDate(entry.updatedAt)} · {relativeTime(entry.updatedAt)}
                  </span>
                  <span className="actions">
                    <button
                      type="button"
                      className="btn-icon"
                      aria-label="Edit entry"
                      title="Edit"
                      onClick={() =>
                        setEditing({ id: entry.id, title: entry.title, body: entry.body })
                      }
                    >
                      <Icon name="pencil" size={15} />
                    </button>
                    <button
                      type="button"
                      className="btn-icon"
                      aria-label="Delete entry"
                      title="Delete"
                      onClick={() => setPendingDelete(entry)}
                    >
                      <Icon name="trash" size={15} />
                    </button>
                  </span>
                </div>
                <p className="journal-item-body">{entry.body}</p>
              </article>
            ))}
          </div>
        </section>
      </div>

      <ConfirmDialog
        open={Boolean(pendingDelete)}
        title="Delete this entry?"
        body={`"${pendingDelete?.title || 'Untitled entry'}" will be permanently removed. This cannot be undone.`}
        confirmLabel="Delete"
        danger
        busy={deleting}
        onConfirm={confirmDelete}
        onCancel={() => setPendingDelete(null)}
      />
    </div>
  )
}
