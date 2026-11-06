import { useEffect, useRef, useState } from 'react'
import { routes } from '../router.js'
import { firstName, initials, relativeTime } from '../greeting.js'
import { BrandMark, ConfirmDialog, Icon } from './ui.jsx'

function ConversationItem({ conversation, active, onOpen, onRename, onDelete }) {
  const [renaming, setRenaming] = useState(false)
  const [draft, setDraft] = useState(conversation.title)
  const inputRef = useRef(null)

  useEffect(() => {
    if (renaming) inputRef.current?.select()
  }, [renaming])

  function commitRename() {
    const title = draft.trim()
    setRenaming(false)
    if (title && title !== conversation.title) onRename(conversation.sessionId, title)
    else setDraft(conversation.title)
  }

  if (renaming) {
    return (
      <div className={`side-item ${active ? 'active' : ''}`}>
        <input
          ref={inputRef}
          className="rename-input"
          value={draft}
          maxLength={80}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commitRename}
          onKeyDown={(e) => {
            if (e.key === 'Enter') commitRename()
            if (e.key === 'Escape') {
              setDraft(conversation.title)
              setRenaming(false)
            }
          }}
          aria-label="Conversation title"
        />
      </div>
    )
  }

  return (
    <div className={`side-item ${active ? 'active' : ''}`}>
      <button
        type="button"
        className="grow"
        style={{ all: 'unset', cursor: 'pointer', flex: 1, minWidth: 0 }}
        onClick={() => onOpen(conversation.sessionId)}
        title={conversation.title}
      >
        <span className="title">{conversation.title}</span>
      </button>
      <span className="when">{relativeTime(conversation.updatedAt)}</span>
      <span className="actions">
        <button
          type="button"
          className="btn-icon"
          aria-label={`Rename ${conversation.title}`}
          title="Rename"
          onClick={() => setRenaming(true)}
        >
          <Icon name="pencil" size={15} />
        </button>
        <button
          type="button"
          className="btn-icon"
          aria-label={`Delete ${conversation.title}`}
          title="Delete"
          onClick={() => onDelete(conversation)}
        >
          <Icon name="trash" size={15} />
        </button>
      </span>
    </div>
  )
}

/**
 * Application sidebar: brand, new conversation, searchable history and the
 * user menu. On mobile it slides out over a scrim (props.open).
 */
export function Sidebar({
  user,
  conversations,
  loadingConversations,
  activeId,
  activeRoute,
  anonymous,
  open,
  onClose,
  onOpenConversation,
  onNewConversation,
  onRename,
  onDelete,
  onSettings,
  onSignOut,
}) {
  const [query, setQuery] = useState('')
  const [menuOpen, setMenuOpen] = useState(false)
  const [pendingDelete, setPendingDelete] = useState(null)
  const [deleting, setDeleting] = useState(false)
  const menuRef = useRef(null)

  useEffect(() => {
    function onClickOutside(event) {
      if (menuRef.current && !menuRef.current.contains(event.target)) {
        setMenuOpen(false)
      }
    }
    document.addEventListener('mousedown', onClickOutside)
    return () => document.removeEventListener('mousedown', onClickOutside)
  }, [])

  const filtered = query.trim()
    ? conversations.filter((c) =>
        c.title.toLowerCase().includes(query.trim().toLowerCase()),
      )
    : conversations

  async function confirmDelete() {
    if (!pendingDelete) return
    setDeleting(true)
    try {
      await onDelete(pendingDelete)
    } finally {
      setDeleting(false)
      setPendingDelete(null)
    }
  }

  return (
    <aside className={`sidebar ${open ? 'open' : ''}`} aria-label="Navigation">
      <button
        type="button"
        className="brand"
        onClick={() => onNewConversation()}
        style={{ background: 'transparent', border: 'none', cursor: 'pointer' }}
      >
        <BrandMark />
        <span>
          <span className="brand-name">Sage</span>
          <br />
          <span className="brand-tag">space to talk things through</span>
        </span>
      </button>

      <button type="button" className="btn btn-primary btn-block" onClick={onNewConversation}>
        <Icon name="plus" size={16} />
        New conversation
      </button>

      {!anonymous && (
        <div className="side-search">
          <Icon name="search" size={15} />
          <input
            type="search"
            value={query}
            placeholder="Search conversations"
            aria-label="Search conversations"
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
      )}

      {!anonymous && (
        <nav className="side-nav" aria-label="Wellness">
          <a
            className={`side-nav-link${activeRoute === 'mood' ? ' active' : ''}`}
            href="#/mood"
            onClick={onClose}
          >
            <Icon name="heart" size={15} />
            Mood check-ins
          </a>
          <a
            className={`side-nav-link${activeRoute === 'journal' ? ' active' : ''}`}
            href="#/journal"
            onClick={onClose}
          >
            <Icon name="book" size={15} />
            Journal
          </a>
          <a
            className={`side-nav-link${activeRoute === 'wellness' ? ' active' : ''}`}
            href="#/wellness"
            onClick={onClose}
          >
            <Icon name="leaf" size={15} />
            Wellness activities
          </a>
        </nav>
      )}

      {!anonymous && (
        <>
          <div className="side-label">Recent</div>
          <div className="side-list">
            {loadingConversations && (
              <>
                <div className="skeleton" style={{ height: 38, margin: '2px 4px' }} />
                <div className="skeleton" style={{ height: 38, margin: '2px 4px' }} />
              </>
            )}
            {!loadingConversations && filtered.length === 0 && (
              <p className="side-empty">
                {query.trim()
                  ? 'No conversations match your search.'
                  : 'Your conversations will appear here.'}
              </p>
            )}
            {filtered.map((conversation) => (
              <ConversationItem
                key={conversation.sessionId}
                conversation={conversation}
                active={conversation.sessionId === activeId}
                onOpen={(id) => {
                  onOpenConversation(id)
                  onClose?.()
                }}
                onRename={onRename}
                onDelete={setPendingDelete}
              />
            ))}
          </div>
        </>
      )}

      <div className="side-footer" ref={menuRef}>
        {menuOpen && (
          <div className="popover" role="menu">
            <button type="button" role="menuitem" onClick={() => { setMenuOpen(false); onSettings() }}>
              <Icon name="settings" size={16} />
              Settings &amp; personalization
            </button>
            <div className="sep" />
            <button type="button" role="menuitem" onClick={onSignOut}>
              <Icon name="logout" size={16} />
              Sign out
            </button>
          </div>
        )}
        <button
          type="button"
          className="user-chip"
          onClick={() => setMenuOpen((v) => !v)}
          aria-haspopup="menu"
          aria-expanded={menuOpen}
        >
          <span className="avatar" aria-hidden="true">
            {initials(user?.displayName ?? '')}
          </span>
          <span className="who">
            <strong>{user?.displayName ?? 'Welcome'}</strong>
            <span>{anonymous ? 'local session' : `@${user?.username ?? ''}`}</span>
          </span>
          <Icon name="chevron" size={16} />
        </button>
      </div>

      <ConfirmDialog
        open={Boolean(pendingDelete)}
        title="Delete conversation?"
        body={`"${pendingDelete?.title ?? ''}" and its saved messages will be permanently removed. This cannot be undone.`}
        confirmLabel="Delete"
        danger
        busy={deleting}
        onConfirm={confirmDelete}
        onCancel={() => setPendingDelete(null)}
      />
    </aside>
  )
}
