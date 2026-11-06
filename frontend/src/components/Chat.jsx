import { useCallback, useEffect, useRef, useState } from 'react'
import {
  fetchMessages,
  fetchSession,
  regenerateChat,
  renameConversation,
  sendChat,
} from '../api.js'
import { toExchange } from '../format.js'
import { routes } from '../router.js'
import { Composer } from './Composer.jsx'
import { MessageList } from './MessageList.jsx'
import { InsightsPanel } from './InsightsPanel.jsx'
import { STARTERS } from './Home.jsx'
import { Icon } from './ui.jsx'

let bubbleSeq = 0
function userBubble(text) {
  bubbleSeq += 1
  return { id: `user-${bubbleSeq}-${Date.now()}`, role: 'user', text, at: new Date().toISOString() }
}

/** Server-stored transcript row -> local bubble shape. */
function fromStored(message) {
  return {
    id: `stored-${message.created_at}-${Math.random()}`,
    role: message.role,
    text: message.content,
    at: message.created_at,
    source: message.meta?.source ?? null,
    risk: message.meta?.risk ?? null,
    guardrail: message.meta?.guardrail ?? null,
    latencyMs: message.meta?.latency_ms ?? null,
  }
}

export function Chat({
  activeId,
  title,
  starter,
  clearStarter,
  anonymous,
  onConversationsChanged,
  onNavigate,
  onOpenMenu,
}) {
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [loadError, setLoadError] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [retryText, setRetryText] = useState(null)
  const [state, setState] = useState(null)
  const [showInsights, setShowInsights] = useState(false)
  const [renaming, setRenaming] = useState(false)
  const [draftTitle, setDraftTitle] = useState('')
  const titleRef = useRef(null)

  // ---- load an existing conversation ------------------------------------
  const load = useCallback(async () => {
    if (!activeId) {
      setMessages([])
      setState(null)
      setLoadError(null)
      return
    }
    if (anonymous) {
      setMessages([])
      setLoadError('Conversation history is unavailable in anonymous mode.')
      return
    }
    setLoading(true)
    setLoadError(null)
    try {
      const [transcript, sessionState] = await Promise.all([
        fetchMessages(activeId),
        fetchSession(activeId).catch(() => null),
      ])
      setMessages(transcript.messages.map(fromStored))
      setState(sessionState)
    } catch (e) {
      setLoadError(e.message)
    } finally {
      setLoading(false)
    }
  }, [activeId, anonymous])

  useEffect(() => {
    load()
  }, [load])

  useEffect(() => {
    if (renaming) titleRef.current?.select()
  }, [renaming])

  // ---- sending -----------------------------------------------------------
  async function deliver(text) {
    setBusy(true)
    setError(null)
    setRetryText(null)
    try {
      const response = await sendChat({ message: text, sessionId: activeId })
      setMessages((prev) => [...prev, toExchange({ ...response, at: new Date().toISOString() })])
      setState(response.state ?? null)
      if (!activeId) {
        onConversationsChanged?.()
        onNavigate(routes.chat(response.session_id))
      } else {
        onConversationsChanged?.()
      }
    } catch (e) {
      setError(e.message)
      setRetryText(text)
    } finally {
      setBusy(false)
    }
  }

  function handleSend(text) {
    setMessages((prev) => [...prev, userBubble(text)])
    deliver(text)
  }

  function handleRetry() {
    if (!retryText) return
    setMessages((prev) => [...prev, userBubble(retryText)])
    deliver(retryText)
  }

  // ---- regenerate (Phase 21) ---------------------------------------------
  /**
   * Ask the backend for a fresh answer to the last user turn. The old reply
   * is restored if the request fails, so the transcript is never lost. The
   * backend re-runs the full pipeline (pre-gate -> model -> post-gate), so
   * regeneration can never bypass a safety check.
   */
  async function handleRegenerate() {
    if (!activeId || anonymous || busy) return
    const last = messages[messages.length - 1]
    if (!last || last.role !== 'assistant') return

    setBusy(true)
    setError(null)
    setRetryText(null)
    try {
      const response = await regenerateChat({ sessionId: activeId })
      const fresh = {
        ...toExchange(response),
        at: new Date().toISOString(), // toExchange has no timestamp
      }
      setMessages((prev) => [...prev.slice(0, -1), fresh])
      setState(response.state ?? null)
      onConversationsChanged?.()
    } catch (e) {
      setError(e.message) // the original reply is still in place
    } finally {
      setBusy(false)
    }
  }

  function pickStarter(text) {
    clearStarter?.()
    handleSend(text)
  }

  // ---- rename ------------------------------------------------------------
  async function commitRename() {
    const next = draftTitle.trim()
    setRenaming(false)
    if (!next || next === title) return
    try {
      await renameConversation(activeId, next)
      onConversationsChanged?.()
    } catch (e) {
      setError(e.message)
    }
  }

  const empty = !loading && !loadError && messages.length === 0

  return (
    <div className="chat-view">
      <header className="chat-head">
        <button
          type="button"
          className="btn-icon mobile-only"
          onClick={onOpenMenu}
          aria-label="Open navigation"
        >
          <Icon name="menu" size={20} />
        </button>
        <div className="chat-title-wrap">
          {renaming ? (
            <div className="rename-inline">
              <input
                ref={titleRef}
                value={draftTitle}
                maxLength={80}
                onChange={(e) => setDraftTitle(e.target.value)}
                onBlur={commitRename}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') commitRename()
                  if (e.key === 'Escape') setRenaming(false)
                }}
                aria-label="Conversation title"
              />
            </div>
          ) : (
            <>
              <div className="chat-title">{title || 'New conversation'}</div>
              <div className="chat-sub">
                {anonymous
                  ? 'anonymous session'
                  : activeId
                    ? 'private to you'
                    : 'starts a private conversation'}
              </div>
            </>
          )}
        </div>
        <div className="chat-head-actions">
          {activeId && !renaming && (
            <button
              type="button"
              className="btn-icon"
              aria-label="Rename conversation"
              title="Rename"
              onClick={() => {
                setDraftTitle(title)
                setRenaming(true)
              }}
            >
              <Icon name="pencil" size={16} />
            </button>
          )}
          <button
            type="button"
            className="btn btn-sm"
            aria-pressed={showInsights}
            onClick={() => setShowInsights((v) => !v)}
          >
            <Icon name="spark" size={15} />
            Insights
          </button>
        </div>
      </header>

      <div className="chat-body">
        <div className="chat-main">
          {loadError && (
            <div style={{ padding: '26px 22px 0' }}>
              <div className="notice notice-error" role="alert">
                <span style={{ flex: 1 }}>{loadError}</span>
                <button type="button" className="btn btn-sm" onClick={load}>
                  Retry
                </button>
              </div>
            </div>
          )}

          {empty ? (
            <div className="chat-empty">
              <span className="avatar avatar-lg avatar-soft" aria-hidden="true">
                <Icon name="spark" size={26} />
              </span>
              <div>
                <h2>What's on your mind?</h2>
                <p className="muted" style={{ margin: '6px 0 0' }}>
                  Share as much or as little as you like. Everything you write
                  stays private to your account.
                </p>
              </div>
              <div className="starters">
                {STARTERS.map((text) => (
                  <button
                    key={text}
                    type="button"
                    className="starter"
                    disabled={busy}
                    onClick={() => pickStarter(text)}
                  >
                    {text}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <MessageList
              messages={messages}
              busy={busy}
              loading={loading}
              onRegenerate={
                activeId && !anonymous && !busy ? handleRegenerate : null
              }
            />
          )}

          <div className="composer-wrap">
            {error && (
              <div className="chat-error">
                <div className="notice notice-error" role="alert" style={{ flex: 1 }}>
                  <span style={{ flex: 1 }}>{error}</span>
                  {retryText && (
                    <button type="button" className="btn btn-sm" onClick={handleRetry}>
                      Retry
                    </button>
                  )}
                  <button
                    type="button"
                    className="btn-icon"
                    onClick={() => setError(null)}
                    aria-label="Dismiss error"
                  >
                    <Icon name="close" size={14} />
                  </button>
                </div>
              </div>
            )}
            <Composer onSend={handleSend} busy={busy} initial={starter ?? ''} />
            <p className="composer-note">
              Sage is a research prototype, not a medical service. In an
              emergency, contact your local emergency services or a trusted
              person nearby.
            </p>
          </div>
        </div>

        {showInsights && (
          <InsightsPanel
            state={state}
            mobile={false}
            onClose={() => setShowInsights(false)}
          />
        )}
      </div>
    </div>
  )
}
