import { useEffect, useRef, useState } from 'react'
import { renderMarkdown } from '../markdown.js'
import { formatClock } from '../greeting.js'
import { copyText } from '../copy.js'
import { riskTone, sourceLabel, sourceTone } from '../format.js'
import { useReadAloud, useVoiceSettings } from '../voice/index.js'
import { Icon } from './ui.jsx'

function AssistantMeta({ message }) {
  const items = []
  if (message.source && message.source !== 'llm') {
    items.push(
      <span key="source" className={`badge ${sourceTone(message.source)}`}>
        {sourceLabel(message.source)}
      </span>,
    )
  }
  if (message.risk) {
    items.push(
      <span key="risk" className={`chip ${riskTone(message.risk)}`}>
        mood context: {message.risk}
      </span>,
    )
  }
  if (message.guardrail) {
    items.push(
      <span key="guardrail" className="chip chip-plain">
        safety check: {message.guardrail}
      </span>,
    )
  }
  if (message.at) {
    items.push(
      <span key="time" className="time">
        {formatClock(message.at)}
      </span>,
    )
  }
  if (items.length === 0) return null
  return <div className="msg-meta">{items}</div>
}

/**
 * Per-message actions: copy any assistant reply, regenerate the newest
 * one, and (Phase 6) read the reply aloud. Copy shows a short "Copied"
 * confirmation; regeneration is only offered when the parent (Chat) can
 * actually perform it (signed in, not busy).
 *
 * Read-aloud controls appear per state: Listen -> Pause/Resume + Stop ->
 * Listen again (replay). Only the bubble currently being spoken shows
 * the active controls; starting a new bubble stops the previous one.
 */
function MessageActions({ message, canRegenerate, onRegenerate, readAloud }) {
  const [copied, setCopied] = useState(false)
  const [copyFailed, setCopyFailed] = useState(false)

  async function handleCopy() {
    const ok = await copyText(message.text)
    if (!ok) {
      setCopyFailed(true)
      window.setTimeout(() => setCopyFailed(false), 1600)
      return
    }
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1600)
  }

  const speaking = readAloud && readAloud.currentText === message.text
  const state = speaking ? readAloud.state : 'idle'
  const voiceOpts = { voiceURI: readAloud?.voiceURI ?? '', rate: readAloud?.rate ?? 1 }

  return (
    <div className="msg-actions">
      <button type="button" className="msg-action" onClick={handleCopy}>
        <Icon name={copied ? 'check' : 'copy'} size={13} />
        {copied ? 'Copied' : copyFailed ? 'Copy failed' : 'Copy'}
      </button>
      {readAloud?.supported && (
        <button
          type="button"
          className={`msg-action${state === 'speaking' ? ' msg-action-active' : ''}`}
          aria-label={
            state === 'speaking'
              ? 'Pause reading'
              : state === 'paused'
                ? 'Resume reading'
                : 'Read this reply aloud'
          }
          onClick={() => readAloud.toggle(message.text, voiceOpts)}
        >
          <Icon name={state === 'speaking' ? 'pause' : 'play'} size={13} />
          {state === 'speaking' ? 'Pause' : state === 'paused' ? 'Resume' : 'Listen'}
        </button>
      )}
      {readAloud?.supported && state !== 'idle' && (
        <button
          type="button"
          className="msg-action"
          aria-label="Stop reading"
          onClick={() => readAloud.stop()}
        >
          <Icon name="stop" size={13} />
          Stop
        </button>
      )}
      {canRegenerate && (
        <button type="button" className="msg-action" onClick={onRegenerate}>
          <Icon name="refresh" size={13} />
          Regenerate
        </button>
      )}
    </div>
  )
}

function Bubble({ message, canRegenerate, onRegenerate, readAloud }) {
  if (message.role === 'user') {
    return (
      <div className="msg msg-user">
        <div className="msg-col">
          <div className="bubble bubble-user">{message.text}</div>
          {message.at && (
            <div className="msg-meta">
              <span className="time">{formatClock(message.at)}</span>
            </div>
          )}
        </div>
      </div>
    )
  }
  const speaking = readAloud && readAloud.currentText === message.text
  return (
    <div className={`msg${speaking && readAloud.state === 'speaking' ? ' msg-speaking' : ''}`}>
      <span className="avatar avatar-soft msg-avatar" aria-hidden="true">
        <Icon name="spark" size={16} />
      </span>
      <div className="msg-col">
        <div className="bubble bubble-assistant">
          <div dangerouslySetInnerHTML={{ __html: renderMarkdown(message.text) }} />
        </div>
        <AssistantMeta message={message} />
        <MessageActions
          message={message}
          canRegenerate={canRegenerate}
          onRegenerate={onRegenerate}
          readAloud={readAloud}
        />
      </div>
    </div>
  )
}

function LoadingSkeleton() {
  return (
    <div className="messages-inner" aria-hidden="true">
      <div className="skeleton" style={{ height: 18, width: '38%' }} />
      <div className="skeleton" style={{ height: 64, width: '72%', alignSelf: 'flex-end' }} />
      <div className="skeleton" style={{ height: 46, width: '55%' }} />
    </div>
  )
}

export function MessageList({ messages, busy, loading, onRegenerate }) {
  const endRef = useRef(null)
  const containerRef = useRef(null)
  const [voice] = useVoiceSettings()
  const readAloudBase = useReadAloud()

  // voice settings ride along so each bubble renders the right controls
  const readAloud =
    voice.ttsEnabled && readAloudBase.supported
      ? { ...readAloudBase, voiceURI: voice.ttsVoice, rate: voice.ttsRate }
      : null

  // Only the newest assistant reply can be regenerated (matches the
  // backend contract: the transcript must end [..., user, assistant]).
  const lastAssistantId = [...messages].reverse().find((m) => m.role === 'assistant')?.id ?? null

  useEffect(() => {
    const el = endRef.current
    const box = containerRef.current
    if (el && box) {
      // instant jump on first paint, smooth follow afterwards
      box.scrollTo({ top: box.scrollHeight, behavior: messages.length > 2 ? 'smooth' : 'auto' })
    }
  }, [messages, busy])

  if (loading) {
    return (
      <div className="messages" ref={containerRef} data-testid="messages">
        <LoadingSkeleton />
      </div>
    )
  }

  return (
    <div className="messages" ref={containerRef} data-testid="messages">
      <div className="messages-inner">
        {messages.map((m) => (
          <Bubble
            key={m.id}
            message={m}
            canRegenerate={
              Boolean(onRegenerate) && !busy && m.role === 'assistant' && m.id === lastAssistantId
            }
            onRegenerate={onRegenerate}
            readAloud={m.role === 'assistant' ? readAloud : null}
          />
        ))}
        {busy && (
          <div className="msg">
            <span className="avatar avatar-soft msg-avatar" aria-hidden="true">
              <Icon name="spark" size={16} />
            </span>
            <div className="msg-col">
              <div className="bubble bubble-pending" role="status" aria-live="polite">
                <span className="typing-dots" aria-hidden="true">
                  <span />
                  <span />
                  <span />
                </span>
                <span>Listening and thinking…</span>
              </div>
            </div>
          </div>
        )}
        <div ref={endRef} />
      </div>
    </div>
  )
}
