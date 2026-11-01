import { useEffect, useRef } from 'react'
import { renderMarkdown } from '../markdown.js'
import { formatClock } from '../greeting.js'
import { riskTone, sourceLabel, sourceTone } from '../format.js'
import { Icon, Spinner } from './ui.jsx'

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

function Bubble({ message }) {
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
  return (
    <div className="msg">
      <span className="avatar avatar-soft msg-avatar" aria-hidden="true">
        <Icon name="spark" size={16} />
      </span>
      <div className="msg-col">
        <div className="bubble bubble-assistant">
          <div dangerouslySetInnerHTML={{ __html: renderMarkdown(message.text) }} />
        </div>
        <AssistantMeta message={message} />
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

export function MessageList({ messages, busy, loading }) {
  const endRef = useRef(null)
  const containerRef = useRef(null)

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
          <Bubble key={m.id} message={m} />
        ))}
        {busy && (
          <div className="msg">
            <span className="avatar avatar-soft msg-avatar" aria-hidden="true">
              <Icon name="spark" size={16} />
            </span>
            <div className="msg-col">
              <div className="bubble bubble-pending">
                <Spinner />
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
