import { SourceBadge } from './SourceBadge.jsx'
import { riskTone } from '../format.js'

function Bubble({ message }) {
  if (message.role === 'user') {
    return (
      <div className="row row-user">
        <div className="bubble bubble-user">{message.text}</div>
      </div>
    )
  }
  return (
    <div className="row row-assistant">
      <div className="bubble bubble-assistant">
        <SourceBadge source={message.source} />
        <p>{message.text}</p>
        <div className="meta">
          {message.risk && (
            <span className={`chip ${riskTone(message.risk)}`}>
              risk: {message.risk}
            </span>
          )}
          {message.guardrail && (
            <span className="chip chip-plain">guardrail: {message.guardrail}</span>
          )}
          {typeof message.latencyMs === 'number' && (
            <span className="chip chip-plain">
              {(message.latencyMs / 1000).toFixed(1)} s
            </span>
          )}
        </div>
      </div>
    </div>
  )
}

export function MessageList({ messages, busy }) {
  return (
    <div className="messages" data-testid="messages">
      {messages.length === 0 && (
        <div className="empty">
          Send a message to start. Replies take roughly 10 seconds warm, or up
          to a minute on a cold start on this hardware.
        </div>
      )}
      {messages.map((m) => (
        <Bubble key={m.id} message={m} />
      ))}
      {busy && (
        <div className="row row-assistant">
          <div className="bubble bubble-assistant bubble-pending">
            Thinking… (state -> pre-gate -> model -> post-gate)
          </div>
        </div>
      )}
    </div>
  )
}
