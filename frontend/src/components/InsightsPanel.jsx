import { riskTone, stateFacts } from '../format.js'
import { Icon } from './ui.jsx'

function Line({ label, value }) {
  return (
    <div className="fact">
      <span className="fact-label">{label}</span>
      <span className="fact-value">{value}</span>
    </div>
  )
}

/**
 * Optional, secondary analytical view. Everything here is what the backend
 * actually produced for this conversation - the UI never computes emotion
 * or risk itself, and nothing here is a diagnosis.
 */
export function InsightsPanel({ state, onClose, mobile = false }) {
  const facts = stateFacts(state)
  return (
    <aside className="insights" aria-label="Conversation insights">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <h3 style={{ margin: 0 }}>Insights</h3>
        {onClose && (
          <button type="button" className="btn-icon" onClick={onClose} aria-label="Close insights">
            <Icon name="close" size={16} />
          </button>
        )}
      </div>
      {!facts ? (
        <p className="muted small">
          Insights appear once the first message of this conversation has been
          analysed.
        </p>
      ) : (
        <>
          <h3>This conversation</h3>
          <Line label="turn" value={facts.turn + 1} />
          <Line
            label="emotional context"
            value={<span className={`chip ${riskTone(facts.risk)}`}>{facts.risk}</span>}
          />
          <Line label="sentiment" value={facts.sentiment} />
          <Line label="emotion" value={facts.emotion} />
          <Line label="feeling trend" value={facts.emotionalTrend} />
          <Line label="safety trend" value={facts.riskTrend} />

          {(facts.learnedName || facts.style || facts.topicsToAvoid.length > 0) && (
            <>
              <h3>Remembered from chat</h3>
              {facts.learnedName && <Line label="name" value={facts.learnedName} />}
              {facts.style && <Line label="style" value={facts.style} />}
              {facts.topicsToAvoid.length > 0 && (
                <Line label="avoid" value={facts.topicsToAvoid.join(', ')} />
              )}
            </>
          )}
          <p className="insights-note">
            These signals help the assistant respond with more care. They are
            conversation hints from a research prototype - not a medical
            assessment, and not a diagnosis. You can review or clear what is
            remembered under Settings → Personalization.
          </p>
        </>
      )}
    </aside>
  )
}
