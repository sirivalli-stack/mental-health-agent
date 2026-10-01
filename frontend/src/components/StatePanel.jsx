import { stateFacts, riskTone } from '../format.js'

function Line({ label, value }) {
  return (
    <div className="fact">
      <span className="fact-label">{label}</span>
      <span className="fact-value">{value}</span>
    </div>
  )
}

export function StatePanel({ state }) {
  const facts = stateFacts(state)
  if (!facts) {
    return (
      <aside className="state-panel">
        <h2>User state</h2>
        <p className="muted">
          The state object the model receives is built after the first message.
        </p>
      </aside>
    )
  }
  return (
    <aside className="state-panel">
      <h2>User state</h2>
      <Line label="turn" value={facts.turn} />
      <Line
        label="risk"
        value={
          <span className={`chip ${riskTone(facts.risk)}`}>{facts.risk}</span>
        }
      />
      <Line label="sentiment" value={facts.sentiment} />
      <Line label="emotion" value={facts.emotion} />
      <Line label="risk trend" value={facts.riskTrend} />
      <Line label="feeling trend" value={facts.emotionalTrend} />
      <Line
        label="rule hits"
        value={facts.ruleHits.length ? facts.ruleHits.join(', ') : 'none'}
      />
      {facts.previousRisk && (
        <Line label="previous risk" value={facts.previousRisk} />
      )}
      {(facts.learnedName || facts.style || facts.topicsToAvoid.length > 0) && (
        <>
          <h2>Learned profile</h2>
          {facts.learnedName && <Line label="name" value={facts.learnedName} />}
          {facts.style && <Line label="style" value={facts.style} />}
          {facts.topicsToAvoid.length > 0 && (
            <Line label="avoid" value={facts.topicsToAvoid.join(', ')} />
          )}
        </>
      )}
      <p className="muted small">
        Prototype heuristic, not a diagnosis. Trends need three turns.
      </p>
    </aside>
  )
}
