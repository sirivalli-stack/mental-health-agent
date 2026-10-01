import { sessionShort } from '../format.js'

export function Header({ sessionId, profile, onProfileChange, onNewSession, chips }) {
  return (
    <header className="header">
      <div className="brand">
        <strong>State-aware support prototype</strong>
        <span className="muted small">research build - not a medical service</span>
      </div>
      <div className="header-controls">
        <label className="small">
          profile{' '}
          <select
            value={profile}
            onChange={(e) => onProfileChange(e.target.value)}
            title="Ablation profile sent with each request; 'default' uses the server setting"
          >
            <option value="">default</option>
            <option value="A">A - LLM only</option>
            <option value="B">B + affect</option>
            <option value="C">C + risk</option>
            <option value="D">D full state</option>
          </select>
        </label>
        <span className="chip chip-plain" title={sessionId || 'no session yet'}>
          session: {sessionShort(sessionId)}
        </span>
        {chips.map((c) => (
          <span
            key={c.name}
            className={`chip ${c.ok ? 'chip-ok' : 'chip-warn'}`}
            title={c.ok ? `${c.name} on` : `${c.name} off`}
          >
            {c.text}
          </span>
        ))}
        <button type="button" onClick={onNewSession}>
          New session
        </button>
      </div>
    </header>
  )
}
