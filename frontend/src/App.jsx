import { useEffect, useState } from 'react'
import { fetchHealth, sendChat, resetSession } from './api.js'
import { healthChips, toExchange } from './format.js'
import { Composer } from './components/Composer.jsx'
import { DisclaimerBanner } from './components/DisclaimerBanner.jsx'
import { Header } from './components/Header.jsx'
import { MessageList } from './components/MessageList.jsx'
import { StatePanel } from './components/StatePanel.jsx'

let bubbleSeq = 0
function userBubble(text) {
  bubbleSeq += 1
  return { id: `user-${bubbleSeq}-${Date.now()}`, role: 'user', text }
}

export default function App() {
  const [messages, setMessages] = useState([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [sessionId, setSessionId] = useState(null)
  const [state, setState] = useState(null)
  const [profile, setProfile] = useState('')
  const [disclaimer, setDisclaimer] = useState('')
  const [chips, setChips] = useState([])

  useEffect(() => {
    let cancelled = false
    fetchHealth()
      .then((health) => {
        if (cancelled) return
        setDisclaimer(health.disclaimer ?? '')
        setChips(healthChips(health.components ?? []))
      })
      .catch((e) => {
        if (!cancelled) setError(e.message)
      })
    return () => {
      cancelled = true
    }
  }, [])

  async function handleSend(text) {
    setMessages((prev) => [...prev, userBubble(text)])
    setBusy(true)
    setError(null)
    try {
      const response = await sendChat({
        message: text,
        sessionId,
        profile: profile || undefined,
      })
      setMessages((prev) => [...prev, toExchange(response)])
      setSessionId(response.session_id)
      setState(response.state ?? null)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  async function handleNewSession() {
    if (sessionId) {
      try {
        await resetSession(sessionId)
      } catch {
        // the session may already be gone server-side; local reset still counts
      }
    }
    setMessages([])
    setSessionId(null)
    setState(null)
    setError(null)
  }

  return (
    <div className="app">
      <DisclaimerBanner text={disclaimer} />
      <Header
        sessionId={sessionId}
        profile={profile}
        onProfileChange={setProfile}
        onNewSession={handleNewSession}
        chips={chips}
      />
      {error && (
        <div className="error" role="alert">
          <span>{error}</span>
          <button type="button" onClick={() => setError(null)}>
            dismiss
          </button>
        </div>
      )}
      <main className="layout">
        <section className="chat">
          <MessageList messages={messages} busy={busy} />
          <Composer onSend={handleSend} busy={busy} />
        </section>
        <StatePanel state={state} />
      </main>
    </div>
  )
}
