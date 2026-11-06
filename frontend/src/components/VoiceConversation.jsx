import { useEffect, useRef, useState } from 'react'
import { LANGUAGE_NAMES, readStoredLanguage } from '../i18n.js'
import { VOICE_STATUS_LABELS } from '../voice/conversation.js'
import { readVoiceSettings, writeVoiceSettings } from '../voice/settings.js'
import { useVoiceSession } from '../voice/useVoiceSession.js'
import { Icon } from './ui.jsx'

/**
 * Voice Conversation Mode (Phase 24). [OURS]
 *
 * A dedicated overlay for the hands-free loop: one mount = one click =
 * mic on, WebSocket connected, VAD turn-taking, Whisper in, Piper out.
 * The transcript panel shows both sides live; the controls cover
 * mute/unmute, pause/resume, settings and end. Text Chat stays mounted
 * underneath - this is an additional interface, never a replacement.
 *
 * Ending the session (or unmounting) releases the microphone tracks,
 * the AudioContext and the WebSocket; every voice turn is persisted by
 * the backend on the same conversation as text chat.
 */
export function VoiceConversation({
  sessionId = null,
  onReply,
  onSessionId,
  onUserText,
  onClose,
}) {
  // The conversation language at start time (same source as dictation).
  const [language] = useState(() => readStoredLanguage())
  const [showSettings, setShowSettings] = useState(false)
  const [rate, setRate] = useState(() => readVoiceSettings().ttsRate)
  const transcriptRef = useRef(null)

  const { state, end, pause, resume, toggleMute } = useVoiceSession({
    language,
    sessionId,
    onReply,
    onSessionId,
    onUserText,
    autoStart: true, // the one click that opened this overlay starts it all
  })

  // Keep the newest line visible.
  useEffect(() => {
    const el = transcriptRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [state.transcript.length, state.aiText, state.status])

  // Escape = end the conversation (and clean up).
  useEffect(() => {
    function onKey(event) {
      if (event.key === 'Escape') handleEnd()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  function handleEnd() {
    end()
    onClose?.()
  }

  const errored = state.status === 'error'
  const languageName = LANGUAGE_NAMES?.[language] ?? language.toUpperCase()

  return (
    <div className="voice-overlay" role="dialog" aria-modal="true" aria-label="Voice conversation">
      <div className="voice-panel">
        <div className="voice-head">
          <div className={`voice-status voice-status-${state.status}`}>
            <span className="voice-dot" aria-hidden="true" />
            <span className="voice-status-text">
              {VOICE_STATUS_LABELS[state.status] ?? state.status}
            </span>
            {state.tts && (
              <span className="voice-chip">
                {state.tts === 'browser' ? 'browser voice' : 'neural voice'}
              </span>
            )}
          </div>
          <button
            type="button"
            className="btn-icon"
            onClick={handleEnd}
            aria-label="Close voice conversation"
          >
            <Icon name="close" size={16} />
          </button>
        </div>

        {/* Mic level bar: proof the mic is live (and a barge-in cue). */}
        <div className="voice-level" aria-hidden="true">
          <div
            className={`voice-level-fill${state.micMuted ? ' voice-level-muted' : ''}`}
            style={{ transform: `scaleX(${state.micMuted ? 0 : state.level})` }}
          />
        </div>

        <div className="voice-transcript" ref={transcriptRef} aria-live="polite">
          {state.transcript.length === 0 && !errored && (
            <p className="voice-hint muted">
              Speak naturally - Sage listens, answers aloud, then listens
              again. You can interrupt while Sage is talking.
            </p>
          )}
          {state.transcript.map((line) => (
            <div key={line.id} className={`voice-line voice-line-${line.role}`}>
              <span className="voice-who">{line.role === 'user' ? 'You' : 'Sage'}</span>
              <p>{line.text}</p>
            </div>
          ))}
          {state.status === 'thinking' && (
            <div className="voice-line voice-line-assistant voice-pending">
              <span className="voice-who">Sage</span>
              <p>Thinking…</p>
            </div>
          )}
          {state.status === 'speaking' && (
            <div className="voice-line voice-line-assistant voice-pending">
              <span className="voice-who">Sage</span>
              <p>Speaking…</p>
            </div>
          )}
        </div>

        {state.error && (
          <div className="notice notice-error" role="alert">
            {state.error.detail}
          </div>
        )}

        <div className="voice-controls">
          <button
            type="button"
            className={`btn btn-sm${state.micMuted ? ' btn-warn' : ''}`}
            onClick={toggleMute}
            aria-pressed={state.micMuted}
            disabled={errored}
          >
            <Icon name="mic" size={15} />
            {state.micMuted ? 'Unmute' : 'Mute'}
          </button>
          {state.status === 'paused' ? (
            <button type="button" className="btn btn-sm" onClick={resume}>
              <Icon name="play" size={15} />
              Resume
            </button>
          ) : (
            <button
              type="button"
              className="btn btn-sm"
              onClick={pause}
              disabled={errored || state.status === 'idle' || state.status === 'connecting'}
            >
              <Icon name="pause" size={15} />
              Pause
            </button>
          )}
          <button
            type="button"
            className="btn btn-sm"
            aria-pressed={showSettings}
            onClick={() => setShowSettings((v) => !v)}
          >
            Settings
            <Icon name="chevron" size={14} />
          </button>
          <button type="button" className="btn btn-sm btn-danger" onClick={handleEnd}>
            <Icon name="stop" size={14} />
            End
          </button>
        </div>

        {showSettings && (
          <div className="voice-settings">
            <div className="voice-setting-row">
              <span>Language</span>
              <span className="voice-chip">{languageName}</span>
            </div>
            <div className="voice-setting-row">
              <span>Speech engine</span>
              <span className="voice-chip">
                {state.tts === 'browser' ? 'browser TTS' : 'Piper (server)'}
              </span>
            </div>
            <label className="voice-setting-row" htmlFor="voice-rate">
              <span>Browser speech rate</span>
              <input
                id="voice-rate"
                type="range"
                min="0.5"
                max="2"
                step="0.1"
                value={rate}
                onChange={(event) => {
                  const value = Number(event.target.value)
                  setRate(value)
                  writeVoiceSettings({ ...readVoiceSettings(), ttsRate: value })
                }}
              />
              <span className="voice-chip">{rate.toFixed(1)}x</span>
            </label>
            <p className="muted voice-note">
              Voice turns are saved to this conversation with the same safety
              checks as typed messages. Sage is a research prototype, not a
              medical service.
            </p>
          </div>
        )}

        <p className="voice-note muted">
          {errored
            ? 'Close this panel and choose Voice again to reconnect.'
            : 'In an emergency, contact your local emergency services.'}
        </p>
      </div>
    </div>
  )
}
