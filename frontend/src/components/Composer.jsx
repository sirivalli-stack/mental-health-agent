import { useEffect, useRef, useState } from 'react'
import { readStoredLanguage } from '../i18n.js'
import { useDictation, useVoiceSettings } from '../voice/hooks.js'
import { Icon } from './ui.jsx'

/**
 * Multiline composer: Enter sends, Shift+Enter adds a newline.
 * `initial` seeds the box once (suggested starters); `disabled` while a
 * reply is in flight.
 *
 * Phase 6 (voice): a microphone button runs browser speech recognition
 * when voice input is enabled. Recognized text is appended to the draft
 * for the user to review and edit - it is NEVER sent automatically.
 * Unsupported browsers simply hide the mic; permission or recognition
 * errors surface as an inline notice with a way back to typing.
 */
export function Composer({ onSend, busy, initial = '' }) {
  const [value, setValue] = useState(initial)
  const [voice, setVoiceSettings] = useVoiceSettings()
  const ref = useRef(null)

  const dictation = useDictation({
    // read the language at start-time so a mid-chat switch applies live
    lang: () => readStoredLanguage(),
    onFinal: (text) => {
      setValue((prev) => {
        const draft = prev.trim()
        return draft ? `${draft} ${text.trim()}` : text.trim()
      })
      ref.current?.focus()
    },
  })

  useEffect(() => {
    if (initial) {
      setValue(initial)
      ref.current?.focus()
    }
  }, [initial])

  // grow with the content, up to a cap handled in CSS
  useEffect(() => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`
  }, [value])

  function submit() {
    const text = value.trim()
    if (!text || busy) return
    setValue('')
    dictation.reset()
    onSend(text)
  }

  const showMic = voice.sttEnabled
  const micListening = dictation.listening

  return (
    <div className="composer-slot">
      <div className="composer">
        {showMic && (
          <button
            type="button"
            className={`mic-btn${micListening ? ' mic-listening' : ''}`}
            onClick={() => (micListening ? dictation.stop() : dictation.start())}
            disabled={busy || !dictation.supported}
            aria-label={micListening ? 'Stop listening' : 'Speak your message'}
            aria-pressed={micListening}
            title={
              dictation.supported
                ? micListening
                  ? 'Stop listening'
                  : 'Speak your message'
                : 'Speech input is not supported in this browser'
            }
          >
            <Icon name="mic" size={16} />
            {micListening && <span className="mic-pulse" aria-hidden="true" />}
          </button>
        )}
        <textarea
          ref={ref}
          value={value}
          rows={1}
          placeholder={micListening ? 'Listening…' : "Share what's on your mind…"}
          aria-label="Message"
          disabled={busy}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              submit()
            }
          }}
        />
        <button
          type="button"
          className="send-btn"
          onClick={submit}
          disabled={busy || !value.trim()}
          aria-label="Send message"
          title="Send (Enter)"
        >
          <Icon name="send" size={18} />
        </button>
      </div>
      {micListening && dictation.interim && (
        <p className="voice-interim" aria-live="polite">
          {dictation.interim}
        </p>
      )}
      {dictation.error && (
        <div className="notice notice-error voice-error" role="alert">
          <span style={{ flex: 1 }}>{dictation.error}</span>
          <button type="button" className="btn btn-sm" onClick={dictation.reset}>
            Dismiss
          </button>
        </div>
      )}
    </div>
  )
}
