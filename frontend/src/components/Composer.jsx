import { useEffect, useRef, useState } from 'react'
import { Icon } from './ui.jsx'

/**
 * Multiline composer: Enter sends, Shift+Enter adds a newline.
 * `initial` seeds the box once (suggested starters); `disabled` while a
 * reply is in flight; `onRetry` shows the retry affordance for a failed
 * send instead of silently dropping the text.
 */
export function Composer({ onSend, busy, initial = '' }) {
  const [value, setValue] = useState(initial)
  const ref = useRef(null)

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
    onSend(text)
  }

  return (
    <div className="composer">
      <textarea
        ref={ref}
        value={value}
        rows={1}
        placeholder="Share what's on your mind…"
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
  )
}
