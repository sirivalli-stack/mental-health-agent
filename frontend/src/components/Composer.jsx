import { useState } from 'react'

export function Composer({ onSend, busy }) {
  const [value, setValue] = useState('')

  function submit() {
    const text = value.trim()
    if (!text || busy) return
    setValue('')
    onSend(text)
  }

  return (
    <div className="composer">
      <textarea
        value={value}
        rows={2}
        placeholder="Type a message. Enter sends, Shift+Enter adds a newline."
        disabled={busy}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            submit()
          }
        }}
      />
      <button type="button" onClick={submit} disabled={busy || !value.trim()}>
        {busy ? 'Working…' : 'Send'}
      </button>
    </div>
  )
}
