/**
 * Speech-to-text via the Web Speech API (Phase 6).
 *
 * Frontend-only: recognition happens in the browser, audio never leaves
 * the device, and nothing is recorded or stored - the interim transcript
 * simply lands in the composer as editable text the user must review
 * before sending (we never auto-send recognized speech).
 *
 * Every failure mode has an explicit, human message: unsupported
 * browser, denied permission, no speech heard, network loss, device
 * error. Nothing here touches the FastAPI backend.
 */

/** Map our language codes to BCP-47 recognition locales. */
const STT_LOCALES = {
  en: 'en-US',
  te: 'te-IN',
  hi: 'hi-IN',
  ta: 'ta-IN',
  kn: 'kn-IN',
}

/** Language code -> speech-recognition locale (falls back to en-US). */
export function sttLocale(code) {
  return STT_LOCALES[code] || 'en-US'
}

/** Constructor of the Web Speech API in this browser, or null. */
export function getSpeechRecognition(win = globalThis) {
  return win.SpeechRecognition || win.webkitSpeechRecognition || null
}

export function isSttSupported(win = globalThis) {
  return Boolean(getSpeechRecognition(win))
}

/**
 * Web Speech error codes -> user-facing messages. Kept as data so the
 * mapping is testable without a browser.
 */
export const STT_ERROR_MESSAGES = {
  'not-allowed': 'Microphone permission was denied. Allow the microphone in your browser settings and try again.',
  'service-not-allowed': 'Speech recognition was blocked by the browser. Check site permissions for the microphone.',
  'no-speech': 'I did not hear anything. Tap the microphone and speak, then try again.',
  'audio-capture': 'No microphone was found. Check that a microphone is connected.',
  network: 'The speech service needs a network connection and it is unavailable right now.',
  aborted: null, // user stopped it on purpose - show nothing
}

export function sttErrorMessage(code) {
  if (code in STT_ERROR_MESSAGES) return STT_ERROR_MESSAGES[code]
  return 'Something went wrong with speech input. You can keep typing instead.'
}

/**
 * Create a configured recognition instance.
 *
 * @param {object} opts
 * @param {string} opts.locale  BCP-47 locale (see sttLocale)
 * @param {(text: string, final: boolean) => void} opts.onTranscript
 * @param {(code: string) => void} opts.onError  Web Speech error code
 * @param {() => void} [opts.onEnd]  fired after stop() or natural end
 * @param {object} [opts.win]  injectable window (tests)
 */
export function createRecognition({
  locale,
  onTranscript,
  onError,
  onEnd,
  win = globalThis,
}) {
  const Ctor = getSpeechRecognition(win)
  if (!Ctor) return null
  const rec = new Ctor()
  rec.lang = locale
  rec.continuous = false // one utterance at a time - review before send
  rec.interimResults = true
  rec.maxAlternatives = 1

  rec.onresult = (event) => {
    let text = ''
    let final = false
    for (let i = event.resultIndex; i < event.results.length; i += 1) {
      const result = event.results[i]
      text += result[0].transcript
      if (result.isFinal) final = true
    }
    if (text) onTranscript(text, final)
  }
  rec.onerror = (event) => onError?.(event.error ?? 'unknown')
  rec.onend = () => onEnd?.()

  return rec
}
