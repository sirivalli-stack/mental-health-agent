/**
 * React glue for the voice layer (Phase 6). Hooks keep all lifecycle
 * work (start/stop/cleanup on unmount) out of the components.
 *
 * Both hooks are fully backend-free: dictation uses the Web Speech API
 * (audio stays in the browser, transcripts land in the composer for
 * review) and read-aloud uses SpeechSynthesis (nothing recorded).
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { readVoiceSettings, writeVoiceSettings } from './settings.js'
import { createRecognition, isSttSupported, sttErrorMessage, sttLocale } from './stt.js'
import { speechController } from './tts.js'

/**
 * Dictation hook: start/stop Web Speech recognition for one language.
 *
 * `onFinal` fires with each final transcript chunk; the component decides
 * what to do (the Composer *appends* it to the draft - the user reviews
 * and sends manually, we never auto-send recognized speech).
 * `interim` holds the in-progress text for the listening indicator.
 *
 * `lang` may be a code or a getter (so a mid-session language switch is
 * picked up on the next start without re-creating the hook).
 *
 * Returns { supported, listening, interim, error, start, stop, reset }.
 */
export function useDictation({ lang = 'en', onFinal } = {}) {
  const [supported] = useState(() => isSttSupported())
  const [listening, setListening] = useState(false)
  const [interim, setInterim] = useState('')
  const [error, setError] = useState(null)
  const recRef = useRef(null)
  const onFinalRef = useRef(onFinal)
  onFinalRef.current = onFinal
  const langRef = useRef(lang)
  langRef.current = lang

  const stop = useCallback(() => {
    const rec = recRef.current
    recRef.current = null
    try {
      rec?.stop()
    } catch {
      /* already stopped */
    }
    setListening(false)
    setInterim('')
  }, [])

  const start = useCallback(() => {
    if (!supported) {
      setError('Speech input is not supported in this browser. You can keep typing instead.')
      return
    }
    if (recRef.current) return // already listening
    setError(null)
    const langCode = typeof langRef.current === 'function' ? langRef.current() : langRef.current
    const rec = createRecognition({
      locale: sttLocale(langCode),
      onTranscript: (text, final) => {
        if (final) {
          setInterim('')
          onFinalRef.current?.(text)
        } else {
          setInterim(text)
        }
      },
      onError: (code) => {
        const message = sttErrorMessage(code)
        if (message) setError(message)
      },
      onEnd: () => {
        recRef.current = null
        setListening(false)
        setInterim('')
      },
    })
    if (!rec) {
      setError('Speech input is not supported in this browser. You can keep typing instead.')
      return
    }
    recRef.current = rec
    try {
      rec.start() // triggers the browser permission prompt on first use
      setListening(true)
    } catch {
      recRef.current = null
      setError('Could not start the microphone. Check your system input settings.')
    }
  }, [supported])

  const reset = useCallback(() => setError(null), [])

  // never leave a recognition session running after unmount
  useEffect(() => stop, [stop])

  return { supported, listening, interim, error, start, stop, reset }
}

/**
 * Read-aloud state for assistant bubbles. Subscribes to the shared
 * SpeechController; the snapshot is { state, currentText } where `state`
 * is 'idle' | 'speaking' | 'paused' and `currentText` is the reply
 * currently being spoken (so each bubble can render its own controls).
 */
export function useReadAloud() {
  const controller = speechController()
  const [snapshot, setSnapshot] = useState(() => ({
    state: controller.state,
    currentText: controller.currentText,
  }))

  useEffect(() => controller.subscribe(setSnapshot), [controller])

  const toggle = useCallback(
    (text, opts) => controller.toggle(text, opts),
    [controller],
  )
  const stop = useCallback(() => controller.stop(), [controller])
  const replay = useCallback(
    (text, opts) => controller.replay(text, opts),
    [controller],
  )

  return {
    supported: controller.supported,
    state: snapshot.state,
    currentText: snapshot.currentText,
    toggle,
    stop,
    replay,
  }
}

/**
 * Device-local voice settings, kept in sync across components: writes go
 * through the shared store (which broadcasts a window event) so the
 * Composer and Settings agree immediately.
 */
export function useVoiceSettings() {
  const [settings, setSettings] = useState(() => readVoiceSettings())

  useEffect(() => {
    const onChange = () => setSettings(readVoiceSettings())
    globalThis.addEventListener?.('mha:voice-changed', onChange)
    return () => globalThis.removeEventListener?.('mha:voice-changed', onChange)
  }, [])

  const update = useCallback((patch) => {
    writeVoiceSettings(patch) // merges, persists, broadcasts mha:voice-changed
    setSettings(readVoiceSettings()) // mirror for instant render
  }, [])

  return [settings, update]
}
