/**
 * Text-to-speech via SpeechSynthesis (Phase 6).
 *
 * A tiny controller around the browser's speech synthesis with an
 * explicit state machine: idle -> speaking <-> paused -> idle. The state
 * is observable (subscribe) so bubbles can show a speaking indicator and
 * only one utterance ever plays at a time - starting a new one stops the
 * previous. Nothing is recorded or persisted; the OS voices read the
 * reply text that is already on screen.
 *
 * `synthesis` and `UtteranceCtor` are injectable so the whole state
 * machine is unit-testable in the node environment.
 */

/** Controller state: 'idle' | 'speaking' | 'paused' */
export class SpeechController {
  constructor({ synthesis, UtteranceCtor } = {}) {
    this.synthesis = synthesis ?? (typeof globalThis.speechSynthesis !== 'undefined' ? globalThis.speechSynthesis : null)
    this.UtteranceCtor = UtteranceCtor ?? (typeof globalThis.SpeechSynthesisUtterance !== 'undefined' ? globalThis.SpeechSynthesisUtterance : null)
    this.state = 'idle'
    this.listeners = new Set()
    this.currentText = null
  }

  get supported() {
    return Boolean(this.synthesis && this.UtteranceCtor)
  }

  subscribe(fn) {
    this.listeners.add(fn)
    return () => this.listeners.delete(fn)
  }

  /** Snapshot subscribers receive: { state, currentText }. */
  _emit() {
    const snapshot = { state: this.state, currentText: this.currentText }
    for (const fn of this.listeners) fn(snapshot)
  }

  /**
   * Transition to `state`, emitting only on a real change. `force`
   * re-emits even when the state is unchanged - used when `currentText`
   * switched (starting another bubble's reply while already speaking).
   */
  _set(state, force = false) {
    const changed = this.state !== state
    this.state = state
    if (changed || force) this._emit()
  }

  /** Voices list; [] when unsupported or the list is not ready yet. */
  voices() {
    try {
      return this.supported ? this.synthesis.getVoices() : []
    } catch {
      return []
    }
  }

  /**
   * Speak `text`. Picks `voiceURI` when that voice still exists,
   * otherwise the browser default (it may auto-pick by locale).
   * Returns false when speech is unsupported or text is empty.
   */
  speak(text, { voiceURI = '', rate = 1 } = {}) {
    if (!this.supported || !text || !text.trim()) return false
    try {
      if (this.state !== 'idle') this.stop() // never two utterances at once
      const utterance = new this.UtteranceCtor(text)
      utterance.rate = rate
      utterance.onend = () => {
        if (this.currentText === text) {
          this.currentText = null
          this._set('idle')
        }
      }
      utterance.onerror = () => {
        this.currentText = null
        this._set('idle')
      }
      const voices = this.voices()
      const picked = voiceURI && voices.find((v) => v.voiceURI === voiceURI)
      if (picked) utterance.voice = picked
      this.currentText = text
      this._set('speaking', true) // force: currentText just changed
      this.synthesis.speak(utterance)
      return true
    } catch {
      this.currentText = null
      this._set('idle')
      return false
    }
  }

  pause() {
    if (!this.supported || this.state !== 'speaking') return
    try {
      this.synthesis.pause()
      this._set('paused')
    } catch {
      /* ignore - state stays as-is */
    }
  }

  resume() {
    if (!this.supported || this.state !== 'paused') return
    try {
      this.synthesis.resume()
      this._set('speaking')
    } catch {
      /* ignore */
    }
  }

  stop() {
    if (!this.supported) return
    try {
      this.synthesis.cancel()
    } catch {
      /* ignore */
    }
    this.currentText = null
    this._set('idle')
  }

  /** Replay the same text from the start (after stop or completion). */
  replay(text, opts) {
    return this.speak(text, opts)
  }

  /** Play/pause toggle bound to one bubble's text. */
  toggle(text, opts) {
    if (this.state === 'speaking' && this.currentText === text) {
      this.pause()
      return
    }
    if (this.state === 'paused' && this.currentText === text) {
      this.resume()
      return
    }
    this.speak(text, opts)
  }
}

/** App-wide singleton (created lazily so tests never touch globals). */
let singleton = null
export function speechController() {
  if (!singleton) singleton = new SpeechController()
  return singleton
}
