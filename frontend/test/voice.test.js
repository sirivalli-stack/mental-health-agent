/**
 * Phase 6: voice layer tests (STT/TTS/settings).
 *
 * Everything here runs in the node environment with injected fakes - no
 * browser, no network. The tests pin the safety-relevant contracts:
 * recognition results are never auto-sent (the composer owns that flow),
 * settings persist device-locally with sane fallbacks, error codes map
 * to human messages, and the read-aloud state machine never overlaps
 * utterances.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  DEFAULT_VOICE_SETTINGS,
  SpeechController,
  VOICE_STORAGE_KEY,
  createRecognition,
  isSttSupported,
  normalizeVoiceSettings,
  readVoiceSettings,
  sttErrorMessage,
  sttLocale,
  writeVoiceSettings,
} from '../src/voice/index.js'

// ---------------------------------------------------------------------------
// settings (device-local, fail-safe)
// ---------------------------------------------------------------------------

function fakeStorage(initial = null) {
  const map = new Map()
  if (initial !== null) map.set(VOICE_STORAGE_KEY, initial)
  return {
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => map.set(k, String(v)),
    removeItem: (k) => map.delete(k),
  }
}

describe('voice settings', () => {
  it('returns defaults when storage is empty or missing', () => {
    expect(readVoiceSettings(fakeStorage())).toEqual(DEFAULT_VOICE_SETTINGS)
    expect(readVoiceSettings({ getItem: () => { throw new Error('blocked') } })).toEqual(
      DEFAULT_VOICE_SETTINGS,
    )
    expect(readVoiceSettings(null)).toEqual(DEFAULT_VOICE_SETTINGS)
  })

  it('round-trips a patch through storage', () => {
    const storage = fakeStorage()
    writeVoiceSettings({ ttsEnabled: false }, storage)
    const reread = readVoiceSettings(storage)
    expect(reread.ttsEnabled).toBe(false)
    expect(reread.sttEnabled).toBe(true) // untouched default survives
  })

  it('survives corrupted JSON', () => {
    const storage = fakeStorage('{not json')
    expect(readVoiceSettings(storage)).toEqual(DEFAULT_VOICE_SETTINGS)
  })

  it('normalizes invalid values instead of trusting them', () => {
    const out = normalizeVoiceSettings({
      ttsRate: 99, // out of range
      ttsVoice: 123, // wrong type
      sttEnabled: 'yes', // wrong type
      evil: 'drop me',
    })
    expect(out.ttsRate).toBe(1)
    expect(out.ttsVoice).toBe('')
    expect(out.sttEnabled).toBe(true)
    expect(out.evil).toBeUndefined()
  })

  it('clamps a valid custom rate and keeps it', () => {
    expect(normalizeVoiceSettings({ ttsRate: 1.5 }).ttsRate).toBe(1.5)
    expect(normalizeVoiceSettings({ ttsRate: 0.75 }).ttsRate).toBe(0.75)
  })
})

// ---------------------------------------------------------------------------
// STT (locale mapping, support detection, error messages, recognition shape)
// ---------------------------------------------------------------------------

describe('speech-to-text', () => {
  it('maps first-class languages to speech locales', () => {
    expect(sttLocale('en')).toBe('en-US')
    expect(sttLocale('te')).toBe('te-IN')
    expect(sttLocale('hi')).toBe('hi-IN')
    expect(sttLocale('ta')).toBe('ta-IN')
    expect(sttLocale('kn')).toBe('kn-IN')
    expect(sttLocale('xx')).toBe('en-US') // unknown -> safe default
    expect(sttLocale(undefined)).toBe('en-US')
  })

  it('detects support from the window shape', () => {
    expect(isSttSupported({})).toBe(false)
    expect(isSttSupported({ SpeechRecognition: class {} })).toBe(true)
    expect(isSttSupported({ webkitSpeechRecognition: class {} })).toBe(true)
    // legacy alias wins if both exist
    expect(
      isSttSupported({ SpeechRecognition: null, webkitSpeechRecognition: class {} }),
    ).toBe(true)
  })

  it('maps every Web Speech error code to a helpful message', () => {
    expect(sttErrorMessage('not-allowed')).toMatch(/permission.*denied/i)
    expect(sttErrorMessage('no-speech')).toMatch(/did not hear/i)
    expect(sttErrorMessage('network')).toMatch(/network/i)
    expect(sttErrorMessage('audio-capture')).toMatch(/microphone/i)
    expect(sttErrorMessage('service-not-allowed')).toMatch(/blocked/i)
    // intentional stop is silent
    expect(sttErrorMessage('aborted')).toBeNull()
    // unknown codes still get something actionable
    expect(sttErrorMessage('mystery-code')).toMatch(/typing instead/i)
  })

  it('creates a recognition bound to one locale with interim results', () => {
    class FakeRec {
      constructor() {
        this.started = false
      }
      start() {
        this.started = true
      }
      stop() {
        this.started = false
      }
    }
    const seen = []
    const rec = createRecognition({
      locale: 'te-IN',
      onTranscript: (text, final) => seen.push([text, final]),
      onError: () => {},
      win: { webkitSpeechRecognition: FakeRec },
    })
    expect(rec).not.toBeNull()
    expect(rec.lang).toBe('te-IN')
    expect(rec.interimResults).toBe(true)
    expect(rec.continuous).toBe(false) // one utterance, then review before send

    // drive result events in the real Web Speech shape: `results` always
    // holds every result so far, `resultIndex` is the first changed one
    const interim = [{ 0: { transcript: 'నాకు బాగా' }, isFinal: false }]
    rec.onresult({ resultIndex: 0, results: interim })
    const final = [
      interim[0],
      { 0: { transcript: ' ఒత్తిడి ఉంది' }, isFinal: true },
    ]
    rec.onresult({ resultIndex: 1, results: final })
    expect(seen).toEqual([['నాకు బాగా', false], [' ఒత్తిడి ఉంది', true]])
  })

  it('returns null (not a throw) when the browser lacks support', () => {
    expect(createRecognition({ locale: 'en-US', onTranscript() {}, onError() {}, win: {} })).toBeNull()
  })

  it('forwards Web Speech error codes to the callback', () => {
    class FakeRec {}
    const errors = []
    const rec = createRecognition({
      locale: 'en-US',
      onTranscript() {},
      onError: (code) => errors.push(code),
      win: { SpeechRecognition: FakeRec },
    })
    rec.onerror({ error: 'not-allowed' })
    expect(errors).toEqual(['not-allowed'])
  })
})

// ---------------------------------------------------------------------------
// TTS controller (state machine with injected synthesis)
// ---------------------------------------------------------------------------

function fakeSpeech() {
  const calls = []
  return {
    calls,
    speak(utterance) {
      calls.push(['speak', utterance.text])
      this._current = utterance
    },
    pause() {
      calls.push(['pause'])
    },
    resume() {
      calls.push(['resume'])
    },
    cancel() {
      calls.push(['cancel'])
      const cur = this._current
      this._current = null
      // browsers fire onend (or onerror) on cancel; simulate onend
      if (cur?.onend) cur.onend()
    },
    getVoices() {
      return [
        { voiceURI: 'v-en', name: 'English US', lang: 'en-US' },
        { voiceURI: 'v-te', name: 'Telugu India', lang: 'te-IN' },
      ]
    },
  }
}

class FakeUtterance {
  constructor(text) {
    this.text = text
    this.rate = 1
    this.voice = null
    this.onend = null
    this.onerror = null
  }
}

function makeController() {
  const synthesis = fakeSpeech()
  const controller = new SpeechController({ synthesis, UtteranceCtor: FakeUtterance })
  return { synthesis, controller }
}

describe('read-aloud controller', () => {
  let controller
  let synthesis

  afterEach(() => {
    controller = null
    synthesis = null
  })

  it('is unsupported when synthesis is missing', () => {
    const bare = new SpeechController({ synthesis: null, UtteranceCtor: null })
    expect(bare.supported).toBe(false)
    expect(bare.speak('hello')).toBe(false)
    expect(bare.voices()).toEqual([])
  })

  it('speaks and returns to idle when the utterance ends', () => {
    const made = makeController()
    controller = made.controller
    synthesis = made.synthesis.calls
    const states = []
    controller.subscribe((snap) => states.push(snap.state))

    expect(controller.speak('hello there')).toBe(true)
    expect(controller.state).toBe('speaking')
    // finish the utterance
    controller.synthesis._current.onend()
    expect(controller.state).toBe('idle')
    expect(states).toEqual(['speaking', 'idle'])
  })

  it('rejects empty text without touching synthesis', () => {
    const made = makeController()
    controller = made.controller
    expect(controller.speak('   ')).toBe(false)
    expect(controller.state).toBe('idle')
    expect(made.synthesis.calls.length).toBe(0)
  })

  it('pause -> resume -> stop transitions are guarded', () => {
    const made = makeController()
    controller = made.controller
    synthesis = made.synthesis.calls

    controller.pause() // not speaking - must be a no-op
    expect(controller.state).toBe('idle')

    controller.speak('one')
    controller.pause()
    expect(controller.state).toBe('paused')
    expect(synthesis).toContainEqual(['pause'])

    controller.resume()
    expect(controller.state).toBe('speaking')
    expect(synthesis).toContainEqual(['resume'])

    controller.stop()
    expect(controller.state).toBe('idle')
    expect(synthesis).toContainEqual(['cancel'])
  })

  it('never overlaps: starting a new reply cancels the previous one', () => {
    const made = makeController()
    controller = made.controller
    synthesis = made.synthesis.calls

    controller.speak('first reply')
    controller.speak('second reply')
    const cancels = synthesis.filter(([op]) => op === 'cancel').length
    expect(cancels).toBe(1)
    expect(synthesis[synthesis.length - 1]).toEqual(['speak', 'second reply'])
    expect(controller.currentText).toBe('second reply')
    expect(controller.state).toBe('speaking')
  })

  it('toggle drives pause/resume for the current text only', () => {
    const made = makeController()
    controller = made.controller

    controller.toggle('alpha')
    expect(controller.state).toBe('speaking')
    controller.toggle('alpha') // same text -> pause
    expect(controller.state).toBe('paused')
    controller.toggle('alpha') // same text -> resume
    expect(controller.state).toBe('speaking')
    controller.toggle('beta') // different text -> restart with beta
    expect(controller.currentText).toBe('beta')
    expect(controller.state).toBe('speaking')
  })

  it('replay restarts the same text from idle', () => {
    const made = makeController()
    controller = made.controller
    controller.speak('again')
    controller.stop()
    expect(controller.replay('again')).toBe(true)
    expect(controller.state).toBe('speaking')
  })

  it('applies the chosen voice when it still exists', () => {
    const made = makeController()
    controller = made.controller
    controller.speak('hello', { voiceURI: 'v-te' })
    expect(made.synthesis.calls[made.synthesis.calls.length - 1]).toEqual(['speak', 'hello'])
    expect(controller.synthesis._current.voice.voiceURI).toBe('v-te')
  })

  it('notifies subscribers with state and currentText snapshots', () => {
    const made = makeController()
    controller = made.controller
    const snaps = []
    controller.subscribe((s) => snaps.push(s))
    controller.speak('hi')
    controller.stop()
    expect(snaps).toEqual([
      { state: 'speaking', currentText: 'hi' },
      { state: 'idle', currentText: null },
    ])
  })

  it('returns to idle when synthesis errors mid-utterance', () => {
    const made = makeController()
    controller = made.controller
    controller.speak('oops')
    controller.synthesis._current.onerror?.()
    expect(controller.state).toBe('idle')
    expect(controller.currentText).toBeNull()
  })
})
