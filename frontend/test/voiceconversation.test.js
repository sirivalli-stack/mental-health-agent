/**
 * Phase 24: voice conversation mode (protocol reducer, capture maths,
 * player queue, socket URL).
 *
 * Node environment with injected fakes - no browser, no network. These
 * tests pin the contracts the hands-free loop depends on: every server
 * event maps to a sane UI state, barge-in/stop discards queued audio
 * instantly, PCM framing matches what the backend VAD expects, and the
 * WebSocket URL follows the same VITE_API_BASE rules as api.js.
 */
import { describe, expect, it } from 'vitest'
import {
  VOICE_STATUS_LABELS,
  initialVoiceState,
  voiceReducer,
  voiceSocketUrl,
} from '../src/voice/conversation.js'
import { floatToInt16, frameRms, int16ToBase64, TARGET_RATE } from '../src/voice/capture.js'
import { createPlayer } from '../src/voice/player.js'

// ---------------------------------------------------------------------------
// reducer: server events -> UI state
// ---------------------------------------------------------------------------

describe('voiceReducer', () => {
  it('starts idle and moves to connecting with a clean slate', () => {
    const connecting = voiceReducer(
      { ...initialVoiceState, transcript: [{ id: 'x', role: 'user', text: 'old' }] },
      { type: 'connect', language: 'ta', sessionId: 's-1' },
    )
    expect(connecting.status).toBe('connecting')
    expect(connecting.transcript).toEqual([])
    expect(connecting.language).toBe('ta')
    expect(connecting.sessionId).toBe('s-1')
    expect(connecting.error).toBeNull()
  })

  it('ready starts listening and adopts the server session/tts', () => {
    const ready = voiceReducer(
      { ...initialVoiceState, status: 'connecting' },
      { type: 'ready', session_id: 'abc', tts: 'piper', language: 'en' },
    )
    expect(ready.status).toBe('listening')
    expect(ready.sessionId).toBe('abc')
    expect(ready.tts).toBe('piper')
  })

  it('maps every server phase onto the status pill', () => {
    let state = { ...initialVoiceState, status: 'listening' }
    for (const phase of ['thinking', 'speaking', 'listening', 'paused', 'listening']) {
      state = voiceReducer(state, { type: 'phase', state: phase })
      expect(state.status).toBe(phase)
      expect(VOICE_STATUS_LABELS[state.status]).toBeTruthy()
    }
  })

  it('builds a two-sided transcript in arrival order', () => {
    let state = voiceReducer({ ...initialVoiceState }, { type: 'connect' })
    state = voiceReducer(state, { type: 'user_transcript', text: 'I feel anxious' })
    state = voiceReducer(state, { type: 'reply', text: 'That sounds hard.', session_id: 's1' })
    state = voiceReducer(state, { type: 'user_transcript', text: 'thank you' })
    state = voiceReducer(state, { type: 'reply', text: 'Anytime.', session_id: 's1' })
    expect(state.transcript.map((l) => [l.role, l.text])).toEqual([
      ['user', 'I feel anxious'],
      ['assistant', 'That sounds hard.'],
      ['user', 'thank you'],
      ['assistant', 'Anytime.'],
    ])
    expect(state.aiText).toBe('Anytime.')
    expect(state.sessionId).toBe('s1')
  })

  it('interrupted only rewinds a speaking state', () => {
    const speaking = { ...initialVoiceState, status: 'speaking' }
    expect(voiceReducer(speaking, { type: 'interrupted' }).status).toBe('listening')
    const thinking = { ...initialVoiceState, status: 'thinking' }
    expect(voiceReducer(thinking, { type: 'interrupted' }).status).toBe('thinking')
  })

  it('skipped no_speech clears the live line but keeps the transcript', () => {
    const base = {
      ...initialVoiceState,
      userText: 'mumbled',
      transcript: [{ id: 'u1', role: 'user', text: 'mumbled' }],
    }
    const cleared = voiceReducer(base, { type: 'skipped', reason: 'no_speech' })
    expect(cleared.userText).toBe('')
    expect(cleared.transcript).toHaveLength(1)
    const deduped = voiceReducer(base, { type: 'skipped', reason: 'duplicate' })
    expect(deduped.userText).toBe('mumbled')
  })

  it('mutes without changing the conversation status', () => {
    const listening = { ...initialVoiceState, status: 'listening' }
    const muted = voiceReducer(listening, { type: 'mute', muted: true })
    expect(muted.micMuted).toBe(true)
    expect(muted.status).toBe('listening')
    expect(voiceReducer(muted, { type: 'mute', muted: false }).micMuted).toBe(false)
  })

  it('keeps non-fatal errors recoverable and fatal ones terminal', () => {
    const listening = { ...initialVoiceState, status: 'listening' }
    const noise = voiceReducer(listening, {
      type: 'error',
      code: 'stt_failed',
      detail: 'boom',
    })
    expect(noise.status).toBe('listening') // session survives
    expect(noise.error.code).toBe('stt_failed')
    const unauthorized = voiceReducer(listening, {
      type: 'error',
      code: 'unauthorized',
      detail: 'sign in',
    })
    expect(unauthorized.status).toBe('error')
    const connecting = voiceReducer({ ...initialVoiceState, status: 'connecting' }, {
      type: 'error',
      code: 'mic_denied',
      detail: 'denied',
    })
    expect(connecting.status).toBe('error')
  })

  it('end resets the session but keeps the resolved conversation id', () => {
    let state = voiceReducer({ ...initialVoiceState }, { type: 'connect' })
    state = voiceReducer(state, { type: 'ready', session_id: 'keep-me', tts: 'piper' })
    state = voiceReducer(state, { type: 'reply', text: 'hi', session_id: 'keep-me' })
    const ended = voiceReducer(state, { type: 'end' })
    expect(ended.status).toBe('idle')
    expect(ended.transcript).toEqual([])
    expect(ended.sessionId).toBe('keep-me')
  })

  it('ignores unknown events instead of crashing', () => {
    const before = { ...initialVoiceState, status: 'listening' }
    expect(voiceReducer(before, { type: 'quantum_teleport' })).toEqual(before)
    expect(voiceReducer(before, null)).toEqual(before)
  })
})

// ---------------------------------------------------------------------------
// socket URL (mirrors VITE_API_BASE handling)
// ---------------------------------------------------------------------------

describe('voiceSocketUrl', () => {
  it('uses same-origin by default', () => {
    // node has no location: the deterministic fallback must still be a
    // valid ws URL pointing at the voice endpoint.
    expect(voiceSocketUrl('')).toMatch(/^wss?:\/\/.+\/ws\/voice$/)
  })

  it('converts an http API base to ws', () => {
    expect(voiceSocketUrl('http://127.0.0.1:8000')).toBe(
      'ws://127.0.0.1:8000/ws/voice',
    )
  })

  it('converts an https API base to wss and trims slashes', () => {
    expect(voiceSocketUrl('https://sage.example/')).toBe(
      'wss://sage.example/ws/voice',
    )
  })
})

// ---------------------------------------------------------------------------
// capture maths (frames the backend VAD/Whisper expect)
// ---------------------------------------------------------------------------

describe('capture framing', () => {
  it('quantises float samples to int16 with clipping', () => {
    const pcm = floatToInt16(new Float32Array([0, 0.5, -0.5, 2, -2]), TARGET_RATE)
    expect(pcm).toHaveLength(5)
    expect(pcm[0]).toBe(0)
    expect(pcm[1]).toBe(Math.trunc(0.5 * 0x7fff)) // 16383.5 -> 16383
    expect(pcm[2]).toBe(-0x8000 / 2) // exactly -16384
    expect(pcm[3]).toBe(0x7fff) // clipped, no wrap-around
    expect(pcm[4]).toBe(-0x8000)
  })

  it('resamples a non-16k input to the backend rate', () => {
    const input = new Float32Array(4800) // 48 kHz, 100 ms
    const pcm = floatToInt16(input, 48_000)
    expect(pcm).toHaveLength(1600) // 100 ms @ 16 kHz
  })

  it('round-trips through base64', () => {
    const pcm = new Int16Array([0, 1, -1, 32767, -32768, 1234])
    const b64 = int16ToBase64(pcm)
    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0))
    expect(bytes).toHaveLength(pcm.byteLength)
    const back = new Int16Array(bytes.buffer)
    expect(Array.from(back)).toEqual(Array.from(pcm))
  })

  it('reports frame RMS for level + barge-in', () => {
    expect(frameRms(new Float32Array(0))).toBe(0)
    expect(frameRms(new Float32Array([0, 0, 0]))).toBe(0)
    expect(frameRms(new Float32Array([1, -1]))).toBeCloseTo(1, 6)
    expect(frameRms(new Float32Array([0.5, 0.5]))).toBeCloseTo(0.5, 6)
  })
})

// ---------------------------------------------------------------------------
// player: ordered playback + instant stop (barge-in)
// ---------------------------------------------------------------------------

function makeFakeAudio() {
  const sources = []
  const ctx = {
    state: 'running',
    resume: async () => {},
    destination: {},
    close: async () => {},
    decodeAudioData: async (buffer) => ({ byteLength: buffer.byteLength }),
    createBufferSource() {
      const source = {
        buffer: null,
        onended: null,
        started: false,
        stopped: false,
        connect() {},
        start() {
          this.started = true
        },
        stop() {
          this.stopped = true
        },
      }
      sources.push(source)
      return source
    },
  }
  return { ctx, sources }
}

/** Regular function (not arrow): the player does `new AudioContextCtor()`. */
function fakeAudioCtor(ctx) {
  return function FakeAudioContext() {
    return ctx
  }
}

const CHUNK = btoa('RIFFfake-wav-bytes')

describe('voice player', () => {
  it('plays queued chunks in order and reports speaking', async () => {
    const { ctx, sources } = makeFakeAudio()
    const player = createPlayer({ AudioContextCtor: fakeAudioCtor(ctx), synthesis: null })
    expect(player.isSpeaking()).toBe(false)
    player.enqueue(CHUNK)
    player.enqueue(CHUNK)
    await new Promise((r) => setTimeout(r, 0))
    // First chunk decoded and started; the second waits its turn.
    expect(sources).toHaveLength(1)
    expect(sources[0].started).toBe(true)
    expect(player.isSpeaking()).toBe(true)
    expect(player.pendingChunks()).toBe(1)
    // Natural end of chunk one promotes chunk two.
    sources[0].onended()
    await new Promise((r) => setTimeout(r, 0))
    expect(sources).toHaveLength(2)
    expect(sources[1].started).toBe(true)
    // Finishing the last chunk clears the speaking flag.
    sources[1].onended()
    expect(player.isSpeaking()).toBe(false)
    player.dispose()
  })

  it('stop() kills playback and drops the queue immediately', async () => {
    const { ctx, sources } = makeFakeAudio()
    const player = createPlayer({ AudioContextCtor: fakeAudioCtor(ctx), synthesis: null })
    player.enqueue(CHUNK)
    await new Promise((r) => setTimeout(r, 0))
    expect(sources[0].started).toBe(true)
    player.enqueue(CHUNK)
    player.enqueue(CHUNK)
    expect(player.pendingChunks()).toBe(2)
    player.stop() // barge-in
    expect(player.isSpeaking()).toBe(false)
    expect(player.pendingChunks()).toBe(0)
    expect(sources[0].stopped).toBe(true)
    // A chunk still decoding when stop() landed must not start later.
    player.enqueue(CHUNK)
    player.stop()
    await new Promise((r) => setTimeout(r, 0))
    expect(sources.every((s) => !s.started || s.stopped)).toBe(true)
    player.dispose()
  })

  it('skips an undecodable chunk instead of stalling the reply', async () => {
    const { ctx, sources } = makeFakeAudio()
    const player = createPlayer({ AudioContextCtor: fakeAudioCtor(ctx), synthesis: null })
    // atob() throws on this one before decodeAudioData is even reached;
    // the good chunk behind it must still play.
    player.enqueue('!!!not-base64!!!')
    player.enqueue(CHUNK)
    await new Promise((r) => setTimeout(r, 0))
    expect(sources).toHaveLength(1)
    expect(sources[0].started).toBe(true)
    player.dispose()
  })
})
