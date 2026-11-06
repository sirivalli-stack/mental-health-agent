/**
 * Voice conversation state machine + protocol (Phase 24). [OURS]
 *
 * Pure module: no DOM, no network - just the reducer that maps WebSocket
 * server events (and local control events) onto the overlay UI state.
 * This is what makes the whole conversation flow unit-testable in node.
 *
 * Server events consumed here (see backend/app/api/voice.py):
 *   ready | phase | user_transcript | reply | audio | tts_start |
 *   tts_end | tts_fallback | interrupted | skipped | error | stopped | pong
 *
 * Client-only events: connect | mute | level | end.
 */

export const initialVoiceState = {
  /** idle | connecting | listening | thinking | speaking | paused | error */
  status: 'idle',
  micMuted: false,
  /** Conversation id (null until the first reply resolves a new one). */
  sessionId: null,
  language: 'en',
  /** Final user utterance of the current turn. */
  userText: '',
  /** Last AI reply text (also what browser-TTS fallback speaks). */
  aiText: '',
  /** Live transcript shown in the panel: {id, role, text}[]. */
  transcript: [],
  /** Backend that will speak: 'piper' | 'browser' (fallback). */
  tts: null,
  /** {code, detail} of the last error, or null. */
  error: null,
  /** Mic level 0..1 for the visualiser. */
  level: 0,
}

export const VOICE_STATUS_LABELS = {
  idle: 'Not connected',
  connecting: 'Connecting…',
  listening: 'Listening…',
  thinking: 'Thinking…',
  speaking: 'Speaking…',
  paused: 'Paused',
  error: 'Voice connection ended',
}

let lineSeq = 0

/**
 * Reduce the previous UI state with one protocol event.
 * Never throws: unknown events are ignored so a newer server message
 * can not crash an older client.
 */
export function voiceReducer(state, event) {
  switch (event?.type) {
    case 'connect':
      return {
        ...state,
        status: 'connecting',
        error: null,
        transcript: [],
        userText: '',
        aiText: '',
        micMuted: false,
        level: 0,
        sessionId: event.sessionId ?? state.sessionId,
        language: event.language ?? state.language,
      }

    case 'ready':
      return {
        ...state,
        status: 'listening',
        sessionId: event.session_id ?? state.sessionId,
        tts: event.tts ?? state.tts,
        language: event.language ?? state.language,
        error: null,
      }

    case 'phase': {
      const next = event.state
      if (next === 'paused') return { ...state, status: 'paused' }
      if (next === 'listening' || next === 'thinking' || next === 'speaking') {
        return { ...state, status: next }
      }
      return state
    }

    case 'user_transcript': {
      lineSeq += 1
      return {
        ...state,
        userText: event.text ?? '',
        transcript: [
          ...state.transcript,
          { id: `u${lineSeq}`, role: 'user', text: event.text ?? '' },
        ],
      }
    }

    case 'reply': {
      lineSeq += 1
      return {
        ...state,
        aiText: event.text ?? '',
        sessionId: event.session_id ?? state.sessionId,
        transcript: [
          ...state.transcript,
          { id: `a${lineSeq}`, role: 'assistant', text: event.text ?? '' },
        ],
      }
    }

    case 'interrupted':
      // The server stopped speaking; `phase listening` normally follows.
      return state.status === 'speaking' ? { ...state, status: 'listening' } : state

    case 'skipped':
      // Noise or a duplicate utterance: nothing to show, stay in the loop.
      return event.reason === 'no_speech' ? { ...state, userText: '' } : state

    case 'mute':
      return { ...state, micMuted: Boolean(event.muted) }

    case 'level':
      return { ...state, level: Math.max(0, Math.min(1, event.value ?? 0)) }

    case 'tts_backend':
      return { ...state, tts: event.value ?? state.tts }

    case 'error': {
      const fatal = ['unauthorized', 'voice_disabled', 'conversation_not_found'].includes(
        event.code,
      )
      const connecting = state.status === 'connecting'
      return {
        ...state,
        error: { code: event.code, detail: event.detail ?? 'Voice error.' },
        status: fatal || connecting ? 'error' : state.status,
      }
    }

    case 'end':
      // Keep the session id (the UI may navigate to it) but reset the rest.
      return {
        ...initialVoiceState,
        sessionId: state.sessionId,
        language: state.language,
      }

    default:
      return state
  }
}

/**
 * WebSocket URL for the voice endpoint.
 *
 * `base` mirrors `VITE_API_BASE` from api.js: '' means same-origin (the
 * vite proxy forwards `/ws` to FastAPI), an http(s) origin is converted
 * to ws(s).
 */
export function voiceSocketUrl(base = '') {
  const trimmed = String(base ?? '').replace(/\/$/, '')
  if (/^https?:\/\//.test(trimmed)) {
    return `${trimmed.replace(/^http/, 'ws')}/ws/voice`
  }
  if (trimmed) return `${trimmed}/ws/voice`
  if (typeof globalThis.location !== 'undefined' && globalThis.location.host) {
    const scheme = globalThis.location.protocol === 'https:' ? 'wss' : 'ws'
    return `${scheme}://${globalThis.location.host}/ws/voice`
  }
  return 'ws://localhost:5173/ws/voice'
}

/** True while the server is (or may be) playing audio right now. */
export function isSpeakingStatus(status) {
  return status === 'speaking'
}
