/**
 * React glue for the voice conversation (Phase 24). [OURS]
 *
 * One click (`start()`) runs the whole hands-free loop:
 *
 *   mic permission -> WebSocket `/ws/voice` (start+token) -> capture
 *   frames -> server VAD/Whisper/pipeline/Piper -> replies as audio ->
 *   client re-listens, forever until `end()`.
 *
 * Responsibilities kept here (all lifecycle work, no rendering):
 * - connect/authenticate the socket and route server events to the UI
 *   reducer (`conversation.js`) and to Chat's transcript callbacks;
 * - own the capture + player instances and release them on end/unmount
 *   (mic tracks, AudioContext, WebSocket);
 * - client-side barge-in: stop playback the instant the user's voice is
 *   loud/long enough, and tell the server (`interrupt`) so it cancels
 *   the stream too. The server's own VAD barge-in (`interrupted`) is
 *   the authoritative fallback.
 */
import { useCallback, useEffect, useReducer, useRef } from 'react'
import { readToken } from '../auth.js'
import { readVoiceSettings } from './settings.js'
import { voiceReducer, initialVoiceState, voiceSocketUrl } from './conversation.js'
import { createCapture } from './capture.js'
import { createPlayer } from './player.js'

/** RMS sustained for 3 frames (~192 ms) counts as talking over the AI. */
const BARGE_RMS = 0.04
const BARGE_FRAMES = 3
/** Or one very loud frame (shout right at the mic). */
const BARGE_STRONG_RMS = 0.12

/**
 * @param {object} props
 * @param {string} [props.language] conversation language code (en/te/hi/ta/kn)
 * @param {string|null} [props.sessionId] continue an existing conversation
 * @param {string} [props.apiBase] mirrors VITE_API_BASE ('' = same-origin)
 * @param {(response: object) => void} [props.onReply] full ChatResponse -> append AI bubble
 * @param {(sessionId: string) => void} [props.onSessionId] first reply resolved the id
 * @param {(text: string) => void} [props.onUserText] final user transcript -> append bubble
 * @param {boolean} [props.autoStart] connect immediately on mount
 */
export function useVoiceSession({
  language = 'en',
  sessionId = null,
  apiBase = '',
  onReply,
  onSessionId,
  onUserText,
  autoStart = false,
} = {}) {
  const [state, dispatch] = useReducer(voiceReducer, initialVoiceState)

  const wsRef = useRef(null)
  const captureRef = useRef(null)
  const playerRef = useRef(null)
  const lastReplyRef = useRef('')
  const bargeRef = useRef({ count: 0, fired: false })
  // Run token: every start()/teardown() bumps it. An async start() that
  // wakes up to a newer token knows it was superseded (StrictMode's dev
  // double-effect runs cleanup + a second effect; a stale run must abort
  // before creating a second socket) and must not dispatch anything.
  const runRef = useRef(0)
  const stateRef = useRef(state)
  stateRef.current = state
  const cbsRef = useRef({ onReply, onSessionId, onUserText, language, sessionId, apiBase })
  cbsRef.current = { onReply, onSessionId, onUserText, language, sessionId, apiBase }

  // ---- outbound ----------------------------------------------------------
  const send = useCallback((payload) => {
    const ws = wsRef.current
    if (ws && ws.readyState === 1 /* OPEN */) {
      ws.send(JSON.stringify(payload))
    }
  }, [])

  // ---- release everything ------------------------------------------------
  const teardown = useCallback(() => {
    runRef.current += 1 // invalidate any in-flight start()
    try {
      captureRef.current?.stop()
    } catch {
      /* already stopped */
    }
    captureRef.current = null
    try {
      playerRef.current?.stop()
      playerRef.current?.dispose?.()
    } catch {
      /* already disposed */
    }
    playerRef.current = null
    const ws = wsRef.current
    wsRef.current = null
    if (ws) {
      try {
        ws.close(1000)
      } catch {
        /* already closed */
      }
    }
  }, [])

  // ---- server events -----------------------------------------------------
  const handleServer = useCallback(
    (msg) => {
      const player = playerRef.current
      switch (msg.type) {
        case 'ready':
          dispatch({ type: 'ready', ...msg })
          break
        case 'phase':
          dispatch({ type: 'phase', state: msg.state })
          break
        case 'user_transcript':
          dispatch({ type: 'user_transcript', text: msg.text })
          cbsRef.current.onUserText?.(msg.text)
          break
        case 'reply':
          lastReplyRef.current = msg.text ?? ''
          dispatch({ type: 'reply', text: msg.text, session_id: msg.session_id })
          cbsRef.current.onSessionId?.(msg.session_id)
          if (msg.response) cbsRef.current.onReply?.(msg.response)
          break
        case 'audio':
          player?.enqueue(msg.data)
          break
        case 'tts_fallback':
          // No Piper voice on the server: the browser reads the reply.
          dispatch({ type: 'tts_backend', value: 'browser' })
          player?.speak(lastReplyRef.current, { rate: readVoiceSettings().ttsRate })
          break
        case 'interrupted':
          // Server-side barge-in: cut playback instantly.
          player?.stop()
          break
        case 'skipped':
          dispatch({ type: 'skipped', reason: msg.reason })
          break
        case 'error':
          dispatch({ type: 'error', code: msg.code, detail: msg.detail })
          if (['unauthorized', 'voice_disabled', 'conversation_not_found'].includes(msg.code)) {
            teardown()
          }
          break
        case 'stopped':
        case 'pong':
          break
        default:
          break // unknown future event: ignore, never crash
      }
    },
    [teardown],
  )

  // ---- capture callbacks (barge-in lives here) ---------------------------
  const onFrame = useCallback(
    (b64, rms) => {
      const player = playerRef.current
      if (player?.isSpeaking()) {
        const barge = bargeRef.current
        if (rms >= BARGE_RMS) barge.count += 1
        else barge.count = 0
        if (!barge.fired && (barge.count >= BARGE_FRAMES || rms >= BARGE_STRONG_RMS)) {
          barge.fired = true
          player.stop() // instant local stop - no round trip
          send({ type: 'interrupt' })
        }
      } else {
        bargeRef.current.count = 0
        bargeRef.current.fired = false
      }
      send({ type: 'audio', data: b64 })
    },
    [send],
  )

  const onLevel = useCallback((rms) => {
    dispatch({ type: 'level', value: rms })
  }, [])

  // ---- start (ONE click; mic permission + socket together) ---------------
  const start = useCallback(async () => {
    if (wsRef.current) return // already connected
    const run = ++runRef.current
    dispatch({ type: 'connect', language: cbsRef.current.language, sessionId: cbsRef.current.sessionId })
    try {
      const player = createPlayer()
      playerRef.current = player
      const capture = createCapture({ onFrame, onLevel })
      captureRef.current = capture
      // Mic first: one permission prompt, and a clear failure if denied.
      await capture.start()
      if (run !== runRef.current) {
        // Superseded while awaiting the mic (unmount / StrictMode cleanup).
        capture.stop()
        return
      }

      const ws = new WebSocket(voiceSocketUrl(cbsRef.current.apiBase))
      wsRef.current = ws
      ws.onmessage = (event) => {
        if (run !== runRef.current) return
        try {
          handleServer(JSON.parse(event.data))
        } catch {
          /* malformed frame from a proxy: ignore */
        }
      }
      ws.onopen = () => {
        const token = readToken()
        ws.send(
          JSON.stringify({
            type: 'start',
            ...(token ? { token } : {}),
            ...(cbsRef.current.sessionId ? { session_id: cbsRef.current.sessionId } : {}),
            language: cbsRef.current.language,
          }),
        )
      }
      ws.onclose = (event) => {
        if (run !== runRef.current) return
        if (event.code === 4401) {
          dispatch({
            type: 'error',
            code: 'unauthorized',
            detail: 'Sign in to start a voice conversation.',
          })
        } else if (!stateRef.current.error) {
          dispatch({
            type: 'error',
            code: 'connection_lost',
            detail: 'The voice connection closed unexpectedly.',
          })
        }
        teardown()
      }
    } catch (err) {
      if (run !== runRef.current) return // superseded: the newer run owns the UI
      const denied =
        err?.name === 'NotAllowedError' || /permission|denied/i.test(err?.message ?? '')
      dispatch({
        type: 'error',
        code: denied ? 'mic_denied' : 'mic_unavailable',
        detail: denied
          ? 'Microphone access was denied. Allow the microphone and try again.'
          : `Could not start the microphone: ${err?.message ?? err}`,
      })
      teardown()
    }
  }, [handleServer, onFrame, onLevel, teardown])

  // ---- controls ----------------------------------------------------------
  const pause = useCallback(() => {
    playerRef.current?.stop()
    send({ type: 'pause' })
  }, [send])

  const resume = useCallback(() => send({ type: 'resume' }), [send])

  const toggleMute = useCallback(() => {
    const next = !stateRef.current.micMuted
    captureRef.current?.setMuted(next)
    dispatch({ type: 'mute', muted: next })
    send({ type: 'mute', muted: next })
  }, [send])

  const end = useCallback(() => {
    send({ type: 'stop' })
    teardown()
    dispatch({ type: 'end' })
  }, [send, teardown])

  // ---- unmount: release mic, audio and socket ---------------------------
  useEffect(() => {
    if (autoStart) start()
    return () => {
      teardown()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return {
    state,
    start,
    end,
    pause,
    resume,
    toggleMute,
    isActive: state.status !== 'idle' && state.status !== 'error',
  }
}
