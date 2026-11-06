/**
 * Microphone capture for the voice conversation (Phase 24). [OURS]
 *
 * Opens the mic once (one permission prompt per session), streams
 * 16 kHz mono int16 PCM frames to the server as base64, and reports the
 * per-frame RMS level for the UI visualiser and the client-side
 * barge-in check (stop the AI's audio the moment the user talks).
 *
 * Design notes:
 * - `AudioContext({sampleRate: 16000})` asks the browser for the rate
 *   the backend expects; when a browser ignores it we resample with
 *   linear interpolation (pure function, unit-tested in node).
 * - Muting stops *sending* frames but keeps the stream alive, so
 *   unmuting is instant (no second permission prompt, no reconnect).
 * - Nothing is recorded or stored: frames are forwarded and forgotten.
 */

/** Rate the backend's VAD + Whisper expect. */
export const TARGET_RATE = 16000

/** ScriptProcessor buffer: 1024 samples @16 kHz = 64 ms per frame. */
const BUFFER_SIZE = 1024

/**
 * Float32 samples in [-1, 1] -> Int16Array at TARGET_RATE.
 * `inputRate` differs only when the browser ignored our requested rate.
 */
export function floatToInt16(input, inputRate = TARGET_RATE) {
  if (!input || input.length === 0) return new Int16Array(0)
  if (inputRate === TARGET_RATE) {
    const out = new Int16Array(input.length)
    for (let i = 0; i < input.length; i += 1) {
      const s = input[i] < -1 ? -1 : input[i] > 1 ? 1 : input[i]
      out[i] = s < 0 ? s * 0x8000 : s * 0x7fff
    }
    return out
  }
  // Linear resample inputRate -> 16 kHz.
  const ratio = inputRate / TARGET_RATE
  const outLen = Math.max(1, Math.floor(input.length / ratio))
  const out = new Int16Array(outLen)
  for (let i = 0; i < outLen; i += 1) {
    const pos = i * ratio
    const i0 = Math.floor(pos)
    const i1 = Math.min(i0 + 1, input.length - 1)
    const frac = pos - i0
    const s = input[i0] * (1 - frac) + input[i1] * frac
    const c = s < -1 ? -1 : s > 1 ? 1 : s
    out[i] = c < 0 ? c * 0x8000 : c * 0x7fff
  }
  return out
}

/** Int16Array -> base64 (chunked; btoa caps argument sizes). */
export function int16ToBase64(samples) {
  if (!samples || samples.length === 0) return ''
  const bytes = new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength)
  let binary = ''
  const CHUNK = 0x8000
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK))
  }
  return btoa(binary)
}

/** RMS of a Float32 frame in 0..1 (0 = digital silence). */
export function frameRms(frame) {
  if (!frame || frame.length === 0) return 0
  let sum = 0
  for (let i = 0; i < frame.length; i += 1) sum += frame[i] * frame[i]
  return Math.sqrt(sum / frame.length)
}

/**
 * Create a capture controller.
 *
 * @param {object} opts
 * @param {(base64: string) => void} opts.onFrame  every PCM frame (when unmuted)
 * @param {(rms: number) => void}   [opts.onLevel] every frame's level (0 when muted)
 * @param {object} [opts.constraints] getUserMedia audio constraints
 * @param {object} [opts.deps]  injection for tests: {getUserMedia, AudioContext}
 * @returns {{start: () => Promise<void>, stop: () => void, setMuted: (m: boolean) => void, isMuted: () => boolean}}
 */
export function createCapture({ onFrame, onLevel, constraints, deps } = {}) {
  const getUserMedia =
    deps?.getUserMedia ??
    ((c) => navigator.mediaDevices.getUserMedia(c))
  const AudioContextCtor =
    deps?.AudioContext ??
    globalThis.AudioContext ??
    globalThis.webkitAudioContext

  let stream = null
  let ctx = null
  let source = null
  let processor = null
  let muted = false
  let running = false

  const audioConstraints = constraints ?? {
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  }

  async function start() {
    if (running) return
    if (!AudioContextCtor) {
      throw new Error('Web Audio is not supported in this browser.')
    }
    stream = await getUserMedia(audioConstraints)
    ctx = new AudioContextCtor({ sampleRate: TARGET_RATE })
    if (ctx.state === 'suspended') {
      try {
        await ctx.resume()
      } catch {
        /* autoplay policies may block until a gesture - the Start click covers it */
      }
    }
    source = ctx.createMediaStreamSource(stream)
    processor = ctx.createScriptProcessor(BUFFER_SIZE, 1, 1)
    processor.onaudioprocess = (event) => {
      const frame = event.inputBuffer.getChannelData(0)
      if (muted) {
        onLevel?.(0)
        return
      }
      const rms = frameRms(frame)
      onLevel?.(rms)
      const pcm = floatToInt16(frame, event.inputBuffer.sampleRate)
      const b64 = int16ToBase64(pcm)
      if (b64) onFrame?.(b64, rms)
    }
    source.connect(processor)
    // The processor must reach the destination to run; its output buffer
    // is never written, so this emits digital silence (no feedback).
    processor.connect(ctx.destination)
    running = true
  }

  function stop() {
    running = false
    try {
      processor?.disconnect()
      source?.disconnect()
    } catch {
      /* already gone */
    }
    processor = null
    source = null
    if (ctx && ctx.state !== 'closed') {
      ctx.close().catch(() => {})
    }
    ctx = null
    if (stream) {
      for (const track of stream.getTracks()) track.stop()
      stream = null
    }
    onLevel?.(0)
  }

  function setMuted(next) {
    muted = Boolean(next)
    if (muted) onLevel?.(0)
  }

  return {
    start,
    stop,
    setMuted,
    isMuted: () => muted,
    isRunning: () => running,
  }
}
