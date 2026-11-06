/**
 * Reply audio playback for the voice conversation (Phase 24). [OURS]
 *
 * Plays the WAV chunks the server streams during TTS, in order, through
 * the Web Audio API, and falls back to the browser's SpeechSynthesis
 * when the backend has no Piper voice (`tts_fallback`).
 *
 * The whole point of this module is *instant* stop: barge-in must cut
 * the audio the moment the user talks. `stop()` bumps a generation
 * counter, clears the queue and kills the current source - any chunk
 * still being decoded is discarded when it lands (stale generation).
 *
 * The AudioContext is injectable so the queue/stop logic is testable
 * in node with a fake.
 */

/** @param {{AudioContextCtor?: Function, synthesis?: SpeechSynthesis|null, UtteranceCtor?: Function}} [opts] */
export function createPlayer(opts = {}) {
  const AudioContextCtor =
    opts.AudioContextCtor ??
    globalThis.AudioContext ??
    globalThis.webkitAudioContext
  const synthesis =
    opts.synthesis !== undefined
      ? opts.synthesis
      : typeof globalThis.speechSynthesis !== 'undefined'
        ? globalThis.speechSynthesis
        : null
  const UtteranceCtor =
    opts.UtteranceCtor ??
    (typeof globalThis.SpeechSynthesisUtterance !== 'undefined'
      ? globalThis.SpeechSynthesisUtterance
      : null)

  let ctx = null
  let queue = []
  let current = null
  let inFlight = false // a decode is running (chunk not yet playing)
  let generation = 0
  let speaking = false // server audio in flight/playing
  let browserSpeaking = false

  function ensureCtx() {
    if (!AudioContextCtor) throw new Error('Web Audio is not supported.')
    if (!ctx) ctx = new AudioContextCtor()
    if (ctx.state === 'suspended') ctx.resume().catch(() => {})
    return ctx
  }

  async function decode(b64) {
    const context = ensureCtx()
    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0))
    // decodeAudioData copies; give it its own ArrayBuffer.
    return context.decodeAudioData(bytes.buffer.slice(0))
  }

  async function pump() {
    // One decode at a time: bursts of chunks must queue, not overlap.
    if (inFlight || current || queue.length === 0) return
    const gen = generation
    const b64 = queue.shift()
    inFlight = true
    try {
      const buffer = await decode(b64)
      if (gen !== generation) return // stopped while decoding
      const source = ctx.createBufferSource()
      source.buffer = buffer
      source.connect(ctx.destination)
      current = source
      speaking = true
      source.onended = () => {
        current = null
        if (gen !== generation) {
          speaking = false
          return
        }
        if (queue.length === 0) {
          speaking = false // the whole reply finished playing
        } else {
          pump().catch(() => {})
        }
      }
      source.start()
    } catch {
      // Undecodable chunk: skip it rather than stall the whole reply.
    } finally {
      inFlight = false
    }
    if (gen === generation && !current && queue.length > 0) {
      pump().catch(() => {})
    }
  }

  return {
    /** Queue one base64 WAV chunk from the server. */
    enqueue(b64) {
      if (!b64) return
      queue.push(b64)
      pump().catch(() => {})
    },

    /** Stop server audio AND browser TTS immediately (barge-in / pause / end). */
    stop() {
      generation += 1
      queue = []
      if (current) {
        try {
          current.onended = null
          current.stop()
        } catch {
          /* already stopped */
        }
        current = null
      }
      speaking = false
      if (synthesis) {
        try {
          synthesis.cancel()
        } catch {
          /* not supported */
        }
      }
      browserSpeaking = false
    },

    /** Browser-TTS fallback when the backend has no Piper voice. */
    speak(text, { rate = 1 } = {}) {
      this.stop()
      if (!text || !synthesis || !UtteranceCtor) return false
      try {
        const utterance = new UtteranceCtor(text)
        utterance.rate = rate
        utterance.onend = () => {
          browserSpeaking = false
        }
        synthesis.speak(utterance)
        browserSpeaking = true
        return true
      } catch {
        return false
      }
    },

    isSpeaking: () => speaking,
    isBrowserSpeaking: () => browserSpeaking,
    pendingChunks: () => queue.length,
    /** Test seam: close the AudioContext. */
    dispose() {
      this.stop()
      if (ctx && ctx.state !== 'closed') ctx.close().catch(() => {})
      ctx = null
    },
  }
}
