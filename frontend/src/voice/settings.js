/**
 * Voice settings (Phase 6) - device-local, never sent to the server.
 *
 * Pure logic lives in `normalizeVoiceSettings` so it is unit-testable in
 * the node test environment; storage access is isolated in the two
 * wrappers below (localStorage can be missing in private mode - every
 * read degrades to defaults, every write is best-effort).
 */

export const VOICE_STORAGE_KEY = 'mha.voice'

export const DEFAULT_VOICE_SETTINGS = {
  /** Read assistant replies aloud (play button on each bubble). */
  ttsEnabled: true,
  /** Show the microphone button in the composer. */
  sttEnabled: true,
  /** System voice URI to use with TTS; '' = browser default. */
  ttsVoice: '',
  /** Speech rate for TTS (0.5 - 2). */
  ttsRate: 1,
}

/**
 * Coerce anything stored (or handed in) into a valid settings object.
 * Unknown keys are dropped; invalid values fall back to defaults so a
 * corrupted store can never break the app.
 */
export function normalizeVoiceSettings(raw) {
  const src = raw && typeof raw === 'object' ? raw : {}
  const rate = Number(src.ttsRate)
  return {
    ttsEnabled: typeof src.ttsEnabled === 'boolean' ? src.ttsEnabled : DEFAULT_VOICE_SETTINGS.ttsEnabled,
    sttEnabled: typeof src.sttEnabled === 'boolean' ? src.sttEnabled : DEFAULT_VOICE_SETTINGS.sttEnabled,
    ttsVoice: typeof src.ttsVoice === 'string' ? src.ttsVoice.slice(0, 300) : '',
    ttsRate: Number.isFinite(rate) && rate >= 0.5 && rate <= 2 ? rate : 1,
  }
}

/** Read settings; missing/broken storage returns defaults. */
export function readVoiceSettings(storage = globalThis.localStorage) {
  try {
    const raw = storage?.getItem(VOICE_STORAGE_KEY)
    return normalizeVoiceSettings(raw ? JSON.parse(raw) : null)
  } catch {
    return { ...DEFAULT_VOICE_SETTINGS }
  }
}

/** Merge a patch into the stored settings; returns the merged result. */
export function writeVoiceSettings(patch, storage = globalThis.localStorage) {
  const next = normalizeVoiceSettings({ ...readVoiceSettings(storage), ...patch })
  try {
    storage?.setItem(VOICE_STORAGE_KEY, JSON.stringify(next))
  } catch {
    // storage unavailable (private mode) - settings still apply for this call
  }
  // let live components (Composer, bubbles) re-read without prop drilling
  try {
    globalThis.dispatchEvent?.(new Event('mha:voice-changed'))
  } catch {
    // no DOM (tests) - nothing to notify
  }
  return next
}
