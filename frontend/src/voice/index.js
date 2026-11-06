/**
 * Voice layer (Phase 6): browser speech in, browser speech out.
 *
 * - `stt.js`   - Web Speech API dictation -> editable composer text
 * - `tts.js`   - SpeechSynthesis read-aloud with play/pause/resume/stop
 * - `settings.js` - device-local voice preferences (localStorage only)
 *
 * Frontend-only by design: no backend endpoint, no audio upload, no
 * recording is stored anywhere. Recognition results are never sent
 * anywhere automatically - they land in the composer for review.
 */

export {
  DEFAULT_VOICE_SETTINGS,
  VOICE_STORAGE_KEY,
  normalizeVoiceSettings,
  readVoiceSettings,
  writeVoiceSettings,
} from './settings.js'

export {
  STT_ERROR_MESSAGES,
  createRecognition,
  getSpeechRecognition,
  isSttSupported,
  sttErrorMessage,
  sttLocale,
} from './stt.js'

export { SpeechController, speechController } from './tts.js'

// React glue (hooks) - components import from '../voice/hooks.js'
export { useDictation, useReadAloud, useVoiceSettings } from './hooks.js'
