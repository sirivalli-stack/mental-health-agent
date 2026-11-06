/**
 * Language helpers (Phase 5) - pure, unit-testable, no React.
 *
 * `detectScript` mirrors the backend's `app.i18n.scripts` routing rule so
 * the UI and the analysis pipeline agree on what "non-Latin input" means.
 * Phase 6 (voice) reuses these codes to pick the right speech-recognition
 * language.
 */

/** The five first-class languages of the product spec. */
export const CORE_LANGUAGES = ['en', 'te', 'hi', 'ta', 'kn']

// ---------------------------------------------------------------------------
// Device-local mirror of the conversation language (Phase 6 voice needs the
// locale at microphone start time without a network round-trip). Settings
// writes it whenever the server preference loads or changes; every read
// falls back to English for anonymous or first-run users.
// ---------------------------------------------------------------------------

const LANG_KEY = 'mha.lang'

/** Persist the current language code (device-local, validated). */
export function writeStoredLanguage(code) {
  try {
    if (LANGUAGE_NAMES[code]) globalThis.localStorage?.setItem(LANG_KEY, code)
  } catch {
    // storage unavailable - voice falls back to English
  }
}

/** Last chosen language code; 'en' when unknown or storage is missing. */
export function readStoredLanguage() {
  try {
    const code = globalThis.localStorage?.getItem(LANG_KEY)
    return LANGUAGE_NAMES[code] ? code : 'en'
  } catch {
    return 'en'
  }
}

/** Code -> endonym (the language's own name), matching Settings options. */
export const LANGUAGE_NAMES = {
  en: 'English',
  es: 'Español',
  fr: 'Français',
  de: 'Deutsch',
  it: 'Italiano',
  pt: 'Português',
  nl: 'Nederlands',
  hi: 'हिन्दी',
  kn: 'ಕನ್ನಡ',
  ar: 'العربية',
  zh: '中文',
  ja: '日本語',
  ko: '한국어',
  ru: 'Русский',
  bn: 'বাংলা',
  ta: 'தமிழ்',
  te: 'తెలుగు',
  mr: 'मराठी',
  ur: 'اردو',
}

const SCRIPT_RANGES = [
  ['devanagari', 0x0900, 0x097f],
  ['bengali', 0x0980, 0x09ff],
  ['gurmukhi', 0x0a00, 0x0a7f],
  ['gujarati', 0x0a80, 0x0aff],
  ['oriya', 0x0b00, 0x0b7f],
  ['tamil', 0x0b80, 0x0bff],
  ['telugu', 0x0c00, 0x0c7f],
  ['kannada', 0x0c80, 0x0cff],
  ['malayalam', 0x0d00, 0x0d7f],
  ['sinhala', 0x0d80, 0x0dff],
  ['arabic', 0x0600, 0x06ff],
  ['hebrew', 0x0590, 0x05ff],
  ['cyrillic', 0x0400, 0x04ff],
  ['greek', 0x0370, 0x03ff],
  ['han', 0x4e00, 0x9fff],
  ['hiragana', 0x3040, 0x309f],
  ['katakana', 0x30a0, 0x30ff],
  ['hangul', 0xac00, 0xd7af],
]

function scriptOf(ch) {
  const code = ch.codePointAt(0)
  if (code >= 0x0041 && code <= 0x024f) return 'latin'
  for (const [name, lo, hi] of SCRIPT_RANGES) {
    if (code >= lo && code <= hi) return name
  }
  if (code >= 0x0750 && code <= 0x077f) return 'arabic'
  return null
}

/** Dominant script: a SCRIPT_RANGES name, 'latin', or 'unknown'. */
export function detectScript(text) {
  const counts = new Map()
  for (const ch of String(text ?? '')) {
    if (!/\p{L}/u.test(ch)) continue
    const name = scriptOf(ch)
    if (name) counts.set(name, (counts.get(name) ?? 0) + 1)
  }
  if (counts.size === 0) return 'unknown'
  const latin = counts.get('latin') ?? 0
  let best = null
  let bestCount = 0
  for (const [name, n] of counts) {
    if (name !== 'latin' && n > bestCount) {
      best = name
      bestCount = n
    }
  }
  return best && bestCount > latin ? best : 'latin'
}

/**
 * True when the message is substantially non-Latin - the same routing
 * rule the backend uses to decide whether analysis runs on a translation.
 */
export function needsAnalysisTranslation(text) {
  let tracked = 0
  let latin = 0
  for (const ch of String(text ?? '')) {
    if (!/\p{L}/u.test(ch)) continue
    const name = scriptOf(ch)
    if (name === 'latin') latin += 1
    else if (name) tracked += 1
  }
  return tracked >= 2 && tracked >= latin
}
