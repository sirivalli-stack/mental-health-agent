/**
 * Phase 5: language helper tests for src/i18n.js.
 *
 * The routing rule must mirror the backend's app.i18n.scripts exactly -
 * these tests pin the contract so the UI (voice language, analysis hints)
 * and the pipeline never disagree about what "non-Latin input" means.
 */
import { describe, expect, it } from 'vitest'
import {
  CORE_LANGUAGES,
  LANGUAGE_NAMES,
  detectScript,
  needsAnalysisTranslation,
} from '../src/i18n'

const TELUGU = 'నాకు చాలా ఒత్తిడిగా ఉంది'
const HINDI = 'आज मुझे बहुत तनाव हो रहा है'
const TAMIL = 'இன்று எனக்கு மிகுந்த மன அழுத்தம்'
const KANNADA = 'ಇಂದು ನನಗೆ ತುಂಬಾ ಒತ್ತಡ ಇದೆ'
const CODE_MIXED = 'aaj bahut tension ho raha hai'

describe('detectScript', () => {
  it('identifies the four first-class Indic scripts', () => {
    expect(detectScript(TELUGU)).toBe('telugu')
    expect(detectScript(HINDI)).toBe('devanagari')
    expect(detectScript(TAMIL)).toBe('tamil')
    expect(detectScript(KANNADA)).toBe('kannada')
  })

  it('reports latin for English and transliterated code-mixing', () => {
    expect(detectScript('I feel stressed about my exams')).toBe('latin')
    expect(detectScript(CODE_MIXED)).toBe('latin')
    expect(detectScript('Bonjour, je suis anxieux')).toBe('latin')
  })

  it('reports unknown when there are no letters', () => {
    expect(detectScript('')).toBe('unknown')
    expect(detectScript('123 ... !!!')).toBe('unknown')
    expect(detectScript(null)).toBe('unknown')
    expect(detectScript(undefined)).toBe('unknown')
  })

  it('lets Latin win ties in mixed input', () => {
    // mostly-English sentence with a couple of native-script words
    expect(detectScript(`I am fine ${KANNADA.slice(0, 4)} ok`)).toBe('latin')
  })
})

describe('needsAnalysisTranslation', () => {
  it('routes substantial non-Latin input to translation', () => {
    expect(needsAnalysisTranslation(TELUGU)).toBe(true)
    expect(needsAnalysisTranslation(HINDI)).toBe(true)
    expect(needsAnalysisTranslation(TAMIL)).toBe(true)
    expect(needsAnalysisTranslation(KANNADA)).toBe(true)
  })

  it('never routes Latin-script input', () => {
    expect(needsAnalysisTranslation('I feel fine today')).toBe(false)
    expect(needsAnalysisTranslation(CODE_MIXED)).toBe(false)
    expect(needsAnalysisTranslation('')).toBe(false)
  })

  it('needs at least two Indic letters (stray word stays untouched)', () => {
    expect(needsAnalysisTranslation(`I feel fine ${KANNADA.slice(0, 1)} ok`)).toBe(
      false,
    )
  })

  it('routes Indic-dominant code-mixed input', () => {
    expect(needsAnalysisTranslation(`${TELUGU} today is hard`)).toBe(true)
  })
})

describe('language metadata', () => {
  it('exposes the five first-class languages', () => {
    expect(CORE_LANGUAGES).toEqual(['en', 'te', 'hi', 'ta', 'kn'])
  })

  it('names every language the settings selector offers', () => {
    for (const code of ['en', 'te', 'hi', 'ta', 'kn', 'mr', 'ur', 'ar', 'zh']) {
      expect(LANGUAGE_NAMES[code]).toBeTruthy()
    }
    expect(LANGUAGE_NAMES.kn).toBe('ಕನ್ನಡ')
    expect(LANGUAGE_NAMES.en).toBe('English')
  })
})
