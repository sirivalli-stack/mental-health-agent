import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  SIZE_OPTIONS,
  THEME_OPTIONS,
  applyAppearance,
  readTextSize,
  readTheme,
  resolveTheme,
  watchSystemTheme,
  writeTextSize,
  writeTheme,
} from '../src/theme.js'

/** Minimal localStorage double; node has none by default. */
function fakeStorage(initial = {}) {
  const map = new Map(Object.entries(initial))
  return {
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => map.set(k, String(v)),
    removeItem: (k) => map.delete(k),
    _map: map,
  }
}

/** Minimal document double for applyAppearance. */
function fakeDocument() {
  const classes = new Set()
  return {
    documentElement: {
      dataset: {},
      classList: {
        add: (c) => classes.add(c),
        remove: (c) => classes.delete(c),
        contains: (c) => classes.has(c),
      },
    },
    _classes: classes,
  }
}

/** matchMedia double that records listeners and lets tests fire changes. */
function fakeMatchMedia(matches = false) {
  const listeners = []
  const mq = {
    matches,
    media: '(prefers-color-scheme: dark)',
    addEventListener: (type, cb) => type === 'change' && listeners.push(cb),
    removeEventListener: (type, cb) => {
      const i = listeners.indexOf(cb)
      if (i >= 0) listeners.splice(i, 1)
    },
    _fire: (nextMatches) => {
      mq.matches = nextMatches // like the real event: matches reflects the new state
      listeners.slice().forEach((cb) => cb({ matches: nextMatches }))
    },
    _listeners: listeners,
  }
  return mq
}

afterEach(() => {
  delete globalThis.localStorage
  delete globalThis.document
  delete globalThis.window
  vi.useRealTimers()
})

describe('theme options', () => {
  it('exposes exactly light, dark, system and standard/large sizes', () => {
    expect(THEME_OPTIONS).toEqual(['light', 'dark', 'system'])
    expect(SIZE_OPTIONS).toEqual(['standard', 'large'])
  })
})

describe('readTheme / writeTheme', () => {
  it('defaults to system when nothing is stored', () => {
    globalThis.localStorage = fakeStorage()
    expect(readTheme()).toBe('system')
  })

  it('persists and reads back a valid choice', () => {
    globalThis.localStorage = fakeStorage()
    writeTheme('dark')
    expect(readTheme()).toBe('dark')
    writeTheme('light')
    expect(readTheme()).toBe('light')
  })

  it('ignores corrupted values (falls back to system)', () => {
    globalThis.localStorage = fakeStorage({ 'mha.theme': 'neon' })
    expect(readTheme()).toBe('system')
  })

  it('falls back to system when storage throws (private mode)', () => {
    globalThis.localStorage = {
      getItem: () => {
        throw new Error('blocked')
      },
    }
    expect(readTheme()).toBe('system')
  })

  it('falls back to system when localStorage does not exist (node tests)', () => {
    expect(readTheme()).toBe('system')
  })

  it('removes the key when writing an unknown value', () => {
    const store = fakeStorage({ 'mha.theme': 'dark' })
    globalThis.localStorage = store
    writeTheme('chartreuse')
    expect(store.getItem('mha.theme')).toBeNull()
    expect(readTheme()).toBe('system')
  })
})

describe('text size', () => {
  it('defaults to standard and persists large', () => {
    globalThis.localStorage = fakeStorage()
    expect(readTextSize()).toBe('standard')
    writeTextSize('large')
    expect(readTextSize()).toBe('large')
  })

  it('rejects unknown sizes', () => {
    const store = fakeStorage({ 'mha.textsize': 'huge' })
    globalThis.localStorage = store
    expect(readTextSize()).toBe('standard')
    writeTextSize('huge')
    expect(store.getItem('mha.textsize')).toBeNull()
  })
})

describe('resolveTheme', () => {
  it('honours explicit preferences regardless of the OS', () => {
    expect(resolveTheme('light', false)).toBe('light')
    expect(resolveTheme('light', true)).toBe('light')
    expect(resolveTheme('dark', false)).toBe('dark')
    expect(resolveTheme('dark', true)).toBe('dark')
  })

  it('follows the OS for system (and unknown) preferences', () => {
    expect(resolveTheme('system', true)).toBe('dark')
    expect(resolveTheme('system', false)).toBe('light')
    expect(resolveTheme('banana', true)).toBe('dark')
    expect(resolveTheme('banana', false)).toBe('light')
  })
})

describe('applyAppearance', () => {
  it('is safe without a DOM and resolves from the media query', () => {
    expect(applyAppearance('system', 'standard', { matches: true })).toBe('dark')
    expect(applyAppearance('system', 'large', { matches: false })).toBe('light')
    expect(applyAppearance('dark', 'standard', { matches: false })).toBe('dark')
  })

  it('writes data-theme and data-textsize on the document root', () => {
    const doc = fakeDocument()
    globalThis.document = doc
    expect(applyAppearance('dark', 'large', { matches: false })).toBe('dark')
    expect(doc.documentElement.dataset.theme).toBe('dark')
    expect(doc.documentElement.dataset.textsize).toBe('large')
  })

  it('does not animate the very first paint', () => {
    const doc = fakeDocument()
    globalThis.document = doc
    applyAppearance('light', 'standard', { matches: false })
    expect(doc._classes.has('theme-anim')).toBe(false)
  })

  it('cross-fades on an actual theme change and cleans the class up', () => {
    vi.useFakeTimers()
    const doc = fakeDocument()
    globalThis.document = doc
    applyAppearance('light', 'standard', { matches: false }) // first paint
    applyAppearance('dark', 'standard', { matches: false }) // change
    expect(doc.documentElement.dataset.theme).toBe('dark')
    expect(doc._classes.has('theme-anim')).toBe(true)
    vi.advanceTimersByTime(500)
    expect(doc._classes.has('theme-anim')).toBe(false)
  })

  it('keeps the class for repeated same-theme writes', () => {
    const doc = fakeDocument()
    globalThis.document = doc
    applyAppearance('dark', 'standard', { matches: false })
    applyAppearance('dark', 'standard', { matches: false })
    expect(doc._classes.has('theme-anim')).toBe(false)
  })
})

describe('watchSystemTheme', () => {
  it('is a no-op without matchMedia and returns an unsubscribe', () => {
    const unsub = watchSystemTheme(() => {})
    expect(typeof unsub).toBe('function')
    expect(() => unsub()).not.toThrow()
  })

  it('fires on OS changes and stops after unsubscribe', () => {
    const mq = fakeMatchMedia(false)
    globalThis.window = { matchMedia: () => mq }
    const seen = []
    const unsub = watchSystemTheme((e) => seen.push(e.matches))
    expect(mq._listeners.length).toBe(1)
    mq._fire(true)
    expect(seen).toEqual([true])
    unsub()
    expect(mq._listeners.length).toBe(0)
    mq._fire(false)
    expect(seen).toEqual([true])
  })

  it('only re-themes the DOM when the stored preference is system', () => {
    vi.useFakeTimers()
    const doc = fakeDocument()
    const mq = fakeMatchMedia(false)
    globalThis.document = doc
    globalThis.window = { matchMedia: () => mq }
    globalThis.localStorage = fakeStorage({ 'mha.theme': 'light' })

    applyAppearance(readTheme(), readTextSize()) // initial: light
    watchSystemTheme(() => {
      if (readTheme() === 'system') applyAppearance('system', readTextSize())
    })

    mq._fire(true) // OS flips to dark...
    expect(doc.documentElement.dataset.theme).toBe('light') // ...but user chose light

    writeTheme('system')
    mq._fire(true)
    expect(doc.documentElement.dataset.theme).toBe('dark') // now it follows
  })
})
