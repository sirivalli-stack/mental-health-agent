/**
 * Appearance persistence: theme (light / dark / system) and text size.
 * These are device-level preferences, so they live in localStorage and are
 * never sent to the backend - unlike conversation preferences, which the
 * server owns.
 */

const THEME_KEY = 'mha.theme'
const SIZE_KEY = 'mha.textsize'

export const THEME_OPTIONS = ['light', 'dark', 'system']
export const SIZE_OPTIONS = ['standard', 'large']

export function readTheme() {
  try {
    const value = localStorage.getItem(THEME_KEY)
    return THEME_OPTIONS.includes(value) ? value : 'system'
  } catch {
    return 'system'
  }
}

export function writeTheme(value) {
  try {
    if (THEME_OPTIONS.includes(value)) localStorage.setItem(THEME_KEY, value)
    else localStorage.removeItem(THEME_KEY)
  } catch {
    /* private mode: appearance simply does not persist */
  }
}

export function readTextSize() {
  try {
    const value = localStorage.getItem(SIZE_KEY)
    return SIZE_OPTIONS.includes(value) ? value : 'standard'
  } catch {
    return 'standard'
  }
}

export function writeTextSize(value) {
  try {
    if (SIZE_OPTIONS.includes(value)) localStorage.setItem(SIZE_KEY, value)
    else localStorage.removeItem(SIZE_KEY)
  } catch {
    /* ignore */
  }
}

/** 'system' + OS preference -> the concrete theme to render. */
export function resolveTheme(pref, prefersDark) {
  if (pref === 'light' || pref === 'dark') return pref
  return prefersDark ? 'dark' : 'light'
}

/**
 * Subscribe to OS light/dark changes. Returns an unsubscribe function.
 * Safe no-op when matchMedia is unavailable (unit tests, old browsers).
 */
export function watchSystemTheme(onChange) {
  const mq =
    typeof window !== 'undefined' && typeof window.matchMedia === 'function'
      ? window.matchMedia('(prefers-color-scheme: dark)')
      : null
  if (!mq) return () => {}
  const handler = (event) => onChange(event ?? { matches: mq.matches })
  if (typeof mq.addEventListener === 'function') {
    mq.addEventListener('change', handler)
    return () => mq.removeEventListener('change', handler)
  }
  if (typeof mq.addListener === 'function') {
    // Safari <= 13 style
    mq.addListener(handler)
    return () => mq.removeListener(handler)
  }
  return () => {}
}

let themeAnimTimer = null

/**
 * Apply theme + text size to the document root; returns the resolved theme.
 * On an actual theme *change* the root briefly gets the `theme-anim` class so
 * styles.css can cross-fade colors instead of snapping (skipped on first paint
 * so initial load never flashes).
 */
export function applyAppearance(pref, textSize, mediaQuery) {
  const dark =
    mediaQuery?.matches ??
    (typeof window !== 'undefined' &&
      typeof window.matchMedia === 'function' &&
      window.matchMedia('(prefers-color-scheme: dark)').matches)
  const resolved = resolveTheme(pref, dark)
  if (typeof document !== 'undefined') {
    const root = document.documentElement
    const previous = root.dataset.theme
    if (previous && previous !== resolved && root.classList) {
      root.classList.add('theme-anim')
      if (themeAnimTimer) clearTimeout(themeAnimTimer)
      themeAnimTimer = setTimeout(() => {
        root.classList.remove('theme-anim')
        themeAnimTimer = null
      }, 420)
      if (typeof themeAnimTimer === 'object' && themeAnimTimer !== null && 'unref' in themeAnimTimer) {
        themeAnimTimer.unref() // never hold a test process open
      }
    }
    root.dataset.theme = resolved
    root.dataset.textsize = textSize || 'standard'
  }
  return resolved
}
