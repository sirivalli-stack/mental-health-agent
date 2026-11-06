/**
 * Copy-to-clipboard helper for message actions (Phase 21).
 *
 * Prefers the async Clipboard API; falls back to the legacy
 * `document.execCommand('copy')` for older browsers and non-secure
 * contexts (http:// on a LAN address) where the async API is absent.
 *
 * Pure-ish: the DOM calls are injected/guarded so the module can be unit
 * tested in node, which has neither `navigator` nor `document`.
 */

/**
 * Copy `text` to the clipboard.
 *
 * @param {string} text
 * @returns {Promise<boolean>} true when the text ended up on the clipboard
 */
export async function copyText(text) {
  const value = String(text ?? '')
  if (!value) return false

  const nav = typeof navigator === 'undefined' ? null : navigator
  if (nav?.clipboard?.writeText) {
    try {
      await nav.clipboard.writeText(value)
      return true
    } catch {
      // permission denied / insecure context - fall through to execCommand
    }
  }

  const doc = typeof document === 'undefined' ? null : document
  if (!doc?.body || typeof doc.createElement !== 'function') return false
  try {
    const area = doc.createElement('textarea')
    area.value = value
    area.setAttribute('readonly', '')
    area.style.position = 'fixed'
    area.style.opacity = '0'
    doc.body.appendChild(area)
    area.select()
    const ok = typeof doc.execCommand === 'function' && doc.execCommand('copy')
    doc.body.removeChild(area)
    return Boolean(ok)
  } catch {
    return false
  }
}
