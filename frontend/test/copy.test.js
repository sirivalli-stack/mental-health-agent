import { afterEach, describe, expect, it, vi } from 'vitest'
import { copyText } from '../src/copy.js'

/** Minimal textarea-creating document double for the execCommand fallback. */
function fakeDocument({ execCommandResult = true } = {}) {
  const appended = []
  return {
    body: {
      appendChild: (node) => appended.push(node),
      removeChild: (node) => {
        const i = appended.indexOf(node)
        if (i >= 0) appended.splice(i, 1)
      },
    },
    createElement: () => ({
      value: '',
      style: {},
      setAttribute: () => {},
      select: () => {},
    }),
    execCommand: vi.fn(() => execCommandResult),
    _appended: appended,
  }
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('copyText', () => {
  it('returns false for empty input without touching the clipboard', async () => {
    const writeText = vi.fn()
    vi.stubGlobal('navigator', { clipboard: { writeText } })
    expect(await copyText('')).toBe(false)
    expect(await copyText(null)).toBe(false)
    expect(writeText).not.toHaveBeenCalled()
  })

  it('uses the async Clipboard API when available', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    vi.stubGlobal('navigator', { clipboard: { writeText } })
    expect(await copyText('hello world')).toBe(true)
    expect(writeText).toHaveBeenCalledWith('hello world')
  })

  it('falls back to execCommand when the Clipboard API is missing', async () => {
    const doc = fakeDocument()
    vi.stubGlobal('navigator', {}) // no clipboard
    vi.stubGlobal('document', doc)
    expect(await copyText('legacy path')).toBe(true)
    expect(doc.execCommand).toHaveBeenCalledWith('copy')
    expect(doc._appended).toEqual([]) // the temp textarea is removed again
  })

  it('falls back when writeText rejects (permission / insecure context)', async () => {
    const doc = fakeDocument()
    vi.stubGlobal('navigator', {
      clipboard: { writeText: vi.fn().mockRejectedValue(new Error('denied')) },
    })
    vi.stubGlobal('document', doc)
    expect(await copyText('denied upstream')).toBe(true)
    expect(doc.execCommand).toHaveBeenCalledWith('copy')
  })

  it('returns false when neither mechanism exists (node tests)', async () => {
    vi.stubGlobal('navigator', {})
    vi.stubGlobal('document', undefined)
    expect(await copyText('nothing available')).toBe(false)
  })

  it('returns false when execCommand reports failure', async () => {
    const doc = fakeDocument({ execCommandResult: false })
    vi.stubGlobal('navigator', {})
    vi.stubGlobal('document', doc)
    expect(await copyText('cannot copy')).toBe(false)
  })

  it('stringifies non-string input', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    vi.stubGlobal('navigator', { clipboard: { writeText } })
    expect(await copyText(42)).toBe(true)
    expect(writeText).toHaveBeenCalledWith('42')
  })
})
