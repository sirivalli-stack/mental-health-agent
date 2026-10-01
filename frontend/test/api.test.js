import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { fetchHealth, sendChat } from '../src/api.js'

describe('api client', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('sendChat posts the payload to /api/chat', async () => {
    const response = { reply: 'hi', source: 'llm', turn_index: 0, session_id: 's1' }
    fetch.mockResolvedValue({ ok: true, json: async () => response })

    const out = await sendChat({ message: 'hello', sessionId: 'abc', profile: 'C' })

    expect(out).toEqual(response)
    const [url, options] = fetch.mock.calls[0]
    expect(url).toBe('/api/chat')
    expect(options.method).toBe('POST')
    expect(JSON.parse(options.body)).toEqual({
      message: 'hello',
      session_id: 'abc',
      profile: 'C',
    })
  })

  it('omits session_id for a brand new session', async () => {
    fetch.mockResolvedValue({ ok: true, json: async () => ({}) })
    await sendChat({ message: 'hello' })
    const body = JSON.parse(fetch.mock.calls[0][1].body)
    expect('session_id' in body).toBe(false)
  })

  it('surfaces validation messages from a 422', async () => {
    fetch.mockResolvedValue({
      ok: false,
      status: 422,
      json: async () => ({
        error: 'validation_error',
        detail: [{ msg: 'message must not be empty or whitespace only' }],
      }),
    })
    await expect(sendChat({ message: 'x' })).rejects.toThrow(
      'message must not be empty or whitespace only',
    )
  })

  it('surfaces string details (404 session_not_found)', async () => {
    fetch.mockResolvedValue({
      ok: false,
      status: 404,
      json: async () => ({ detail: 'session_not_found' }),
    })
    await expect(fetchHealth()).rejects.toThrow('session_not_found')
  })

  it('wraps network failures as a friendly error', async () => {
    fetch.mockRejectedValue(new TypeError('Failed to fetch'))
    await expect(fetchHealth()).rejects.toThrow('Backend unreachable')
  })
})
