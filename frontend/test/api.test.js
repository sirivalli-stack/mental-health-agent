import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  deleteAllConversations,
  fetchHealth,
  fetchSettings,
  fromServerPreferences,
  fromServerSettings,
  login,
  register,
  regenerateChat,
  savePreferences,
  saveSettings,
  sendChat,
  toServerPreferences,
  toServerSettings,
} from '../src/api.js'

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

describe('auth payloads', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('register posts the snake_case body', async () => {
    fetch.mockResolvedValue({ ok: true, json: async () => ({ token: 't' }) })
    await register({
      displayName: 'Alex Rivera',
      username: 'alex',
      email: 'alex@example.com',
      password: 'correct-horse-1',
    })
    const [url, options] = fetch.mock.calls[0]
    expect(url).toBe('/api/auth/register')
    expect(JSON.parse(options.body)).toEqual({
      display_name: 'Alex Rivera',
      username: 'alex',
      email: 'alex@example.com',
      password: 'correct-horse-1',
    })
  })

  it('login sends the identifier, password and remember flag', async () => {
    fetch.mockResolvedValue({ ok: true, json: async () => ({ token: 't' }) })
    await login({ identifier: 'alex', password: 'pw', remember: true })
    expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual({
      identifier: 'alex',
      password: 'pw',
      remember: true,
    })
  })
})

describe('preference mapping', () => {
  const ui = {
    responseLength: 'short',
    tone: 'gentle',
    asksQuestions: false,
    language: 'es',
    personalizationEnabled: true,
  }

  it('maps the UI shape onto the backend contract', () => {
    expect(toServerPreferences(ui)).toEqual({
      response_length: 'short',
      tone: 'gentle',
      asks_questions: false,
      preferred_language: 'es',
      personalization_enabled: true,
    })
  })

  it('maps the backend contract back onto the UI shape', () => {
    expect(
      fromServerPreferences({
        response_length: 'detailed',
        tone: 'direct',
        asks_questions: true,
        preferred_language: 'en',
        personalization_enabled: false,
        updated_at: '2026-03-10T12:00:00Z',
      }),
    ).toEqual({
      responseLength: 'detailed',
      tone: 'direct',
      asksQuestions: true,
      language: 'en',
      personalizationEnabled: false,
      updatedAt: '2026-03-10T12:00:00Z',
    })
  })

  it('round-trips through savePreferences', async () => {
    vi.stubGlobal('fetch', vi.fn())
    fetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        response_length: 'short',
        tone: 'gentle',
        asks_questions: false,
        preferred_language: 'es',
        personalization_enabled: true,
      }),
    })
    const saved = await savePreferences(ui)
    expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual(toServerPreferences(ui))
    expect(saved.responseLength).toBe('short')
    vi.unstubAllGlobals()
  })
})

// ---------------------------------------------------------------------------
// Phase 21: regenerate, app settings, bulk delete
// ---------------------------------------------------------------------------

describe('regenerate', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('posts the session id to /api/chat/regenerate', async () => {
    const response = { reply: 'fresh', source: 'llm', turn_index: 3, session_id: 's1' }
    fetch.mockResolvedValue({ ok: true, json: async () => response })

    const out = await regenerateChat({ sessionId: 's1' })

    expect(out).toEqual(response)
    const [url, options] = fetch.mock.calls[0]
    expect(url).toBe('/api/chat/regenerate')
    expect(options.method).toBe('POST')
    expect(JSON.parse(options.body)).toEqual({ session_id: 's1' })
  })

  it('surfaces the backend error message', async () => {
    fetch.mockResolvedValue({
      ok: false,
      status: 400,
      json: async () => ({
        detail: { error: 'nothing_to_regenerate', detail: 'no trailing reply' },
      }),
    })
    await expect(regenerateChat({ sessionId: 's1' })).rejects.toThrow('no trailing reply')
  })
})

describe('app settings', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('maps the backend contract onto the UI shape', () => {
    expect(fromServerSettings({ daily_reminders: false, updated_at: 't' })).toEqual({
      dailyReminders: false,
      updatedAt: 't',
    })
    expect(fromServerSettings({ daily_reminders: true })).toEqual({
      dailyReminders: true,
      updatedAt: null,
    })
    expect(toServerSettings({ dailyReminders: false })).toEqual({ daily_reminders: false })
  })

  it('fetchSettings reads GET /api/settings', async () => {
    fetch.mockResolvedValue({
      ok: true,
      json: async () => ({ daily_reminders: false, updated_at: null }),
    })
    const out = await fetchSettings()
    expect(fetch.mock.calls[0][0]).toBe('/api/settings')
    expect(out.dailyReminders).toBe(false)
  })

  it('saveSettings GETs the current state before PUTting', async () => {
    fetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ daily_reminders: true, updated_at: null }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ daily_reminders: false, updated_at: 'now' }),
      })

    const out = await saveSettings({ dailyReminders: false })

    expect(fetch.mock.calls[0][1]?.method ?? 'GET').toBe('GET')
    const [putUrl, putOptions] = fetch.mock.calls[1]
    expect(putUrl).toBe('/api/settings')
    expect(putOptions.method).toBe('PUT')
    // the PUT carries every field (the backend resets omitted ones)
    expect(JSON.parse(putOptions.body)).toEqual({ daily_reminders: false })
    expect(out.dailyReminders).toBe(false)
    expect(out.updatedAt).toBe('now')
  })
})

describe('bulk conversation deletion', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('sends DELETE to /api/conversations and returns the count', async () => {
    fetch.mockResolvedValue({ ok: true, json: async () => ({ deleted: 3, status: 'deleted' }) })
    const out = await deleteAllConversations()
    const [url, options] = fetch.mock.calls[0]
    expect(url).toBe('/api/conversations')
    expect(options.method).toBe('DELETE')
    expect(out.deleted).toBe(3)
  })
})
