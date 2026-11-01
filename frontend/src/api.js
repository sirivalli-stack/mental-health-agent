/**
 * Thin client for the API.
 *
 * Same-origin by default: in `vite dev` / `vite preview` the proxy forwards
 * `/api` and `/health` to FastAPI, so no CORS configuration exists anywhere.
 * Set VITE_API_BASE to point at another origin (e.g. a deployed backend).
 *
 * Every request carries the bearer token when one is stored (Phase 20); a
 * 401 while a token is present notifies the app so it can return to the
 * sign-in screen (expired or revoked session). Login/register failures are
 * 401 too, but no token is held there, so they never trigger a redirect.
 */

import { buildChatPayload } from './format.js'
import { readToken } from './auth.js'

const BASE = (import.meta.env?.VITE_API_BASE ?? '').replace(/\/$/, '')

let unauthorizedHandler = null

/** Register a callback invoked when a held token is rejected (401). */
export function onUnauthorized(fn) {
  unauthorizedHandler = typeof fn === 'function' ? fn : null
}

function authHeaders() {
  const token = readToken()
  return token ? { Authorization: `Bearer ${token}` } : {}
}

async function request(path, options = {}) {
  let response
  try {
    response = await fetch(`${BASE}${path}`, {
      ...options,
      headers: { 'Content-Type': 'application/json', ...authHeaders(), ...(options.headers ?? {}) },
    })
  } catch (networkError) {
    throw new Error(`Backend unreachable: ${networkError.message}`)
  }
  if (response.status === 401 && unauthorizedHandler) {
    unauthorizedHandler()
  }
  const body = await response.json().catch(() => null)
  if (!response.ok) {
    const detail = body?.detail
    const text = Array.isArray(detail)
      ? detail.map((d) => d.msg).join('; ')
      : typeof detail === 'string'
        ? detail
        : detail?.detail ?? body?.error ?? `HTTP ${response.status}`
    throw new Error(text)
  }
  return body
}

// ---------------------------------------------------------------------------
// System
// ---------------------------------------------------------------------------

export function fetchHealth() {
  return request('/health')
}

// ---------------------------------------------------------------------------
// Chat + sessions (Phase 12 surface, unchanged payloads)
// ---------------------------------------------------------------------------

export function sendChat({ message, sessionId, profile }) {
  return request('/api/chat', {
    method: 'POST',
    body: JSON.stringify(buildChatPayload({ message, sessionId, profile })),
  })
}

export function fetchSession(sessionId) {
  return request(`/api/sessions/${encodeURIComponent(sessionId)}`)
}

export function resetSession(sessionId) {
  return request(`/api/sessions/${encodeURIComponent(sessionId)}`, {
    method: 'DELETE',
  })
}

// ---------------------------------------------------------------------------
// Auth (Phase 20)
// ---------------------------------------------------------------------------

/** Backend user row -> the UI shape (camelCase). Never carries a password. */
export function toUser(row) {
  if (!row) return null
  return {
    id: row.id,
    username: row.username,
    email: row.email,
    displayName: row.display_name ?? '',
    createdAt: row.created_at ?? null,
  }
}

export function register({ displayName, username, email, password }) {
  return request('/api/auth/register', {
    method: 'POST',
    body: JSON.stringify({ display_name: displayName, username, email, password }),
  }).then((body) => ({ ...body, user: toUser(body.user) }))
}

export function login({ identifier, password, remember }) {
  return request('/api/auth/login', {
    method: 'POST',
    body: JSON.stringify({ identifier, password, remember: Boolean(remember) }),
  }).then((body) => ({ ...body, user: toUser(body.user) }))
}

export function logout() {
  return request('/api/auth/logout', { method: 'POST' })
}

export function fetchMe() {
  return request('/api/auth/me').then(toUser)
}

export function updateProfile({ displayName, username, email }) {
  const body = {}
  if (displayName !== undefined) body.display_name = displayName
  if (username !== undefined) body.username = username
  if (email !== undefined) body.email = email
  return request('/api/auth/profile', { method: 'PATCH', body: JSON.stringify(body) }).then(toUser)
}

export function changePassword({ currentPassword, newPassword }) {
  return request('/api/auth/password', {
    method: 'POST',
    body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
  })
}

export function deleteAccount() {
  return request('/api/auth/account', { method: 'DELETE' })
}

// ---------------------------------------------------------------------------
// Conversations (Phase 20)
// ---------------------------------------------------------------------------

export function listConversations() {
  return request('/api/conversations')
}

export function fetchConversation(sessionId) {
  return request(`/api/conversations/${encodeURIComponent(sessionId)}`)
}

export function fetchMessages(sessionId) {
  return request(`/api/conversations/${encodeURIComponent(sessionId)}/messages`)
}

export function renameConversation(sessionId, title) {
  return request(`/api/conversations/${encodeURIComponent(sessionId)}`, {
    method: 'PATCH',
    body: JSON.stringify({ title }),
  })
}

export function deleteConversation(sessionId) {
  return request(`/api/conversations/${encodeURIComponent(sessionId)}`, {
    method: 'DELETE',
  })
}

// ---------------------------------------------------------------------------
// Preferences + personalization (Phase 20)
// ---------------------------------------------------------------------------

/** UI shape -> the backend contract (snake_case, explicit booleans). */
export function toServerPreferences(prefs) {
  return {
    response_length: prefs.responseLength,
    tone: prefs.tone,
    asks_questions: prefs.asksQuestions,
    preferred_language: prefs.language,
    personalization_enabled: prefs.personalizationEnabled,
  }
}

/** Backend contract -> the UI shape. */
export function fromServerPreferences(prefs) {
  return {
    responseLength: prefs.response_length,
    tone: prefs.tone,
    asksQuestions: prefs.asks_questions,
    language: prefs.preferred_language,
    personalizationEnabled: prefs.personalization_enabled,
    updatedAt: prefs.updated_at ?? null,
  }
}

export function fetchPreferences() {
  return request('/api/preferences').then(fromServerPreferences)
}

export function savePreferences(prefs) {
  return request('/api/preferences', {
    method: 'PUT',
    body: JSON.stringify(toServerPreferences(prefs)),
  }).then(fromServerPreferences)
}

export function fetchPersonalization() {
  return request('/api/personalization')
}

export function clearPersonalization() {
  return request('/api/personalization', { method: 'DELETE' })
}
