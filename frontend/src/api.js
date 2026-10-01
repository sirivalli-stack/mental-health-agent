/**
 * Thin client for the Phase 12 API.
 *
 * Same-origin by default: in `vite dev` / `vite preview` the proxy forwards
 * `/api` and `/health` to FastAPI, so no CORS configuration exists anywhere.
 * Set VITE_API_BASE to point at another origin (e.g. a deployed backend).
 */

import { buildChatPayload } from './format.js'

const BASE = (import.meta.env?.VITE_API_BASE ?? '').replace(/\/$/, '')

async function request(path, options = {}) {
  let response
  try {
    response = await fetch(`${BASE}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    })
  } catch (networkError) {
    throw new Error(`Backend unreachable: ${networkError.message}`)
  }
  const body = await response.json().catch(() => null)
  if (!response.ok) {
    const detail = body?.detail
    const text = Array.isArray(detail)
      ? detail.map((d) => d.msg).join('; ')
      : typeof detail === 'string'
        ? detail
        : body?.error ?? `HTTP ${response.status}`
    throw new Error(text)
  }
  return body
}

export function fetchHealth() {
  return request('/health')
}

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
