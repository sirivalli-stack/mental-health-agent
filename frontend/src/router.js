/**
 * Minimal hash router - no dependency, fully unit-testable.
 *
 * Routes:
 *   #/              home (personalized welcome)
 *   #/chat          chat workspace (no active conversation)
 *   #/chat/:id      one conversation
 *   #/settings      settings + personalization
 *   #/login         sign in
 *   #/register      create account
 */

const ROUTE_NAMES = new Set(['home', 'chat', 'settings', 'login', 'register'])

export function parseRoute(hash) {
  const raw = String(hash ?? '').replace(/^#/, '')
  const parts = raw.split('/').filter(Boolean)

  if (parts.length === 0) return { name: 'home', params: {} }
  const [head, ...rest] = parts

  if (head === 'chat') {
    if (rest.length === 0) return { name: 'chat', params: {} }
    if (rest.length === 1) return { name: 'chat', params: { id: rest[0] } }
    return { name: 'home', params: {} }
  }
  if (ROUTE_NAMES.has(head) && rest.length === 0) {
    return { name: head, params: {} }
  }
  return { name: 'home', params: {} }
}

export const routes = {
  home: () => '#/',
  chat: (id) => (id ? `#/chat/${encodeURIComponent(id)}` : '#/chat'),
  settings: () => '#/settings',
  login: () => '#/login',
  register: () => '#/register',
}

export function navigate(hash) {
  if (typeof window !== 'undefined') window.location.hash = hash
}

export function currentHash() {
  return typeof window !== 'undefined' ? window.location.hash : '#/'
}

export function isAuthRoute(name) {
  return name === 'login' || name === 'register'
}
