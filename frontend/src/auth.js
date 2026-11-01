/**
 * Bearer-token storage.
 *
 * The token - never the password - is kept in localStorage so a reload
 * keeps the session. Passwords only ever travel to the login/register
 * endpoints and are not stored client-side in any form. Storage access is
 * guarded because the unit-test environment (node) has no localStorage.
 */

const TOKEN_KEY = 'mha.token'

export function readToken() {
  try {
    return typeof localStorage === 'undefined'
      ? null
      : localStorage.getItem(TOKEN_KEY)
  } catch {
    return null
  }
}

export function writeToken(token) {
  try {
    if (typeof localStorage === 'undefined') return
    if (token) localStorage.setItem(TOKEN_KEY, token)
    else localStorage.removeItem(TOKEN_KEY)
  } catch {
    /* private mode: the session simply does not survive reloads */
  }
}

export function clearToken() {
  writeToken(null)
}
