import { describe, expect, it } from 'vitest'
import { isAuthRoute, parseRoute, routes } from '../src/router.js'

describe('parseRoute', () => {
  it('maps the empty hash to home', () => {
    expect(parseRoute('')).toEqual({ name: 'home', params: {} })
    expect(parseRoute('#')).toEqual({ name: 'home', params: {} })
    expect(parseRoute('#/')).toEqual({ name: 'home', params: {} })
    expect(parseRoute(undefined)).toEqual({ name: 'home', params: {} })
  })

  it('maps the chat routes, with and without an id', () => {
    expect(parseRoute('#/chat')).toEqual({ name: 'chat', params: {} })
    expect(parseRoute('#/chat/abc123')).toEqual({
      name: 'chat',
      params: { id: 'abc123' },
    })
  })

  it('maps settings, login and register', () => {
    expect(parseRoute('#/settings').name).toBe('settings')
    expect(parseRoute('#/login').name).toBe('login')
    expect(parseRoute('#/register').name).toBe('register')
  })

  it('falls back to home for unknown routes', () => {
    expect(parseRoute('#/nope').name).toBe('home')
    expect(parseRoute('#/chat/one/two').name).toBe('home')
  })
})

describe('route builders', () => {
  it('builds hashes and encodes ids', () => {
    expect(routes.home()).toBe('#/')
    expect(routes.chat()).toBe('#/chat')
    expect(routes.chat('abc')).toBe('#/chat/abc')
    expect(routes.chat('a/b')).toBe('#/chat/a%2Fb')
    expect(routes.settings()).toBe('#/settings')
    expect(routes.login()).toBe('#/login')
    expect(routes.register()).toBe('#/register')
  })

  it('knows which routes are the auth screens', () => {
    expect(isAuthRoute('login')).toBe(true)
    expect(isAuthRoute('register')).toBe(true)
    expect(isAuthRoute('chat')).toBe(false)
  })
})
