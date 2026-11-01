import { useState } from 'react'
import { login, register } from '../api.js'
import { BrandMark, Field, PasswordField, Spinner } from './ui.jsx'
import { routes } from '../router.js'

/**
 * Welcome / sign-in / create-account screen.
 *
 * The form never stores what is typed beyond component state; passwords go
 * straight to the auth endpoint and are never kept anywhere client-side.
 */
export function AuthScreen({ mode, onAuthenticated, onNavigate, health, notice }) {
  const isRegister = mode === 'register'
  const [identifier, setIdentifier] = useState('')
  const [displayName, setDisplayName] = useState('')
  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [remember, setRemember] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  async function submit(e) {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      const result = isRegister
        ? await register({ displayName, username, email, password })
        : await login({ identifier, password, remember })
      onAuthenticated(result.user, result.token)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="auth">
      <div className="auth-brand">
        <BrandMark size={44} />
        <div>
          <div className="brand-name" style={{ fontSize: '1.4rem' }}>Sage</div>
          <div className="brand-tag">space to talk things through</div>
        </div>
      </div>

      <main className="card auth-card">
        {isRegister ? (
          <>
            <h1>Create your space</h1>
            <p className="sub">
              A calm, private place to reflect. Your conversations stay yours.
            </p>
          </>
        ) : (
          <>
            <h1>Welcome back</h1>
            <p className="sub">Sign in to continue where you left off.</p>
          </>
        )}

        <form className="auth-form" onSubmit={submit}>
          {notice && (
            <div className="notice" role="status">
              {notice}
            </div>
          )}
          {isRegister ? (
            <>
              <Field label="Name" id="au-name" hint="How Sage should greet you.">
                <input
                  id="au-name"
                  type="text"
                  value={displayName}
                  maxLength={80}
                  autoComplete="name"
                  autoFocus
                  required
                  onChange={(e) => setDisplayName(e.target.value)}
                />
              </Field>
              <Field label="Username" id="au-user" hint="3–32 characters; letters, digits, '_', '.' or '-'">
                <input
                  id="au-user"
                  type="text"
                  value={username}
                  maxLength={32}
                  autoComplete="username"
                  required
                  onChange={(e) => setUsername(e.target.value)}
                />
              </Field>
              <Field label="Email" id="au-email">
                <input
                  id="au-email"
                  type="email"
                  value={email}
                  autoComplete="email"
                  required
                  onChange={(e) => setEmail(e.target.value)}
                />
              </Field>
              <PasswordField
                id="au-pass"
                label="Password"
                hint="At least 8 characters."
                value={password}
                autoComplete="new-password"
                onChange={setPassword}
              />
            </>
          ) : (
            <>
              <Field label="Email or username" id="au-id">
                <input
                  id="au-id"
                  type="text"
                  value={identifier}
                  autoComplete="username"
                  autoFocus
                  required
                  onChange={(e) => setIdentifier(e.target.value)}
                />
              </Field>
              <PasswordField
                id="au-pass"
                label="Password"
                value={password}
                autoComplete="current-password"
                onChange={setPassword}
              />
              <div className="auth-row">
                <label>
                  <input
                    type="checkbox"
                    checked={remember}
                    onChange={(e) => setRemember(e.target.checked)}
                  />
                  Keep me signed in on this device
                </label>
              </div>
            </>
          )}

          {error && (
            <div className="notice notice-error" role="alert">
              {error}
            </div>
          )}

          <button type="submit" className="btn btn-primary btn-lg btn-block" disabled={busy}>
            {busy ? <Spinner /> : isRegister ? 'Create account' : 'Sign in'}
          </button>
        </form>

        <div className="auth-switch">
          {isRegister ? (
            <>
              Already have an account?{' '}
              <a
                href={routes.login()}
                onClick={() => onNavigate(routes.login())}
              >
                Sign in
              </a>
            </>
          ) : (
            <>
              New here?{' '}
              <a
                href={routes.register()}
                onClick={() => onNavigate(routes.register())}
              >
                Create an account
              </a>
            </>
          )}
        </div>
      </main>

      <p className="auth-foot">
        Sage is a research prototype for emotional support and reflection. It is
        not a medical device, not a diagnostic system, and not a replacement for
        a mental-health professional. If you are in crisis, contact your local
        emergency services.
        {health && (
          <>
            {' '}
            Backend: <span className="chip">{health.status}</span>
          </>
        )}
      </p>
    </div>
  )
}
