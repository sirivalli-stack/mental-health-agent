import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  deleteConversation,
  fetchHealth,
  fetchMe,
  listConversations,
  logout as apiLogout,
  renameConversation,
  onUnauthorized,
} from './api.js'
import { clearToken, readToken, writeToken } from './auth.js'
import { applyAppearance, readTextSize, readTheme, watchSystemTheme } from './theme.js'
import { currentHash, isAuthRoute, navigate, parseRoute, routes } from './router.js'
import { AuthScreen } from './components/AuthScreen.jsx'
import { Chat } from './components/Chat.jsx'
import { DisclaimerBanner } from './components/DisclaimerBanner.jsx'
import { Home } from './components/Home.jsx'
import { JournalPage } from './components/Journal.jsx'
import { MoodPage } from './components/MoodPage.jsx'
import { Settings } from './components/Settings.jsx'
import { Sidebar } from './components/Sidebar.jsx'
import { WellnessPage } from './components/Wellness.jsx'
import { BrandMark, Spinner } from './components/ui.jsx'

/** Server conversation row -> the UI shape (camelCase). */
function toConversation(row) {
  return {
    sessionId: row.session_id,
    title: row.title,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    messageCount: row.message_count,
  }
}

export default function App() {
  // ---- bootstrap state ----------------------------------------------------
  const [booting, setBooting] = useState(true)
  const [user, setUser] = useState(null)
  const [authEnabled, setAuthEnabled] = useState(true)
  const [disclaimer, setDisclaimer] = useState('')
  const [health, setHealth] = useState(null)

  // ---- routing ------------------------------------------------------------
  const [route, setRoute] = useState(parseRoute(currentHash()))

  // ---- data ---------------------------------------------------------------
  const [conversations, setConversations] = useState([])
  const [loadingConversations, setLoadingConversations] = useState(false)
  const [listError, setListError] = useState(null)

  // ---- shell --------------------------------------------------------------
  const [menuOpen, setMenuOpen] = useState(false)
  const [starter, setStarter] = useState(null)
  const [authNotice, setAuthNotice] = useState(null)
  const refreshSeq = useRef(0)

  // ---- appearance ---------------------------------------------------------
  useEffect(() => {
    applyAppearance(readTheme(), readTextSize())
    // live-follow the OS when the preference is "system"
    return watchSystemTheme(() => {
      if (readTheme() === 'system') applyAppearance('system', readTextSize())
    })
  }, [])

  // ---- hash routing -------------------------------------------------------
  useEffect(() => {
    function onHashChange() {
      setRoute(parseRoute(currentHash()))
    }
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  // ---- health + session bootstrap ----------------------------------------
  useEffect(() => {
    let cancelled = false
    const token = readToken()

    async function bootstrap() {
      try {
        const h = await fetchHealth()
        if (cancelled) return
        setHealth(h)
        setDisclaimer(h.disclaimer ?? '')
        const authComponent = (h.components ?? []).find((c) => c.name === 'auth')
        // anonymous mode (AUTH_ENABLED=false) reports auth as not loaded
        const enabled = authComponent ? Boolean(authComponent.loaded) : true
        setAuthEnabled(enabled)

        if (!enabled) {
          // no accounts on this deployment: run the app directly
          setBooting(false)
          return
        }
        if (token) {
          const me = await fetchMe()
          if (!cancelled) {
            setUser(me)
            setBooting(false)
          }
        } else if (!cancelled) {
          setBooting(false)
        }
      } catch (e) {
        if (cancelled) return
        if (token && /401|unauthori/i.test(e.message)) {
          clearToken()
        }
        setBooting(false)
      }
    }
    bootstrap()
    return () => {
      cancelled = true
    }
  }, [])

  // ---- 401 while holding a token -> back to sign-in -----------------------
  useEffect(() => {
    onUnauthorized(() => {
      clearToken()
      setUser(null)
      setConversations([])
      setAuthNotice('Your session ended. Please sign in again.')
      navigate(routes.login())
    })
    return () => onUnauthorized(null)
  }, [])

  // ---- conversation list --------------------------------------------------
  const refreshConversations = useCallback(async () => {
    if (!user || !authEnabled) return
    const seq = (refreshSeq.current += 1)
    setLoadingConversations(true)
    try {
      const body = await listConversations()
      if (seq !== refreshSeq.current) return // a newer refresh won
      setConversations((body.conversations ?? []).map(toConversation))
      setListError(null)
    } catch (e) {
      if (seq === refreshSeq.current) setListError(e.message)
    } finally {
      if (seq === refreshSeq.current) setLoadingConversations(false)
    }
  }, [user, authEnabled])

  useEffect(() => {
    refreshConversations()
  }, [refreshConversations])

  // ---- auth actions -------------------------------------------------------
  const authenticated = useCallback((signedInUser, token) => {
    writeToken(token)
    setUser(signedInUser)
    setAuthNotice(null)
    navigate(routes.home())
  }, [])

  async function signOut() {
    try {
      await apiLogout()
    } catch {
      // the token may already be invalid server-side; clear locally regardless
    }
    clearToken()
    setUser(null)
    setConversations([])
    navigate(routes.login())
  }

  // ---- conversation actions ----------------------------------------------
  const openConversation = useCallback((id) => {
    setStarter(null)
    navigate(routes.chat(id))
  }, [])

  const newConversation = useCallback(() => {
    setStarter(null)
    navigate(routes.chat())
  }, [])

  const pickStarter = useCallback((text) => {
    setStarter(text)
    navigate(routes.chat())
  }, [])

  const handleRename = useCallback(
    async (id, title) => {
      // optimistic: reflect immediately, roll back on failure
      const previous = conversations
      setConversations((cs) => cs.map((c) => (c.sessionId === id ? { ...c, title } : c)))
      try {
        await renameConversation(id, title)
      } catch {
        setConversations(previous)
      }
    },
    [conversations],
  )

  const handleDelete = useCallback(
    async (conversation) => {
      await deleteConversation(conversation.sessionId)
      setConversations((cs) => cs.filter((c) => c.sessionId !== conversation.sessionId))
      if (route.name === 'chat' && route.params.id === conversation.sessionId) {
        navigate(routes.chat())
      }
    },
    [route],
  )

  // ---- route guards --------------------------------------------------------
  const needsAuth = authEnabled && !user
  useEffect(() => {
    if (booting) return
    if (needsAuth && !isAuthRoute(route.name)) {
      navigate(routes.login())
    }
    if (!needsAuth && isAuthRoute(route.name)) {
      navigate(routes.home())
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [booting, needsAuth, route.name])

  // ---- boot splash ---------------------------------------------------------
  if (booting) {
    return (
      <div className="auth">
        <div className="auth-brand">
          <BrandMark size={44} />
          <div className="brand-name" style={{ fontSize: '1.4rem' }}>Sage</div>
        </div>
        <Spinner />
      </div>
    )
  }

  // ---- auth screens ---------------------------------------------------------
  if (needsAuth) {
    if (!isAuthRoute(route.name)) return null // guard navigation in flight
    return (
      <AuthScreen
        mode={route.name}
        health={health}
        onAuthenticated={authenticated}
        onNavigate={navigate}
        notice={authNotice}
      />
    )
  }

  // ---- main shell ------------------------------------------------------------
  const activeId = route.name === 'chat' ? (route.params.id ?? null) : null
  const active = conversations.find((c) => c.sessionId === activeId)

  let page = null
  if (route.name === 'home') {
    page = (
      <Home
        user={user}
        anonymous={!authEnabled}
        conversations={conversations}
        onNewConversation={newConversation}
        onPickStarter={pickStarter}
        onOpenMenu={() => setMenuOpen(true)}
      />
    )
  } else if (route.name === 'chat') {
    page = (
      <Chat
        activeId={activeId}
        title={active?.title ?? ''}
        starter={starter}
        clearStarter={() => setStarter(null)}
        anonymous={!authEnabled}
        onConversationsChanged={refreshConversations}
        onNavigate={navigate}
        onOpenMenu={() => setMenuOpen(true)}
      />
    )
  } else if (route.name === 'settings') {
    page = (
      <Settings
        user={user}
        onUserChanged={setUser}
        onConversationsChanged={refreshConversations}
        onOpenMenu={() => setMenuOpen(true)}
        onAccountDeleted={() => {
          clearToken()
          setUser(null)
          setConversations([])
          navigate(routes.register())
        }}
      />
    )
  } else if (route.name === 'mood') {
    page = <MoodPage onOpenMenu={() => setMenuOpen(true)} />
  } else if (route.name === 'journal') {
    page = <JournalPage onOpenMenu={() => setMenuOpen(true)} />
  } else if (route.name === 'wellness') {
    page = <WellnessPage onOpenMenu={() => setMenuOpen(true)} />
  }

  return (
    <>
      <DisclaimerBanner text={disclaimer} />
      <div className="shell">
        <Sidebar
          user={user}
          conversations={conversations}
          loadingConversations={loadingConversations}
          activeId={activeId}
          activeRoute={route.name}
          anonymous={!authEnabled}
          open={menuOpen}
          onClose={() => setMenuOpen(false)}
          onOpenConversation={openConversation}
          onNewConversation={newConversation}
          onRename={handleRename}
          onDelete={handleDelete}
          onSettings={() => {
            setMenuOpen(false)
            navigate(routes.settings())
          }}
          onSignOut={signOut}
        />
        <main className="main">
          {listError && (
            <div style={{ padding: '10px 14px 0' }}>
              <div className="notice notice-error" role="alert">
                Could not refresh conversations: {listError}
              </div>
            </div>
          )}
          {page}
        </main>
      </div>
      {menuOpen && (
        <button
          type="button"
          className="scrim"
          aria-label="Close navigation"
          onClick={() => setMenuOpen(false)}
        />
      )}
    </>
  )
}
