import { useEffect, useState } from 'react'
import {
  changePassword,
  clearPersonalization,
  deleteAccount,
  deleteAllConversations,
  fetchPersonalization,
  fetchPreferences,
  fetchSettings,
  savePreferences,
  saveSettings,
  updateProfile,
} from '../api.js'
import { initials } from '../greeting.js'
import { readStoredLanguage, writeStoredLanguage } from '../i18n.js'
import {
  SIZE_OPTIONS,
  THEME_OPTIONS,
  applyAppearance,
  readTextSize,
  readTheme,
  writeTextSize,
  writeTheme,
} from '../theme.js'
import { ConfirmDialog, Field, Icon, PasswordField, Spinner, Switch } from './ui.jsx'
import { isSttSupported, speechController, useVoiceSettings } from '../voice/index.js'

const LENGTH_OPTIONS = [
  { value: 'short', label: 'Short', hint: 'two or three sentences' },
  { value: 'balanced', label: 'Balanced', hint: 'a few sentences' },
  { value: 'detailed', label: 'Detailed', hint: 'thorough reflections' },
]
const TONE_OPTIONS = [
  { value: 'warm', label: 'Warm', hint: 'caring and supportive' },
  { value: 'gentle', label: 'Gentle', hint: 'soft and reassuring' },
  { value: 'direct', label: 'Direct', hint: 'clear and to the point' },
]
// every code the backend's extractor supports (never guess others)
const LANGUAGE_OPTIONS = [
  ['en', 'English'],
  ['es', 'Español'],
  ['fr', 'Français'],
  ['de', 'Deutsch'],
  ['it', 'Italiano'],
  ['pt', 'Português'],
  ['nl', 'Nederlands'],
  ['hi', 'हिन्दी'],
  ['ar', 'العربية'],
  ['zh', '中文'],
  ['ja', '日本語'],
  ['ko', '한국어'],
  ['ru', 'Русский'],
  ['bn', 'বাংলা'],
  ['ta', 'தமிழ்'],
  ['te', 'తెలుగు'],
  ['kn', 'ಕನ್ನಡ'],
  ['mr', 'मराठी'],
  ['ur', 'اردو'],
]

function ChoiceGroup({ name, options, value, onChange }) {
  return (
    <div className="choice-row" role="radiogroup" aria-label={name}>
      {options.map((option) => (
        <label
          key={option.value}
          className={`choice ${value === option.value ? 'selected' : ''}`}
        >
          <input
            type="radio"
            name={name}
            value={option.value}
            checked={value === option.value}
            onChange={() => onChange(option.value)}
          />
          <span>
            {option.label}
            {option.hint && (
              <span className="hint" style={{ display: 'block', fontWeight: 400 }}>
                {option.hint}
              </span>
            )}
          </span>
        </label>
      ))}
    </div>
  )
}

function Notice({ kind, children }) {
  if (!children) return null
  return (
    <div className={`notice ${kind === 'error' ? 'notice-error' : 'notice-ok'}`} role="status">
      {children}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Profile section
// ---------------------------------------------------------------------------

function ProfileSection({ user, onUserChanged }) {
  const [displayName, setDisplayName] = useState(user?.displayName ?? '')
  const [username, setUsername] = useState(user?.username ?? '')
  const [email, setEmail] = useState(user?.email ?? '')
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [error, setError] = useState(null)

  const [currentPassword, setCurrentPassword] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [pwBusy, setPwBusy] = useState(false)
  const [pwDone, setPwDone] = useState(false)
  const [pwError, setPwError] = useState(null)

  useEffect(() => {
    setDisplayName(user?.displayName ?? '')
    setUsername(user?.username ?? '')
    setEmail(user?.email ?? '')
  }, [user])

  async function saveProfile(e) {
    e.preventDefault()
    setSaving(true)
    setSaved(false)
    setError(null)
    try {
      const updated = await updateProfile({
        displayName: displayName.trim(),
        username: username.trim(),
        email: email.trim(),
      })
      onUserChanged(updated)
      setSaved(true)
    } catch (err) {
      setError(err.message)
    } finally {
      setSaving(false)
    }
  }

  async function savePassword(e) {
    e.preventDefault()
    setPwError(null)
    setPwDone(false)
    if (newPassword !== confirmPassword) {
      setPwError('The new passwords do not match.')
      return
    }
    setPwBusy(true)
    try {
      await changePassword({ currentPassword, newPassword })
      setCurrentPassword('')
      setNewPassword('')
      setConfirmPassword('')
      setPwDone(true)
    } catch (err) {
      setPwError(err.message)
    } finally {
      setPwBusy(false)
    }
  }

  return (
    <section className="card settings-card" aria-label="Profile">
      <h2>Profile</h2>
      <p className="desc">How Sage addresses you. This never changes what is remembered from your chats.</p>
      <div className="profile-head">
        <span className="avatar avatar-lg" aria-hidden="true">
          {initials(displayName || username)}
        </span>
        <div className="who">
          <h2>{displayName || username}</h2>
          <p>@{username} · {email}</p>
        </div>
      </div>
      <form onSubmit={saveProfile} className="settings-group">
        <Field label="Display name" id="pf-name">
          <input
            id="pf-name"
            type="text"
            value={displayName}
            maxLength={80}
            onChange={(e) => setDisplayName(e.target.value)}
          />
        </Field>
        <Field label="Username" id="pf-user" hint="3–32 characters; letters, digits, '_', '.' or '-'">
          <input
            id="pf-user"
            type="text"
            value={username}
            maxLength={32}
            onChange={(e) => setUsername(e.target.value)}
          />
        </Field>
        <Field label="Email" id="pf-email">
          <input
            id="pf-email"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
        </Field>
        <div className="settings-actions">
          <button type="submit" className="btn btn-primary" disabled={saving}>
            {saving ? <Spinner /> : 'Save profile'}
          </button>
          <Notice kind="ok">{saved ? 'Profile updated.' : null}</Notice>
          <Notice kind="error">{error}</Notice>
        </div>
      </form>

      <div style={{ marginTop: 26 }}>
        <h2 style={{ fontSize: '1rem' }}>Change password</h2>
        <p className="desc">You will be signed out of every device after the change.</p>
        <form onSubmit={savePassword} className="settings-group">
          <PasswordField
            id="pw-current"
            label="Current password"
            value={currentPassword}
            autoComplete="current-password"
            onChange={setCurrentPassword}
          />
          <PasswordField
            id="pw-new"
            label="New password"
            hint="At least 8 characters."
            value={newPassword}
            autoComplete="new-password"
            onChange={setNewPassword}
          />
          <PasswordField
            id="pw-confirm"
            label="Confirm new password"
            value={confirmPassword}
            autoComplete="new-password"
            onChange={setConfirmPassword}
          />
          <div className="settings-actions">
            <button type="submit" className="btn" disabled={pwBusy}>
              {pwBusy ? <Spinner /> : 'Change password'}
            </button>
            <Notice kind="ok">{pwDone ? 'Password changed. All sessions were signed out.' : null}</Notice>
            <Notice kind="error">{pwError}</Notice>
          </div>
        </form>
      </div>
    </section>
  )
}

// ---------------------------------------------------------------------------
// Conversation preferences
// ---------------------------------------------------------------------------

function PreferencesSection() {
  const [prefs, setPrefs] = useState(null)
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false
    fetchPreferences()
      .then((loaded) => {
        if (!cancelled) {
          setPrefs(loaded)
          // device-local mirror so voice input picks the right locale
          writeStoredLanguage(loaded.language)
        }
      })
      .catch((e) => {
        if (!cancelled) setError(e.message)
      })
    return () => {
      cancelled = true
    }
  }, [])

  function update(patch) {
    setPrefs((prev) => ({ ...prev, ...patch }))
    setSaved(false)
  }

  async function persist() {
    setSaving(true)
    setSaved(false)
    setError(null)
    try {
      const stored = await savePreferences(prefs)
      setPrefs(stored)
      writeStoredLanguage(stored.language)
      setSaved(true)
    } catch (err) {
      setError(err.message)
    } finally {
      setSaving(false)
    }
  }

  if (!prefs && error) {
    return (
      <section className="card settings-card" aria-label="Conversation preferences">
        <h2>Conversation preferences</h2>
        <div className="notice notice-error" role="alert">{error}</div>
      </section>
    )
  }
  if (!prefs) {
    return (
      <section className="card settings-card" aria-label="Conversation preferences">
        <h2>Conversation preferences</h2>
        <div className="skeleton" style={{ height: 120 }} />
      </section>
    )
  }

  return (
    <section className="card settings-card" aria-label="Conversation preferences">
      <h2>Conversation preferences</h2>
      <p className="desc">
        These settings seed every new conversation. Conversations you have already
        started keep the style they began with.
      </p>
      <div className="settings-group">
        <div className="field">
          <label id="lbl-length">Reply length</label>
          <ChoiceGroup
            name="length"
            options={LENGTH_OPTIONS}
            value={prefs.responseLength}
            onChange={(responseLength) => update({ responseLength })}
          />
        </div>
        <div className="field">
          <label id="lbl-tone">Tone</label>
          <ChoiceGroup
            name="tone"
            options={TONE_OPTIONS}
            value={prefs.tone}
            onChange={(tone) => update({ tone })}
          />
        </div>
        <div className="switch-row">
          <span>
            <span style={{ fontWeight: 600, fontSize: '0.92rem' }}>Ask reflective questions</span>
            <span className="hint" style={{ display: 'block' }}>
              Let Sage ask questions that help you explore what you feel.
            </span>
          </span>
          <Switch
            checked={prefs.asksQuestions}
            onChange={(asksQuestions) => update({ asksQuestions })}
            label="Ask reflective questions"
          />
        </div>
        <Field label="Language" id="pref-lang" hint="Sage replies in this language when it can.">
          <select
            id="pref-lang"
            value={prefs.language}
            onChange={(e) => update({ language: e.target.value })}
          >
            {LANGUAGE_OPTIONS.map(([code, label]) => (
              <option key={code} value={code}>
                {label}
              </option>
            ))}
          </select>
        </Field>
        <div className="settings-actions">
          <button type="button" className="btn btn-primary" onClick={persist} disabled={saving}>
            {saving ? <Spinner /> : 'Save preferences'}
          </button>
          <Notice kind="ok">{saved ? 'Saved. New conversations will use these settings.' : null}</Notice>
          <Notice kind="error">{error && !saved ? error : null}</Notice>
        </div>
      </div>
    </section>
  )
}

// ---------------------------------------------------------------------------
// Personalization (transparency + control)
// ---------------------------------------------------------------------------

function PersonalizationSection() {
  const [view, setView] = useState(null)
  const [enabled, setEnabled] = useState(null)
  const [toggling, setToggling] = useState(false)
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)
  const [confirmClear, setConfirmClear] = useState(false)
  const [cleared, setCleared] = useState(false)

  async function load() {
    try {
      const [personalization, prefs] = await Promise.all([
        fetchPersonalization(),
        fetchPreferences(),
      ])
      setView(personalization)
      setEnabled(prefs.personalizationEnabled)
      setError(null)
    } catch (e) {
      setError(e.message)
    }
  }

  useEffect(() => {
    load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  async function toggleLearning(next) {
    setToggling(true)
    setError(null)
    try {
      const prefs = await fetchPreferences()
      await savePreferences({ ...prefs, personalizationEnabled: next })
      setEnabled(next)
      await load()
    } catch (e) {
      setError(e.message)
    } finally {
      setToggling(false)
    }
  }

  async function doClear() {
    setBusy(true)
    try {
      await clearPersonalization()
      await load()
      setCleared(true)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
      setConfirmClear(false)
    }
  }

  return (
    <section className="card settings-card" aria-label="Personalization">
      <h2>Personalization</h2>
      <p className="desc">
        Over time Sage may remember small, useful things you share — your name, how
        you like to be spoken to, topics to steer around. Here is everything it has
        remembered, and you can erase it all at once.
      </p>

      {enabled !== null && (
        <div className="switch-row" style={{ marginBottom: 16 }}>
          <span>
            <span style={{ fontWeight: 600, fontSize: '0.92rem' }}>
              Remember details across chats
            </span>
            <span className="hint" style={{ display: 'block' }}>
              When off, Sage stops learning new details. Your preferences above still apply.
            </span>
          </span>
          <Switch
            checked={enabled}
            onChange={(next) => toggleLearning(next)}
            label="Remember details across chats"
          />
        </div>
      )}

      {error && <div className="notice notice-error" role="alert">{error}</div>}

      {!view && !error && <div className="skeleton" style={{ height: 60 }} />}

      {view && view.items.length === 0 && (
        <p className="muted small">
          Nothing is remembered yet. Anything you see here will always come from
          what you actually said — never invented.
        </p>
      )}

      {view && view.items.length > 0 && (
        <>
          <ul className="learned-list">
            {view.items.map((item) => (
              <li key={`${item.kind}:${item.value}`} className="learned-item">
                <span className="kind">{item.kind.replace(/_/g, ' ')}</span>
                <span className="value">{item.value}</span>
                <span className="chip chip-plain">
                  {item.conversations} conversation{item.conversations === 1 ? '' : 's'}
                </span>
              </li>
            ))}
          </ul>
          <div className="settings-actions">
            <button
              type="button"
              className="btn btn-soft-danger"
              onClick={() => setConfirmClear(true)}
              disabled={busy}
            >
              Forget everything Sage has remembered
            </button>
            <Notice kind="ok">
              {cleared ? 'Cleared. No learned details remain.' : null}
            </Notice>
          </div>
        </>
      )}

      <ConfirmDialog
        open={confirmClear}
        title="Forget remembered details?"
        body="Every personal detail Sage learned from your conversations will be removed. Your conversations themselves are kept, and your explicit preferences above are not affected."
        confirmLabel="Forget details"
        danger
        busy={busy}
        onConfirm={doClear}
        onCancel={() => setConfirmClear(false)}
      />
    </section>
  )
}

// ---------------------------------------------------------------------------
// Appearance (device-local)
// ---------------------------------------------------------------------------

function AppearanceSection({ onThemeChanged }) {
  const [theme, setTheme] = useState(readTheme())
  const [textSize, setTextSize] = useState(readTextSize())

  function chooseTheme(value) {
    setTheme(value)
    writeTheme(value)
    // re-theme the DOM immediately (the same way chooseSize applies text size)
    applyAppearance(value, readTextSize())
    onThemeChanged?.(value)
  }
  function chooseSize(value) {
    setTextSize(value)
    writeTextSize(value)
    applyAppearance(readTheme(), value)
  }

  return (
    <section className="card settings-card" aria-label="Appearance">
      <h2>Appearance</h2>
      <p className="desc">Saved on this device only — nothing here is sent to the server.</p>
      <div className="settings-group">
        <div className="field">
          <label>Theme</label>
          <ChoiceGroup
            name="theme"
            options={[
              { value: 'light', label: 'Light' },
              { value: 'dark', label: 'Dark' },
              { value: 'system', label: 'Match system' },
            ]}
            value={theme}
            onChange={chooseTheme}
          />
        </div>
        <div className="field">
          <label>Text size</label>
          <ChoiceGroup
            name="textsize"
            options={[
              { value: 'standard', label: 'Standard' },
              { value: 'large', label: 'Large' },
            ]}
            value={textSize}
            onChange={chooseSize}
          />
        </div>
      </div>
    </section>
  )
}

// ---------------------------------------------------------------------------
// Voice (Phase 6) - device-local only; nothing here touches the server
// ---------------------------------------------------------------------------

function VoiceSection() {
  const [settings, update] = useVoiceSettings()
  const [voices, setVoices] = useState(() => speechController().voices())
  const sttSupported = isSttSupported()
  const ttsSupported = speechController().supported

  // voices can arrive asynchronously (Chrome populates them late)
  useEffect(() => {
    const synth = globalThis.speechSynthesis
    if (!synth) return undefined
    const refresh = () => setVoices(synth.getVoices())
    refresh()
    synth.addEventListener?.('voiceschanged', refresh)
    // one delayed retry covers browsers that never fire the event
    const timer = globalThis.setTimeout?.(refresh, 800)
    return () => {
      synth.removeEventListener?.('voiceschanged', refresh)
      if (timer) globalThis.clearTimeout?.(timer)
    }
  }, [])

  // only voices matching the current conversation language are offered
  const lang = readStoredLanguage()
  const matching = voices.filter((v) => v.lang?.toLowerCase().startsWith(lang.toLowerCase()))
  const shown = matching.length > 0 ? matching : voices

  return (
    <section className="card settings-card" aria-label="Voice">
      <h2>Voice</h2>
      <p className="desc">
        Speech runs entirely in your browser. Audio is never uploaded and no
        recording is stored — recognized text simply appears in the message
        box for you to review before sending.
      </p>
      <div className="settings-group">
        <div className="switch-row">
          <span>
            <span style={{ fontWeight: 600, fontSize: '0.92rem' }}>Microphone input</span>
            <span className="hint" style={{ display: 'block' }}>
              {sttSupported
                ? 'Show the microphone button so you can speak instead of typing.'
                : 'Not supported in this browser — Chrome and Edge support speech input.'}
            </span>
          </span>
          <Switch
            checked={settings.sttEnabled && sttSupported}
            onChange={(sttEnabled) => update({ sttEnabled })}
            label="Microphone input"
          />
        </div>
        <div className="switch-row">
          <span>
            <span style={{ fontWeight: 600, fontSize: '0.92rem' }}>Read replies aloud</span>
            <span className="hint" style={{ display: 'block' }}>
              {ttsSupported
                ? 'Add a play button under each reply to hear it spoken.'
                : 'Not supported in this browser.'}
            </span>
          </span>
          <Switch
            checked={settings.ttsEnabled && ttsSupported}
            onChange={(ttsEnabled) => update({ ttsEnabled })}
            label="Read replies aloud"
          />
        </div>
        {settings.ttsEnabled && ttsSupported && (
          <>
            <Field
              label="Voice"
              id="voice-select"
              hint={matching.length > 0 ? `Voices for your conversation language (${lang}).` : 'Your language has no dedicated voice here; the browser default is used.'}
            >
              <select
                id="voice-select"
                value={settings.ttsVoice}
                onChange={(e) => update({ ttsVoice: e.target.value })}
              >
                <option value="">Browser default</option>
                {shown.map((v) => (
                  <option key={v.voiceURI} value={v.voiceURI}>
                    {v.name} ({v.lang})
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Speed" id="voice-rate" hint="1 is normal speaking pace.">
              <select
                id="voice-rate"
                value={String(settings.ttsRate)}
                onChange={(e) => update({ ttsRate: Number(e.target.value) })}
              >
                <option value="0.75">Slow</option>
                <option value="1">Normal</option>
                <option value="1.25">Brisk</option>
                <option value="1.5">Fast</option>
              </select>
            </Field>
          </>
        )}
      </div>
    </section>
  )
}

// ---------------------------------------------------------------------------
// Notifications (in-app, server-persisted)
// ---------------------------------------------------------------------------

function NotificationsSection() {
  const [settings, setSettings] = useState(null)
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false
    fetchSettings()
      .then((loaded) => {
        if (!cancelled) setSettings(loaded)
      })
      .catch((e) => {
        if (!cancelled) setError(e.message)
      })
    return () => {
      cancelled = true
    }
  }, [])

  async function toggle(next) {
    setSaving(true)
    setSaved(false)
    setError(null)
    // optimistic: flip immediately, roll back if the write fails
    setSettings((prev) => ({ ...prev, dailyReminders: next }))
    try {
      const stored = await saveSettings({ dailyReminders: next })
      setSettings(stored)
      setSaved(true)
    } catch (e) {
      setSettings((prev) => ({ ...prev, dailyReminders: !next }))
      setError(e.message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <section className="card settings-card" aria-label="Notifications">
      <h2>Notifications</h2>
      <p className="desc">
        Gentle in-app nudges — never email or SMS. Sage never sends you
        anything outside this app.
      </p>
      {!settings && !error && <div className="skeleton" style={{ height: 54 }} />}
      {settings && (
        <div className="switch-row">
          <span>
            <span style={{ fontWeight: 600, fontSize: '0.92rem' }}>
              Daily check-in reminder
            </span>
            <span className="hint" style={{ display: 'block' }}>
              A small prompt on the dashboard to log how you're feeling today.
            </span>
          </span>
          <Switch
            checked={settings.dailyReminders}
            onChange={(next) => toggle(next)}
            label="Daily check-in reminder"
          />
        </div>
      )}
      <Notice kind="ok">{saved ? 'Saved.' : null}</Notice>
      <Notice kind="error">{error}</Notice>
    </section>
  )
}

// ---------------------------------------------------------------------------
// Privacy / danger zone
// ---------------------------------------------------------------------------

function PrivacySection({ onDeleted, onConversationsChanged }) {
  const [confirming, setConfirming] = useState(false)
  const [confirmClearAll, setConfirmClearAll] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [cleared, setCleared] = useState(false)

  async function doDelete() {
    setBusy(true)
    setError(null)
    try {
      await deleteAccount()
      onDeleted()
    } catch (e) {
      setError(e.message)
      setBusy(false)
      setConfirming(false)
    }
  }

  async function doDeleteAll() {
    setBusy(true)
    setError(null)
    try {
      await deleteAllConversations()
      setCleared(true)
      onConversationsChanged?.() // sidebar + dashboard recents refresh
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
      setConfirmClearAll(false)
    }
  }

  return (
    <section className="card settings-card danger-zone" aria-label="Privacy and data">
      <h2>Privacy &amp; data</h2>
      <p className="desc">
        Your conversations belong to you. You can remove them at any time —
        individually from the sidebar, all at once below, or together with
        your account.
      </p>
      <div className="settings-actions">
        <button
          type="button"
          className="btn btn-soft-danger"
          onClick={() => setConfirmClearAll(true)}
          disabled={busy}
        >
          Delete all conversations
        </button>
        <button type="button" className="btn btn-soft-danger" onClick={() => setConfirming(true)}>
          Delete account and all data
        </button>
        <Notice kind="ok">
          {cleared ? 'All conversations deleted.' : null}
        </Notice>
        <Notice kind="error">{error}</Notice>
      </div>
      <ConfirmDialog
        open={confirmClearAll}
        title="Delete all conversations?"
        body="Every conversation, message and learned detail will be removed permanently. Your account and settings stay."
        confirmLabel="Delete all"
        danger
        busy={busy}
        onConfirm={doDeleteAll}
        onCancel={() => setConfirmClearAll(false)}
      />
      <ConfirmDialog
        open={confirming}
        title="Delete your account?"
        body="This permanently removes your login, every conversation, and every saved message. There is no way to recover them."
        confirmLabel="Delete everything"
        danger
        busy={busy}
        onConfirm={doDelete}
        onCancel={() => setConfirming(false)}
      />
    </section>
  )
}

// ---------------------------------------------------------------------------

export function Settings({
  user,
  onUserChanged,
  onAccountDeleted,
  onThemeChanged,
  onConversationsChanged,
  onOpenMenu,
}) {
  return (
    <div className="settings-scroll">
      <div className="settings-inner">
        <button
          type="button"
          className="btn-icon mobile-only page-menu-btn"
          onClick={onOpenMenu}
          aria-label="Open navigation"
        >
          <Icon name="menu" size={20} />
        </button>
        <h1>Settings</h1>
        <ProfileSection user={user} onUserChanged={onUserChanged} />
        <PreferencesSection />
        <PersonalizationSection />
        <VoiceSection />
        <NotificationsSection />
        <AppearanceSection onThemeChanged={onThemeChanged} />
        <PrivacySection
          onDeleted={onAccountDeleted}
          onConversationsChanged={onConversationsChanged}
        />
      </div>
    </div>
  )
}
