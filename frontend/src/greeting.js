/**
 * Greetings and time formatting - pure, unit-testable.
 * All formatting uses the viewer's local time zone (the browser's).
 */

export function greetingFor(date = new Date()) {
  const hour = date.getHours()
  if (hour < 5) return 'Good evening'
  if (hour < 12) return 'Good morning'
  if (hour < 18) return 'Good afternoon'
  return 'Good evening'
}

export function firstName(name) {
  const trimmed = String(name ?? '').trim()
  if (!trimmed) return ''
  return trimmed.split(/\s+/)[0]
}

export function greetingMessage(name, date = new Date()) {
  const first = firstName(name)
  const greeting = greetingFor(date)
  return first ? `${greeting}, ${first}.` : `${greeting}.`
}

function startOfDay(date) {
  const d = new Date(date)
  d.setHours(0, 0, 0, 0)
  return d
}

/** Compact, human relative label: "just now", "12m ago", "3h ago", ... */
export function relativeTime(iso, now = new Date()) {
  const then = new Date(iso)
  if (Number.isNaN(then.getTime())) return ''
  const seconds = Math.max(0, (now.getTime() - then.getTime()) / 1000)
  if (seconds < 45) return 'just now'
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`
  const days = Math.round(
    (startOfDay(now).getTime() - startOfDay(then).getTime()) / 86_400_000,
  )
  if (days <= 0) return `${Math.round(seconds / 3600)}h ago`
  if (days === 1) return 'Yesterday'
  if (days < 7) return `${days}d ago`
  return then.toLocaleDateString(undefined, { day: 'numeric', month: 'short' })
}

export function formatClock(iso) {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return ''
  return date.toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
  })
}

export function formatDate(iso) {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return ''
  return date.toLocaleDateString(undefined, {
    day: 'numeric',
    month: 'long',
    year: 'numeric',
  })
}

/** Initials for the avatar chip: "Alex Rivera" -> "AR". */
export function initials(name) {
  const parts = String(name ?? '').trim().split(/\s+/).filter(Boolean)
  if (parts.length === 0) return '?'
  const first = parts[0][0]
  const last = parts.length > 1 ? parts[parts.length - 1][0] : ''
  return (first + last).toUpperCase()
}
