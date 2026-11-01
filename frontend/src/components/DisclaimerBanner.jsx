import { Icon } from './ui.jsx'

/** Slim, always-visible safety note (the backend's own disclaimer text). */
export function DisclaimerBanner({ text }) {
  if (!text) return null
  return (
    <div className="disclaimer" role="note">
      <Icon name="info" size={14} />
      <span>{text}</span>
    </div>
  )
}
