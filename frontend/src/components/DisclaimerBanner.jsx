export function DisclaimerBanner({ text }) {
  if (!text) return null
  return (
    <div className="disclaimer" role="note">
      {text}
    </div>
  )
}
