import { useEffect, useRef, useState } from 'react'

/** Small icon set (inline SVG, stroke = currentColor). */
export function Icon({ name, size = 18 }) {
  const paths = {
    plus: <path d="M10 4v12M4 10h12" />,
    search: (
      <>
        <circle cx="9" cy="9" r="5.5" />
        <path d="M13.5 13.5 17 17" />
      </>
    ),
    settings: (
      <>
        <circle cx="10" cy="10" r="2.6" />
        <path d="M10 2.8v2M10 15.2v2M2.8 10h2M15.2 10h2M4.9 4.9l1.4 1.4M13.7 13.7l1.4 1.4M15.1 4.9l-1.4 1.4M6.3 13.7l-1.4 1.4" />
      </>
    ),
    logout: (
      <>
        <path d="M8 17H5a1.5 1.5 0 0 1-1.5-1.5v-11A1.5 1.5 0 0 1 5 3h3" />
        <path d="M13 13.5 16.5 10 13 6.5M16.5 10H7.5" />
      </>
    ),
    trash: (
      <>
        <path d="M4 5.5h12M8 5.5V4h4v1.5M6 5.5l.7 10h6.6l.7-10" />
      </>
    ),
    pencil: <path d="m12.6 3.9 3.5 3.5-8.6 8.6-4.2.7.7-4.2zM11.4 5.1l3.5 3.5" />,
    send: <path d="M3.5 10 16.5 4l-4 12-2.6-4.8z" />,
    close: <path d="m5 5 10 10M15 5 5 15" />,
    menu: <path d="M3.5 6h13M3.5 10h13M3.5 14h13" />,
    user: (
      <>
        <circle cx="10" cy="7" r="3" />
        <path d="M4.5 16.5c.8-2.7 3-4 5.5-4s4.7 1.3 5.5 4" />
      </>
    ),
    spark: (
      <path d="M10 3v4M10 13v4M3 10h4M13 10h4M5.8 5.8l2.1 2.1M12.1 12.1l2.1 2.1M14.2 5.8l-2.1 2.1M7.9 12.1l-2.1 2.1" />
    ),
    info: (
      <>
        <circle cx="10" cy="10" r="7" />
        <path d="M10 9v4.5M10 6.6v.2" />
      </>
    ),
    chevron: <path d="m6 8 4 4 4-4" />,
    eye: (
      <>
        <path d="M2.5 10S5.5 5 10 5s7.5 5 7.5 5-3 5-7.5 5-7.5-5-7.5-5z" />
        <circle cx="10" cy="10" r="2" />
      </>
    ),
    eyeOff: (
      <>
        <path d="M4 4l12 12" />
        <path d="M7.7 7.8A2 2 0 0 0 10 12a2 2 0 0 0 1.9-1.3M5.2 6.4C3.6 7.6 2.5 10 2.5 10s3 5 7.5 5c1.2 0 2.3-.3 3.2-.8M14.5 12.6c1.6-1.2 3-2.6 3-2.6s-3-5-7.5-5" />
      </>
    ),
  }
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 20 20"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {paths[name] ?? null}
    </svg>
  )
}

/** Brand mark: a leaf inside a rounded square. */
export function BrandMark({ size = 34 }) {
  return (
    <span className="brand-mark" style={{ width: size, height: size }} aria-hidden="true">
      <svg
        width={size * 0.58}
        height={size * 0.58}
        viewBox="0 0 20 20"
        fill="none"
        aria-hidden="true"
      >
        <path
          d="M16 4c0 6.6-3.4 10.4-9 11.2M16 4C9.6 4.4 5.4 7.2 4.6 12.4M16 4c-5.2.6-8.4 3.4-9.4 7"
          stroke="#FAF9F6"
          strokeWidth="1.5"
          strokeLinecap="round"
        />
        <path
          d="M5.6 15.2C7 12 10 9.6 13.4 8.4"
          stroke="#FAF9F6"
          strokeWidth="1.5"
          strokeLinecap="round"
          opacity="0.75"
        />
      </svg>
    </span>
  )
}

export function Spinner() {
  return <span className="spinner" role="status" aria-label="loading" />
}

/** Text field with label, optional hint and error. */
export function Field({ label, hint, error, id, children }) {
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      {children}
      {hint && !error && <span className="hint">{hint}</span>}
      {error && (
        <span className="hint" style={{ color: 'var(--danger)' }} role="alert">
          {error}
        </span>
      )}
    </div>
  )
}

/** Password input with a show/hide toggle. */
export function PasswordField({ id, label, hint, error, value, onChange, autoComplete, autoFocus }) {
  const [visible, setVisible] = useState(false)
  return (
    <Field label={label} hint={hint} error={error} id={id}>
      <span className="pw-wrap">
        <input
          id={id}
          type={visible ? 'text' : 'password'}
          value={value}
          autoComplete={autoComplete}
          autoFocus={autoFocus}
          onChange={(e) => onChange(e.target.value)}
        />
        <button
          type="button"
          className="pw-toggle"
          onClick={() => setVisible((v) => !v)}
          aria-label={visible ? 'Hide password' : 'Show password'}
          title={visible ? 'Hide password' : 'Show password'}
        >
          <Icon name={visible ? 'eyeOff' : 'eye'} size={16} />
        </button>
      </span>
    </Field>
  )
}

/** Accessible toggle switch. */
export function Switch({ checked, onChange, label }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      className="switch"
      onClick={() => onChange(!checked)}
    />
  )
}

/** Confirmation dialog; renders nothing until opened. */
export function ConfirmDialog({ open, title, body, confirmLabel = 'Confirm', danger = false, busy = false, onConfirm, onCancel }) {
  const ref = useRef(null)
  useEffect(() => {
    if (open) ref.current?.focus()
  }, [open])
  if (!open) return null
  return (
    <div className="modal-scrim" role="presentation" onClick={onCancel}>
      <div
        className="modal"
        role="alertdialog"
        aria-modal="true"
        aria-label={title}
        onClick={(e) => e.stopPropagation()}
      >
        <h2>{title}</h2>
        <p>{body}</p>
        <div className="modal-actions">
          <button type="button" className="btn" onClick={onCancel} disabled={busy}>
            Cancel
          </button>
          <button
            ref={ref}
            type="button"
            className={`btn ${danger ? 'btn-danger' : 'btn-primary'}`}
            onClick={onConfirm}
            disabled={busy}
          >
            {busy ? <Spinner /> : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  )
}
