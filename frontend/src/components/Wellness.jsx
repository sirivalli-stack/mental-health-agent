import { useEffect, useRef, useState } from 'react'
import { Icon } from './ui.jsx'

/**
 * Wellness activities (Phase 7): four lightweight, client-only exercises -
 * breathing, grounding, relaxation and mindfulness. Nothing here talks to
 * the backend; they are guided prompts and timers, not treatment. If a
 * practice feels like too much, stopping is always one tap away.
 */

const ACTIVITIES = [
  { key: 'breathing', name: 'Guided breathing', blurb: 'Box breathing to settle your body.', icon: 'wind' },
  { key: 'grounding', name: 'Grounding', blurb: 'The 5-4-3-2-1 senses exercise.', icon: 'leaf' },
  { key: 'relaxation', name: 'Relaxation', blurb: 'A short muscle-release sequence.', icon: 'sun' },
  { key: 'mindfulness', name: 'Mindfulness', blurb: 'A one-minute mindful pause.', icon: 'spark' },
]

// ---------------------------------------------------------------------------
// Breathing: 4-4-4-4 box breathing with an animated circle
// ---------------------------------------------------------------------------

const BREATH_PHASES = [
  { label: 'Breathe in', seconds: 4, scale: 1.35 },
  { label: 'Hold', seconds: 4, scale: 1.35 },
  { label: 'Breathe out', seconds: 4, scale: 0.85 },
  { label: 'Hold', seconds: 4, scale: 0.85 },
]

function BreathingActivity() {
  const [running, setRunning] = useState(false)
  const [phaseIndex, setPhaseIndex] = useState(0)
  const [remaining, setRemaining] = useState(BREATH_PHASES[0].seconds)
  const [cycles, setCycles] = useState(0)
  const timerRef = useRef(null)

  useEffect(() => {
    if (!running) return undefined
    timerRef.current = globalThis.setInterval(() => {
      setRemaining((prev) => {
        if (prev > 1) return prev - 1
        // advance to the next phase
        setPhaseIndex((idx) => {
          const next = (idx + 1) % BREATH_PHASES.length
          if (next === 0) setCycles((c) => c + 1)
          setRemaining(BREATH_PHASES[next].seconds)
          return next
        })
        return BREATH_PHASES[phaseIndex].seconds
      })
    }, 1000)
    return () => globalThis.clearInterval(timerRef.current)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running])

  function stop() {
    setRunning(false)
    setPhaseIndex(0)
    setRemaining(BREATH_PHASES[0].seconds)
  }

  const phase = BREATH_PHASES[phaseIndex]

  return (
    <div className="activity-body">
      <p className="activity-lead">
        Follow the circle. Breathe in as it grows, hold, breathe out as it
        shrinks, hold again - four counts each. A few rounds is plenty.
      </p>
      <div className="breathing-stage" aria-live="polite">
        <span
          className={`breathing-circle${running ? ' running' : ''}`}
          style={{ transform: `scale(${running ? phase.scale : 1})` }}
        />
        <span className="breathing-label">{running ? phase.label : 'Ready?'}</span>
        <span className="breathing-count">{running ? `${remaining}` : ''}</span>
      </div>
      <div className="activity-actions">
        {!running ? (
          <button type="button" className="btn btn-primary" onClick={() => setRunning(true)}>
            <Icon name="play" size={15} />
            Start
          </button>
        ) : (
          <button type="button" className="btn" onClick={stop}>
            <Icon name="stop" size={15} />
            Stop
          </button>
        )}
        {cycles > 0 && <span className="faint">{cycles} cycle{cycles === 1 ? '' : 's'} completed</span>}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Grounding: 5-4-3-2-1 senses
// ---------------------------------------------------------------------------

const GROUNDING_STEPS = [
  { count: 5, sense: 'things you can see', prompt: 'Look around and name five things you can see.' },
  { count: 4, sense: 'things you can feel', prompt: 'Notice four things you can feel - feet on the floor, fabric, air.' },
  { count: 3, sense: 'things you can hear', prompt: 'Listen for three sounds, near or far.' },
  { count: 2, sense: 'things you can smell', prompt: 'Find two things you can smell, or two smells you like.' },
  { count: 1, sense: 'thing you can taste', prompt: 'Notice one thing you can taste right now.' },
]

function GroundingActivity() {
  const [step, setStep] = useState(0)
  const current = GROUNDING_STEPS[step]
  const done = step >= GROUNDING_STEPS.length

  return (
    <div className="activity-body">
      <p className="activity-lead">
        Work through the senses in order - there is no rush, and you can stop
        whenever you like.
      </p>
      {done ? (
        <div className="activity-done" role="status">
          <Icon name="check" size={18} />
          <div>
            <strong>That's the exercise.</strong>
            <p className="faint">Come back to it any time things feel fuzzy.</p>
          </div>
          <button type="button" className="btn btn-sm" onClick={() => setStep(0)}>
            Start over
          </button>
        </div>
      ) : (
        <div className="grounding-step" aria-live="polite">
          <span className="grounding-count">{current.count}</span>
          <div>
            <div className="grounding-sense">{current.sense}</div>
            <p className="activity-lead">{current.prompt}</p>
          </div>
        </div>
      )}
      {!done && (
        <div className="activity-actions">
          <button
            type="button"
            className="btn"
            disabled={step === 0}
            onClick={() => setStep((s) => s - 1)}
          >
            Back
          </button>
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => setStep((s) => s + 1)}
          >
            {step === GROUNDING_STEPS.length - 1 ? 'Finish' : 'Next'}
          </button>
          <span className="faint">
            Step {step + 1} of {GROUNDING_STEPS.length}
          </span>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Relaxation: progressive muscle release
// ---------------------------------------------------------------------------

const RELAX_STEPS = [
  'Let your shoulders drop away from your ears. Unclench your jaw.',
  'Soften your forehead and the space between your eyebrows.',
  'Release your hands - let them rest open, whatever they are doing.',
  'Breathe out slowly and let your chest feel heavy and easy.',
  'Notice your feet on the ground and let them be completely still.',
  'Take one slow breath and check in: anything left to soften?',
]

function RelaxationActivity() {
  const [step, setStep] = useState(0)
  const done = step >= RELAX_STEPS.length

  return (
    <div className="activity-body">
      <p className="activity-lead">
        A short release sequence - work down the body, one line at a time.
      </p>
      {done ? (
        <div className="activity-done" role="status">
          <Icon name="check" size={18} />
          <div>
            <strong>Well done.</strong>
            <p className="faint">Carry that softness with you.</p>
          </div>
          <button type="button" className="btn btn-sm" onClick={() => setStep(0)}>
            Repeat
          </button>
        </div>
      ) : (
        <div className="relax-step" aria-live="polite">
          <span className="relax-num">{step + 1}</span>
          <p className="relax-text">{RELAX_STEPS[step]}</p>
        </div>
      )}
      {!done && (
        <div className="activity-actions">
          <button
            type="button"
            className="btn"
            disabled={step === 0}
            onClick={() => setStep((s) => s - 1)}
          >
            Back
          </button>
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => setStep((s) => s + 1)}
          >
            {step === RELAX_STEPS.length - 1 ? 'Finish' : 'Next'}
          </button>
          <span className="faint">
            {step + 1} of {RELAX_STEPS.length}
          </span>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Mindfulness: one-minute observation pause
// ---------------------------------------------------------------------------

const MINDFUL_PROMPTS = [
  'Pick one ordinary thing near you - a cup, a lamp, a leaf outside the window.',
  'Look at it as if you have never seen it before. Notice colour, shape, texture.',
  'When your mind wanders (it will), gently come back to the one thing.',
  'Take a slow breath. Carry that same attention into your next task.',
]

function MindfulnessActivity() {
  const [step, setStep] = useState(0)
  const [seconds, setSeconds] = useState(60)
  const [running, setRunning] = useState(false)
  const timerRef = useRef(null)

  useEffect(() => {
    if (!running) return undefined
    timerRef.current = globalThis.setInterval(() => {
      setSeconds((prev) => {
        if (prev <= 1) {
          globalThis.clearInterval(timerRef.current)
          setRunning(false)
          return 0
        }
        return prev - 1
      })
    }, 1000)
    return () => globalThis.clearInterval(timerRef.current)
  }, [running])

  const finished = seconds === 0

  return (
    <div className="activity-body">
      <p className="activity-lead">
        A minute of ordinary attention - that is all it takes.
      </p>
      <div className="mindful-prompt" aria-live="polite">
        <span className="relax-num">{step + 1}</span>
        <p className="relax-text">{MINDFUL_PROMPTS[step]}</p>
      </div>
      <div className="activity-actions">
        <button
          type="button"
          className="btn"
          disabled={step === 0}
          onClick={() => setStep((s) => s - 1)}
        >
          Back
        </button>
        <button
          type="button"
          className="btn btn-primary"
          onClick={() => setStep((s) => Math.min(s + 1, MINDFUL_PROMPTS.length - 1))}
          disabled={step === MINDFUL_PROMPTS.length - 1}
        >
          Next
        </button>
        <span className="timer-chip" aria-live="polite">
          {finished ? 'Minute complete' : `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`}
        </span>
        {!running && !finished && (
          <button type="button" className="btn btn-sm" onClick={() => setRunning(true)}>
            <Icon name="play" size={14} />
            Start 1-minute pause
          </button>
        )}
        {running && (
          <button
            type="button"
            className="btn btn-sm"
            onClick={() => {
              setRunning(false)
              setSeconds(60)
            }}
          >
            <Icon name="stop" size={14} />
            Stop
          </button>
        )}
        {finished && (
          <button type="button" className="btn btn-sm" onClick={() => setSeconds(60)}>
            Go again
          </button>
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------

const BODIES = {
  breathing: BreathingActivity,
  grounding: GroundingActivity,
  relaxation: RelaxationActivity,
  mindfulness: MindfulnessActivity,
}

/**
 * Wellness hub: pick an activity, run it in place. Every exercise is a
 * gentle prompt with a start and a stop - support, not treatment.
 */
export function WellnessPage({ onOpenMenu }) {
  const [active, setActive] = useState(null)
  const Body = active ? BODIES[active] : null
  const meta = ACTIVITIES.find((a) => a.key === active)

  return (
    <div className="page-scroll">
      <div className="page-inner">
        <header className="page-head">
          <button
            type="button"
            className="btn-icon mobile-only page-menu-btn"
            onClick={onOpenMenu}
            aria-label="Open navigation"
          >
            <Icon name="menu" size={20} />
          </button>
          <h1>Wellness activities</h1>
          <p className="page-sub">
            Short, gentle practices for hard moments. They run entirely on your
            device - nothing is recorded or sent anywhere.
          </p>
        </header>

        {!active && (
          <div className="wellness-grid">
            {ACTIVITIES.map((a, i) => (
              <button
                key={a.key}
                type="button"
                className="dash-card wellness-card dash-fade"
                style={{ '--delay': `${i * 60}ms` }}
                onClick={() => setActive(a.key)}
              >
                <span className="wellness-icon" aria-hidden="true">
                  <Icon name={a.icon} size={20} />
                </span>
                <span className="wellness-name">{a.name}</span>
                <span className="wellness-blurb">{a.blurb}</span>
                <span className="wellness-go">
                  Open <Icon name="chevron" size={14} />
                </span>
              </button>
            ))}
          </div>
        )}

        {active && (
          <section className="card dash-card" aria-label={meta?.name}>
            <div className="activity-head">
              <button type="button" className="btn btn-sm" onClick={() => setActive(null)}>
                <Icon name="chevron" size={14} style={{ transform: 'rotate(180deg)' }} />
                All activities
              </button>
              <div className="section-title" style={{ margin: 0 }}>
                {meta?.name}
              </div>
            </div>
            <Body />
            <p className="mood-disclaimer">
              These are wellness exercises, not medical care. If things feel
              heavier than a practice can reach, please reach out to someone
              you trust or a professional.
            </p>
          </section>
        )}
      </div>
    </div>
  )
}
