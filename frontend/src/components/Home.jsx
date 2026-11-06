import { formatDate, greetingMessage, relativeTime } from '../greeting.js'
import { routes } from '../router.js'
import { MoodCheckIn } from './MoodCheckIn.jsx'
import { Icon } from './ui.jsx'

/** Suggested conversation starters - prompts, never fabricated history. */
export const STARTERS = [
  "I've been feeling overwhelmed lately and need to talk",
  'Help me reflect on my week',
  "I'm having trouble sleeping because of stress",
  'I feel anxious about an upcoming exam',
  'I want to work through a difficult decision',
  'I could use some encouragement today',
]

function StartersGrid({ onPick, compact = false }) {
  return (
    <div className="starters">
      {STARTERS.slice(0, compact ? 4 : STARTERS.length).map((text) => (
        <button key={text} type="button" className="starter" onClick={() => onPick(text)}>
          {text}
        </button>
      ))}
    </div>
  )
}

function RecentPanel({ conversations }) {
  const recent = conversations.slice(0, 5)
  if (recent.length === 0) {
    return (
      <p className="recent-empty">
        Nothing here yet. Anything you share starts a private conversation you
        can return to anytime.
      </p>
    )
  }
  return (
    <div className="recent-list">
      {recent.map((conversation) => (
        <a
          key={conversation.sessionId}
          className="recent-item"
          href={routes.chat(conversation.sessionId)}
        >
          <span className="grow">
            <span className="title">{conversation.title}</span>
            <span className="meta">
              {conversation.messageCount} messages · {relativeTime(conversation.updatedAt)}
            </span>
          </span>
          <Icon name="chevron" size={16} />
        </a>
      ))}
    </div>
  )
}

/**
 * Dashboard: today's date, time-of-day greeting with the user's real name, a
 * warm invitation, the primary action, and a grid of suggested starters,
 * recent chats, the daily mood check-in and wellness shortcuts. Every card
 * renders real data from its API - no mock content.
 */
export function Home({ user, anonymous, conversations, onNewConversation, onPickStarter, onOpenMenu }) {
  const name = anonymous ? '' : (user?.displayName ?? user?.username ?? '')
  const today = formatDate(new Date().toISOString())

  return (
    <div className="home">
      <div className="home-inner">
        <header className="dash-head dash-fade">
          <button
            type="button"
            className="btn-icon mobile-only page-menu-btn"
            onClick={onOpenMenu}
            aria-label="Open navigation"
          >
            <Icon name="menu" size={20} />
          </button>
          <p className="dash-date">{today}</p>
          <h1 className="home-greeting">{greetingMessage(name)}</h1>
          <p className="home-sub">
            This is your space to pause, reflect, and talk things through - at your
            own pace, whenever you're ready.
          </p>
          <div className="home-cta">
            <button type="button" className="btn btn-primary btn-lg" onClick={onNewConversation}>
              <Icon name="plus" size={17} />
              Start a new conversation
            </button>
          </div>
        </header>

        <div className={`dash-grid${anonymous ? ' dash-grid-single' : ''}`}>
          <section className="dash-card dash-fade" style={{ '--delay': '70ms' }}>
            <div className="section-title">Where would you like to start?</div>
            <StartersGrid onPick={onPickStarter} />
          </section>

          {!anonymous && (
            <section className="dash-card dash-fade" style={{ '--delay': '130ms' }}>
              <div className="section-title">Continue where you left off</div>
              <RecentPanel conversations={conversations} />
            </section>
          )}
        </div>

        {!anonymous && (
          <div className="dash-grid dash-grid-wellness">
            <section className="dash-card dash-fade" style={{ '--delay': '190ms' }}>
              <div className="section-title">Daily mood check-in</div>
              <MoodCheckIn />
              <a className="dash-link" href={routes.mood()}>
                See your history <Icon name="chevron" size={14} />
              </a>
            </section>

            <section className="dash-card dash-fade" style={{ '--delay': '250ms' }}>
              <div className="section-title">Wellness activities</div>
              <p className="recent-empty" style={{ marginBottom: 12 }}>
                Short practices for hard moments - breathing, grounding,
                relaxation and mindfulness.
              </p>
              <div className="wellness-mini">
                {[
                  { href: routes.wellness(), icon: 'wind', label: 'Guided breathing' },
                  { href: routes.wellness(), icon: 'leaf', label: 'Grounding' },
                  { href: routes.wellness(), icon: 'sun', label: 'Relaxation' },
                  { href: routes.wellness(), icon: 'spark', label: 'Mindfulness' },
                ].map((item) => (
                  <a key={item.label} className="wellness-mini-item" href={item.href}>
                    <Icon name={item.icon} size={16} />
                    {item.label}
                  </a>
                ))}
              </div>
              <a className="dash-link" href={routes.wellness()}>
                Open activities <Icon name="chevron" size={14} />
              </a>
            </section>
          </div>
        )}
      </div>
    </div>
  )
}
