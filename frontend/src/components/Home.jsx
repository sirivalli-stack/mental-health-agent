import { firstName, greetingMessage, relativeTime } from '../greeting.js'
import { routes } from '../router.js'
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

/**
 * Personalized home: time-of-day greeting with the user's real name, a warm
 * invitation, the primary action, suggested starters and recent chats.
 */
export function Home({ user, anonymous, conversations, onNewConversation, onPickStarter }) {
  const name = anonymous ? '' : (user?.displayName ?? user?.username ?? '')
  const recent = conversations.slice(0, 5)

  return (
    <div className="home">
      <div className="home-inner">
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

        <div className="section-title">Where would you like to start?</div>
        <StartersGrid onPick={onPickStarter} />

        {!anonymous && (
          <>
            <div className="section-title">Continue where you left off</div>
            {recent.length === 0 ? (
              <p className="recent-empty">
                Nothing here yet. Anything you share starts a private
                conversation you can return to anytime.
              </p>
            ) : (
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
            )}
          </>
        )}
      </div>
    </div>
  )
}
