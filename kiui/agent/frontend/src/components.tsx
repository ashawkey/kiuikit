import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from 'react'

import type { Prompt, SessionSummary } from './types'
import type { Theme } from './theme'

export function SessionSidebar({
  sessions,
  activeId,
  onSelect,
}: {
  sessions: SessionSummary[]
  activeId: string | null
  onSelect: (id: string) => void
}) {
  return (
    <nav className="sidebar" aria-label="Agent sessions">
      <div className="sidebar-title">Agents</div>
      {sessions.length === 0 ? (
        <p className="sidebar-empty">No agents connected. Run <code>kia</code>.</p>
      ) : (
        <ul className="session-list">
          {sessions.map((session) => {
            const dir = session.cwd
              ? session.cwd.split(/[\\/]/).filter(Boolean).pop() || session.cwd
              : session.title
            return (
              <li key={session.id}>
                <button
                  type="button"
                  className={session.id === activeId ? 'session-tab active' : 'session-tab'}
                  onClick={() => onSelect(session.id)}
                  title={session.cwd}
                >
                  <span className="session-name">{dir}</span>
                </button>
              </li>
            )
          })}
        </ul>
      )}
    </nav>
  )
}

export function ThemeToggle({ theme, onToggle }: { theme: Theme; onToggle: () => void }) {
  const goingLight = theme === 'dark'
  return (
    <button
      className="theme-toggle"
      type="button"
      onClick={onToggle}
      aria-label={goingLight ? 'Switch to light theme' : 'Switch to dark theme'}
      title={goingLight ? 'Light theme' : 'Dark theme'}
    >
      {goingLight ? '☀' : '☾'}
    </button>
  )
}

// Scroll-to-top affordance in the fixed top controls. Hidden until the user
// has scrolled down a meaningful amount so it never clutters the initial view.
export function ScrollTopButton({ onClick }: { onClick: () => void }) {
  const [visible, setVisible] = useState(false)
  useEffect(() => {
    const onScroll = () => setVisible(window.scrollY > 400)
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])
  if (!visible) return null
  return (
    <button
      className="scroll-top"
      type="button"
      onClick={onClick}
      aria-label="Scroll to top"
      title="Scroll to top"
    >
      ↑
    </button>
  )
}

// Light shell-style highlight: distinguish the program name, flags, and
// quoted strings. Purely presentational — React escapes every text node.
function HighlightedCommand({ text }: { text: string }) {
  const tokens = text.match(/\s+|"[^"]*"|'[^']*'|[^\s]+/g) ?? [text]
  let namePlaced = false
  return (
    <code className="command">
      {tokens.map((token, index) => {
        if (/^\s+$/.test(token)) return <span key={index}>{token}</span>
        let kind = 'cmd-arg'
        if (/^["']/.test(token)) kind = 'cmd-str'
        else if (/^-/.test(token)) kind = 'cmd-flag'
        else if (!namePlaced) {
          kind = 'cmd-name'
          namePlaced = true
        }
        return <span key={index} className={kind}>{token}</span>
      })}
    </code>
  )
}

function CommandPreview({ detail }: { detail: string }) {
  // Summaries arrive as "<tool>: <detail>"; peel the tool label off so the
  // command/path renders on its own with highlighting.
  const separator = detail.indexOf(': ')
  const label = separator > 0 ? detail.slice(0, separator) : ''
  const body = separator > 0 ? detail.slice(separator + 2) : detail
  return (
    <div className="prompt-command">
      {label ? <span className="cmd-label">{label}</span> : null}
      <HighlightedCommand text={body} />
    </div>
  )
}

export function Thinking() {
  const [seconds, setSeconds] = useState(0)
  useEffect(() => {
    const start = Date.now()
    const id = window.setInterval(() => {
      setSeconds(Math.floor((Date.now() - start) / 1000))
    }, 1000)
    return () => window.clearInterval(id)
  }, [])
  return (
    <div className="working" aria-label="working">
      <span /><span /><span /><em>Working… {seconds}s</em>
    </div>
  )
}

export function Login({ onSuccess }: { onSuccess: () => void }) {
  const [token, setToken] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  async function submit(event: FormEvent) {
    event.preventDefault()
    if (!token || busy) return
    setBusy(true)
    setError('')
    try {
      const response = await fetch('/api/login', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ token }),
      })
      if (!response.ok) {
        setError(response.status === 429 ? 'Too many attempts. Try again shortly.' : 'That token is not valid.')
        return
      }
      setToken('')
      onSuccess()
    } catch {
      setError('Could not reach kia.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className="login-shell">
      <form className="login" onSubmit={submit}>
        <input
          type="password"
          autoComplete="current-password"
          enterKeyHint="go"
          aria-label="Access token"
          placeholder="Access token"
          value={token}
          disabled={busy}
          onChange={(event) => setToken(event.target.value)}
          required
        />
        <button className="login-submit" type="submit" disabled={busy || !token}>
          {busy ? 'Submitting…' : 'Submit'}
        </button>
      </form>
      <p className="form-error" role="alert">{error}</p>
    </main>
  )
}

export function PromptDialog({
  prompt,
  onAnswer,
}: {
  prompt: Prompt
  onAnswer: (answer: string) => void
}) {
  const [answer, setAnswer] = useState(prompt.default)
  useEffect(() => setAnswer(prompt.default), [prompt.id, prompt.default])

  // Highlight the default choice; fall back to the first when the default is
  // absent from the list so exactly one button always reads as primary.
  const primary = prompt.choices.includes(prompt.default) ? prompt.default : prompt.choices[0]

  return (
    <div className="prompt-backdrop">
      <section className="prompt" role="dialog" aria-modal="true" aria-labelledby="prompt-message">
        <p className="prompt-kicker">{prompt.kind === 'select' ? 'Action required' : 'Your input'}</p>
        {(() => {
          const [head, ...rest] = prompt.message.split('\n')
          const detail = rest.join('\n').trim()
          return (
            <>
              <div id="prompt-message" className="prompt-message">{head}</div>
              {detail ? <CommandPreview detail={detail} /> : null}
            </>
          )
        })()}
        <div className="prompt-options">
          {prompt.kind === 'select' ? prompt.choices.map((choice) => (
            <button
              type="button"
              key={choice}
              className={choice === primary ? 'primary' : undefined}
              onClick={() => onAnswer(choice)}
            >
              {choice}
            </button>
          )) : (
            <>
              <textarea autoFocus rows={3} value={answer} onChange={(event) => setAnswer(event.target.value)} />
              <button type="button" onClick={() => onAnswer(answer)}>Submit response</button>
            </>
          )}
        </div>
      </section>
    </div>
  )
}

export function Composer({
  pending,
  operationId,
  draft,
  onDraftChange,
  onSend,
  onCancel,
}: {
  pending: number
  operationId: string | null
  draft: string
  onDraftChange: (text: string) => void
  onSend: (text: string) => void
  onCancel: () => void
}) {
  const text = draft
  const setText = onDraftChange
  const field = useRef<HTMLTextAreaElement>(null)
  const shell = useRef<HTMLElement>(null)

  // Grow the single-line field to fit wrapped/multi-line input, up to the CSS
  // max-height (then it scrolls). Runs on every value change.
  useEffect(() => {
    const el = field.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${el.scrollHeight}px`
  }, [text])

  // Keep the composer above the on-screen keyboard on mobile: the visual
  // viewport shrinks when the keyboard opens, so lift the fixed bar by the
  // overlapped amount. A no-op on desktop (overlap is 0).
  useEffect(() => {
    const vv = window.visualViewport
    if (!vv) return
    const update = () => {
      const overlap = Math.max(0, window.innerHeight - vv.height - vv.offsetTop)
      shell.current?.style.setProperty('bottom', `${overlap}px`)
    }
    update()
    vv.addEventListener('resize', update)
    vv.addEventListener('scroll', update)
    return () => {
      vv.removeEventListener('resize', update)
      vv.removeEventListener('scroll', update)
    }
  }, [])

  function submit() {
    const value = text.trim()
    if (!value) return
    onSend(value)
    setText('')
    field.current?.focus()
  }

  function keyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      submit()
    }
  }

  return (
    <section className="composer-shell" ref={shell}>
      <div className="composer">
        {pending ? <div className="queue-chip">{pending} message{pending === 1 ? '' : 's'} queued</div> : null}
        <textarea
          ref={field}
          rows={1}
          maxLength={32768}
          placeholder="Type Anything..."
          value={text}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={keyDown}
        />
        {operationId ? <button className="stop-button" type="button" onClick={onCancel}><span /> Stop</button> : null}
        <button className="send-button" type="button" onClick={submit}>Send</button>
      </div>
    </section>
  )
}
