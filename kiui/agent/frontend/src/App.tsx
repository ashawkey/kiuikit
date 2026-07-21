import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'

import { Composer, Login, PromptDialog, ScrollTopButton, SessionSidebar, Thinking, ThemeToggle } from './components'
import { EventCard } from './renderers'
import { applyTheme, hasStoredTheme, resolveInitialTheme, storeTheme } from './theme'
import type { Theme } from './theme'
import type { AgentEvent, ClientAction, DisplayEvent, PendingMessage, Prompt, SessionSummary, StateMessage } from './types'
import { displayTypes, isPrompt } from './types'
import { appendDelta, finalizeStream } from './streaming'

function appendEvent(events: DisplayEvent[], event: DisplayEvent) {
  const previous = events.at(-1)
  if (previous && previous.type === event.type && ['output', 'system'].includes(event.type)) {
    const separator = previous.text && event.text ? '\n' : ''
    return [
      ...events.slice(0, -1),
      { ...previous, text: `${previous.text}${separator}${event.text}`, data: { ...previous.data, ...event.data } },
    ]
  }
  return [...events, event]
}

/**
 * One agent session. Every session gets a pane that stays mounted with its own
 * always-open websocket, so switching tabs only toggles visibility — no
 * reconnect and no history replay. Composer/prompt render only for the active
 * pane, so exactly one of each exists in the DOM.
 */
function SessionPane({
  sessionId,
  active,
  draft,
  restoreScroll,
  onDraftChange,
}: {
  sessionId: string
  active: boolean
  draft: string
  restoreScroll: number | undefined
  onDraftChange: (text: string) => void
}) {
  const [events, setEvents] = useState<DisplayEvent[]>([])
  const [operationId, setOperationId] = useState<string | null>(null)
  const [prompt, setPrompt] = useState<Prompt | null>(null)
  const [pending, setPending] = useState<PendingMessage | null>(null)
  const [submitAction, setSubmitAction] = useState<string | null>(null)
  const [withdrawAction, setWithdrawAction] = useState<string | null>(null)
  const submitActionRef = useRef<string | null>(null)
  const withdrawActionRef = useRef<string | null>(null)
  const [thinking, setThinking] = useState(false)
  const [thinkingStatus, setThinkingStatus] = useState({
    suffix: '',
    contextTokens: 0,
    contextLimit: 0,
    totalTokensUsed: 0,
    label: 'Working',
    progress: false,
    startedAt: undefined as number | undefined,
  })
  const socket = useRef<WebSocket | null>(null)
  const lastSeq = useRef(0)
  const streamKey = useRef('')
  const localKey = useRef(0)
  const pinned = useRef(true)
  const activeRef = useRef(active)
  activeRef.current = active
  // Window scrolling is asynchronous. Guard every scheduled scroll with this
  // pane's current visibility so a callback from a pane that was just hidden
  // cannot move the newly selected session.
  const scrollToTail = useCallback(() => {
    requestAnimationFrame(() => {
      if (activeRef.current) {
        window.scrollTo({ top: document.documentElement.scrollHeight })
      }
    })
  }, [])

  const showEvent = useCallback((type: string, text: string, data = {}, seq?: number) => {
    const key = seq ? `event-${seq}` : `local-${++localKey.current}`
    setEvents((current) => appendEvent(current, { key, type, text, data }))
  }, [])

  const handleMessage = useCallback((message: AgentEvent) => {
    if (message.type === 'state') {
      const state = message as StateMessage
      const key = `${state.session}:${state.stream_id}`
      if (streamKey.current && streamKey.current !== key) {
        lastSeq.current = 0
        setEvents([])
      }
      streamKey.current = key
      setOperationId(state.operation_id)
      setPrompt(state.prompt)
      setPending(state.pending)
      const submitId = submitActionRef.current
      if (submitId !== null && state.pending?.action_id === submitId) {
        onDraftChange('')
        submitActionRef.current = null
      }
      setSubmitAction(null)
      setWithdrawAction(null)
      if (state.replay_truncated) {
        showEvent(
          'warning',
          `Earlier events are no longer available; replay starts at event ${state.oldest_seq}.`,
        )
      }
      return
    }
    if (message.type === 'rejected') {
      const actionId = typeof message.action_id === 'string' ? message.action_id : ''
      if (actionId && actionId === submitActionRef.current) {
        submitActionRef.current = null
        setSubmitAction(null)
      } else if (actionId && actionId === withdrawActionRef.current) {
        withdrawActionRef.current = null
        setWithdrawAction(null)
        showEvent('error', message.error || 'Pending message is unavailable.')
      } else {
        showEvent('error', message.error || 'Message rejected')
      }
      return
    }
    if (message.type === 'cancel_ack') {
      if (!message.ok) setOperationId(null)
      return
    }
    if (message.type === 'accepted' || message.type === 'prompt_ack') return

    if (message.seq) {
      if (message.seq <= lastSeq.current) return
      lastSeq.current = message.seq
    }

    const data = message.data ?? {}
    switch (message.type) {
      case 'pending_set':
        setPending({
          id: typeof data.id === 'string' ? data.id : '',
          text: typeof data.text === 'string' ? data.text : '',
          source: typeof data.source === 'string' ? data.source : '',
          action_id: typeof data.action_id === 'string' ? data.action_id : null,
        })
        if (typeof data.action_id === 'string' && data.action_id === submitActionRef.current) {
          onDraftChange('')
          submitActionRef.current = null
          setSubmitAction(null)
        }
        break
      case 'pending_cleared':
        setPending(null)
        if (
          data.reason === 'withdrawn' &&
          typeof data.action_id === 'string' &&
          data.action_id === withdrawActionRef.current
        ) {
          onDraftChange(typeof data.text === 'string' ? data.text : '')
          withdrawActionRef.current = null
          setWithdrawAction(null)
        }
        break
      case 'submission_rejected':
        if (typeof data.action_id === 'string' && data.action_id === submitActionRef.current) {
          submitActionRef.current = null
          setSubmitAction(null)
        }
        break
      case 'withdrawal_rejected':
        if (typeof data.action_id === 'string' && data.action_id === withdrawActionRef.current) {
          withdrawActionRef.current = null
          setWithdrawAction(null)
          showEvent('error', typeof data.error === 'string' ? data.error : 'Pending message is unavailable.')
        }
        break
      case 'prompt_open':
        if (isPrompt(data)) setPrompt(data)
        break
      case 'prompt_resolved':
        setPrompt(null)
        break
      case 'operation_start':
        setOperationId(typeof data.id === 'string' ? data.id : null)
        break
      case 'operation_end':
        setOperationId(null)
        break
      case 'thinking_start':
        setThinking(true)
        setThinkingStatus({
          suffix: typeof data.suffix === 'string' ? data.suffix : '',
          contextTokens: typeof data.context_tokens === 'number' ? data.context_tokens : 0,
          contextLimit: typeof data.context_limit === 'number' ? data.context_limit : 0,
          totalTokensUsed: typeof data.total_tokens_used === 'number' ? data.total_tokens_used : 0,
          label: typeof data.label === 'string' ? data.label : 'Working',
          startedAt: typeof data.started_at === 'number' ? data.started_at * 1000 : Date.now(),
          progress: data.progress === true,
        })
        break
      case 'thinking_stop':
        setThinking(false)
        break
      case 'timeline_reset':
        setThinking(false)
        setEvents([])
        break
      case 'assistant_delta':
      case 'thinking_delta':
        setThinking(false)
        setEvents((current) => appendDelta(current, message.type, typeof data.text === 'string' ? data.text : ''))
        break
      case 'assistant_message':
      case 'thinking':
        setEvents((current) => finalizeStream(
          current,
          message.type,
          typeof data.text === 'string' ? data.text : '',
          message.seq ? `event-${message.seq}` : `local-${++localKey.current}`,
        ))
        break
      default:
        if (displayTypes.has(message.type)) {
          showEvent(message.type, typeof data.text === 'string' ? data.text : '', data, message.seq)
        }
    }
  }, [onDraftChange, showEvent])

  // Open the session socket once and keep it open for the pane's whole life.
  useEffect(() => {
    let disposed = false
    let reconnectTimer: number | undefined

    function connect() {
      if (disposed) return
      const protocol = location.protocol === 'https:' ? 'wss' : 'ws'
      const id = encodeURIComponent(sessionId)
      const next = new WebSocket(
        `${protocol}://${location.host}/api/ws?session=${id}&after=${lastSeq.current}`,
      )
      socket.current = next
      next.onmessage = (event) => {
        if (disposed || socket.current !== next) return
        try {
          handleMessage(JSON.parse(event.data) as AgentEvent)
        } catch {
          // Ignore malformed server frames.
        }
      }
      next.onclose = (event) => {
        if (disposed || socket.current !== next) return
        setSubmitAction(null)
        setWithdrawAction(null)
        // 4404: the agent exited; the control channel will prune this pane.
        if (event.code !== 4403 && event.code !== 4404) {
          reconnectTimer = window.setTimeout(connect, 1200)
        }
      }
    }

    connect()
    return () => {
      disposed = true
      if (reconnectTimer) window.clearTimeout(reconnectTimer)
      socket.current?.close()
      socket.current = null
    }
  }, [sessionId, handleMessage])

  // Track whether new output should keep following the tail. App owns the
  // actual per-session offsets because it can capture before switching panes.
  useEffect(() => {
    if (!active) return
    const onScroll = () => {
      if (!activeRef.current) return
      const distance =
        document.documentElement.scrollHeight - window.innerHeight - window.scrollY
      pinned.current = distance < 220
    }
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [active])

  // App captures the outgoing position before changing tabs, while its pane is
  // still visible and the browser has not clamped the window scroll offset.
  useLayoutEffect(() => {
    if (!active) return
    if (restoreScroll === undefined) {
      scrollToTail()
    } else {
      window.scrollTo({ top: restoreScroll })
    }
  }, [active, restoreScroll, scrollToTail])

  // Follow new output only. Including `active` here would schedule a second
  // tail scroll on tab activation and overwrite the restoration above.
  useEffect(() => {
    if (activeRef.current && pinned.current) scrollToTail()
  }, [events, thinking, scrollToTail])

  const send = useCallback((action: ClientAction): boolean => {
    const current = socket.current
    if (current?.readyState !== WebSocket.OPEN) return false
    try {
      current.send(JSON.stringify(action))
      return true
    } catch {
      return false
    }
  }, [])

  return (
    <>
      <section className="workspace" style={active ? undefined : { display: 'none' }}>
        <div className="timeline" aria-live="polite">
          {events.map((event) => <EventCard event={event} key={event.key} />)}
        </div>
        {active && thinking ? <Thinking {...thinkingStatus} /> : null}
      </section>
      {active && prompt ? (
        <PromptDialog prompt={prompt} onAnswer={(answer) => send({ type: 'prompt_response', id: prompt.id, answer })} />
      ) : null}
      {active ? (
        <Composer
          operationId={operationId}
          pending={pending}
          busy={submitAction !== null || withdrawAction !== null}
          draft={draft}
          onDraftChange={onDraftChange}
          onSend={(text) => {
            if (pending || submitAction) return
            const actionId = crypto.randomUUID()
            if (!send({ type: 'submit', text, action_id: actionId })) return
            submitActionRef.current = actionId
            setSubmitAction(actionId)
            pinned.current = true
            scrollToTail()
          }}
          onWithdraw={() => {
            if (!pending || draft || withdrawAction) return
            const actionId = crypto.randomUUID()
            if (!send({ type: 'withdraw_pending', id: pending.id, action_id: actionId })) return
            withdrawActionRef.current = actionId
            setWithdrawAction(actionId)
          }}
          onCancel={() => send({ type: 'cancel', operation_id: operationId })}
        />
      ) : null}
    </>
  )
}

export default function App() {
  const [authenticated, setAuthenticated] = useState<boolean | null>(null)
  const [connectVersion, setConnectVersion] = useState(0)
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [activeSession, setActiveSession] = useState<string | null>(null)
  const [theme, setTheme] = useState<Theme>(resolveInitialTheme)
  // Per-session composer drafts. Panes unmount their Composer when inactive, so
  // the in-progress text lives here to survive tab switches.
  const [drafts, setDrafts] = useState<Record<string, string>>({})
  const controlSocket = useRef<WebSocket | null>(null)
  const csrf = useRef('')
  const scrollPositions = useRef(new Map<string, number>())

  // Control channel: session list + auth confirmation.
  useEffect(() => {
    let disposed = false
    let reconnectTimer: number | undefined

    function connect() {
      if (disposed) return
      const protocol = location.protocol === 'https:' ? 'wss' : 'ws'
      const next = new WebSocket(`${protocol}://${location.host}/api/ws`)
      controlSocket.current = next

      next.onmessage = (event) => {
        if (disposed || controlSocket.current !== next) return
        try {
          const message = JSON.parse(event.data) as AgentEvent
          if (message.type !== 'sessions') return
          if (message.csrf) csrf.current = message.csrf
          setAuthenticated(true)
          const list = message.sessions ?? []
          setSessions(list)
          setActiveSession((current) => {
            if (current && list.some((s) => s.id === current)) return current
            return list[0]?.id ?? null
          })
        } catch {
          // Ignore malformed frames.
        }
      }
      next.onclose = (event) => {
        if (disposed || controlSocket.current !== next) return
        if (event.code === 4403) setAuthenticated(false)
        else reconnectTimer = window.setTimeout(connect, 1200)
      }
    }

    connect()
    return () => {
      disposed = true
      if (reconnectTimer) window.clearTimeout(reconnectTimer)
      controlSocket.current?.close()
      controlSocket.current = null
    }
  }, [connectVersion])

  useEffect(() => { applyTheme(theme) }, [theme])

  useEffect(() => {
    const query = window.matchMedia('(prefers-color-scheme: dark)')
    const onChange = () => {
      if (!hasStoredTheme()) setTheme(query.matches ? 'dark' : 'light')
    }
    query.addEventListener('change', onChange)
    return () => query.removeEventListener('change', onChange)
  }, [])

  const toggleTheme = useCallback(() => {
    setTheme((current) => {
      const next = current === 'dark' ? 'light' : 'dark'
      storeTheme(next)
      return next
    })
  }, [])

  const setDraft = useCallback((sessionId: string, text: string) => {
    setDrafts((current) => ({ ...current, [sessionId]: text }))
  }, [])

  const selectSession = useCallback((sessionId: string) => {
    if (activeSession) scrollPositions.current.set(activeSession, window.scrollY)
    setActiveSession(sessionId)
  }, [activeSession])

  const scrollToTop = useCallback(() => {
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }, [])

  const logout = useCallback(async () => {
    try {
      await fetch('/api/logout', {
        method: 'POST',
        headers: { 'x-csrf-token': csrf.current },
      })
    } catch {
      // Ignore network failures: the local state reset below still signs out.
    }
    const control = controlSocket.current
    controlSocket.current = null
    control?.close()
    // Clearing sessions unmounts every pane, closing their sockets.
    setSessions([])
    setActiveSession(null)
    setDrafts({})
    scrollPositions.current.clear()
    setAuthenticated(false)
  }, [])

  if (authenticated === false) {
    return <Login onSuccess={() => setConnectVersion((value) => value + 1)} />
  }
  if (authenticated === null) return <main className="loading" aria-label="Connecting" />

  return (
    <main className="app-shell">
      <div className="top-controls">
        <ScrollTopButton onClick={scrollToTop} />
        <ThemeToggle theme={theme} onToggle={toggleTheme} />
        <button className="logout" type="button" onClick={logout} aria-label="Sign out" title="Sign out">⏻</button>
      </div>
      <div className="layout">
        <SessionSidebar sessions={sessions} activeId={activeSession} onSelect={selectSession} />
        <div className="panes">
          {sessions.length === 0 ? (
            <section className="workspace">
              <p className="sidebar-empty">Waiting for an agent to connect…</p>
            </section>
          ) : null}
          {sessions.map((session) => (
            <SessionPane
              key={session.id}
              sessionId={session.id}
              active={session.id === activeSession}
              draft={drafts[session.id] ?? ''}
              restoreScroll={scrollPositions.current.get(session.id)}
              onDraftChange={(text) => setDraft(session.id, text)}
            />
          ))}
        </div>
      </div>
    </main>
  )
}
