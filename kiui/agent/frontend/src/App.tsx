import { useCallback, useEffect, useRef, useState } from 'react'

import { Composer, Login, PromptDialog, SessionSidebar, Thinking, ThemeToggle } from './components'
import { EventCard } from './renderers'
import { applyTheme, hasStoredTheme, resolveInitialTheme, storeTheme } from './theme'
import type { Theme } from './theme'
import type { AgentEvent, ClientAction, DisplayEvent, Prompt, SessionSummary, StateMessage } from './types'
import { displayTypes, isPrompt } from './types'

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

function scrollToBottom() {
  requestAnimationFrame(() => {
    window.scrollTo({ top: document.documentElement.scrollHeight })
  })
}

/**
 * One agent session. Every session gets a pane that stays mounted with its own
 * always-open websocket, so switching tabs only toggles visibility — no
 * reconnect and no history replay. Composer/prompt render only for the active
 * pane, so exactly one of each exists in the DOM.
 */
function SessionPane({ sessionId, active }: { sessionId: string; active: boolean }) {
  const [events, setEvents] = useState<DisplayEvent[]>([])
  const [pending, setPending] = useState(0)
  const [operationId, setOperationId] = useState<string | null>(null)
  const [prompt, setPrompt] = useState<Prompt | null>(null)
  const [thinking, setThinking] = useState(false)
  const socket = useRef<WebSocket | null>(null)
  const lastSeq = useRef(0)
  const streamKey = useRef('')
  const localKey = useRef(0)
  const pinned = useRef(true)

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
      setPending(state.pending)
      setOperationId(state.operation_id)
      setPrompt(state.prompt)
      if (state.replay_truncated) {
        showEvent(
          'warning',
          `Earlier events are no longer available; replay starts at event ${state.oldest_seq}.`,
        )
      }
      return
    }
    if (message.type === 'rejected') {
      showEvent('error', message.error || 'Message rejected')
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
      case 'queue_changed':
        setPending(typeof data.pending === 'number' ? data.pending : 0)
        break
      case 'thinking_start':
        setThinking(true)
        break
      case 'thinking_stop':
        setThinking(false)
        break
      default:
        if (displayTypes.has(message.type)) {
          showEvent(message.type, typeof data.text === 'string' ? data.text : '', data, message.seq)
        }
    }
  }, [showEvent])

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

  // Only the visible pane tracks scroll position and follows the tail, so a
  // background session receiving events never yanks the active view.
  useEffect(() => {
    if (!active) return
    const onScroll = () => {
      const distance =
        document.documentElement.scrollHeight - window.innerHeight - window.scrollY
      pinned.current = distance < 220
    }
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [active])

  useEffect(() => {
    if (active) {
      pinned.current = true
      scrollToBottom()
    }
  }, [active])

  useEffect(() => {
    if (active && pinned.current) scrollToBottom()
  }, [events, thinking, active])

  const send = useCallback((action: ClientAction) => {
    if (socket.current?.readyState === WebSocket.OPEN) {
      socket.current.send(JSON.stringify(action))
    }
  }, [])

  return (
    <>
      <section className="workspace" style={active ? undefined : { display: 'none' }}>
        <div className="timeline" aria-live="polite">
          {events.map((event) => <EventCard event={event} key={event.key} />)}
        </div>
        {active && thinking ? <Thinking /> : null}
      </section>
      {active && prompt ? (
        <PromptDialog prompt={prompt} onAnswer={(answer) => send({ type: 'prompt_response', id: prompt.id, answer })} />
      ) : null}
      {active ? (
        <Composer
          pending={pending}
          operationId={operationId}
          onSend={(text) => {
            send({ type: 'submit', text })
            pinned.current = true
            scrollToBottom()
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
  const controlSocket = useRef<WebSocket | null>(null)
  const csrf = useRef('')

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
    setAuthenticated(false)
  }, [])

  if (authenticated === false) {
    return <Login onSuccess={() => setConnectVersion((value) => value + 1)} />
  }
  if (authenticated === null) return <main className="loading" aria-label="Connecting" />

  return (
    <main className="app-shell">
      <div className="top-controls">
        <ThemeToggle theme={theme} onToggle={toggleTheme} />
        <button className="logout" type="button" onClick={logout}>Sign out</button>
      </div>
      <div className="layout">
        <SessionSidebar sessions={sessions} activeId={activeSession} onSelect={setActiveSession} />
        <div className="panes">
          {sessions.length === 0 ? (
            <section className="workspace">
              <p className="sidebar-empty">Waiting for an agent to connect…</p>
            </section>
          ) : null}
          {sessions.map((session) => (
            <SessionPane key={session.id} sessionId={session.id} active={session.id === activeSession} />
          ))}
        </div>
      </div>
    </main>
  )
}
