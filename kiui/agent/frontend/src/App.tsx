import { useCallback, useEffect, useRef, useState } from 'react'

import { Composer, Login, PromptDialog, Thinking, ThemeToggle } from './components'
import { EventCard } from './renderers'
import { applyTheme, hasStoredTheme, resolveInitialTheme, storeTheme } from './theme'
import type { Theme } from './theme'
import type { AgentEvent, ClientAction, DisplayEvent, Prompt, StateMessage } from './types'
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

export default function App() {
  const [authenticated, setAuthenticated] = useState<boolean | null>(null)
  const [connectVersion, setConnectVersion] = useState(0)
  const [events, setEvents] = useState<DisplayEvent[]>([])
  const [pending, setPending] = useState(0)
  const [operationId, setOperationId] = useState<string | null>(null)
  const [prompt, setPrompt] = useState<Prompt | null>(null)
  const [thinking, setThinking] = useState(false)
  const [theme, setTheme] = useState<Theme>(resolveInitialTheme)
  const socket = useRef<WebSocket | null>(null)
  const csrf = useRef('')
  // Start every fresh page load from seq 0 so the server replays its retained
  // history and a refresh rebuilds the whole timeline. The ref advances as
  // events arrive, so in-session reconnects still resume without duplicates.
  const lastSeq = useRef(0)
  const localKey = useRef(0)
  // Whether the view is pinned to the bottom; only then do we auto-scroll, so a
  // user reading scrollback is not yanked down by new activity.
  const pinned = useRef(true)

  const scrollToBottom = useCallback(() => {
    // Defer to the next frame so freshly committed content is measured first.
    requestAnimationFrame(() => {
      window.scrollTo({ top: document.documentElement.scrollHeight })
    })
  }, [])

  const showEvent = useCallback((type: string, text: string, data = {}, seq?: number) => {
    const key = seq ? `event-${seq}` : `local-${++localKey.current}`
    setEvents((current) => appendEvent(current, { key, type, text, data }))
  }, [])

  const handleMessage = useCallback((message: AgentEvent) => {
    if (message.type === 'state') {
      const state = message as StateMessage
      csrf.current = state.csrf
      // The server sends `state` only for an authenticated session, so this
      // frame — not the raw socket open — is what confirms we are signed in.
      setAuthenticated(true)
      const oldStream = sessionStorage.getItem('kia-stream')
      if (oldStream && oldStream !== state.stream_id) {
        lastSeq.current = 0
        setEvents([])
      }
      sessionStorage.setItem('kia-stream', state.stream_id)
      setPending(state.pending)
      setOperationId(state.operation_id)
      setPrompt(state.prompt)
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

  useEffect(() => {
    let disposed = false
    let reconnectTimer: number | undefined

    function connect() {
      if (disposed) return
      const protocol = location.protocol === 'https:' ? 'wss' : 'ws'
      const stream = encodeURIComponent(sessionStorage.getItem('kia-stream') || '')
      const next = new WebSocket(`${protocol}://${location.host}/api/ws?after=${lastSeq.current}&stream=${stream}`)
      socket.current = next

      next.onmessage = (event) => {
        if (disposed || socket.current !== next) return
        try {
          handleMessage(JSON.parse(event.data) as AgentEvent)
        } catch {
          // Ignore malformed server frames and keep the live connection.
        }
      }
      next.onclose = (event) => {
        if (disposed || socket.current !== next) return
        if (event.code === 4403) setAuthenticated(false)
        else reconnectTimer = window.setTimeout(connect, 1200)
      }
    }

    connect()
    return () => {
      disposed = true
      if (reconnectTimer) window.clearTimeout(reconnectTimer)
      socket.current?.close()
    }
  }, [connectVersion, handleMessage])

  useEffect(() => {
    const onScroll = () => {
      const distance =
        document.documentElement.scrollHeight - window.innerHeight - window.scrollY
      // Reserve the composer's fixed footer height in the threshold.
      pinned.current = distance < 220
    }
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  // Follow the tail as content grows (new events, streaming tokens, thinking dots).
  useEffect(() => {
    if (pinned.current) scrollToBottom()
  }, [events, thinking, scrollToBottom])

  useEffect(() => { applyTheme(theme) }, [theme])

  useEffect(() => {
    // Follow the OS preference until the user makes an explicit choice.
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

  const send = useCallback((action: ClientAction) => {
    if (socket.current?.readyState === WebSocket.OPEN) {
      socket.current.send(JSON.stringify(action))
    }
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
    sessionStorage.removeItem('kia-stream')
    lastSeq.current = 0
    // Null the ref before closing so the socket's onclose handler bails out
    // (its guard checks socket.current === next) and no reconnect is scheduled.
    const active = socket.current
    socket.current = null
    active?.close()
    setEvents([])
    setPrompt(null)
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
      <section className="workspace">
        <div className="timeline" aria-live="polite">
          {events.map((event) => <EventCard event={event} key={event.key} />)}
        </div>
        {thinking ? <Thinking /> : null}
      </section>

      {prompt ? <PromptDialog prompt={prompt} onAnswer={(answer) => send({ type: 'prompt_response', id: prompt.id, answer })} /> : null}
      <Composer
        pending={pending}
        operationId={operationId}
        onSend={(text) => {
          send({ type: 'submit', text })
          // Sending is an explicit intent to follow the conversation tail.
          pinned.current = true
          scrollToBottom()
        }}
        onCancel={() => send({ type: 'cancel', operation_id: operationId })}
      />
    </main>
  )
}
