import { useCallback, useEffect, useRef, useState } from 'react'

export type ConnectionStatus = 'connecting' | 'connected' | 'reconnecting'

type SocketMessage = { type?: string; [key: string]: unknown }

type Options = {
  enabled?: boolean
  getUrl: () => string
  onMessage: (message: SocketMessage) => void
  shouldReconnect?: (event: CloseEvent) => boolean
  onTerminalClose?: (event: CloseEvent) => void
}

const CONNECT_TIMEOUT = 8_000
const HEARTBEAT_INTERVAL = 10_000
const HEARTBEAT_TIMEOUT = 8_000
const RECONNECT_DELAY = 1_200

export function useConnectionSocket({
  enabled = true,
  getUrl,
  onMessage,
  shouldReconnect = () => true,
  onTerminalClose,
}: Options) {
  const [status, setStatus] = useState<ConnectionStatus>('connecting')
  const [retryVersion, setRetryVersion] = useState(0)
  const socket = useRef<WebSocket | null>(null)
  const callbacks = useRef({ getUrl, onMessage, shouldReconnect, onTerminalClose })
  callbacks.current = { getUrl, onMessage, shouldReconnect, onTerminalClose }

  useEffect(() => {
    if (!enabled) {
      socket.current?.close()
      socket.current = null
      setStatus('connecting')
      return
    }

    let disposed = false
    let connectedOnce = false
    let reconnectTimer: number | undefined
    let connectTimer: number | undefined
    let heartbeatInterval: number | undefined
    let heartbeatTimeout: number | undefined

    const clearHeartbeat = () => {
      if (connectTimer) window.clearTimeout(connectTimer)
      if (heartbeatInterval) window.clearInterval(heartbeatInterval)
      connectTimer = undefined
      if (heartbeatTimeout) window.clearTimeout(heartbeatTimeout)
      heartbeatInterval = undefined
      heartbeatTimeout = undefined
    }

    const markHealthy = () => {
      if (heartbeatTimeout) window.clearTimeout(heartbeatTimeout)
      heartbeatTimeout = undefined
      connectedOnce = true
      setStatus('connected')
    }

    const ping = (target: WebSocket) => {
      if (disposed || socket.current !== target || target.readyState !== WebSocket.OPEN) return
      try {
        target.send(JSON.stringify({ type: 'ping' }))
      } catch {
        target.close()
        return
      }
      if (heartbeatTimeout) window.clearTimeout(heartbeatTimeout)
      heartbeatTimeout = window.setTimeout(() => {
        // Background tabs heavily throttle timers. Check promptly when the tab
        // becomes visible instead of declaring a hidden connection dead.
        if (document.visibilityState === 'hidden') return
        if (socket.current === target) {
          socket.current = null
          target.close()
          clearHeartbeat()
          setStatus('reconnecting')
          reconnectTimer = window.setTimeout(connect, RECONNECT_DELAY)
        }
      }, HEARTBEAT_TIMEOUT)
    }

    const connect = () => {
      if (disposed) return
      if (reconnectTimer) window.clearTimeout(reconnectTimer)
      reconnectTimer = undefined
      clearHeartbeat()
      setStatus(connectedOnce ? 'reconnecting' : 'connecting')

      const next = new WebSocket(callbacks.current.getUrl())
      socket.current = next
      connectTimer = window.setTimeout(() => {
        if (socket.current === next) {
          socket.current = null
          next.close()
          clearHeartbeat()
          setStatus('reconnecting')
          reconnectTimer = window.setTimeout(connect, RECONNECT_DELAY)
        }
      }, CONNECT_TIMEOUT)
      next.onopen = () => {
        if (disposed || socket.current !== next) return
        if (connectTimer) window.clearTimeout(connectTimer)
        connectTimer = undefined
        ping(next)
        heartbeatInterval = window.setInterval(() => ping(next), HEARTBEAT_INTERVAL)
      }
      next.onmessage = (event) => {
        if (disposed || socket.current !== next) return
        try {
          const message = JSON.parse(event.data) as SocketMessage
          if (!message || typeof message !== 'object') return
          markHealthy()
          if (message.type !== 'pong') callbacks.current.onMessage(message)
        } catch {
          // Ignore malformed server frames; they do not prove the connection healthy.
        }
      }
      next.onclose = (event) => {
        if (disposed || socket.current !== next) return
        clearHeartbeat()
        socket.current = null
        setStatus('reconnecting')
        if (callbacks.current.shouldReconnect(event)) {
          reconnectTimer = window.setTimeout(connect, RECONNECT_DELAY)
        } else {
          callbacks.current.onTerminalClose?.(event)
        }
      }
    }

    const checkNow = () => {
      if (disposed) return
      const current = socket.current
      if (current?.readyState === WebSocket.OPEN) {
        ping(current)
      } else if (!reconnectTimer) {
        connect()
      } else {
        window.clearTimeout(reconnectTimer)
        reconnectTimer = undefined
        connect()
      }
    }
    const onVisible = () => {
      if (document.visibilityState === 'visible') checkNow()
    }

    connect()
    window.addEventListener('online', checkNow)
    document.addEventListener('visibilitychange', onVisible)
    return () => {
      disposed = true
      if (reconnectTimer) window.clearTimeout(reconnectTimer)
      clearHeartbeat()
      window.removeEventListener('online', checkNow)
      document.removeEventListener('visibilitychange', onVisible)
      const current = socket.current
      if (current) {
        socket.current = null
        current.close()
      }
    }
  }, [enabled, retryVersion])

  const send = useCallback((message: unknown): boolean => {
    const current = socket.current
    if (status !== 'connected' || current?.readyState !== WebSocket.OPEN) return false
    try {
      current.send(JSON.stringify(message))
      return true
    } catch {
      setStatus('reconnecting')
      current.close()
      return false
    }
  }, [status])

  const retry = useCallback(() => setRetryVersion((value) => value + 1), [])
  const close = useCallback(() => {
    const current = socket.current
    socket.current = null
    current?.close()
  }, [])

  return { status, send, retry, close }
}
