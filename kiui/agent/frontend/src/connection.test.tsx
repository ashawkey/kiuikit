import { act, renderHook } from '@testing-library/react'
import { vi } from 'vitest'

import { useConnectionSocket } from './connection'

class MockWebSocket {
  static instances: MockWebSocket[] = []
  static OPEN = 1
  readyState = 0
  onopen: (() => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null
  onclose: ((event: CloseEvent) => void) | null = null
  send = vi.fn()

  constructor(public url: string) {
    MockWebSocket.instances.push(this)
  }

  open() {
    this.readyState = MockWebSocket.OPEN
    this.onopen?.()
  }

  message(message: unknown) {
    this.onmessage?.({ data: JSON.stringify(message) })
  }

  close() {
    if (this.readyState === 3) return
    this.readyState = 3
    this.onclose?.({ code: 1006 } as CloseEvent)
  }
}

describe('useConnectionSocket', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    MockWebSocket.instances = []
    vi.stubGlobal('WebSocket', MockWebSocket)
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  it('becomes connected only after a valid server frame', () => {
    const onMessage = vi.fn()
    const { result } = renderHook(() => useConnectionSocket({ getUrl: () => 'ws://test', onMessage }))
    const socket = MockWebSocket.instances[0]
    act(() => socket.open())
    expect(result.current.status).toBe('connecting')
    expect(socket.send).toHaveBeenCalledWith('{"type":"ping"}')

    act(() => socket.message({ type: 'state' }))
    expect(result.current.status).toBe('connected')
    expect(onMessage).toHaveBeenCalledWith({ type: 'state' })
  })

  it('closes and reconnects after a missing heartbeat', () => {
    const { result } = renderHook(() => useConnectionSocket({ getUrl: () => 'ws://test', onMessage: () => undefined }))
    const socket = MockWebSocket.instances[0]
    act(() => {
      socket.open()
      socket.message({ type: 'pong' })
      vi.advanceTimersByTime(10_000)
      vi.advanceTimersByTime(8_000)
    })
    expect(result.current.status).toBe('reconnecting')
    act(() => vi.advanceTimersByTime(1_200))
    expect(MockWebSocket.instances).toHaveLength(2)
  })
})
