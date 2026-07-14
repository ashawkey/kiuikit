import { appendDelta, finalizeStream } from './streaming'
import type { DisplayEvent } from './types'

function run(events: DisplayEvent[], step: (e: DisplayEvent[]) => DisplayEvent[]) {
  return step(events)
}

describe('streaming fold', () => {
  it('keeps thinking and content as separate coexisting buffers', () => {
    let events: DisplayEvent[] = []
    // Thinking streams first, then content — the classic turn order.
    events = run(events, (e) => appendDelta(e, 'thinking_delta', 'Let me '))
    events = run(events, (e) => appendDelta(e, 'thinking_delta', 'reason.'))
    events = run(events, (e) => appendDelta(e, 'assistant_delta', 'The '))
    events = run(events, (e) => appendDelta(e, 'assistant_delta', 'answer.'))

    expect(events).toHaveLength(2)
    expect(events[0]).toMatchObject({ type: 'thinking', text: 'Let me reason.' })
    expect(events[1]).toMatchObject({ type: 'assistant_message', text: 'The answer.' })
  })

  it('finalizes the thinking buffer without duplicating it after content streams', () => {
    let events: DisplayEvent[] = []
    events = run(events, (e) => appendDelta(e, 'thinking_delta', 'reasoning'))
    events = run(events, (e) => appendDelta(e, 'assistant_delta', 'reply'))

    // Final events arrive in the same order (thinking, then assistant_message).
    events = run(events, (e) => finalizeStream(e, 'thinking', 'reasoning', 'event-10'))
    events = run(events, (e) => finalizeStream(e, 'assistant_message', 'reply', 'event-11'))

    // No new rows: both live buffers were replaced in place, thinking preserved.
    expect(events).toHaveLength(2)
    expect(events[0]).toMatchObject({ type: 'thinking', text: 'reasoning', key: 'event-10' })
    expect(events[0].data.streaming).toBeUndefined()
    expect(events[1]).toMatchObject({ type: 'assistant_message', text: 'reply', key: 'event-11' })
    expect(events[1].data.streaming).toBeUndefined()
  })

  it('does not fold a delta into an already-finalized buffer', () => {
    let events: DisplayEvent[] = [
      { key: 'event-1', type: 'thinking', text: 'old turn', data: {} },
    ]
    // A new turn's thinking delta must open a fresh buffer, not extend the
    // finalized one from a previous turn.
    events = run(events, (e) => appendDelta(e, 'thinking_delta', 'new turn'))
    expect(events).toHaveLength(2)
    expect(events[0].text).toBe('old turn')
    expect(events[1]).toMatchObject({ type: 'thinking', text: 'new turn' })
    expect(events[1].data.streaming).toBe(true)
  })
})
