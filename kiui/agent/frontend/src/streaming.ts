import type { DisplayEvent } from './types'

// Streaming deltas (assistant_delta / thinking_delta) fold into a single live
// event of the matching finalized type so tokens appear progressively. The
// authoritative final event (assistant_message / thinking) replaces that live
// buffer, keeping reconnect replays idempotent.
//
// Within one turn the thinking buffer and the content buffer coexist (thinking
// streams first, then content streams below it), so the matching open buffer is
// found by type anywhere in the tail — not just at the very end.
export const streamFinal: Record<string, string> = {
  assistant_delta: 'assistant_message',
  thinking_delta: 'thinking',
}

// Find an open (still-streaming) event of *finalType*, scanning from the end.
// Only the current turn's buffers are open, so this stays O(1) in practice.
function findOpenStream(events: DisplayEvent[], finalType: string): number {
  for (let i = events.length - 1; i >= 0; i--) {
    if (events[i].data.streaming && events[i].type === finalType) return i
    if (!events[i].data.streaming) break
  }
  return -1
}

export function appendDelta(events: DisplayEvent[], type: string, text: string) {
  const finalType = streamFinal[type]
  const index = findOpenStream(events, finalType)
  if (index >= 0) {
    const target = events[index]
    const next = events.slice()
    next[index] = { ...target, text: target.text + text }
    return next
  }
  return [
    ...events,
    { key: `stream-${finalType}-${events.length}`, type: finalType, text, data: { streaming: true } },
  ]
}

export function finalizeStream(
  events: DisplayEvent[],
  finalType: string,
  text: string,
  key: string,
) {
  const index = findOpenStream(events, finalType)
  if (index >= 0) {
    const next = events.slice()
    next[index] = { key, type: finalType, text, data: {} }
    return next
  }
  return [...events, { key, type: finalType, text, data: {} }]
}
