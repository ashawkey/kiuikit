export type EventData = {
  text?: string
  path?: string
  old_text?: string
  new_text?: string
  line_num?: number | null
  count?: number
  success?: boolean
  id?: string
  pending?: number
  [key: string]: unknown
}

export type Prompt = {
  id: string
  kind: 'select' | 'text'
  message: string
  choices: string[]
  default: string
}

export type StateMessage = {
  type: 'state'
  csrf: string
  stream_id: string
  latest_seq: number
  pending: number
  operation_id: string | null
  prompt: Prompt | null
}

export type AgentEvent = {
  type: string
  seq?: number
  data?: EventData
  error?: string
  ok?: boolean
  csrf?: string
  stream_id?: string
  latest_seq?: number
  pending?: number
  operation_id?: string | null
  prompt?: Prompt | null
}

export function isPrompt(value: unknown): value is Prompt {
  if (!value || typeof value !== 'object') return false
  const candidate = value as Record<string, unknown>
  return (
    typeof candidate.id === 'string' &&
    (candidate.kind === 'select' || candidate.kind === 'text') &&
    typeof candidate.message === 'string' &&
    Array.isArray(candidate.choices) &&
    typeof candidate.default === 'string'
  )
}

export const displayTypes = new Set([
  'assistant_message',
  'user_message',
  'system',
  'warning',
  'error',
  'tool_start',
  'tool_result',
  'output',
  'debug',
  'diff',
])

export type DisplayEvent = {
  key: string
  type: string
  text: string
  data: EventData
}

export type ClientAction =
  | { type: 'submit'; text: string }
  | { type: 'prompt_response'; id: string; answer: string }
  | { type: 'cancel'; operation_id: string | null }
