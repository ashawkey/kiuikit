export type EventData = {
  text?: string
  path?: string
  old_text?: string
  new_text?: string
  line_num?: number | null
  count?: number
  success?: boolean
  streaming?: boolean
  id?: string
  context_tokens?: number
  context_limit?: number
  total_tokens_used?: number
  label?: string
  progress?: boolean
  [key: string]: unknown
}

export type PendingMessage = {
  id: string
  text: string
  source: string
  action_id?: string | null
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
  session: string
  stream_id: string
  latest_seq: number
  oldest_seq: number
  replay_truncated: boolean
  operation_id: string | null
  prompt: Prompt | null
  pending: PendingMessage | null
}

export type SessionSummary = {
  id: string
  title: string
  cwd: string
  model: string
  host: string
}

export type SessionsMessage = {
  type: 'sessions'
  csrf?: string
  sessions: SessionSummary[]
}

export type AgentEvent = {
  type: string
  seq?: number
  data?: EventData
  error?: string
  ok?: boolean
  action_id?: string
  csrf?: string
  session?: string
  sessions?: SessionSummary[]
  stream_id?: string
  latest_seq?: number
  operation_id?: string | null
  prompt?: Prompt | null
  pending?: PendingMessage | null
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
  'thinking',
])

export type DisplayEvent = {
  key: string
  type: string
  text: string
  data: EventData
}

export type ClientAction =
  | { type: 'submit'; text: string; action_id: string }
  | { type: 'withdraw_pending'; id: string; action_id: string }
  | { type: 'prompt_response'; id: string; answer: string }
  | { type: 'cancel'; operation_id: string | null }
