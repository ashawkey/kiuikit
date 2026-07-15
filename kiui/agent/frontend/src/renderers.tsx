import { diffLines } from 'diff'
import type { ReactNode } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

import type { DisplayEvent, EventData } from './types'
import { highlightLine, languageForPath } from './highlight'

// Matches ANSI/VT control sequences (CSI colour codes, cursor moves, OSC, etc.).
const ANSI_PATTERN =
  // eslint-disable-next-line no-control-regex
  /[][[\]()#;?]*(?:(?:(?:[a-zA-Z\d]*(?:;[a-zA-Z\d]*)*)?)|(?:(?:\d{1,4}(?:;\d{0,4})*)?[\dA-PR-TZcf-ntqry=><~]))/g

export function stripAnsi(input: string): string {
  return input.replace(ANSI_PATTERN, '')
}

const labels: Record<string, string> = {
  assistant_message: 'kia',
  user_message: 'you',
  system: 'system',
  warning: 'notice',
  error: 'error',
  output: 'output',
  diff: 'file change',
}

// Activity events (tool calls, results, streamed output, debug, reasoning)
// render compact without the uppercase label head; their meaning is carried by
// inline marks or a foldable summary.
const activityTypes = new Set(['tool_start', 'tool_result', 'output', 'debug', 'thinking'])

// Output longer than this collapses behind a <details> toggle.
const COLLAPSE_LINES = 14

export function MarkdownMessage({ children }: { children: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        a: ({ node: _node, ...props }) => (
          <a {...props} target="_blank" rel="noreferrer noopener" />
        ),
      }}
    >
      {children}
    </ReactMarkdown>
  )
}

export function AnsiOutput({ children }: { children: string }) {
  // ANSI colour rendering was unreliable, so control sequences are stripped and
  // the output is shown as clean monospace text.
  return <span className="ansi-output">{stripAnsi(children)}</span>
}

function countLines(value: string) {
  if (!value) return 0
  const lines = value.split('\n')
  return lines.at(-1) === '' ? lines.length - 1 : lines.length
}

export function DiffView({ data }: { data: EventData }) {
  const oldText = data.old_text ?? ''
  const newText = data.new_text ?? ''
  const changes = diffLines(oldText, newText)
  const language = languageForPath(data.path)
  // Full-file writes send a null line number and are numbered from line one.
  let oldLine = data.line_num ?? 1
  let newLine = data.line_num ?? 1
  const removed = countLines(oldText)
  const added = countLines(newText)

  return (
    <div className="diff-view">
      <div className="diff-title">
        <strong>{data.path ?? 'File change'}</strong>
        {data.count && data.count > 1 ? <span>{data.count} occurrences</span> : null}
      </div>
      <div className="diff-lines" role="table" aria-label={`Changes to ${data.path ?? 'file'}`}>
        {changes.flatMap((change, changeIndex) => {
          const lines = change.value.split('\n')
          if (lines.at(-1) === '') lines.pop()
          return lines.map((line, lineIndex) => {
            const kind = change.added ? 'added' : change.removed ? 'removed' : 'context'
            const currentOld = change.added ? null : oldLine++
            const currentNew = change.removed ? null : newLine++
            return (
              <div className={`diff-line ${kind}`} role="row" key={`${changeIndex}-${lineIndex}`}>
                <span className="diff-number" role="cell">{currentOld ?? ''}</span>
                <span className="diff-number" role="cell">{currentNew ?? ''}</span>
                <span className="diff-sign" aria-hidden="true">
                  {change.added ? '+' : change.removed ? '−' : ' '}
                </span>
                <code role="cell">{line ? highlightLine(line, language) : ' '}</code>
              </div>
            )
          })
        })}
      </div>
      <div className="diff-summary">
        {removed > 0 ? `${removed} removed` : null}
        {removed > 0 && added > 0 ? ' · ' : null}
        {added > 0 ? `${added} added` : null}
        {removed === 0 && added === 0 ? 'No textual changes' : null}
      </div>
    </div>
  )
}

export function Collapsible({ text, children }: { text: string; children: ReactNode }) {
  if (countLines(text) <= COLLAPSE_LINES) return <>{children}</>
  return (
    <details className="foldable">
      <summary>{countLines(text)} lines — click to expand</summary>
      <div className="foldable-body">{children}</div>
    </details>
  )
}

// Reasoning stream: always a foldable block (like tool output) with a plain
// "thinking" summary, open by default so the live stream is visible, and
// collapsible once the answer arrives.
export function ThinkingBlock({ text }: { text: string }) {
  return (
    <details className="foldable" open>
      <summary>thinking</summary>
      <div className="foldable-body"><span className="thinking-text">{text}</span></div>
    </details>
  )
}

function EventBody({ type, text, data, failed }: {
  type: string
  text: string
  data: EventData
  failed: boolean
}) {
  // Activity/output text can carry ANSI control sequences from the terminal;
  // strip them everywhere they are shown as plain text.
  const clean = stripAnsi(text)
  switch (type) {
    case 'assistant_message':
      return <MarkdownMessage>{text}</MarkdownMessage>
    case 'thinking':
      return <ThinkingBlock text={clean} />
    case 'diff':
      return <DiffView data={data} />
    case 'tool_start':
      return (
        <>
          <span className="activity-mark" aria-hidden="true">▸</span>
          <code className="tool-code">{clean}</code>
        </>
      )
    case 'tool_result':
      return (
        <>
          <span className={`activity-mark ${failed ? 'fail' : 'ok'}`} aria-hidden="true">
            {failed ? '✗' : '✓'}
          </span>
          <Collapsible text={clean}><span className="activity-text">{clean}</span></Collapsible>
        </>
      )
    case 'output':
      return <Collapsible text={clean}><AnsiOutput>{text}</AnsiOutput></Collapsible>
    case 'debug':
      return <Collapsible text={clean}><span className="debug-text">{clean}</span></Collapsible>
    default:
      return <>{clean}</>
  }
}

export function EventCard({ event }: { event: DisplayEvent }) {
  const failed = event.data.success === false
  const showHead = !activityTypes.has(event.type)
  return (
    <article className={`event ${event.type}${failed ? ' failed' : ''}`}>
      {showHead ? <div className="event-head"><i /><span>{labels[event.type] ?? event.type}</span></div> : null}
      <div className="event-body">
        <EventBody type={event.type} text={event.text} data={event.data} failed={failed} />
      </div>
    </article>
  )
}
