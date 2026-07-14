import { render, screen } from '@testing-library/react'

import { AnsiOutput, DiffView, EventCard, MarkdownMessage } from './renderers'
import type { DisplayEvent } from './types'

function makeEvent(type: string, text: string, data: DisplayEvent['data'] = {}): DisplayEvent {
  return { key: `k-${type}`, type, text, data }
}

describe('content renderers', () => {
  it('renders CommonMark and GFM features', () => {
    const { container } = render(
      <MarkdownMessage>{'before\n\n---\n\n- [x] done\n\n| A | B |\n| - | - |\n| 1 | 2 |'}</MarkdownMessage>,
    )
    expect(container.querySelector('hr')).toBeInTheDocument()
    expect(container.querySelector('input[type="checkbox"]')).toBeChecked()
    expect(container.querySelector('table')).toBeInTheDocument()
  })

  it('strips ANSI control sequences to clean text', () => {
    const { container } = render(
      <AnsiOutput>{'plain [31mred[0m [1;32mbold[0m'}</AnsiOutput>,
    )
    expect(container.textContent).toBe('plain red bold')
    expect(container.querySelector('span[style]')).toBeNull()
  })

  it('renders structured additions and removals', () => {
    render(<DiffView data={{ path: 'demo.py', old_text: 'old\n', new_text: 'new\n', line_num: 4 }} />)
    expect(screen.getByText('demo.py')).toBeInTheDocument()
    expect(screen.getByText('old').closest('.diff-line')).toHaveClass('removed')
    expect(screen.getByText('new').closest('.diff-line')).toHaveClass('added')
  })

  it('describes a new-file diff without an empty removal count', () => {
    render(<DiffView data={{ path: 'new.py', old_text: '', new_text: 'one\ntwo\n', line_num: null }} />)
    expect(screen.getByText('2 added')).toBeInTheDocument()
    expect(screen.queryByText(/removed/)).toBeNull()
  })

  it('renders a tool call as a compact code line without a label head', () => {
    const { container } = render(<EventCard event={makeEvent('tool_start', 'exec_command(ls)')} />)
    expect(container.querySelector('.event-head')).toBeNull()
    expect(container.querySelector('.tool-code')).toHaveTextContent('exec_command(ls)')
  })

  it('renders a thinking event as a foldable block without a label head', () => {
    const { container } = render(<EventCard event={makeEvent('thinking', 'Let me reason about this.')} />)
    expect(container.querySelector('.event-head')).toBeNull()
    const details = container.querySelector('details.foldable')
    expect(details).not.toBeNull()
    expect(details).toHaveAttribute('open')
    expect(details?.querySelector('summary')).toHaveTextContent('thinking')
    expect(container.querySelector('.thinking-text')).toHaveTextContent('Let me reason about this.')
  })

  it('collapses long output behind a details toggle', () => {
    const short = makeEvent('output', 'line one\nline two')
    const { container, rerender } = render(<EventCard event={short} />)
    expect(container.querySelector('details.foldable')).toBeNull()

    const long = makeEvent('output', Array.from({ length: 30 }, (_, i) => `line ${i}`).join('\n'))
    rerender(<EventCard event={long} />)
    const details = container.querySelector('details.foldable')
    expect(details).not.toBeNull()
    expect(details?.querySelector('summary')).toHaveTextContent('30 lines — click to expand')
  })
})
