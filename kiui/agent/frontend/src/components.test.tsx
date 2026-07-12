import { act, fireEvent, render, screen } from '@testing-library/react'
import { vi } from 'vitest'

import { Composer, PromptDialog, Thinking, ThemeToggle } from './components'

describe('interaction components', () => {
  it('toggles the theme and labels the next choice', () => {
    const onToggle = vi.fn()
    const { rerender } = render(<ThemeToggle theme="dark" onToggle={onToggle} />)
    const button = screen.getByRole('button', { name: 'Switch to light theme' })
    fireEvent.click(button)
    expect(onToggle).toHaveBeenCalled()
    rerender(<ThemeToggle theme="light" onToggle={onToggle} />)
    expect(screen.getByRole('button', { name: 'Switch to dark theme' })).toBeInTheDocument()
  })

  it('sends composer text with Enter', () => {
    const onSend = vi.fn()
    render(<Composer pending={0} operationId={null} onSend={onSend} onCancel={() => undefined} />)
    const field = screen.getByPlaceholderText('Type anything…')
    fireEvent.change(field, { target: { value: 'hello' } })
    fireEvent.keyDown(field, { key: 'Enter' })
    expect(onSend).toHaveBeenCalledWith('hello')
  })

  it('highlights the command in a permission prompt', () => {
    const { container } = render(
      <PromptDialog
        prompt={{
          id: '3',
          kind: 'select',
          message: 'Allow this call?\nexec_command: ls -la',
          choices: ['Yes', 'No'],
          default: 'Yes',
        }}
        onAnswer={() => undefined}
      />,
    )
    expect(screen.getByText('Allow this call?')).toBeInTheDocument()
    expect(container.querySelector('.cmd-label')).toHaveTextContent('exec_command')
    expect(container.querySelector('.command .cmd-name')).toHaveTextContent('ls')
    expect(container.querySelector('.command .cmd-flag')).toHaveTextContent('-la')
  })

  it('counts elapsed seconds while working', () => {
    vi.useFakeTimers()
    try {
      render(<Thinking />)
      expect(screen.getByText('Working… 0s')).toBeInTheDocument()
      act(() => { vi.advanceTimersByTime(2000) })
      expect(screen.getByText('Working… 2s')).toBeInTheDocument()
    } finally {
      vi.useRealTimers()
    }
  })

  it('renders prompt choices as separate buttons', () => {
    const answer = vi.fn()
    render(
      <PromptDialog
        prompt={{ id: '1', kind: 'select', message: 'Allow?', choices: ['Allow', 'Deny'], default: '' }}
        onAnswer={answer}
      />,
    )
    fireEvent.click(screen.getByRole('button', { name: 'Allow' }))
    expect(answer).toHaveBeenCalledWith('Allow')
    expect(screen.getByRole('button', { name: 'Deny' }).parentElement).toHaveClass('prompt-options')
  })

  it('marks the default choice as primary regardless of position', () => {
    render(
      <PromptDialog
        prompt={{ id: '2', kind: 'select', message: 'Allow?', choices: ['Allow', 'Deny'], default: 'Deny' }}
        onAnswer={() => undefined}
      />,
    )
    expect(screen.getByRole('button', { name: 'Deny' })).toHaveClass('primary')
    expect(screen.getByRole('button', { name: 'Allow' })).not.toHaveClass('primary')
  })
})
