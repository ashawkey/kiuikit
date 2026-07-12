import { isPrompt } from './types'

describe('isPrompt guard', () => {
  it('accepts a well-formed prompt payload', () => {
    expect(
      isPrompt({ id: '1', kind: 'select', message: 'Allow?', choices: ['Allow'], default: 'Allow' }),
    ).toBe(true)
  })

  it('rejects malformed or partial payloads', () => {
    expect(isPrompt(null)).toBe(false)
    expect(isPrompt({ id: '1', kind: 'bogus', message: 'x', choices: [], default: '' })).toBe(false)
    expect(isPrompt({ id: '1', kind: 'text', message: 'x', choices: 'nope', default: '' })).toBe(false)
    expect(isPrompt({ kind: 'text', message: 'x', choices: [], default: '' })).toBe(false)
  })
})
