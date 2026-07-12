export type Theme = 'light' | 'dark'

const STORAGE_KEY = 'kia-theme'
// Browser-chrome colors matching each palette's page background.
const chrome: Record<Theme, string> = { light: '#ffffff', dark: '#22231e' }

export function hasStoredTheme(): boolean {
  const stored = localStorage.getItem(STORAGE_KEY)
  return stored === 'light' || stored === 'dark'
}

export function resolveInitialTheme(): Theme {
  const stored = localStorage.getItem(STORAGE_KEY)
  if (stored === 'light' || stored === 'dark') return stored
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

export function applyTheme(theme: Theme): void {
  document.documentElement.setAttribute('data-theme', theme)
  const tag = document.querySelector('meta[name="theme-color"]')
  if (tag) tag.setAttribute('content', chrome[theme])
}

export function storeTheme(theme: Theme): void {
  localStorage.setItem(STORAGE_KEY, theme)
}
