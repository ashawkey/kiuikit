import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import App from './App'
import './styles.css'
import { applyTheme, resolveInitialTheme } from './theme'

// Resolve the theme before first paint so there is no flash of the wrong palette.
applyTheme(resolveInitialTheme())

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
