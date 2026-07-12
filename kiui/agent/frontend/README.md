# kia Web UI

The Web UI is a React + TypeScript application built with Vite. FastAPI serves
the generated `dist/` directory; Node.js is needed only for frontend
development and release builds, not by users running `kia`.

## Development

Start the Python Web UI on its normal port in one terminal:

```bash
kia --web
```

Then start Vite in this directory:

```bash
npm install
npm run dev
```

Open the Vite URL. Requests and WebSockets under `/api` are proxied to
`127.0.0.1:8765`.

## Commands

```bash
npm run typecheck
npm test
npm run build
```

`npm run build` writes production assets to `dist/`. Commit those assets when
the frontend changes because Python package builds do not run Node.js.

## Rendering model

The frontend treats server events semantically:

- assistant messages use `react-markdown` with GitHub-flavored Markdown;
- command output uses `ansi-to-react`;
- file edits use structured old/new text rendered from `diff` changes;
- prompts and operation state use dedicated React components.

Keep raw HTML disabled in Markdown and avoid inline scripts so the server's
strict Content Security Policy remains effective.

## Session model

Login exchanges the access token for an httponly session cookie plus a CSRF
token. The CSRF token is delivered again on every `state` frame (a new tab
shares the cookie but not the in-memory copy) and is required by `/api/logout`.
"Sign out" posts to `/api/logout`, drops the local socket, and returns to the
login screen.
