# AGENTS.md — Kiuikit (`kiui`)

## Project

Kiuikit is a flat Python package of CV/3D utilities plus the `kia` terminal/web AI agent, published as `kiui`.

- Python **>=3.10**; setuptools build (`pyproject.toml` is authoritative).
- Install: `pip install -e .`; agent: `pip install -e ".[kia]"`; all optional CV/3D dependencies: `pip install -e ".[full]"`.
- CLI entry points are declared in `pyproject.toml`; do not duplicate or guess them.
- Keep minimal installs importable. Do not import heavy optional dependencies at package/module import time unless that module requires them.

## Repository map

- `kiui/*.py`: standalone toolkit modules; `kiui/utils.py`: root-exposed utilities.
- `kiui/nn/`: PyTorch components; `kiui/gridencoder/`: PyTorch wrapper and CUDA/C++ extension.
- `kiui/agent/`: `kia`/`kib`; `kiui/agent/frontend/`: React/TypeScript/Vite UI.
- `tests/`: pytest suite, mainly agent coverage; `docs/source/`: Sphinx docs.

Inspect the relevant module rather than relying on this high-level map.

## Important contracts

### Root package

`kiui/__init__.py` uses `lazy_loader` to expose modules, `env`, and functions parsed from `kiui/utils.py`.

- New public functions in `utils.py` automatically become `kiui.<name>`; do not manually re-export them.
- Preserve lazy imports and the minimal-install path.
- Importing `kiui` loads `kiui.conf` from `./.kiui.yaml`, then `~/.kiui.yaml`.

### Agent

- `backend/`: provider-neutral loop, commands, goals, sessions; `providers/`: provider/auth implementations.
- `tools/registry.py` is the source of truth for tool schema, handler, and advertising. Built-in schemas live in `tools/schemas.py`; execution routes through `tools/executor.py`.
- Managed processes live in `tools/process_manager.py` and `tools/process_util.py`; active monitoring is defined by the bundled `monitor` skill.
- `context.py` owns compaction. `session_store.py` owns the append-only message/code-revision DAG and object storage. Rewind is implemented in `utils/rewind.py` and coordinated by `backend/sessions.py`.
- `skills.py` and `personas.py` own discovery/validation; bundled resources are under `bundled_skills/` and `bundled_personas/`, project resources under `.kia/`.
- `terminal.py` owns prompt lifecycle. A session must have one prompt task; pause/ask/restart remain one locked operation.
- `hub.py` serves the committed frontend build and multiplexes agents; `hubclient.py` connects terminal agents.
- For shared agent behavior, trace terminal/web paths, persistence/replay, and cancellation. Use `formatting.describe_tool_call` for live and replay tool labels.

### Frontend

FastAPI serves `kiui/agent/frontend/dist`; Python builds do not run Node.

- Preserve disabled raw HTML, CSP, httponly cookies, and CSRF protections.
- After source changes, run the frontend checks and commit rebuilt `dist/` assets.

## Verification

Run the smallest relevant check:

- Focused agent test: `python -m pytest tests/test_agent_<area>.py -q`
- Broad/shared agent change: `python -m pytest tests -q`
- Package/import change: `python -c "import kiui"`
- Build/metadata change: `python -m build`
- Frontend: from `kiui/agent/frontend`, run `npm ci`, `npm run typecheck`, `npm test`, `npm run build`
- Docs: `pip install -r docs/requirements.txt`, then `sphinx-build docs/source docs/build -b dirhtml`

Outside `kiui.agent`, use focused smoke tests for the changed module and optional dependency; do not install `.[full]` for unrelated checks.

Update `kiui/agent/readme.md`, `README.md`, or `docs/source/` for user-facing behavior. Never commit local `.kiui.yaml`, `.kia/`, caches, `node_modules/`, build output, or credentials; `kiui/agent/frontend/dist/` is the generated-artifact exception.
