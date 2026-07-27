# AGENTS.md — Kiuikit (`kiui`)

## Project

Kiuikit is a flat Python package for computer-vision and 3D utilities, plus the `kia` terminal/web AI agent. It is published as `kiui`.

- Python: **>=3.10** (source of truth: `pyproject.toml`)
- Build: setuptools
- Minimal install: `pip install -e .`
- Agent install: `pip install -e ".[kia]"`
- All optional CV/3D dependencies: `pip install -e ".[full]"`
- CLI entry points are declared in `pyproject.toml`; do not duplicate or guess them.

Many CV/3D dependencies are intentionally optional. Keep the minimal package importable: avoid importing heavy optional dependencies at package or module import time unless that module requires them.

## Repository map

- `kiui/*.py`: standalone toolkit modules (mesh, camera, image/video, rendering, geometry, Slurm, etc.).
- `kiui/utils.py`: general utilities exposed at the package root.
- `kiui/nn/`: standalone PyTorch components.
- `kiui/gridencoder/`: PyTorch wrapper and CUDA/C++ extension sources.
- `kiui/agent/`: `kia` and `kib` implementation.
- `kiui/agent/frontend/`: React/TypeScript/Vite Web UI.
- `tests/`: pytest suite, currently focused on `kiui.agent`.
- `docs/source/`: Sphinx documentation.

Prefer inspecting the relevant module over relying on a static exhaustive file list.

## Important contracts

### Root package

`kiui/__init__.py` uses `lazy_loader` to expose top-level modules, `env`, and top-level functions parsed from `kiui/utils.py`.

- A new top-level public function in `utils.py` automatically becomes `kiui.<name>`; no manual re-export is needed.
- Preserve lazy imports and the minimal-install import path.
- `kiui.conf` is loaded from `./.kiui.yaml`, then `~/.kiui.yaml`, at import time.

### Agent (`kiui/agent`)

- `backend/`: provider-neutral API loop, commands, goals, and session coordination.
- `providers/`: provider implementations and authentication.
- `tools/registry.py`: single source of truth for tool schema, handler, permissions, and advertising. Built-in schemas alone live in `tools/schemas.py`; execution routes through `tools/executor.py`.
- Managed-process internals live in `tools/process_manager.py` and `tools/process_util.py`; user-facing process tools come from the bundled `monitor` skill.
- `context.py` owns conversation/token compaction. `session_store.py` owns the append-only message/code revision DAG and object storage. Rewind planning/application lives in `utils/rewind.py` and is coordinated by `backend/sessions.py`.
- `skills.py` and `personas.py` implement discovery/validation. Bundled resources are under `bundled_skills/` and `bundled_personas/`; project resources are under `.kia/`.
- `terminal.py` owns prompt lifecycle. Preserve the invariant that a session has one prompt task; pause/ask/restart must remain one locked operation.
- `hub.py` serves the committed frontend build and multiplexes agents; `hubclient.py` connects terminal agents to it.

When changing shared agent behavior, trace terminal and web paths, session persistence/replay, cancellation, and permission handling. Keep `formatting.describe_tool_call` as the common tool-call label for live execution and replay.

### Frontend

Source is in `kiui/agent/frontend/src`; FastAPI serves `frontend/dist` in installed packages. Python package builds do not run Node.

- Keep raw HTML disabled in Markdown and preserve CSP, httponly-cookie, and CSRF protections.
- After frontend source changes, run the checks below and commit the rebuilt `dist/` assets.

## Verification

Run the smallest relevant check first.

```bash
# Focused Python test
python -m pytest tests/test_agent_<area>.py -q

# Full Python suite for broad/shared agent changes
python -m pytest tests -q

# Minimal import smoke test for package/import changes
python -c "import kiui"

# Package build/metadata changes
python -m build
```

Frontend:

```bash
cd kiui/agent/frontend
npm ci
npm run typecheck
npm test
npm run build
```

Documentation:

```bash
pip install -r docs/requirements.txt
sphinx-build docs/source docs/build -b dirhtml
```

There is limited automated coverage outside `kiui.agent`; use focused smoke tests for the modified module and optional dependency. Do not require `.[full]` merely to test unrelated agent/core changes.

## Change discipline

- Keep changes focused and consistent with nearby code; do not refactor unrelated modules.
- Add or update focused tests for agent behavior changes.
- Update `kiui/agent/readme.md`, `README.md`, or `docs/source/` when user-facing commands, configuration, or behavior changes.
- Do not commit local `.kiui.yaml`, `.kia/`, caches, `node_modules/`, package build output, or credentials. `kiui/agent/frontend/dist/` is the deliberate generated-artifact exception.
