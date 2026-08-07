# AGENTS.md — Kiuikit (`kiui`)

## Project

Kiuikit is a flat Python package of computer-vision and 3D utilities, published as `kiui`.

- Python **>=3.10**; setuptools build (`pyproject.toml` is authoritative).
- Install: `pip install -e .`; all optional CV/3D dependencies: `pip install -e ".[full]"`.
- CLI entry points are declared in `pyproject.toml`; do not duplicate or guess them.
- Keep minimal installs importable. Do not import heavy optional dependencies at package/module import time unless that module requires them.

## Repository map

- `kiui/*.py`: standalone toolkit modules; `kiui/utils.py`: root-exposed utilities.
- `kiui/cli/`: additional command-line utilities.
- `kiui/nn/`: PyTorch components; `kiui/gridencoder/`: PyTorch wrapper and CUDA/C++ extension.
- `docs/source/`: Sphinx documentation.

Inspect the relevant module rather than relying on this high-level map.

## Important contracts

### Root package

`kiui/__init__.py` uses `lazy_loader` to expose modules, `env`, and functions parsed from `kiui/utils.py`.

- New public functions in `utils.py` automatically become `kiui.<name>`; do not manually re-export them.
- Preserve lazy imports and the minimal-install path.
- Importing `kiui` loads `kiui.conf` from `./.kiui.yaml`, then `~/.kiui.yaml`.

## Verification

Run the smallest relevant check:

- Package/import change: `python -c "import kiui"`
- Build/metadata change: `python -m build`
- Docs: `pip install -r docs/requirements.txt`, then `sphinx-build docs/source docs/build -b dirhtml`

Use focused smoke tests for the changed module and optional dependency; do not install `.[full]` for unrelated checks.

Update `README.md` or `docs/source/` for user-facing behavior. Never commit local `.kiui.yaml`, caches, `node_modules`, build output, or credentials.
