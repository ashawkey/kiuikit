# AGENTS.md — Kiuikit (`kiui`)

## Overview
Kiuikit is a **niche Python toolkit**. It is a single package (`kiui`) distributed via PyPI. The project follows a **flat** (non-namespace) package layout with a **lazy loader** at the root.

- **Install**: `pip install kiui` (minimal) / `pip install kiui[full]` (all deps)
- **Python**: ≥3.8
- **Build**: setuptools (declared in `pyproject.toml`)

---

## Entry Points (CLI commands → module)
Defined in `pyproject.toml` `[project.scripts]`:

| Command | Module | Purpose |
|---------|--------|---------|
| `kire` | `kiui.render:main` | GUI 3D mesh viewer (nvdiffrast + dearpygui) |
| `kisr` | `kiui.sr:main` | Super-resolution (Real-ESRGAN) |
| `kivi` | `kiui.video:main` | Video/Image info & processing |
| `kiss` | `kiui.sys:main` | System information (OS, GPU, torch, etc.) |
| `ks` | `kiui.slurm:main` | Slurm job management |
| `kia` | `kiui.agent.cli:main` | Terminal AI agent (LLM + tools + web) |
| `kib` | `kiui.agent.library_cli:main` | Git-backed personal skill library manager |

---

## Package Architecture

### Lazy Loading System (`kiui/__init__.py`)
The root `__init__.py` uses `lazy_loader.attach()` to:
- Expose **all submodules** and **all public functions from `utils.py`** at the `kiui` namespace.
- `kiui.lo(x)` works without explicit import of `kiui.utils`.
- Reads `~/.kiui.yaml` or `./.kiui.yaml` into `kiui.conf` on import.

**Implication**: when adding a new public function to `utils.py`, it automatically appears as `kiui.funcname`. New submodules (top-level `.py` files in `kiui/`) are also auto-exposed.

### Key Module Map

#### Core (always available, minimal deps)
| File | Purpose |
|------|---------|
| `config.py` | Load `~/.kiui.yaml` into dict `conf` |
| `typing.py` | Re-exports common type hints (`Tensor`, `ndarray`, `Union`, `Optional`, etc.) |
| `utils.py` | **Big misc utility module**: `lo()` (array inspection with `rich`), `seed_everything()`, `read_image/write_image`, `read_json/write_json`, `read_pickle/write_pickle`, `load_file_from_url`, `batch_process_files`, etc. |
| `env.py` | `kiui.env('torch')` — auto-import libraries into the caller's globals. Saves boilerplate in notebooks/scripts. |
| `op.py` | Vector math (`dot`, `length`, `safe_normalize`), image scaling helpers, `uv_padding`, `inverse_sigmoid/softplus`. Works on both torch tensors and numpy arrays. |
| `timer.py` | `sync_timer` — CUDA-synchronized timer (context manager + decorator). Gated by env var `TIMER=1`. |

#### Mesh (3D feature)
| File | Purpose |
|------|---------|
| `mesh.py` | **`Mesh` class** (~1200 lines). Torch-native mesh with `v/f/vn/vt/vc/albedo/metallicRoughness`. Loads `.obj/.ply/.glb/.fbx` via own parser or trimesh fallback. Exports `.obj/.ply/.glb`. Auto-normalize, auto-UV (xatlas), auto-size, remap-UV. |
| `mesh_utils.py` | `clean_mesh()`, `decimate_mesh()`, `remesh()` via pymeshlab. |

#### Visualization & Rendering
| File | Purpose |
|------|---------|
| `vis.py` | `plot_image()` (matplotlib), `map_color()` (matplotlib colormaps) |
| `render.py` | **`kire` CLI**: GUI mesh viewer (nvdiffrast + dearpygui + OrbitCamera). PBR rendering, auto-rotate, save video. |
| `render_viser.py` | Web-based mesh viewer using `viser` (alternative to dearpygui). |
| `video.py` | `read_video()`, video info/stats, CLI processing. ~1200 lines. |
| `lpips.py` | Clean LPIPS perceptual loss using SqueezeNet. |

#### Camera & Geometry
| File | Purpose |
|------|---------|
| `cam.py` | `OrbitCamera`, `to_homo()`, `intr_to_K()`, projection/unprojection utilities. |
| `geocalib.py` | Single-image camera intrinsics estimation (GeoCalib, pinhole model). ~1400 lines, self-contained. |
| `equirect.py` | Equirectangular (360°) utilities, HDR tonemapping, cubemap conversion. |
| `quaternion.py` | Quaternion math: norm, normalize, conjugate, inverse, multiply, rotate, slerp, to/from matrices. Supports both torch and numpy. |

#### Pose & Skeleton
| File | Purpose |
|------|---------|
| `poser.py` | Interactive 3D pose viewer with preset skeletons (2head–8head proportions) and OpenPose joint mapping. Uses dearpygui + nvdiffrast. ~700 lines. |

#### Neural Networks (`kiui/nn/`)
A collection of **standalone PyTorch NN building blocks** — not a framework, just reusable modules:
| File | Purpose |
|------|---------|
| `__init__.py` | `MLP` class, `TruncExp` autograd function |
| `utils.py` | Cosine LR schedule with warmup, parameter counting |
| `attention_flash.py` | Flash attention wrapper |
| `attention_xformers.py` | xFormers attention wrapper |
| `dit.py` | Diffusion Transformer (DiT) blocks |
| `encoder.py` | Encoder network utilities |
| `flow_matching.py` | Flow matching utilities |
| `llm.py` | LLM-related network components |
| `perciever.py` | Perceiver architecture |
| `sparse.py` | Sparse tensor utilities |
| `unet_2d.py` | 2D UNet |
| `unet_2d_cond.py` | Conditional 2D UNet |
| `unet_3d_cond.py` | Conditional 3D UNet |
| `vae_2d.py` | 2D VAE |
| `vae_3d.py` | 3D VAE |

#### Grid Encoder (`kiui/gridencoder/`)
| File | Purpose |
|------|---------|
| `grid.py` | `GridEncoder` torch module (hash-grid encoding) |
| `backend.py` | Backend binding |
| `src/` | CUDA C++ kernel (`gridencoder.cu`, `.h`, `bindings.cpp`) |

#### AI Agent (`kiui/agent/`)
A self-contained **terminal AI agent** (comparable to Claude Code / Codex CLI). Features include file and shell tools, managed background processes, web access, sub-agents, skills and a Git-backed skill library, personas, standing goals, context compaction, rewind, three-tier permissions, and a shared Web UI. See `kiui/agent/readme.md` for full docs.

The implementation is split into focused packages; there are no longer top-level `backend.py`, `tools.py`, `prompts.py`, `io.py`, `interrupt.py`, or `rewind.py` modules.

| Path | Purpose |
|------|---------|
| `cli.py` | `kia` entry point and Tyro argument parsing; starts chats or the hub, resumes sessions, lists models, and manages project `.kia` storage. |
| `backend/__init__.py` | `LLMAgent`: core OpenAI-compatible API loop, retries, streaming, context/tool orchestration, cancellation, and persona setup. |
| `backend/commands.py` | Slash-command routing, including model, persona, permission, context, and rewind commands. |
| `backend/goals.py` | Standing-goal state and automatic goal-check iteration. |
| `backend/sessions.py` | Session persistence, selection, resume, and replay. |
| `backend/skill_commands.py` | Skill listing, reload, and manual loading commands. |
| `tools/` | Tool subsystem. `schemas.py` defines the OpenAI tool schemas and `executor.py` dispatches to focused file, shell, managed-process, search, web, and agent-session mixins. Also contains result formatting/artifact storage, limits, and process supervision. |
| `context.py` | Conversation representation, token estimation, tool-result ingress compaction, context pruning, and LLM compaction. |
| `personas/` | Persona registry and prompt ownership. `coder.py`, `chatter.py`, and `reviewer.py` define complete system prompts and tool whitelists; `common.py` provides shared prompt builders. |
| `skills.py` / `bundled_skills/` | Agent Skills discovery, validation, progressive loading, and bundled skill packs. Project and personal skills live under `.kia/skills/`. |
| `library.py` / `library_cli.py` | Git-backed personal skill library and its `kib` CLI (`list`, `install`, `update`, `upload`, and `remove`). |
| `subagent.py` | Synchronous, in-process sub-agent spawning with isolated working-directory context. |
| `permissions.py` | `PermissionMode` (auto/default/strict), confirmation policy, and destructive-command safety guard. |
| `models.py` | Model capability profiles and provider-specific reasoning configuration. |
| `ui.py` | Rich terminal rendering (`AgentConsole`), status indicators, and streamed responses. |
| `terminal.py` | Prompt-toolkit input, file-path completion, history, and keyboard bindings. |
| `utils/io.py` | Thread-safe `EventHub`, `InputBroker`, `PromptBroker`, and `CancellationToken` shared by terminal and web clients. |
| `utils/interrupt.py` | Cancellation of in-flight API calls, retry waits, and foreground commands via Escape/Ctrl+C. |
| `utils/rewind.py` | Per-round file change tracking, backups, and rollback; conversation rewind is coordinated by `backend/commands.py`. |
| `utils/paths.py`, `utils/storage.py`, `utils/streaming.py` | `.kia` path/storage helpers and OpenAI-compatible streamed-response accumulation. |
| `hub.py` | `kia --hub` web daemon: browser authentication, agent/session registry, API/WebSockets, and multiplexing many agents into one UI. |
| `hubclient.py` | Agent-side client. A normal terminal `kia` automatically links to a reachable hub, forwards events, and injects browser actions through the shared brokers. |
| `frontend/` | React + TypeScript/Vite Web UI; FastAPI serves the committed production build from `frontend/dist/`. |

#### CLI Tools (`kiui/cli/`)
Standalone CLI scripts (not exposed via the main package namespace, run directly as scripts):
| File | Purpose |
|------|---------|
| `aes.py` | Aesthetic Predictor V2 |
| `bg.py` | Background removal (rembg) |
| `blender_*.py` | Blender integration (qremesh, render) |
| `blip.py` | BLIP image captioning |
| `clip_sim.py` / `clip_sim_text.py` | CLIP similarity |
| `convert.py` / `convert_fp16.py` | File/format conversion |
| `depth_midas.py` / `depth_zoe.py` | Monocular depth estimation |
| `dircmp.py` | Directory comparison |
| `hed.py` | HED edge detection |
| `lock_version.py` | Lock dependency versions |
| `remesh.py` | Mesh remeshing |
| `timer.py` | CLI timer tool |
| `pose.py` | OpenPose CLI wrapper |
| `openpose/` | OpenPose body/face/hand model wrappers |

#### Other Top-Level Modules
| File | Purpose |
|------|---------|
| `sys.py` | `kiss` CLI: rich system info (OS, CPU, GPU, disk, torch, CUDA, conda, etc.) |
| `slurm.py` | `ks` CLI: Slurm job submission, monitoring, management. ~1100 lines. |
| `sr.py` | `kisr` CLI: Real-ESRGAN super-resolution. Downloads models from HuggingFace Hub. |
| `grid_put.py` | `scatter_add_nd` and related sparse grid scatter operations. |
| `assets/` | Bundled assets: HDRI environment maps (`blender_lights/` with city, courtyard, forest, interior, night, studio, sunrise, sunset EXR files; `lights/` with BSDF LUT and HDR). |

---

## Configuration
- `kiui.conf` — loaded at import time from `./.kiui.yaml` or `~/.kiui.yaml`. Used by the agent (`kia`) for model API keys/URLs, and potentially by other modules.

---

## Documentation
Built with Sphinx, source in `docs/source/`. Published at https://kit.kiui.moe/. Each major module has a corresponding `.md` file in the docs.


## Coding Style: Fail Fast, Trust Contracts, and Keep Comments High-Signal

Keep implementation code lean, explicit, and predictable. Validate and sanitize untrusted data at system boundaries, such as API endpoints, file parsers, CLI handlers, and external integrations. Once data enters the core system, trust the established contracts, type hints, and upstream validation rather than repeatedly checking, coercing, or normalizing it.

Do not add speculative safeguards, silent exception handling, arbitrary fallback values, redundant runtime type conversions, or defensive branches for states that should be impossible under the contract. Invalid internal input should fail loudly with a clear exception so the root cause can be corrected at its source. Catch exceptions only when the code can meaningfully recover, add useful context, perform required cleanup, or translate an error at a boundary.

Keep comments concise, accurate, and useful. Comments should explain intent, invariants, non-obvious constraints, tradeoffs, or why an implementation differs from the obvious approach; they should not restate what the code already expresses. Whenever code behavior changes, update or remove any affected comments in the same change. Never leave stale comments that contradict or misrepresent the implementation. Prefer clear names and straightforward structure over explanatory commentary, and delete comments that no longer add insight.