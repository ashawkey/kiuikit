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
A self-contained **terminal AI agent** (comparable to Claude Code / Codex CLI). Features: file ops, shell exec, web search, sub-agents, skills, context compaction, rewind, 3-tier permissions. See `kiui/agent/readme.md` for full docs.

| File | Purpose |
|------|---------|
| `cli.py` | Entry point (`kia` command). Argument parsing via `tyro`. |
| `backend.py` | Core agent loop, tool dispatch, API calls |
| `tools.py` | Tool definitions (`read_file`, `write_file`, `edit_file`, `exec_command`, `glob_files`, `grep_files`, `web_search`, `web_fetch`, `remove_file`, `spawn_subagent`, `load_skill`) |
| `ui.py` | Rich-based terminal UI (`AgentConsole`) |
| `models.py` | Model profile resolution from `kiui.conf` |
| `permissions.py` | `PermissionMode` enum (auto/default/strict) + hard safety guard |
| `io.py` | Thread-safe shared I/O primitives (`EventHub`, `InputBroker`, `PromptBroker`, `CancellationToken`) bridging terminal + web |
| `hub.py` | `kia --hub` shared web hub daemon: owns the public port, browser auth, session registry, multiplexes many agents into one UI |
| `hubclient.py` | Agent-side client that links a `kia --web` terminal agent to the hub (registers a session, forwards events, injects browser actions) |
| `context.py` | Context window management (pruning, LLM compaction) |
| `subagent.py` | Sub-agent spawning |
| `skills.py` | Skill pack loading from `.kia/skills/` |
| `rewind.py` | Rewind/rollback conversation and file changes |
| `prompts.py` | System prompt construction |
| `interrupt.py` | Interrupt/signal handling |
| `terminal.py` | Terminal I/O helpers |

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
