<p align="center">
    <picture>
    <img alt="kiuikit_logo" src="docs/source/_static/logo.png" width="50%">
    </picture>
    </br>
    <b>Kiuikit</b>
    </br>
    <code>pip install kiui</code>
    &nbsp;&nbsp;&bull;&nbsp;&nbsp;
    <a href="https://kit.kiui.moe/">Documentation</a>
</p>

A niche toolkit for computer vision (especially 3D vision) tasks.

### Install

```bash
# released
pip install kiui # install the minimal package
pip install kiui[full] # install optional dependencies

# latest
pip install git+https://github.com/ashawkey/kiuikit.git # only the minimal package
```

### Highlights

**CLI tools:**

| Command | Description |
|---------|-------------|
| `kire` | GUI 3D mesh viewer with PBR rendering, auto-rotate, save video |
| `kisr` | Super-resolution (Real-ESRGAN) |
| `kivi` | Video/image info, resize, preview (capped-CRF), split |
| `kiss` | System information — OS, CPU, GPU, torch, CUDA, conda |
| `ks` | Slurm job management — queue, history, logs, cancel, usage |
| `kia` | AI coding agent — terminal-native with web UI, sub-agents, skills, and more |

### AI Agent (`kia`)

`kia` is a lightweight yet powerful coding agent.

```bash
pip install "kiui[kia]"
kia
```

What makes it different:
- Native terminal + synchronized web UI, remote access from any browser.
- Personal skill library `kib`: upload your own skills, download in any projects later.
- Fully customizable personas.

See the [agent documentation](https://kit.kiui.moe/agent.html) for full details.

**Python library:**

| Module | Highlights |
|--------|-----------|
| `kiui.lo()` | Rich-based inspection of arrays, tensors, and any object |
| `kiui.Mesh` | Torch-native 3D mesh — load `.obj/.glb/.ply/.fbx`, export, auto-UV, auto-normalize |
| `kiui.read_video` / `write_video` | Video I/O with numpy/torch support |
| `kiui.read_image` / `write_image` | Image I/O with float/HDR support |
| `kiui.op` | Vector math (`dot`, `length`, `safe_normalize`) working on both torch & numpy |
| `kiui.cam` | `OrbitCamera`, projection / unprojection utilities |
| `kiui.quaternion` | Quaternion math (norm, multiply, slerp, to/from matrices) for torch & numpy |
| `kiui.nn` | Standalone PyTorch NN blocks — MLP, DiT, UNet 2D/3D, VAE, attention, flow matching |
| `kiui.gridencoder` | Hash-grid encoding with CUDA backend |
| `kiui.lpips` | Clean LPIPS perceptual loss (SqueezeNet) |
| `kiui.timer` | CUDA-synchronized timer (context manager & decorator) |
| `kiui.equirect` | Equirectangular (360°) / cubemap utilities |
| `kiui.geocalib` | Single-image camera intrinsics estimation |

See the [documentation](https://kit.kiui.moe/) for full API details.