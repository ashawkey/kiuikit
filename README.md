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

### Basic Usage

The package comes with many helpful CLI tools:

```bash
# print detailed information of a video or image
kivi --help
kivi info <video_path>

# print system information
kiss --help
kiss os # print os, cpu, gpu, etc.
kiss torch # print torch version, cuda availability, etc.

# llm agent utils
kia --help
kia list # list available models (APIs should be defined in ~/.kiui.yaml)
kia chat --model <name> # start interactive chat mode
kia exec --model <name> "What does kiui mean?" # execute a single query

# open a GUI to render a mesh (extra dep: nvdiffrast)
kire --help
kire mesh.obj
kire mesh.glb --pbr # render with PBR (metallic + roughness)
kire mesh.obj --save_video out.mp4 --wogui # save 360 degree rotating video
```

It can also be used as a Python library:

```python
import kiui

# quick inspection of array-like object
x = torch.tensor(...)
y = np.array(...)

kiui.lo(x)
kiui.lo(x, y) # support multiple objects
kiui.lo(kiui) # or any other object (just print with name)

# visualization tools
img_tensor = torch.rand(3, 256, 256) 
# support tensor of [3, H, W], [1, H, W], [H, W] / np.ndarray of [H, W ,3], [H, W, 1], [H, W] in [0, 1]
kiui.vis.plot_image(img)
kiui.vis.plot_image(img_tensor)

# mesh utils
from kiui.mesh import Mesh
mesh = Mesh.load('model.obj')
kiui.lo(mesh.v, mesh.f) # CUDA torch.Tensor
mesh.write('new.glb') # support exporting to GLB/GLTF too (texture embedded).
```