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

A toolkit for computer vision (especially 3D vision) tasks.

**Features**:
* Collection of useful & reusable code snippets.
* Always using lazy import so the code is not slowed down by `import kiui`.

### Install

```bash
# released
pip install kiui # install the minimal package
pip install kiui[full] # install optional dependencies

# latest
pip install git+https://github.com/ashawkey/kiuikit.git # only the minimal package
```

### Basic Usage

```python
import kiui

### quick inspection of array-like object
x = torch.tensor(...)
y = np.array(...)

kiui.lo(x)
kiui.lo(x, y) # support multiple objects
kiui.lo(kiui) # or any other object (just print with name)

### visualization tools
img_tensor = torch.rand(3, 256, 256) 
# tensor of [3, H, W], [1, H, W], [H, W] / array of [H, W ,3], [H, W, 1], [H, W] in [0, 1]
kiui.vis.plot_image(img)
kiui.vis.plot_image(img_tensor)

### mesh utils
from kiui.mesh import Mesh
mesh = Mesh.load('model.obj')
kiui.lo(mesh.v, mesh.f) # CUDA torch.Tensor
mesh.write('new.glb') # support exporting to GLB/GLTF too (texture embedded).
```

CLI tools:
```bash
# sr (Real-ESRGAN from https://github.com/ai-forever/Real-ESRGAN/tree/main)
python -m kiui.sr --help
python -m kiui.sr image.jpg --scale 2 # save to image_2x.jpg
kisr image.jpg --scale 2 # short cut cmd

# mesh format conversion (only for a single textured mesh in obj/glb)
python -m kiui.cli.convert input.obj output.glb
kico input.obj output.glb # short cut cmd
kico mesh_folder video_folder --in_fmt .glb --out_fmt .mp4 # render all glb meshes into rotating videos

# aesthetic predictor v2 (https://github.com/christophschuhmann/improved-aesthetic-predictor)
python -m kiui.cli.aes --help

# compare content of two directories
python -m kiui.cli.dircmp <dir1> <dir2>

# lock requirements.txt package versions based on current environment
python -m kiui.cli.lock_version <requirements.txt>

# render mesh with blender
python -m kiui.cli.blender_render --mesh input.glb --gpu 0 --depth --normal --pbr --camera --blend
python -m kiui.cli.blender_render --mesh input.glb --wireframe
```

GUI tools:
```bash
# open a GUI to render a mesh (extra dep: nvdiffrast)
python -m kiui.render --help
python -m kiui.render mesh.obj
python -m kiui.render mesh.glb --pbr # render with PBR (metallic + roughness)
python -m kiui.render mesh.obj --save_video out.mp4 --wogui # save 360 degree rotating video
kire --help # short cut cmd

# open a GUI to render and edit pose (openpose convention, controlnet compatible)
python -m kiui.poser --help
python -m kiui.poser --load 3head # load preset 3 headed skeleton
```

https://github.com/ashawkey/kiuikit/assets/25863658/d8cbcf0f-a6d8-4fa7-aee9-afbbf25ed167

> ["Seahourse3"](https://skfb.ly/6TwFv) by seanhepburn is licensed under [Creative Commons Attribution](http://creativecommons.org/licenses/by/4.0/).
