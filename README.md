# kiui kit

A toolkit for personal use.

### Install

```bash
# released
pip install kiui # install the minimal package
pip install kiui[full] # install optional dependencies

# latest
pip install git+https://github.com/ashawkey/kiuikit.git # only the minimal package
```

### Usage

```python
import kiui

### auto import
kiui.env() # os, glob, math, time, random, argparse
kiui.env('data') # above + np, plt, cv2, Image, ...
kiui.env('torch') # above + torch, nn, F, ...

### quick inspection of array-like object
x = torch.tensor(...)
y = np.array(...)

kiui.lo(x)
kiui.lo(x, y) # support multiple objects

kiui.lo(kiui) # or any other object (just print with name)

### io utils
# read image as-is in RGB order
img = kiui.read_image('image.png', mode='float') # mode: float (default), pil, uint8, tensor
# write image
kiui.write_image('image.png', img)

### visualization tools
img_tensor = torch.rand(3, 256, 256) 
# tensor of [3, H, W], [1, H, W], [H, W] / array of [H, W ,3], [H, W, 1], [H, W] in [0, 1]
kiui.vis.plot_image(img)
kiui.vis.plot_image(img_tensor)

### mesh utils
from kiui.mesh import Mesh
mesh = Mesh.load('model.obj')
kiui.lo(mesh.v, mesh.f) # CUDA torch.Tensor suitable for nvdiffrast
mesh.write('new.obj')

```

CLI tools:
```bash
# background removal utils
python -m kiui.cli.bg --help
python -m kiui.cli.bg input.png output.png
python -m kiui.cli.bg input_folder output_folder

# openpose detector
python -m kiui.cli.pose --help

# hed edge detector
python -m kiui.cli.hed --help

# open a GUI to render a mesh (extra dependency: nvdiffrast)
python -m kiui.cli.renderer --help
python -m kiui.cli.renderer mesh.obj
```