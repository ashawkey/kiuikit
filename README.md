# kiui kit

Utils for self-use.

### Install

```bash
# released
pip install kiui

# latest
pip install git+https://github.com/ashawkey/kiuikit.git
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
kiui.lo(x, y)

kiui.lo(kiui) # or any other object (just print with name)

### visualization tools
img = torch.rand(3, 256, 256) 
# tensor of [3, H, W], [1, H, W], [H, W] / array of [H, W ,3], [H, W, 1], [H, W] in [0, 1]
kiui.vis.plot_image(img)

### io utils
# read image as-is in RGB order
img = kiui.read_image('path/to/image', mode='float') # pil, float, uint8, tensor
# write image
kiui.write_image('path/to/image', img)
```