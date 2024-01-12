# Mesh

`kiui.mesh.Mesh` class provides an implementation of a triangular mesh in PyTorch.

`kiui.mesh_utils` provides utility functions for mesh processing and loss functions.

### Examples

```python
import kiui
from kiui.mesh import Mesh

# load mesh
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mesh = Mesh.load('model.glb', device=device)

# inspect attributes
kiui.lo(mesh.v, mesh.f, mesh.albedo, mesh.vt, mesh.ft, mesh.vn, mesh.fn)

# write mesh (can change to obj format)
mesh.write('model.obj')
```


### API

.. automodule:: kiui.mesh
   :members:

.. automodule:: kiui.mesh_utils
   :members:
