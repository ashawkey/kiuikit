import lazy_loader

# lazy import that equals:
# from . import op, vis, bg, mesh
# from .utils import *
# from .env import env

__getattr__, __dir__, _ = lazy_loader.attach(
    __name__,
    submodules=["op", "vis", "cam", "mesh", "cli"],
    submod_attrs={
        "utils": ["lo", "read_json", "write_json", "read_image", "write_image"],
        "env": ["env"],
    },
)
