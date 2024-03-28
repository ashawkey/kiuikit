import os
import lazy_loader

# lazy import that equals:
# from . import op, vis, bg, mesh, ...
# from .utils import *
# from .env import env

module_path = os.path.dirname(os.path.abspath(__file__))
submodules = [m.strip('.py') for m in os.listdir(module_path) if not m.startswith('__')]
submodules.append('gridencoder')
submodules.append('nn')

# find out all function names without importing the module
utils_path = os.path.join(module_path, 'utils.py')
with open(utils_path, 'r', encoding='utf-8') as f:
    utils_code = f.readlines()
utils_funcnames = []
for line in utils_code:
    if line.startswith('def '):
        utils_funcnames.append(line.split('(')[0].split(' ')[1])

__getattr__, __dir__, _ = lazy_loader.attach(
    __name__,
    submodules=submodules,
    submod_attrs={
        "utils": utils_funcnames,
        "env": ["env"],
    },
)
