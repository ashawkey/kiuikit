# Remesh

[[source]](https://github.com/ashawkey/kiuikit/blob/main/kiui/cli/remesh.py)

We provide a wrapper of [Continuous Remeshing](https://github.com/Profactor/continuous-remeshing) for convenience.

```bash
# install torch_scatter following https://github.com/rusty1s/pytorch_scatter
# e.g., for torch2.1.0+cu121
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# help
python -m kiui.cli.remesh --help

# example
python -m kiui.cli.remesh --mesh chest.glb --iters 1000
```