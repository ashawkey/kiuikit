import os
import shutil
import torch
from rich.console import Console
from torch.utils.cpp_extension import load, _get_build_directory

_src_path = os.path.dirname(os.path.abspath(__file__))

cpp_standard = 17 # assume CUDA > 11.0, change to 14 otherwise
nvcc_flags = [
    '-O3', f'-std=c++{cpp_standard}',
    '-U__CUDA_NO_HALF_OPERATORS__', 
    '-U__CUDA_NO_HALF_CONVERSIONS__', 
    '-U__CUDA_NO_HALF2_OPERATORS__',
]

# get CUDA compute capability
assert torch.cuda.is_available(), "CUDA is not available"
major, minor = torch.cuda.get_device_capability()
compute_capability = major * 10 + minor

nvcc_flags += [f"-gencode=arch=compute_{compute_capability},code={code}_{compute_capability}" for code in ["compute", "sm"]]

if os.name == "posix":
    c_flags = ['-O3', f'-std=c++{cpp_standard}']
elif os.name == "nt":
    c_flags = ['/O2', f'/std:c++{cpp_standard}']

    # find cl.exe
    def find_cl_path():
        import glob
        for edition in ["Enterprise", "Professional", "BuildTools", "Community"]:
            paths = sorted(glob.glob(r"C:\\Program Files (x86)\\Microsoft Visual Studio\\*\\%s\\VC\\Tools\\MSVC\\*\\bin\\Hostx64\\x64" % edition), reverse=True)
            if paths:
                return paths[0]

    # If cl.exe is not on path, try to find it.
    if os.system("where cl.exe >nul 2>nul") != 0:
        cl_path = find_cl_path()
        if cl_path is None:
            raise RuntimeError("Could not locate a supported Microsoft Visual C++ installation")
        os.environ["PATH"] += ";" + cl_path

name = '_grid_encoder'
build_dir = _get_build_directory(name, verbose=False)

if os.listdir(build_dir) != []:
    _backend = load(
        name=name,
        extra_cflags=c_flags,
        extra_cuda_cflags=nvcc_flags,
        sources=[os.path.join(_src_path, 'src', f) for f in [
            'gridencoder.cu',
            'bindings.cpp',
        ]],
    )
else:
    shutil.rmtree(build_dir)
    with Console().status(
        "[bold yellow]GridEncoder: Setting up CUDA (This may take a few minutes the first time)",
        spinner="bouncingBall",
    ):
        _backend = load(
            name=name,
            extra_cflags=c_flags,
            extra_cuda_cflags=nvcc_flags,
            sources=[os.path.join(_src_path, 'src', f) for f in [
                'gridencoder.cu',
                'bindings.cpp',
            ]],
        )


__all__ = ['_backend']