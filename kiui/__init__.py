# import 
import importlib
import inspect

from .torch_utils import *


LIBS = {
    'standard': {
        'os': 'os',
        'sys': 'sys',
        'math': 'math',
        'copy': 'copy',
        'glob': 'glob',
    },
    'utils': {
        'time': 'time',
        'tqdm': 'tqdm',
        'shutil': 'shutil',
        'argparse': 'argparse',
    },
    'data': {
        'np': ['numpytorch', 'numpy'], # alternatives
        'pd': 'pandas',
        'cv2': 'cv2',
        'plt': 'matplotlib.pyplot',
        'Image': 'PIL.Image',
    },
    'torch': {
        'torch': 'torch',
        'nn': 'torch.nn',
        'F': 'torch.nn.functional',
        'Dataset': ('torch.utils.data', 'Dataset'), # single class/function
        'DataLoader': ('torch.utils.data', 'DataLoader'),
        'torch_vis_2d': torch_vis_2d, # live object
    },
    'ddp': {
        'dist': 'torch.distributed',
        'mp': 'torch.multiprocessing',
        'DDP': ('torch.nn.parallel', 'DistributedDataParallel'),
    }
}


LEVELS = {}
for k in LIBS.keys():
    LEVELS[k] = [k]
LEVELS[0] = ['standard']
LEVELS[1] = LEVELS[0] + ['utils']
LEVELS[2] = LEVELS[1] + ['data']
LEVELS[3] = LEVELS[2] + ['torch']
LEVELS['all'] = list(LIBS.keys())

G = None

def retrieve_globals(verbose=False):
    # locate and set G to the globals which directly `import kiui`. (only once)
    # ref: https://stackoverflow.com/questions/40652688/how-to-access-globals-of-parent-module-into-a-sub-module/50381748
    global G
    stack = inspect.stack()
    frame_id = 1 
    while frame_id < len(stack):
        g = dict(inspect.getmembers(stack[frame_id][0]))["f_globals"]
        if 'kiui' in g:
            G = g
            if verbose:
                print(f'[INFO] located global frame at {frame_id}')
            break
        frame_id += 1
    if G is None:
        raise RuntimeError('Cannot locate global frame, make sure you called exactly `import kiui`!')


def try_import(target, sources, verbose=False):

    if G is None:
        retrieve_globals(verbose)
    
    if target in G:
        if verbose:
            print(f'[INFO] {target} is already present, skipped.')
        return

    if not isinstance(sources, list):
        sources = [sources]

    for source in sources:
        try:
            if verbose:
                print(f'[INFO] try to import {source}')
            if isinstance(source, tuple):
                source = getattr(importlib.import_module(source[0]), source[1])
            elif isinstance(source, str):
                source = importlib.import_module(source)
            G[target] = source
            if verbose:
                print(f'[INFO] succeed to import {source} as {target}')
            break
        except ImportError as e:
            print(f'[WARN] failed to import {source} as {target}: {str(e)}')
            

def import_libs(libs, verbose=False):
    for k, v in LIBS[libs].items():
        try_import(k, v, verbose)


def init(level=3, verbose=False):
    if level not in LEVELS:
        raise ValueError(f'invalid level, availables: {LEVELS.keys()}')
    for libs in LEVELS[level]:
        import_libs(libs, verbose)
    