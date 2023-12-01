import importlib
import inspect


LIBS = {
    "base": [
        {
            "os": "os",
            "glob": "glob",
            "math": "math",
            "time": "time",
            "random": "random",
            "argparse": "argparse",
        },
        (),
    ],
    "utils": [
        {
            "tqdm": "tqdm",
            "rich": "rich",
        },
        ("base"),
    ],  # dependency
    "data": [
        {
            "np": "numpy",  # alternatives
            "cv2": "cv2",
            "plt": "matplotlib.pyplot",
            "Image": "PIL.Image",
            # 'vis': vis, # live object
        },
        ("base", "utils"),
    ],
    "torch": [
        {
            "torch": "torch",
            "nn": "torch.nn",
            "F": "torch.nn.functional",
            "Dataset": ("torch.utils.data", "Dataset"),  # single class/function
            "DataLoader": ("torch.utils.data", "DataLoader"),
        },
        ("base", "utils", "data"),
    ],
}

LIBS["all"] = [{}, tuple(LIBS.keys())]

G = None


def retrieve_globals(verbose=False):
    # locate and set G to the globals which directly `import kiui`. (only once)
    # ref: https://stackoverflow.com/questions/40652688/how-to-access-globals-of-parent-module-into-a-sub-module/50381748
    global G
    stack = inspect.stack()
    frame_id = 1
    while frame_id < len(stack):
        g = dict(inspect.getmembers(stack[frame_id][0]))["f_globals"]
        if "kiui" in g:
            G = g
            if verbose:
                print(f"[INFO] located global frame at {frame_id}")
                # print(G)
            break
        frame_id += 1
    if G is None:
        raise RuntimeError(
            "Cannot locate global frame, make sure you called exactly `import kiui`!"
        )

def is_imported(target, verbose=False):

    if G is None:
        retrieve_globals(verbose)

    return target in G

def try_import(target, sources, verbose=False):

    if G is None:
        retrieve_globals(verbose)

    if target in G:
        if verbose:
            print(f"[INFO] {target} is already present, skipped.")
        return

    if not isinstance(sources, list):
        sources = [sources]

    for source in sources:
        try:
            if verbose:
                print(f"[INFO] try to import {source}")

            # (module, component) or ("module", component)
            if isinstance(source, tuple):
                source_module, source_component = source
                if isinstance(source_module, str):
                    source_module = importlib.import_module(source_module)
                source = getattr(source_module, source_component)
            # "module"
            elif isinstance(source, str):
                source = importlib.import_module(source)
            # module
            else:
                pass

            G[target] = source

            if verbose:
                print(f"[INFO] succeed to import {source} as {target}")
            break

        except ImportError as e:
            print(f"[WARN] failed to import {source} as {target}: {str(e)}")


def import_libs(pack, verbose=False):
    for k, v in LIBS[pack][0].items():
        try_import(k, v, verbose)


"""
setup all import in one line.
usage:
    kiui.env() # setup base env
    kiui.env('torch') # pytorch with all regular dependencies
    kiui.env('[torch]') # pytorch only
"""


def env(*packs, verbose=False):
    if len(packs) == 0:
        packs = ["base"]

    def check_pack(pack):
        if pack not in LIBS:
            raise RuntimeError(
                f"[Kiui-ERROR] unknown pack {pack}, availables: {list(LIBS.keys())}"
            )

    def resolve_env(packs):
        res = set()
        for pack in packs:
            if pack[0] == "[" and pack[-1] == "]":
                pack = pack[1:-1]
                check_pack(pack)
                res.add(pack)
            else:
                check_pack(pack)
                for dep in LIBS[pack][1]:
                    res.add(dep)
                res.add(pack)
        return list(res)

    packs = resolve_env(packs)
    for pack in packs:
        import_libs(pack, verbose)
