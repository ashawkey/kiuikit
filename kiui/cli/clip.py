import numpy as np

# reusable lazy session
SESSION = None

def process(img_or_txt, **kwargs):
    # TODO
    return

if __name__ == "__main__":
    import argparse
    from ..utils import batch_process_files

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default=None)
    parser.add_argument("out_path", type=str, default=None)
    args = parser.parse_args()

    batch_process_files(
        process,
        args.path,
        args.out_path,
        in_format=[".jpg", ".jpeg", ".png", ".txt"],
        out_format=".npy",
    )
