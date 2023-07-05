import numpy as np
from skimage.measure import label

import rembg

# ref: https://stackoverflow.com/questions/47540926/get-the-largest-connected-component-of-segmentation-image
# ref: https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.label
def largest_connected_component(mask, background=0, connectivity=None):
    # mask: [h, w], int
    labels = label(mask, background=background, connectivity=connectivity)
    assert labels.max() != 0, "assume at least one connected component!"
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC


# reusable lazy session
SESSION = None

# a make-more-sense wrapper of rembg
def remove(img, mode="rgba", lcc=False, post_process=True, **kwargs):
    # img: np.ndarray, (h, w, 3), uint8, BGR

    global SESSION
    if SESSION is None:
        SESSION = rembg.new_session()

    res = rembg.remove(
        img, session=SESSION, post_process=post_process, **kwargs
    )  # (h, w, 4), BGRA

    # largest-connected-component
    if lcc:
        mask = largest_connected_component((res[..., 3] > 10).astype(np.uint8))
        res = res * mask[..., None].astype(np.uint8)

    if mode == "rgb":
        # mix masked image with white background
        # res = cv2.cvtColor(res, cv2.COLOR_BGRA2BGR)
        res = res[..., :3] * (res[..., 3:] / 255.0) + 255.0 * (
            1.0 - res[..., 3:] / 255.0
        )
    elif mode == "a":
        res = res[..., 3]  # (h, w), uint8

    return res


if __name__ == "__main__":
    import argparse
    from kiui.utils import batch_process_files

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default=None)
    parser.add_argument("out_path", type=str, default=None)
    parser.add_argument("--lcc", action="store_true")
    parser.add_argument(
        "--mode", type=str, choices=["rgba", "a", "rgb"], default="rgba"
    )
    args = parser.parse_args()

    batch_process_files(
        remove,
        args.path,
        args.out_path,
        color_order="BGR",
        out_format=".png" if args.mode == "rgba" else None,
        mode=args.mode,
        lcc=args.lcc,
    )
