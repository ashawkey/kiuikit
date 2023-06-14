import os
import glob
import cv2
import tqdm
import numpy as np
from skimage.measure import label

import rembg

# ref: https://stackoverflow.com/questions/47540926/get-the-largest-connected-component-of-segmentation-image
# ref: https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.label
def largest_connected_component(mask, background=0, connectivity=None):
    # mask: [h, w], int
    labels = label(mask, background=background, connectivity=connectivity)
    assert labels.max() != 0, 'assume at least one connected component!'
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC

# reusable lazy session
SESSION = None

# a make-more-sense wrapper of rembg
def remove(img, return_mask=False, lcc=False, **kwargs):
    # img: np.ndarray, (h, w, 3), uint8, BGR

    global SESSION
    if SESSION is None:
        SESSION = rembg.new_session()

    res = rembg.remove(img, session=SESSION, **kwargs) # (h, w, 4), BGRA

    # largest-connected-component
    if lcc:
        mask = largest_connected_component(res[:, :, 3])
        res *= mask[:, :, None].astype(np.uint8)

    if return_mask:
        res = res[:, :, 3] # (h, w), uint8
    
    return res


def remove_file(path, out_path, **kwargs):
    img = cv2.imread(path)
    res = remove(img, **kwargs)
    cv2.imwrite(out_path, res)


def remove_folder(path, out_path, **kwargs):
    os.makedirs(out_path, exist_ok=True)
    img_paths = glob.glob(os.path.join(path, '*'))
    for img_path in tqdm.tqdm(img_paths):
        try:
            img_out_path = os.path.join(out_path, os.path.basename(img_path))
            if img_out_path.endswith('.jpg'):
                img_out_path = img_out_path.replace('.jpg', '.png')
            elif img_out_path.endswith('.jpeg'):
                img_out_path = img_out_path.replace('.jpeg', '.png')
            remove_file(img_path, img_out_path, **kwargs)
        except Exception as e:
            print(e)
