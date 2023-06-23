import os
import cv2
import glob
import tqdm

from .openpose import OpenposeDetector

# reusable lazy session
SESSION = None

def detect(img, hand_and_face=True, **kwargs):
    # img: np.ndarray, (h, w, 3), uint8, RGB

    global SESSION
    if SESSION is None:
        SESSION = OpenposeDetector()

    res = SESSION(img, hand_and_face=hand_and_face)

    return res


def detect_file(path, out_path, **kwargs):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = detect(img, **kwargs)
    res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, res)


def detect_folder(path, out_path, **kwargs):
    os.makedirs(out_path, exist_ok=True)
    img_paths = glob.glob(os.path.join(path, '*'))
    for img_path in tqdm.tqdm(img_paths):
        try:
            img_out_path = os.path.join(out_path, os.path.basename(img_path))
            detect_file(img_path, img_out_path, **kwargs)
        except Exception as e:
            print(e)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, default=None)
    parser.add_argument('out_path', type=str, default=None)
    parser.add_argument('--body_only', action='store_true')
    args = parser.parse_args()

    if os.path.isfile(args.path):
        detect_file(args.path, args.out_path, hand_and_face=not args.body_only)
    elif os.path.isdir(args.path):
        detect_folder(args.path, args.out_path, hand_and_face=not args.body_only)