import cv2
import torch
from .openpose import OpenposeDetector

# reusable lazy session
SESSION = None

@torch.no_grad()
def detect(img, reso=None, body_only=False, **kwargs):
    # img: np.ndarray, (h, w, 3), uint8, RGB

    global SESSION
    if SESSION is None:
        SESSION = OpenposeDetector()

    if img.shape[-1] > 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    if reso is not None:
        img = cv2.resize(img, (reso, reso), interpolation=cv2.INTER_LINEAR)
    res = SESSION(img, hand_and_face=not body_only)

    return res


if __name__ == "__main__":
    import argparse
    from kiui.utils import batch_process_files

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default=None)
    parser.add_argument("out_path", type=str, default=None)
    parser.add_argument("--reso", type=int, default=None)
    parser.add_argument("--body_only", action="store_true")
    args = parser.parse_args()

    batch_process_files(detect, args.path, args.out_path, reso=args.reso, body_only=args.body_only)
