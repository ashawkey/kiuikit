from .openpose import OpenposeDetector
import cv2

# reusable lazy session
SESSION = None


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
    from .utils import batch_process_image

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default=None)
    parser.add_argument("out_path", type=str, default=None)
    parser.add_argument("--reso", type=int, default=None)
    parser.add_argument("--body_only", action="store_true")
    args = parser.parse_args()

    batch_process_image(detect, args.path, args.out_path, reso=args.reso, body_only=args.body_only)
