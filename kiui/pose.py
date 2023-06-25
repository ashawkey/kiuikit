from .openpose import OpenposeDetector

# reusable lazy session
SESSION = None


def detect(img, body_only=False, **kwargs):
    # img: np.ndarray, (h, w, 3), uint8, RGB

    global SESSION
    if SESSION is None:
        SESSION = OpenposeDetector()

    res = SESSION(img, hand_and_face=not body_only)

    return res


if __name__ == "__main__":
    import argparse
    from .utils import batch_process_image

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default=None)
    parser.add_argument("out_path", type=str, default=None)
    parser.add_argument("--body_only", action="store_true")
    args = parser.parse_args()

    batch_process_image(detect, args.path, args.out_path, body_only=args.body_only)
