import torch
import numpy as np


# reusable lazy session
SESSION = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def process(image, **kwargs):
    # image: np array uint8

    global SESSION
    if SESSION is None:
        # NOTE: requires timm==0.6.11 ...
        SESSION = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True, trust_repo=True).to(DEVICE)
    
    image = torch.from_numpy(image.astype(np.float32) / 255).to(DEVICE)
    image = image.permute(2, 0, 1).contiguous().unsqueeze(0) # [1, 3, H, W]

    # rgba to rgb
    if image.shape[1] == 4:
        image = image[:, :3] * image[:, 3:] + (1 - image[:, 3:])

    depth = SESSION.infer(image).detach().cpu().numpy()[0, 0] # numpy array [H, W]

    # ref: https://github.com/lllyasviel/ControlNet-v1-1-nightly/blob/main/annotator/zoe/__init__.py
    vmin = np.percentile(depth, 2)
    vmax = np.percentile(depth, 85)

    depth = (depth - vmin) / (vmax - vmin + 1e-20)
    depth = 1.0 - depth # actually disparity !!!

    depth = (depth * 255.0).clip(0, 255).astype(np.uint8)

    return depth


if __name__ == "__main__":
    import argparse
    from kiui.utils import batch_process_files

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default=None)
    parser.add_argument("out_path", type=str, default=None)
    args = parser.parse_args()

    batch_process_files(
        process,
        args.path,
        args.out_path,
        in_format=[".jpg", ".jpeg", ".png"],
        image_mode='uint8',
        out_format=".jpg",
    )
