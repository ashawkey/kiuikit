import torch
import numpy as np

from transformers import DPTForDepthEstimation, DPTFeatureExtractor


# reusable lazy session
SESSION = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def process(image, **kwargs):

    global SESSION
    if SESSION is None:
        model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(DEVICE)
        feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
        SESSION = (model, feature_extractor)
    
    model, feature_extractor = SESSION

    if image.shape[-1] == 4:
        image = image[..., :3] * image[..., 3:] + (1 - image[..., 3:])

    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(DEVICE)

    # disparity!    
    depth = model(pixel_values=pixel_values).predicted_depth

    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1),
        size=image.shape[:2][::-1],
        mode="bicubic",
        align_corners=False,
    ).detach().cpu().numpy()[0, 0] # numpy array [H, W]

    vmin = depth.min()
    vmax = depth.max()

    depth = (depth - vmin) / (vmax - vmin + 1e-20)
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
