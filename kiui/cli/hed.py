# This is an improved version and model of HED edge detection with Apache License, Version 2.0.
# Please use this implementation in your products
# This implementation may produce slightly different results from Saining Xie's official implementations,
# but it generates smoother edges and is more suitable for ControlNet as well as other image-to-image translations.
# Different from official models and other implementations, this is an RGB-input model (rather than BGR)
# and in this way it works better for gradio's RGB protocol

import cv2
import torch
import numpy as np

from kiui.utils import load_file_from_url


def nms(x, t, s):
    x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), s)

    f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)

    y = np.zeros_like(x)

    for f in [f1, f2, f3, f4]:
        np.putmask(y, cv2.dilate(x, kernel=f) == x, x)

    z = np.zeros_like(y, dtype=np.uint8)
    z[y > t] = 255
    return z


def safe_step(x, step=2):
    y = x.astype(np.float32) * float(step + 1)
    y = y.astype(np.int32).astype(np.float32) / float(step)
    return y


class DoubleConvBlock(torch.nn.Module):
    def __init__(self, input_channel, output_channel, layer_number):
        super().__init__()
        self.convs = torch.nn.Sequential()
        self.convs.append(
            torch.nn.Conv2d(
                in_channels=input_channel,
                out_channels=output_channel,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
            )
        )
        for i in range(1, layer_number):
            self.convs.append(
                torch.nn.Conv2d(
                    in_channels=output_channel,
                    out_channels=output_channel,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=1,
                )
            )
        self.projection = torch.nn.Conv2d(
            in_channels=output_channel,
            out_channels=1,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
        )

    def __call__(self, x, down_sampling=False):
        h = x
        if down_sampling:
            h = torch.nn.functional.max_pool2d(h, kernel_size=(2, 2), stride=(2, 2))
        for conv in self.convs:
            h = conv(h)
            h = torch.nn.functional.relu(h)
        return h, self.projection(h)


class ControlNetHED_Apache2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.Parameter(torch.zeros(size=(1, 3, 1, 1)))
        self.block1 = DoubleConvBlock(
            input_channel=3, output_channel=64, layer_number=2
        )
        self.block2 = DoubleConvBlock(
            input_channel=64, output_channel=128, layer_number=2
        )
        self.block3 = DoubleConvBlock(
            input_channel=128, output_channel=256, layer_number=3
        )
        self.block4 = DoubleConvBlock(
            input_channel=256, output_channel=512, layer_number=3
        )
        self.block5 = DoubleConvBlock(
            input_channel=512, output_channel=512, layer_number=3
        )

    def __call__(self, x):
        h = x - self.norm
        h, projection1 = self.block1(h)
        h, projection2 = self.block2(h, down_sampling=True)
        h, projection3 = self.block3(h, down_sampling=True)
        h, projection4 = self.block4(h, down_sampling=True)
        h, projection5 = self.block5(h, down_sampling=True)
        return projection1, projection2, projection3, projection4, projection5


class HEDdetector:
    def __init__(self):
        remote_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetHED.pth"
        self.netNetwork = ControlNetHED_Apache2().float().cuda().eval()
        self.netNetwork.load_state_dict(
            torch.load(load_file_from_url(remote_model_path))
        )

    @torch.no_grad()
    def __call__(self, input_image, safe=False):
        assert input_image.ndim == 3
        H, W, C = input_image.shape

        image_hed = torch.from_numpy(input_image.copy()).float().cuda()
        image_hed = (
            image_hed.permute(2, 0, 1).contiguous().unsqueeze(0)
        )  # rearrange(image_hed, 'h w c -> 1 c h w')
        edges = self.netNetwork(image_hed)
        edges = [e.detach().cpu().numpy().astype(np.float32)[0, 0] for e in edges]
        edges = [cv2.resize(e, (W, H), interpolation=cv2.INTER_LINEAR) for e in edges]
        edges = np.stack(edges, axis=2)
        edge = 1 / (1 + np.exp(-np.mean(edges, axis=2).astype(np.float64)))
        if safe:
            edge = safe_step(edge)
        edge = (edge * 255.0).clip(0, 255).astype(np.uint8)

        edge = nms(edge, 127, 3.0)
        edge = cv2.GaussianBlur(edge, (0, 0), 3.0)
        edge[edge > 4] = 255
        edge[edge < 255] = 0

        # use white bg, black fg
        edge = 255 - edge

        return edge


# reusable lazy session
SESSION = None

@torch.no_grad()
def detect(img, **kwargs):
    # img: np.ndarray, (h, w, 3), uint8, RGB

    global SESSION
    if SESSION is None:
        SESSION = HEDdetector()

    res = SESSION(img, safe=True)

    return res


if __name__ == "__main__":
    import argparse
    from kiui.utils import batch_process_files

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default=None)
    parser.add_argument("out_path", type=str, default=None)
    args = parser.parse_args()

    batch_process_files(detect, args.path, args.out_path)
