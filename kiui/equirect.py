import torch
import numpy as np
import torch.nn.functional as F

def tonemap_hdr_to_ldr(
    image: np.ndarray,
    percentile: float = 90.0,
    max_mapping: float = 0.8,
    gamma: float = 2.0,
    clip: bool = True,
) -> np.ndarray:
    """
    Tonemap an HDR image to an LDR image.

    Args:
        image: HDR image, [0, inf], float32
        percentile: use this percentile of brightness for exposure normalization, 
            ignoring extreme highlights (e.g., sun, specular reflections).
        max_mapping: target brightness after normalization, <1.0 leaves headroom 
            for soft clipping instead of harsh cutoff.
        gamma: gamma correction exponent, compensates for human perception being 
            logarithmic (boosts darks, compresses brights).
        clip: clip the image to [0, 1]

    Returns:
        LDR image, [0, 1], float32
    """
    image = image.astype(np.float32)
    # robust exposure normalization
    image = max_mapping * image / (np.percentile(image, percentile) + 1e-10)
    if clip:
        image = image.clip(0, 1)
    if gamma != 1.0:
        image = image ** (1 / gamma)
    return image # ldr image, [0, 1]
    

def render_pinhole(
    pano: torch.Tensor,
    # pinhole camera parameters
    height: int,
    width: int,
    vfov: torch.Tensor,
    # panorama camera orientation
    roll: torch.Tensor,
    pitch: torch.Tensor,
    yaw: torch.Tensor,
) -> torch.Tensor:
    """
    Render pinhole images from a panorama given the camera parameters.
    We use CUDA and F.grid_sample for efficient rendering.

    Args:
        pano: [3, Hp, Wp], float tensor, panorama image.
        height: int, image height
        width: int, image width
        vfov: [B,], float tensor, vertical field of view in radians
        roll: [B,], float tensor, roll angle in radians
        pitch: [B,], float tensor, pitch angle in radians
        yaw: [B,], float tensor, yaw angle in radians

    Returns:
        images: [B, 3, H, W], float tensor, pinhole images rendered from the panorama.
    """
    device = pano.device
    dtype = pano.dtype
    B = vfov.shape[0]
    
    # ensure all tensors have the same dtype and device
    vfov = vfov.to(device=device, dtype=dtype)
    roll = roll.to(device=device, dtype=dtype)
    pitch = pitch.to(device=device, dtype=dtype)
    yaw = yaw.to(device=device, dtype=dtype)
    
    # compute focal length from vfov
    fy = height / (2 * torch.tan(vfov / 2))  # [B,]
    
    # create pixel grid, centered at (0, 0)
    y = torch.arange(height, device=device, dtype=dtype) - (height - 1) / 2  # [H,]
    x = torch.arange(width, device=device, dtype=dtype) - (width - 1) / 2  # [W,]
    yy, xx = torch.meshgrid(y, x, indexing='ij')  # [H, W]
    
    # compute ray directions in camera space (z forward, y down, x right)
    # shape: [B, H, W, 3]
    dirs = torch.stack([
        xx.unsqueeze(0).expand(B, -1, -1) / fy[:, None, None],  # x / f
        yy.unsqueeze(0).expand(B, -1, -1) / fy[:, None, None],  # y / f
        torch.ones(B, height, width, device=device, dtype=dtype),  # z = 1
    ], dim=-1)  # [B, H, W, 3]
    
    # normalize ray directions
    dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)  # [B, H, W, 3]
    
    # build rotation matrix from roll, pitch, yaw
    # roll: rotation around z-axis (forward)
    # pitch: rotation around x-axis (right)
    # yaw: rotation around y-axis (up)
    cos_r, sin_r = torch.cos(roll), torch.sin(roll)
    cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
    
    # Rz (roll)
    Rz = torch.zeros(B, 3, 3, device=device, dtype=dtype)
    Rz[:, 0, 0] = cos_r
    Rz[:, 0, 1] = -sin_r
    Rz[:, 1, 0] = sin_r
    Rz[:, 1, 1] = cos_r
    Rz[:, 2, 2] = 1
    
    # Rx (pitch)
    Rx = torch.zeros(B, 3, 3, device=device, dtype=dtype)
    Rx[:, 0, 0] = 1
    Rx[:, 1, 1] = cos_p
    Rx[:, 1, 2] = -sin_p
    Rx[:, 2, 1] = sin_p
    Rx[:, 2, 2] = cos_p
    
    # Ry (yaw)
    Ry = torch.zeros(B, 3, 3, device=device, dtype=dtype)
    Ry[:, 0, 0] = cos_y
    Ry[:, 0, 2] = sin_y
    Ry[:, 1, 1] = 1
    Ry[:, 2, 0] = -sin_y
    Ry[:, 2, 2] = cos_y
    
    # combined rotation: R = Ry @ Rx @ Rz (yaw, then pitch, then roll)
    R = Ry @ Rx @ Rz  # [B, 3, 3]
    
    # apply rotation to ray directions
    dirs = torch.einsum('bij,bhwj->bhwi', R, dirs)  # [B, H, W, 3]
    
    # convert to spherical coordinates (longitude, latitude)
    # longitude (phi): angle in xz plane from z-axis, range [-pi, pi]
    # latitude (theta): angle from xz plane, range [-pi/2, pi/2]
    lon = torch.atan2(dirs[..., 0], dirs[..., 2])  # [B, H, W]
    lat = torch.asin(dirs[..., 1].clamp(-1, 1))  # [B, H, W]
    
    # convert to panorama UV coordinates, range [-1, 1] for grid_sample
    u = lon / np.pi  # [-1, 1]
    v = lat / (np.pi / 2)  # [-1, 1]
    
    # stack to grid format [B, H, W, 2]
    grid = torch.stack([u, v], dim=-1)
    
    # sample from panorama
    pano_expanded = pano.unsqueeze(0).expand(B, -1, -1, -1)  # [B, 3, Hp, Wp]
    images = F.grid_sample(
        pano_expanded, grid, mode='bilinear', padding_mode='border', align_corners=True
    )  # [B, 3, H, W]
    
    return images



def main():
    import argparse
    import kiui 
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="path to equirectangular image (exr, hdr, png, jpg, ...)")
    parser.add_argument('--height', type=int, default=512, help="image height")
    parser.add_argument('--width', type=int, default=512, help="image width")
    parser.add_argument('--vfov', type=float, default=60.0, help="vertical field of view in degrees")
    parser.add_argument('--roll', type=float, default=0.0, help="roll angle in degrees")
    parser.add_argument('--pitch', type=float, default=0.0, help="pitch angle in degrees")
    parser.add_argument('--yaw', type=float, default=0.0, help="yaw angle in degrees")
    parser.add_argument('--output', type=str, default='./results', help="output directory")
    args = parser.parse_args()

    equirect = kiui.read_image(args.path, mode="float", order="RGB")
    kiui.lo(equirect)
    
    if args.path.endswith('.exr'):
        equirect = tonemap_hdr_to_ldr(equirect)
        kiui.write_image(os.path.join(args.output, f'{os.path.basename(args.path).split(".")[0]}_ldr.jpg'), equirect)
        print(f"[INFO] tonemapped to LDR:")
        kiui.lo(equirect)
    
    equirect = torch.from_numpy(equirect).permute(2, 0, 1)  # [3, Hp, Wp]
    height = args.height
    width = args.width
    vfov = torch.tensor([np.deg2rad(args.vfov)])
    roll = torch.tensor([np.deg2rad(args.roll)])
    pitch = torch.tensor([np.deg2rad(args.pitch)])
    yaw = torch.tensor([np.deg2rad(args.yaw)])
    
    images = render_pinhole(equirect, height, width, vfov, roll, pitch, yaw)
    print(f"[INFO] rendered pinhole images:")
    kiui.lo(images)

    os.makedirs(args.output, exist_ok=True)
    # convert from CHW to HWC for write_image
    image_out = images[0].permute(1, 2, 0)  # [H, W, 3]
    kiui.write_image(os.path.join(args.output, f'vfov{args.vfov:.2f}_roll{args.roll:.2f}_pitch{args.pitch:.2f}_yaw{args.yaw:.2f}.jpg'), image_out)


if __name__ == '__main__':
    main()
