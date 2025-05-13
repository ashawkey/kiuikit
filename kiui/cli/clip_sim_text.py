# eval the clip-similarity for an input image and a geneated mesh
import cv2
import torch
import numpy as np
from torchvision import transforms as T
from transformers import CLIPModel, CLIPTokenizer, CLIPProcessor

import kiui
from kiui.render import GUI

class CLIP:
    def __init__(self, device, model_name='openai/clip-vit-large-patch14'):

        self.device = device

        self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
    
    def encode_image(self, image):
        # image: PIL, np.ndarray uint8 [H, W, 3]

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        image_features = self.clip_model.get_image_features(pixel_values=pixel_values)

        image_features = image_features / image_features.norm(dim=-1,keepdim=True)  # normalize features

        return image_features

    def encode_text(self, text):
        # text: str

        inputs = self.processor(text=[text], padding=True, return_tensors="pt")
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)
        text_features = self.clip_model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1,keepdim=True)  # normalize features

        return text_features


if __name__ == '__main__':
    import os
    import tqdm
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str, help="prompt for mesh")
    parser.add_argument('mesh', type=str, help="path to mesh (obj, glb, ...)")
    parser.add_argument('--pbr', action='store_true', help="enable PBR material")
    parser.add_argument('--envmap', type=str, default=None, help="hdr env map path for pbr")
    parser.add_argument('--front_dir', type=str, default='+z', help="mesh front-facing dir")
    parser.add_argument('--mode', default='albedo', type=str, choices=['lambertian', 'albedo', 'normal', 'depth', 'pbr'], help="rendering mode")
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--ssaa', type=float, default=1, help="super-sampling anti-aliasing ratio")
    parser.add_argument('--radius', type=float, default=3, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument("--force_cuda_rast", action='store_true', help="force to use RasterizeCudaContext.")
    parser.add_argument('--elevation', type=int, default=0, help="rendering elevation")
    parser.add_argument('--num_azimuth', type=int, default=8, help="number of images to render from different azimuths")
    
    opt = parser.parse_args()
    opt.wogui = True

    # clip = CLIP('cuda')
    clip = CLIP('cuda', model_name='laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')
    
    gui = GUI(opt)

    with torch.no_grad():
        ref_features = clip.encode_text(opt.prompt)

    # render from random views and evaluate similarity
    results = []

    elevation = [opt.elevation,]
    azimuth = np.linspace(0, 360, opt.num_azimuth, dtype=np.int32, endpoint=False)
    for ele in tqdm.tqdm(elevation):
        for azi in tqdm.tqdm(azimuth):
            gui.cam.from_angle(ele, azi)
            gui.need_update = True
            gui.step()
            image = (gui.render_buffer * 255).astype(np.uint8)
            with torch.no_grad():
                cur_features = clip.encode_image(image)
        
            similarity = (ref_features * cur_features).sum(dim=-1).mean().item()

            results.append(similarity)
    
    avg_similarity = np.mean(results)
    print(avg_similarity)

            
