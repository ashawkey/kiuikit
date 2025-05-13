''' Improved Aesthetic Predictor V2
ref: https://github.com/christophschuhmann/improved-aesthetic-predictor/
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPModel, CLIPTokenizer, CLIPProcessor

from kiui.utils import load_file_from_url

class CLIP:
    def __init__(self, device, model_name='openai/clip-vit-large-patch14'):

        self.device = device

        self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
    
    def encode_image(self, image):
        # image: PIL, np.ndarray uint8 [H, W, 3] or [B, H, W, 3]

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        image_features = self.clip_model.get_image_features(pixel_values=pixel_values)

        image_features = image_features / image_features.norm(dim=-1,keepdim=True)  # normalize features

        return image_features


class MLP(nn.Module):
    def __init__(self, input_size=768):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            #nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)


class AES:
    def __init__(self, device):
        super().__init__()

        self.device = device
        self.clip = CLIP(device, model_name='openai/clip-vit-large-patch14')
        self.mlp = MLP(input_size=768).to(device)

        # load pretrained checkpoint
        remote_model_path = 'https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth'
        self.mlp.load_state_dict(torch.load(load_file_from_url(remote_model_path)))


    def __call__(self, x):
        # x: np.ndarray, (h, w, 3) / (b, h, w, 3), uint8, RGB
        # return: y: aesthetic score

        features = self.clip.encode_image(x)
        y = self.mlp(features)
            
        return y


if __name__ == '__main__':
    import argparse
    import cv2
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, help="path to image")

    opt = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    aes = AES(device)

    image = cv2.imread(opt.image, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image.shape[-1] == 4:
        image = image.astype(np.float32) / 255
        image = image[..., :3] * image[..., 3:] + (1 - image[..., 3:])
        image = (image * 255).astype(np.uint8)
    elif image.shape[-1] == 1:
        image = np.concatenate([image] * 3, axis=-1)
    
    aes_score = aes(image[None, :])
    print(f'Aesthetic score: {aes_score} <-- {opt.image}')
