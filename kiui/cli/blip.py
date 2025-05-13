import cv2
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

class BLIP2():
    def __init__(self, device='cuda'):
        self.device = device
        from transformers import AutoProcessor, Blip2ForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to(device)

    @torch.no_grad()
    def __call__(self, image):
        image = Image.fromarray(image)
        inputs = self.processor(image, return_tensors="pt").to(self.device, torch.float16)

        generated_ids = self.model.generate(**inputs, max_new_tokens=20)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        return generated_text

# reusable lazy session
SESSION = None

@torch.no_grad()
def detect(img, **kwargs):
    # img: np.ndarray, (h, w, 3), uint8, RGB

    global SESSION
    if SESSION is None:
        SESSION = BLIP2()

    if img.shape[-1] > 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    res = SESSION(img)
    
    print(res)
    return res


if __name__ == "__main__":
    import argparse
    from kiui.utils import batch_process_files

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default=None)
    parser.add_argument("out_path", type=str, default=None)
    args = parser.parse_args()

    batch_process_files(detect, args.path, args.out_path)
