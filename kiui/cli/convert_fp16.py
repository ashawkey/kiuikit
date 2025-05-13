import torch
from safetensors.torch import load_file, save_file
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="model path")
    opt = parser.parse_args()

    # load ckpt
    postfix = opt.path.split('.')[-1]
    if postfix == 'pth':
        model = torch.load(opt.path)
    elif postfix == 'safetensors':
        model = load_file(opt.path)
    else:
        raise ValueError(f"Unknown file format: {postfix}")
    
    # convert to fp16
    for k, v in model.items():
        model[k] = v.half()
    
    # save ckpt
    save_path = opt.path.split('.')[0] + '_fp16.' + postfix
    if postfix == 'pth':
        torch.save(model, save_path)
    elif postfix == 'safetensors':
        save_file(model, save_path)
    
