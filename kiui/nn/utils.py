import math
import torch

from kiui.typing import *

# cosine lr scheduler with warm up
# ref: https://github.com/huggingface/diffusers/blob/main/src/diffusers/optimization.py#L154C1-L185C54
def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of periods of the cosine function in a schedule (the default is to just decrease from the max
            value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        # linear warmup (0 --> 1)
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        # cosine annealing (1 -- half cosine period --> 0)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def count_parameters(model: torch.nn.Module):
    total, trainable = 0, 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    return total, trainable


def tolerant_load(model: torch.nn.Module, ckpt: dict, verbose: bool=False):
    """loading params from ckpt into model with matching shape (warn instead of error for mismatched shapes compared to torch.load unstrict mode)

    Args:
        model (torch.nn.Module): model.
        ckpt (Dict): state_dict to load.
        verbose (bool): whether to log mismatching params. Defaults to False.
    """
    state_dict = model.state_dict()
    seen = {k: False  for k in state_dict}
    for k, v in ckpt.items():
        if k in state_dict: 
            if state_dict[k].shape == v.shape:
                state_dict[k].copy_(v)
            else:
                if verbose: print(f'[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.')
            seen[k] = True
        else:
            if verbose: print(f'[WARN] unexpected param {k} in ckpt: {v.shape}')
            
    if verbose: 
        for k, v in seen.items():
            if not v:
                print(f'[WARN] missing param {k} in model: {state_dict[k].shape}')