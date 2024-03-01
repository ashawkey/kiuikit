import torch

class _TruncExp(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))

trunc_exp = _TruncExp.apply
