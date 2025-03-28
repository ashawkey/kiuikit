import numpy as np
import torch


class FlowMatchingScheduler:
    def __init__(self, num_train_timesteps: int = 1000, shift: float = 1):
        # set timesteps
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift

        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        sigmas = timesteps / num_train_timesteps
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.sigmas = sigmas  # 1 --> 0
        self.timesteps = sigmas * num_train_timesteps  # num_train_timesteps --> 1

    # set device
    def to(self, device):
        self.sigmas = self.sigmas.to(device=device)
        self.timesteps = self.timesteps.to(device=device)

    # add random noise to latent during training
    def add_noise(self, latent: torch.Tensor, logit_mean: float = 1.0, logit_std: float = 1.0):
        # latent: [B, ...]
        # timesteps: [B]
        # return: [B, ...] noisy_latent, [B, ...] noise, [B] timesteps

        # logit-normal sampling
        u = torch.normal(mean=logit_mean, std=logit_std, size=(latent.shape[0],), device=self.sigmas.device)
        u = torch.nn.functional.sigmoid(u)

        step_indices = (u * self.num_train_timesteps).long()
        timesteps = self.timesteps[step_indices]

        sigmas = self.sigmas[step_indices].flatten()

        while len(sigmas.shape) < latent.ndim:
            sigmas = sigmas.unsqueeze(-1)

        noise = torch.randn_like(latent)
        noisy_latent = (1.0 - sigmas) * latent + sigmas * noise

        return noisy_latent, noise, timesteps
