import kiui

kiui.seed_everything(42, True)

import random
kiui.seed_everything(42, True)
a = random.random()
kiui.seed_everything(42, True)
b = random.random()
assert a == b

import numpy as np
kiui.seed_everything(42, True)
a = np.random.randn(10)
kiui.seed_everything(42, True)
b = np.random.randn(10)
assert np.allclose(a, b)

import torch
kiui.seed_everything(42, True)
a = torch.randn(10)
kiui.seed_everything(42, True)
b = torch.randn(10)
assert torch.allclose(a, b)