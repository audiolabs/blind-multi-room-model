from typing import List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange


class NoiseQuantization(nn.Module):
    """Adding quantization noise to latent vectors"""

    def __init__(
        self,
        levels: Union[int, None],
        downstream: bool = False,
    ) -> None:
        super().__init__()
        self.levels = levels
        self.downstream = downstream

    def quantize(self, x: Tensor) -> Tensor:
        # quantize to levels levels
        x = (self.levels - 1) * ((x + 1) / 2)
        x = 2 * (torch.round(x) / (self.levels - 1)) - 1
        return x

    def forward(self, x: Tensor) -> Tensor:
        # squeeze latent betwen [-1, 1]
        x = F.tanh(x)

        if self.levels is not None:
            assert self.levels >= 2, "Quantization to 2 levels at least"

            if self.downstream:
                # flag to apply quant during downstream training
                x = self.quantize(x)

            else:
                if self.training:
                    x = (self.levels - 1) * (((x + 1) / 2))
                    noise = torch.rand_like(x) - 0.5
                    x = x + noise
                    x = (x / (self.levels - 1)) * 2 - 1
                    x = torch.clamp(x, min=-1, max=1)
                else:
                    # quantize during inference
                    x = self.quantize(x)
        return x
