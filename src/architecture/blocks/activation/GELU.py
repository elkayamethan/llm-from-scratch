import torch
import math
from torch import nn, Tensor

_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)

class GELU(nn.Module):
    """Approximated GELU activation function"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return 0.5 * x * (
            1 + torch.tanh(
                _SQRT_2_OVER_PI * (x + 0.044715 * torch.pow(x, 3))
            )
        )
