from typing import Type
import torch
from torch import nn, Tensor


class MLPBlock(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 mlp_dim: int,
                 activation: Type[nn.Module] = nn.GELU
                 ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.act = activation()
        self.fc2 = nn.Linear(mlp_dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))
    

class LayerNorm2d(nn.Module):
    def __init__(self,
                 num_chanels: int,
                 eps: float = 1e-6
                 ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_chanels))
        self.bias = nn.Parameter(torch.zeros(num_chanels))
        self.eps = eps
    
    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(1, keepdim=True)
        std = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(std + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x