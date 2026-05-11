from __future__ import annotations
from typing import Literal
import torch
import torch.nn as nn
ProjectorType = Literal["linear", "mlp"]
class LinearProjector(nn.Module):
    """最简单的线性投影模块。"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
class MLPProjector(nn.Module):
    """多层感知机投影模块。"""
    def __init__(self, in_dim: int, out_dim: int, depth: int = 2):
        super().__init__()
        layers = [nn.Linear(in_dim, out_dim), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(out_dim, out_dim), nn.GELU()]
        layers.pop()  # 去掉最后一个 GELU，保持线性输出
        self.net = nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
def build_projector(projector_type: ProjectorType, in_dim: int, out_dim: int, depth: int = 2) -> nn.Module:
    if projector_type == "linear":
        return LinearProjector(in_dim, out_dim)
    elif projector_type == "mlp":
        return MLPProjector(in_dim, out_dim, depth=depth)
    else:
        raise ValueError(f"未知的 projector 类型: {projector_type}")