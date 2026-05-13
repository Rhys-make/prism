from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def _factor_grid(num_queries: int) -> Tuple[int, int]:
    h = int(math.sqrt(num_queries))
    while h > 1 and num_queries % h != 0:
        h -= 1
    return h, num_queries // h


def _build_grid_coords(height: int, width: int) -> torch.Tensor:
    ys = (torch.arange(height, dtype=torch.float32) + 0.5) / float(height)
    xs = (torch.arange(width, dtype=torch.float32) + 0.5) / float(width)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1).reshape(height * width, 2)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        hidden = dim * mult
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatialCrossAttention(nn.Module):
    """Cross-attention with a distance bias from fixed query positions to token centers."""

    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner = dim_head * heads
        self.norm_q = nn.LayerNorm(dim)
        self.norm_x = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_kv = nn.Linear(dim, inner * 2, bias=False)
        self.to_out = nn.Linear(inner, dim, bias=False)
        self.size_bias = nn.Linear(1, heads, bias=False)
        self.spatial_bias_log_scale = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        queries: torch.Tensor,
        tokens: torch.Tensor,
        query_centers: torch.Tensor,
        token_centers: torch.Tensor,
        token_sizes: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        local_topk: Optional[int] = None,
    ) -> torch.Tensor:
        bsz, num_queries, _ = queries.shape
        num_tokens = tokens.shape[1]
        heads = self.heads

        q = self.to_q(self.norm_q(queries))
        k, v = self.to_kv(self.norm_x(tokens)).chunk(2, dim=-1)

        q = q.reshape(bsz, num_queries, heads, -1).transpose(1, 2)
        k = k.reshape(bsz, num_tokens, heads, -1).transpose(1, 2)
        v = v.reshape(bsz, num_tokens, heads, -1).transpose(1, 2)
        q = q * self.scale

        sim = torch.einsum("b h m d, b h n d -> b h m n", q, k)

        dist2 = (query_centers[None, :, None, :] - token_centers[:, None, :, :]).pow(2).sum(dim=-1)
        spatial_scale = torch.nn.functional.softplus(self.spatial_bias_log_scale)
        sim = sim - spatial_scale * dist2[:, None, :, :]

        size_bias = self.size_bias(token_sizes.unsqueeze(-1)).transpose(1, 2)
        sim = sim + size_bias[:, :, None, :]

        valid_mask = None
        if attention_mask is not None:
            valid_mask = attention_mask.bool()
            sim = sim.masked_fill(~valid_mask[:, None, None, :], torch.finfo(sim.dtype).min)

        if local_topk is not None and local_topk > 0 and num_tokens > 0:
            k_top = min(local_topk, num_tokens)
            local_scores = -dist2
            if valid_mask is not None:
                local_scores = local_scores.masked_fill(~valid_mask[:, None, :], torch.finfo(local_scores.dtype).min)
            top_idx = local_scores.topk(k=k_top, dim=-1).indices
            local_mask = torch.zeros((bsz, num_queries, num_tokens), dtype=torch.bool, device=tokens.device)
            local_mask.scatter_(dim=-1, index=top_idx, value=True)
            if valid_mask is not None:
                local_mask = local_mask & valid_mask[:, None, :]
            sim = sim.masked_fill(~local_mask[:, None, :, :], torch.finfo(sim.dtype).min)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = torch.einsum("b h m n, b h n d -> b h m d", attn, v)
        out = out.transpose(1, 2).contiguous().reshape(bsz, num_queries, -1)
        return self.to_out(out)


class SpatialMixingBlock(nn.Module):
    def __init__(self, dim: int, grid_size: Tuple[int, int]):
        super().__init__()
        self.grid_size = grid_size
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.pointwise = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, num_queries, dim = x.shape
        h, w = self.grid_size
        if num_queries != h * w:
            return x
        residual = x
        x = self.norm(x).transpose(1, 2).reshape(bsz, dim, h, w)
        x = self.pointwise(self.act(self.depthwise(x)))
        x = x.reshape(bsz, dim, num_queries).transpose(1, 2)
        return residual + x


class SourceAwareTokenPackerLite(nn.Module):
    """ToMe-source-aware coarse-to-fine visual projector.

    Inputs:
        tokens: [B, N, D]
        token_centers: [B, N, 2], normalized x/y centers in [0, 1]
        token_sizes: [B, N], normalized covered patch counts in [0, 1]
    Output:
        [B, M, out_dim], where M is num_queries.
    """

    def __init__(
        self,
        *,
        in_dim: int,
        out_dim: int,
        num_queries: int = 128,
        depth: int = 2,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        local_topk: int = 8,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_queries = num_queries
        self.local_topk = local_topk
        self.query_grid = _factor_grid(num_queries)

        query_centers = _build_grid_coords(*self.query_grid)
        self.register_buffer("query_centers", query_centers, persistent=False)

        self.query_tokens = nn.Parameter(torch.randn(num_queries, in_dim) * 0.02)
        self.query_pos = nn.Sequential(
            nn.Linear(2, in_dim),
            nn.SiLU(),
            nn.Linear(in_dim, in_dim),
        )
        self.token_pos = nn.Sequential(
            nn.Linear(2, in_dim),
            nn.SiLU(),
            nn.Linear(in_dim, in_dim),
        )
        self.token_size = nn.Sequential(
            nn.Linear(1, in_dim),
            nn.SiLU(),
            nn.Linear(in_dim, in_dim),
        )

        self.layers = nn.ModuleList()
        for _ in range(max(1, depth)):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "global_attn": SpatialCrossAttention(in_dim, dim_head=dim_head, heads=heads),
                        "local_attn": SpatialCrossAttention(in_dim, dim_head=dim_head, heads=heads),
                        "mix": SpatialMixingBlock(in_dim, self.query_grid),
                        "ff": FeedForward(in_dim, mult=ff_mult),
                    }
                )
            )

        self.norm = nn.LayerNorm(in_dim)
        self.to_out = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim, bias=False)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_centers: Optional[torch.Tensor] = None,
        token_sizes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"tokens should be [B, N, D], got {tuple(tokens.shape)}")

        bsz, num_tokens, _ = tokens.shape
        dtype = tokens.dtype
        device = tokens.device

        if attention_mask is None:
            attention_mask = torch.ones((bsz, num_tokens), dtype=torch.long, device=device)
        if token_centers is None:
            token_centers = torch.zeros((bsz, num_tokens, 2), dtype=dtype, device=device)
        if token_sizes is None:
            token_sizes = attention_mask.to(dtype=dtype) / max(1, num_tokens)

        token_centers = token_centers.to(device=device, dtype=dtype)
        token_sizes = token_sizes.to(device=device, dtype=dtype)
        attention_mask = attention_mask.to(device=device)

        tokens = tokens + self.token_pos(token_centers) + self.token_size(token_sizes.unsqueeze(-1))

        query_centers = self.query_centers.to(device=device, dtype=dtype)
        queries = self.query_tokens.to(dtype=dtype).unsqueeze(0).expand(bsz, -1, -1)
        queries = queries + self.query_pos(query_centers).unsqueeze(0)

        for layer in self.layers:
            queries = queries + layer["global_attn"](
                queries,
                tokens,
                query_centers=query_centers,
                token_centers=token_centers,
                token_sizes=token_sizes,
                attention_mask=attention_mask,
            )
            queries = queries + layer["local_attn"](
                queries,
                tokens,
                query_centers=query_centers,
                token_centers=token_centers,
                token_sizes=token_sizes,
                attention_mask=attention_mask,
                local_topk=self.local_topk,
            )
            queries = layer["mix"](queries)
            queries = queries + layer["ff"](queries)

        queries = self.norm(queries)
        return self.to_out(queries)
