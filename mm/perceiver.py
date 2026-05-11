from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim, bias=False),
            nn.GELU(),
            nn.Linear(inner_dim, dim, bias=False),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
class PerceiverAttention(nn.Module):
    def __init__(self, *, dim: int, dim_head: int = 64, heads: int = 8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
    def forward(self, x: torch.Tensor, latents: torch.Tensor, media_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Cross-attention from latents to media tokens.
        Args:
            x: [B, N, D] media tokens
            latents: [B, M, D] latent queries
            media_padding_mask: [B, N], 1 for valid tokens, 0 for padding tokens
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)
        if media_padding_mask is not None:
            if media_padding_mask.ndim != 2 or media_padding_mask.shape[:2] != x.shape[:2]:
                raise ValueError(
                    f"media_padding_mask shape must be [B, N] and match x; got mask={tuple(media_padding_mask.shape)}, x={tuple(x.shape)}"
                )
            x = x * media_padding_mask.to(dtype=x.dtype).unsqueeze(-1)
        h = self.heads
        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q = q.reshape(q.shape[0], q.shape[1], h, -1).transpose(1, 2)
        k = k.reshape(k.shape[0], k.shape[1], h, -1).transpose(1, 2)
        v = v.reshape(v.shape[0], v.shape[1], h, -1).transpose(1, 2)
        q = q * self.scale
        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = out.transpose(1, 2).contiguous().reshape(out.shape[0], out.shape[2], -1)
        return self.to_out(out)
class PerceiverResampler(nn.Module):
    """Flamingo 风格的 PerceiverResampler。
    输入:
        x: [B, N, D] 或 [B, T, N, D]
        mask: [B, N] 或 [B, T, N]
    输出:
        latents: [B, M, D] 或 [B, T, M, D]
    """
    def __init__(
        self,
        *,
        dim: int,
        depth: int = 6,
        dim_head: int = 64,
        heads: int = 8,
        num_latents: int = 64,
        max_num_media: Optional[int] = None,
        max_num_frames: Optional[int] = None,
        ff_mult: int = 4,
        out_dim: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim or dim
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.randn(num_latents, dim) * 0.02)
        self.frame_embs = (
            nn.Parameter(torch.randn(max_num_frames, dim) * 0.02) if max_num_frames is not None else None
        )
        self.media_time_embs = (
            nn.Parameter(torch.randn(max_num_media, 1, dim) * 0.02) if max_num_media is not None else None
        )
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )
        self.norm = nn.LayerNorm(dim)
        self.to_out = nn.Identity() if self.out_dim == dim else nn.Linear(dim, self.out_dim, bias=False)
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Resample media tokens into fixed number of latent tokens.
        Args:
            x: [B, N, D] or [B, T, N, D]
            attention_mask: [B, N] or [B, T, N], 1 for valid tokens, 0 for padding tokens
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)  # [B, 1, N, D]
            if attention_mask is not None and attention_mask.ndim == 2:
                attention_mask = attention_mask.unsqueeze(1)
        elif x.ndim != 4:
            raise ValueError(f"Expected x to be [B, N, D] or [B, T, N, D], got {tuple(x.shape)}")
        b, T, N, d = x.shape
        if attention_mask is not None and attention_mask.shape[:3] != (b, T, N):
            raise ValueError(
                f"attention_mask shape must match x prefix dimensions. got mask={tuple(attention_mask.shape)}, x={tuple(x.shape)}"
            )
        if self.frame_embs is not None:
            if x.shape[2] > self.frame_embs.shape[0]:
                raise ValueError(
                    f"Input frames ({x.shape[2]}) exceed max_num_frames ({self.frame_embs.shape[0]})."
                )
            frame_embs = self.frame_embs[: x.shape[2]].view(1, 1, x.shape[2], 1, d)
            x = x + frame_embs
        x = x.view(b, T, N, d)
        if self.media_time_embs is not None:
            if T > self.media_time_embs.shape[0]:
                raise ValueError(
                    f"Input media count ({T}) exceed max_num_media ({self.media_time_embs.shape[0]})."
                )
            x = x + self.media_time_embs[:T].view(1, T, 1, d)
        x = x.reshape(b * T, N, d)
        if attention_mask is not None:
            attention_mask = attention_mask.reshape(b * T, N)
        latents = self.latents.unsqueeze(0).unsqueeze(0).expand(b, T, -1, -1).reshape(b * T, self.num_latents, d)
        for attn, ff in self.layers:
            latents = attn(x, latents, media_padding_mask=attention_mask) + latents
            latents = ff(latents) + latents
        latents = self.norm(latents)
        latents = latents.reshape(b, T, self.num_latents, d)
        latents = self.to_out(latents)
        if latents.shape[1] == 1:
            return latents[:, 0]
        return latents