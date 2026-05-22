from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _factor_grid(num_queries: int) -> Tuple[int, int]:
    """Choose a compact 2D grid whose product equals num_queries."""
    if num_queries <= 0:
        raise ValueError(f"num_queries must be positive, got {num_queries}")

    h = int(math.sqrt(num_queries))
    while h > 1 and num_queries % h != 0:
        h -= 1
    return h, num_queries // h


def _build_grid_coords(height: int, width: int) -> torch.Tensor:
    """Build normalized x/y centers for a regular query grid."""
    ys = (torch.arange(height, dtype=torch.float32) + 0.5) / float(height)
    xs = (torch.arange(width, dtype=torch.float32) + 0.5) / float(width)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1).reshape(height * width, 2)


def _metadata_mlp(input_dim: int, dim: int) -> nn.Sequential:
    """Small MLP for source metadata.

    The visual embedding dimension is 4096 for LLaVA-1.5. A full 4096 -> 4096
    metadata MLP would be unnecessarily large, so metadata is lifted through a
    bottleneck before being added to visual tokens.
    """
    hidden = min(512, max(64, dim // 8))
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.SiLU(),
        nn.Linear(hidden, dim),
    )


class FeedForward(nn.Module):
    """Transformer-style channel refinement for reconstructed tokens."""

    def __init__(self, dim: int, mult: int = 2, dropout: float = 0.0, max_hidden_dim: int = 2048):
        super().__init__()

        # For LLaVA-1.5, dim=4096. A standard 4096 -> 8192 -> 4096 FFN is
        # expensive for a cloud-side compensation module, so we cap the hidden
        # width. This keeps the module compact while still adding nonlinearity.
        hidden = min(int(dim * mult), int(max_hidden_dim))
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SourceAwareTokenEncoder(nn.Module):
    """Inject ToMe source metadata into projected visual embeddings.

    Input tokens are expected to be LLaVA-space embeddings produced by the
    frozen LLaVA projector: [B, N, 4096]. Source metadata tells the reconstructor
    where each compressed token came from and how much original patch area it
    covers.
    """

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.center_embed = _metadata_mlp(2, dim)
        self.size_embed = _metadata_mlp(1, dim)
        self.ratio_embed = _metadata_mlp(1, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        token_centers: torch.Tensor,
        token_sizes: torch.Tensor,
        retain_ratio: torch.Tensor,
    ) -> torch.Tensor:
        if visual_tokens.ndim != 3:
            raise ValueError(f"visual_tokens must be [B, N, D], got {tuple(visual_tokens.shape)}")
        if token_centers.shape[:2] != visual_tokens.shape[:2] or token_centers.shape[-1] != 2:
            raise ValueError(
                "token_centers must be [B, N, 2] and match visual_tokens; "
                f"got centers={tuple(token_centers.shape)}, tokens={tuple(visual_tokens.shape)}"
            )
        if token_sizes.shape != visual_tokens.shape[:2]:
            raise ValueError(
                "token_sizes must be [B, N] and match visual_tokens; "
                f"got sizes={tuple(token_sizes.shape)}, tokens={tuple(visual_tokens.shape)}"
            )

        dtype = visual_tokens.dtype
        device = visual_tokens.device
        token_centers = token_centers.to(device=device, dtype=dtype)
        token_sizes = token_sizes.to(device=device, dtype=dtype)
        retain_ratio = retain_ratio.to(device=device, dtype=dtype)

        if retain_ratio.ndim == 1:
            retain_ratio = retain_ratio[:, None]
        if retain_ratio.ndim != 2 or retain_ratio.shape[0] != visual_tokens.shape[0] or retain_ratio.shape[1] != 1:
            raise ValueError(f"retain_ratio must be [B] or [B, 1], got {tuple(retain_ratio.shape)}")

        source_bias = (
            self.center_embed(token_centers)
            + self.size_embed(token_sizes.unsqueeze(-1))
            + self.ratio_embed(retain_ratio).unsqueeze(1)
        )
        return self.norm(visual_tokens + self.dropout(source_bias))


class SpatialCrossAttention(nn.Module):
    """Cross-attention from compact reconstruction queries to source-aware tokens.

    Locality is enforced only by an optional source-map radius/top-k mask. We
    intentionally do not add an extra distance-based spatial bias to the
    attention logits, so the model can decide the relative importance of local
    candidate tokens from QK similarity.
    """

    def __init__(self, dim: int, dim_head: int = 128, heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner = dim_head * heads

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_kv = nn.Linear(dim, inner * 2, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner, dim, bias=False),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        source_tokens: torch.Tensor,
        query_centers: torch.Tensor,
        token_centers: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        local_topk: Optional[int] = None,
        local_radius: Optional[float] = None,
    ) -> torch.Tensor:
        bsz, num_queries, _ = queries.shape
        num_tokens = source_tokens.shape[1]
        heads = self.heads

        q = self.to_q(self.norm_q(queries))
        k, v = self.to_kv(self.norm_kv(source_tokens)).chunk(2, dim=-1)

        q = q.reshape(bsz, num_queries, heads, -1).transpose(1, 2)
        k = k.reshape(bsz, num_tokens, heads, -1).transpose(1, 2)
        v = v.reshape(bsz, num_tokens, heads, -1).transpose(1, 2)

        sim = torch.einsum("b h m d, b h n d -> b h m n", q * self.scale, k)

        valid_mask = None
        if attention_mask is not None:
            if attention_mask.shape != source_tokens.shape[:2]:
                raise ValueError(
                    "attention_mask must be [B, N] and match source_tokens; "
                    f"got mask={tuple(attention_mask.shape)}, tokens={tuple(source_tokens.shape)}"
                )
            valid_mask = attention_mask.to(device=source_tokens.device).bool()
            if not bool(valid_mask.any(dim=1).all()):
                raise ValueError("Each sample must contain at least one valid compressed visual token.")
            sim = sim.masked_fill(~valid_mask[:, None, None, :], torch.finfo(sim.dtype).min)

        # Optional source-guided radius/top-k mask. This decides which
        # compressed tokens each compact query may read from; within that local
        # candidate set, attention weights are still determined by semantic QK
        # similarity.
        no_candidate_mask = None
        use_topk = local_topk is not None and local_topk > 0
        use_radius = local_radius is not None and local_radius > 0
        if use_topk or use_radius:
            dist2 = (query_centers[None, :, None, :] - token_centers[:, None, :, :]).pow(2).sum(dim=-1)

            candidate_mask = torch.ones((bsz, num_queries, num_tokens), dtype=torch.bool, device=source_tokens.device)
            if valid_mask is not None:
                candidate_mask = candidate_mask & valid_mask[:, None, :]

            if use_radius:
                candidate_mask = candidate_mask & dist2.le(float(local_radius) ** 2)

            if use_topk:
                k_top = min(int(local_topk), num_tokens)
                local_scores = -dist2
                if valid_mask is not None:
                    local_scores = local_scores.masked_fill(
                        ~valid_mask[:, None, :],
                        torch.finfo(local_scores.dtype).min,
                    )
                top_idx = local_scores.topk(k=k_top, dim=-1).indices
                topk_mask = torch.zeros((bsz, num_queries, num_tokens), dtype=torch.bool, device=source_tokens.device)
                topk_mask.scatter_(dim=-1, index=top_idx, value=True)
                candidate_mask = candidate_mask & topk_mask

            # If a query has no token inside its radius, do not pull distant
            # tokens across the image. We temporarily unmask valid tokens only
            # to keep softmax finite, then zero the cross-attention update for
            # those empty local regions below.
            no_candidate_mask = ~candidate_mask.any(dim=-1)
            if valid_mask is not None:
                fallback_mask = valid_mask[:, None, :].expand_as(candidate_mask)
            else:
                fallback_mask = torch.ones_like(candidate_mask)
            safe_mask = torch.where(no_candidate_mask[:, :, None], fallback_mask, candidate_mask)

            sim = sim.masked_fill(~safe_mask[:, None, :, :], torch.finfo(sim.dtype).min)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = torch.einsum("b h m n, b h n d -> b h m d", attn, v)
        out = out.transpose(1, 2).contiguous().reshape(bsz, num_queries, -1)
        if no_candidate_mask is not None:
            out = out.masked_fill(no_candidate_mask[:, :, None], 0.0)
        return self.to_out(out)


class SpatialSmoothBlock(nn.Module):
    """Lightweight smoothness block on the compact semantic grid."""

    def __init__(self, dim: int, grid_size: Tuple[int, int]):
        super().__init__()
        self.grid_size = grid_size
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)

        # Start as an identity mapping. Training can open the gate if smoothing is useful.
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, num_queries, dim = x.shape
        h, w = self.grid_size
        if num_queries != h * w:
            return x

        y = self.norm(x).transpose(1, 2).reshape(bsz, dim, h, w)
        y = self.depthwise(y).reshape(bsz, dim, num_queries).transpose(1, 2)
        return x + self.gate.tanh() * y


class ReconstructionBlock(nn.Module):
    """One reconstruction refinement layer: cross-attention, smooth, then FFN."""

    def __init__(
        self,
        dim: int,
        grid_size: Tuple[int, int],
        dim_head: int = 128,
        heads: int = 8,
        ff_mult: int = 2,
        dropout: float = 0.0,
        local_topk: int = 0,
        local_radius: float = 0.0,
    ):
        super().__init__()
        self.local_topk = local_topk
        self.local_radius = local_radius
        self.cross_attn = SpatialCrossAttention(dim=dim, dim_head=dim_head, heads=heads, dropout=dropout)
        self.smooth = SpatialSmoothBlock(dim=dim, grid_size=grid_size)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout)

    def forward(
        self,
        queries: torch.Tensor,
        source_tokens: torch.Tensor,
        query_centers: torch.Tensor,
        token_centers: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        queries = queries + self.cross_attn(
            queries=queries,
            source_tokens=source_tokens,
            query_centers=query_centers,
            token_centers=token_centers,
            attention_mask=attention_mask,
            local_topk=self.local_topk,
            local_radius=self.local_radius,
        )
        queries = self.smooth(queries)
        queries = queries + self.ff(queries)
        return queries


class SourceGuidedCompactSemanticReconstructor(nn.Module):
    """Source-Guided Compact Semantic Reconstruction module.

    This module is intentionally placed after the frozen LLaVA projector:

        compressed CLIP features [B, N, 1024]
            -> frozen LLaVA projector
            -> compressed visual embeddings [B, N, 4096]
            -> this module
            -> reconstructed semantic tokens [B, K, 4096]

    It does not recover all 576 original patch tokens. Instead, K learnable
    reconstruction queries form a compact semantic grid, e.g. K=144 for 12x12.
    """

    def __init__(
        self,
        dim: int = 4096,
        num_queries: int = 144,
        depth: int = 2,
        dim_head: int = 128,
        heads: int = 8,
        ff_mult: int = 2,
        dropout: float = 0.0,
        local_topk: int = 0,
        local_radius: float = 0.0,
        grid_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_queries = num_queries
        self.grid_size = grid_size or _factor_grid(num_queries)
        if self.grid_size[0] * self.grid_size[1] != num_queries:
            raise ValueError(f"grid_size={self.grid_size} does not match num_queries={num_queries}")

        query_centers = _build_grid_coords(*self.grid_size)
        self.register_buffer("query_centers", query_centers, persistent=False)

        self.source_encoder = SourceAwareTokenEncoder(dim=dim, dropout=dropout)
        self.query_tokens = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.query_pos = _metadata_mlp(2, dim)

        self.layers = nn.ModuleList(
            [
                ReconstructionBlock(
                    dim=dim,
                    grid_size=self.grid_size,
                    dim_head=dim_head,
                    heads=heads,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    local_topk=local_topk,
                    local_radius=local_radius,
                )
                for _ in range(max(1, depth))
            ]
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(
        self,
        visual_embeddings: torch.Tensor,
        token_centers: torch.Tensor,
        token_sizes: torch.Tensor,
        retain_ratio: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Reconstruct compact semantic tokens.

        Args:
            visual_embeddings: [B, N, D], after frozen LLaVA projector.
            token_centers: [B, N, 2], normalized ToMe source centers.
            token_sizes: [B, N], normalized source area per compressed token.
            retain_ratio: [B] or [B, 1], e.g. 0.2 / 0.4 / 0.6 / 0.8.
            attention_mask: [B, N], 1 for valid tokens and 0 for padding.
        """
        if visual_embeddings.ndim != 3:
            raise ValueError(f"visual_embeddings must be [B, N, D], got {tuple(visual_embeddings.shape)}")
        if visual_embeddings.shape[-1] != self.dim:
            raise ValueError(
                f"visual_embeddings last dim is {visual_embeddings.shape[-1]}, but reconstructor dim is {self.dim}. "
                "Call the frozen LLaVA projector before this module."
            )

        bsz, num_tokens, _ = visual_embeddings.shape
        device = visual_embeddings.device
        dtype = visual_embeddings.dtype

        if attention_mask is None:
            attention_mask = torch.ones((bsz, num_tokens), dtype=torch.long, device=device)
        else:
            attention_mask = attention_mask.to(device=device)

        token_centers = token_centers.to(device=device, dtype=dtype)
        token_sizes = token_sizes.to(device=device, dtype=dtype)
        source_tokens = self.source_encoder(
            visual_tokens=visual_embeddings,
            token_centers=token_centers,
            token_sizes=token_sizes,
            retain_ratio=retain_ratio,
        )

        query_centers = self.query_centers.to(device=device, dtype=dtype)
        queries = self.query_tokens.to(dtype=dtype).unsqueeze(0).expand(bsz, -1, -1)
        queries = queries + self.query_pos(query_centers).unsqueeze(0)

        for layer in self.layers:
            queries = layer(
                queries=queries,
                source_tokens=source_tokens,
                query_centers=query_centers,
                token_centers=token_centers,
                attention_mask=attention_mask,
            )

        return self.out_norm(queries)


def pool_teacher_visual_tokens(
    teacher_tokens: torch.Tensor,
    output_grid: Tuple[int, int],
    teacher_grid: Tuple[int, int] = (24, 24),
) -> torch.Tensor:
    """Pool no-ToMe teacher visual tokens to the compact student grid.

    Teacher tokens should be the frozen LLaVA projected visual embeddings:
    [B, 576, 4096] for a 24x24 CLIP grid. For K=144, output_grid=(12, 12),
    this becomes [B, 144, 4096] through average pooling.
    """
    if teacher_tokens.ndim != 3:
        raise ValueError(f"teacher_tokens must be [B, T, D], got {tuple(teacher_tokens.shape)}")

    bsz, num_tokens, dim = teacher_tokens.shape
    th, tw = teacher_grid
    oh, ow = output_grid
    if num_tokens != th * tw:
        raise ValueError(f"teacher token count {num_tokens} does not match teacher_grid={teacher_grid}")

    x = teacher_tokens.reshape(bsz, th, tw, dim).permute(0, 3, 1, 2).contiguous()
    x = F.adaptive_avg_pool2d(x, output_size=(oh, ow))
    return x.permute(0, 2, 3, 1).reshape(bsz, oh * ow, dim).contiguous()


def compact_feature_distillation_loss(
    student_tokens: torch.Tensor,
    teacher_compact_tokens: torch.Tensor,
    mse_weight: float = 1.0,
    cosine_weight: float = 0.1,
) -> torch.Tensor:
    """Feature distillation loss for compact semantic reconstruction."""
    if student_tokens.shape != teacher_compact_tokens.shape:
        raise ValueError(
            "student_tokens and teacher_compact_tokens must have the same shape; "
            f"got student={tuple(student_tokens.shape)}, teacher={tuple(teacher_compact_tokens.shape)}"
        )

    teacher_compact_tokens = teacher_compact_tokens.detach()
    # Compute the reconstruction objective in fp32 even when the model runs in
    # fp16/bf16. This avoids tiny feature differences being rounded away inside
    # the distillation loss, while still allowing the module itself to use the
    # dtype chosen by the training script.
    mse = F.mse_loss(student_tokens.float(), teacher_compact_tokens.float())
    cosine = 1.0 - F.cosine_similarity(student_tokens.float(), teacher_compact_tokens.float(), dim=-1).mean()
    return mse_weight * mse + cosine_weight * cosine
