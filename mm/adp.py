from __future__ import annotations

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = dim * expansion
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdaptiveQueryBlock(nn.Module):
    """单个交叉注意力块，让可学习 query 从压缩后的视觉 token 中读取信息。"""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.ffn = FeedForward(dim, expansion=4, dropout=dropout)
        self.norm_ffn = nn.LayerNorm(dim)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, kv_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        qn = self.norm_q(q)
        kvn = self.norm_kv(kv)
        attn_out, _ = self.attn(
            query=qn,
            key=kvn,
            value=kvn,
            key_padding_mask=kv_padding_mask,
            need_weights=False,
        )
        q = q + self.gate(q) * attn_out
        q = q + self.ffn(self.norm_ffn(q))
        return q


class SemanticResampler(nn.Module):
    """压缩感知的自适应语义重采样器。

    设计目标：
    - 适配 ToMe 压缩后不定长的视觉 token
    - 输出固定数量的 query token 供 LLM 使用
    - 用可学习 query 主动抽取显著语义
    - 使用少量 refinement block 提升表达能力
    """

    def __init__(
        self,
        in_dim: int,
        hidden_size: int,
        num_queries: int = 128,
        num_heads: int = 8,
        depth: int = 2,
        dropout: float = 0.0,
        use_input_skip: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_size = hidden_size
        self.num_queries = num_queries
        self.use_input_skip = use_input_skip

        self.input_proj = nn.Linear(in_dim, hidden_size)
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, hidden_size) * 0.02)
        self.pos_mlp = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.blocks = nn.ModuleList([
            AdaptiveQueryBlock(hidden_size, num_heads=num_heads, dropout=dropout)
            for _ in range(depth)
        ])

        self.out_norm = nn.LayerNorm(hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.out_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
        )

    def _build_positional_encoding(self, n_tokens: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """构造连续位置编码，使模块可以适配任意长度的 token 输入。"""
        if n_tokens <= 1:
            pos = torch.zeros(1, 1, device=device, dtype=dtype)
        else:
            pos = torch.linspace(0, 1, steps=n_tokens, device=device, dtype=dtype).unsqueeze(-1)
        return self.pos_mlp(pos).unsqueeze(0)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, N, D_in]
        if x.ndim != 3:
            raise ValueError(f"输入 x 应为 [B, N, D]，但得到的形状是 {tuple(x.shape)}")

        bsz, n_tokens, _ = x.shape
        x = self.input_proj(x)

        # 给 token 加入轻量连续位置提示，帮助 query 区分顺序信息。
        x = x + self._build_positional_encoding(n_tokens, x.device, x.dtype)

        # 如果传入了 attention_mask，则将 padding 位置归零。
        # 这样可以让后续的 mean pooling 和 cross-attention 都忽略 padding token。
        kv_padding_mask = None
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                if attention_mask.shape[1] != n_tokens:
                    raise ValueError(
                        f"attention_mask 长度与 token 数不匹配：mask={tuple(attention_mask.shape)} tokens={n_tokens}"
                    )
                valid = attention_mask.to(dtype=x.dtype).unsqueeze(-1)
                x = x * valid
                kv_padding_mask = attention_mask == 0
            else:
                raise ValueError(f"attention_mask 应为 [B, N]，但得到 {tuple(attention_mask.shape)}")

        # 从固定学习得到的 query token 开始。
        q = self.query_tokens.expand(bsz, -1, -1).contiguous()

        # 先让 query 直接读取视觉 token，再经过多层 refinement 进行细化。
        for block in self.blocks:
            q = block(q, x, kv_padding_mask=kv_padding_mask)

        # 可选的输入残差信息：把有效视觉 token 的均值信息加回来，作为全局语义保底。
        if self.use_input_skip:
            if attention_mask is not None:
                denom = attention_mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=x.dtype)
                pooled = (x * attention_mask.unsqueeze(-1).to(dtype=x.dtype)).sum(dim=1, keepdim=True) / denom.unsqueeze(-1)
            else:
                pooled = x.mean(dim=1, keepdim=True)
            q = q + pooled

        # 输出投影与门控，增加一点表达能力。
        q = self.out_norm(q)
        q = q + self.out_gate(q) * self.out_proj(q)
        return q
