import torch
import torch.nn as nn

class SemanticResampler(nn.Module):
    def __init__(self, in_dim=1024, llm_dim=2048, num_queries=128, num_heads=8):
        """
        Prism-VLM 核心云端解码器
        :param in_dim: 边缘端传来的特征维度 (比如 1024)
        :param llm_dim: 云端 TinyLlama 的隐层维度 (2048)
        :param num_queries: 固定输出的 Token 数量 (M=128)
        """
        super().__init__()
        self.num_queries = num_queries
        self.llm_dim = llm_dim
        
        # 1. 云端侦探：可学习的 Query 向量 [1, 128, 2048]
        self.queries = nn.Parameter(torch.randn(1, num_queries, llm_dim))
        
        # 2. 维度投影：先把边缘端 1024 维拉升到 2048 维，方便做 Attention
        self.kv_proj = nn.Linear(in_dim, llm_dim)
        
        # 3. 交叉注意力机制 (核心)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=llm_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        # 4. FFN (前馈网络) 用于增强语义表达
        self.ln_1 = nn.LayerNorm(llm_dim)
        self.ln_2 = nn.LayerNorm(llm_dim)
        self.ffn = nn.Sequential(
            nn.Linear(llm_dim, llm_dim * 4),
            nn.GELU(),
            nn.Linear(llm_dim * 4, llm_dim)
        )

    def forward(self, edge_features):
        """
        :param edge_features: 形状 [Batch, N, 1024] (N是波动的)
        :return: 形状 [Batch, 128, 2048] (绝对固定)
        """
        B = edge_features.shape[0]
        
        # 对齐维度作为 Key 和 Value
        kv = self.kv_proj(edge_features)  # [B, N, 2048]
        
        # 扩展 Query 以匹配 Batch Size
        q = self.queries.expand(B, -1, -1)  # [B, 128, 2048]
        
        # 执行 Cross-Attention
        # attn_out: [B, 128, 2048]
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
        
        # 残差连接 + LayerNorm
        out = self.ln_1(q + attn_out)
        out = out + self.ffn(self.ln_2(out))
        
        return out

