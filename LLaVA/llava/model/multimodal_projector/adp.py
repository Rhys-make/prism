import torch
import torch.nn as nn

class SemanticResampler(nn.Module):
    # 【修改点】：llm_dim 从 2048 改为 4096，迎合 7B 级别大模型
    def __init__(self, in_dim=1024, llm_dim=4096, num_queries=128, num_heads=8):
        super().__init__()
        self.num_queries = num_queries
        self.llm_dim = llm_dim
        
        # 剩下的代码完全不用动，因为所有线性层和 Attention 都会自动读取这个 4096
        self.queries = nn.Parameter(torch.randn(1, num_queries, llm_dim))
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

