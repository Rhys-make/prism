import torch
import torch.nn as nn

class AsymmetricDemergingProjector(nn.Module):
    def __init__(self, in_dim=1024, llm_dim=4096, expand_ratio=4):
        """
        初始化非对称重构投影器
        in_dim: 边缘端 ViT 的输出维度 (如 CLIP ViT-Large 为 1024)
        llm_dim: 云端 LLM 的输入维度 (如 LLaMA-7B 为 4096)
        expand_ratio: 序列膨胀倍数 (如 25 个 Token 变 100 个，设为 4)
        """
        super().__init__()
        self.expand_ratio = expand_ratio
        self.in_dim = in_dim
        self.llm_dim = llm_dim
        
        # 1. 联合升维与特征膨胀层
        # 将原始维度直接投射到 (LLM维度 * 膨胀倍数) 的超大空间
        # 相当于给每个未来的新 Token 都预留了特征槽位
        self.up_proj = nn.Linear(in_dim, llm_dim * expand_ratio)
        
        # 2. 激活函数
        # 提供非线性能力，这是让网络学会“语义解纠缠”的关键
        self.act = nn.GELU()
        
        # 3. 特征平滑层
        # 让拆解后的特征在各自的槽位里进行混合与微调，更好地贴合 LLM 的输入分布
        self.smooth = nn.Linear(llm_dim * expand_ratio, llm_dim * expand_ratio)

    def forward(self, x):
        """
        前向传播
        输入 x 形状: [B, N, in_dim]  (例如: [1, 25, 1024])
        输出形状: [B, N * expand_ratio, llm_dim] (例如: [1, 100, 4096])
        """
        B, N, _ = x.shape
        
        # Step 1 & 2 & 3: 特征空间的映射与非线性处理
        # 形状变化: [B, 25, 1024] -> [B, 25, 16384] (注: 4096 * 4)
        x = self.up_proj(x)
        x = self.act(x)
        x = self.smooth(x)
        
        # --- 核心空间折叠魔法 (Reshape) ---
        
        # Step 4: 将巨大的隐藏层维度拆开，分离出“倍数”维度
        # 形状变化: [B, 25, 16384] -> [B, 25, 4, 4096]
        x = x.view(B, N, self.expand_ratio, self.llm_dim)
        
        # Step 5: 融合前两个维度，将原本并行的特征转化为序列长度
        # 形状变化: [B, 25, 4, 4096] -> [B, 100, 4096]
        x = x.contiguous().view(B, N * self.expand_ratio, self.llm_dim)
        
        return x

