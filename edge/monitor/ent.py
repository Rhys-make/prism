import torch
import torch.nn as nn
import numpy as np

class EntropyEstimator:
    """视觉特征冗余度评估器 (基于 Patch Embedding 的全局余弦相似度)"""
    
    def __init__(self, vit_model):
        self.model = vit_model
        self.model.eval()

    @torch.no_grad()
    def evaluate_image(self, image_tensor):
        """
        极速截断式前向传播：测算图像的“原生合并潜力”
        """
        # 1. 仅过 Patch Embedding (无位置编码污染)
        # [1, 196, 768]
        patches = self.model.patch_embed(image_tensor) 

        # 2. 计算 196 个图块的归一化特征
        patches_norm = torch.nn.functional.normalize(patches, p=2, dim=-1)
        
        # 3. 矩阵乘法算出 196x196 的相似度矩阵
        # sim_matrix 里面的每一个值，代表第 i 个块和第 j 个块有多像 (0~1)
        sim_matrix = torch.bmm(patches_norm, patches_norm.transpose(1, 2)) 

        # 4. 剔除对角线 (自己和自己永远是 1.0，必须排除以免拉高平均值)
        B, N, _ = sim_matrix.shape
        mask = torch.eye(N, dtype=torch.bool, device=sim_matrix.device).unsqueeze(0)
        sim_matrix.masked_fill_(mask, 0.0)

        # 5. 计算全局平均相似度 (除以 196 * 195 个有效配对)
        avg_sim = sim_matrix.sum() / (B * N * (N - 1))
        
        # 此时的 avg_sim 完美契合 H_norm 的定义：
        # 天空图 (全相似) -> avg_sim 逼近 0.9 -> H_norm 高 -> 激进压缩
        # 客厅图 (全不同) -> avg_sim 逼近 0.1 -> H_norm 低 -> 保守压缩
        # 白底特写图 (部分相似) -> avg_sim 在 0.4 左右 -> 中度压缩
        
        print(f"   [底层调试] 图像原生合并潜力 (平均相似度): {avg_sim.item():.4f}")
        
        # 稍微拉伸一下对比度，确保能充满 0.1 ~ 0.9 的区间反馈给大脑
        h_norm_final = np.clip(avg_sim.item() * 1.5 - 0.1, 0.05, 0.95)
        
        return h_norm_final