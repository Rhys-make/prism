import math
import numpy as np

# =====================================================================
# 边缘侧决策中枢：CNA-Allocator (支持任意层数与动态高斯自适应)
# =====================================================================

class CNA_Allocator:
    """
    结合图像冗余度 (H) 与网络带宽 (B)，动态生成任意层数 ViT 的 Token 裁减清单
    """
    def __init__(self, num_layers=24, total_tokens=576, max_drop=450):
        self.num_layers = num_layers
        self.total_tokens = total_tokens
        self.max_drop = max_drop  
        
        # ---------------------------------------------------
        # 🌟 核心升级：动态计算自适应截断式高斯分布
        # ---------------------------------------------------
        x = np.arange(num_layers)
        
        # 1. 动态寻找绝对中心 (24层就是 11.5，12层就是 5.5)
        mu = (num_layers - 1) / 2.0 
        
        # 2. 动态缩放方差 (确保钟形曲线覆盖中间的大部分层)
        sigma = num_layers / 6.0 
        
        weights = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        
        # 3. 动态计算保护区跨度 (按总层数的 1/6 设定绝对安全区)
        # 例如：12层保护前后各 2 层；24层保护前后各 4 层
        protect_len = max(1, num_layers // 6)
        
        weights[:protect_len] = 0.0
        weights[-protect_len:] = 0.0
        
        # 4. 强制归一化，确保所有权重相加严格等于 1.0
        self.weights = weights / np.sum(weights)

    def generate_r_list(self, h_norm: float, bandwidth_mbps: float) -> list:
        # ==========================================
        # 1. 宏观定编 (Macro-Budgeting) - 修复数学饱和溢出
        # ==========================================
        # 优化后的平滑超参数
        B_ref = 2.0  # 参照带宽降为 2Mbps，作为敏感分水岭
        alpha = 1.2  # 降低激进倍率
        
        # 新公式：确保正常网速下不会轻易触发极限裁员
        tanh_input = alpha * h_norm / ((bandwidth_mbps / B_ref) + 0.1)
        
        # 根据动态上限算出本次总共需要砍掉多少 Token
        R_target = int(self.max_drop * math.tanh(tanh_input))
        
        # ==========================================
        # 2. 微观派发 (Micro-Allocation)
        # ==========================================
        r_list_np = np.round(R_target * self.weights).astype(int)
        
        current_sum = np.sum(r_list_np)
        diff = R_target - current_sum
        
        if diff != 0:
            max_idx = np.argmax(self.weights)
            r_list_np[max_idx] += diff 
            r_list_np[max_idx] = max(0, r_list_np[max_idx]) 

        return r_list_np.tolist()

