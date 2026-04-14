import torch
import time
import torch.nn as nn
from transformers import AutoModelForCausalLM

# 导入你的 ADP 核心模块
from cloud.adp import SemanticResampler

# ---------------------------------------------------------
# 1. 官方 LLaVA 原装 MLP 桥梁
# ---------------------------------------------------------
class LLaVAMLPProjector(nn.Module):
    def __init__(self, in_dim=1024, out_dim=4096):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )
    def forward(self, x):
        return self.mlp(x)

# ---------------------------------------------------------
# 2. 核心测速函数
# ---------------------------------------------------------
def measure_performance(model_name, projector, llm, edge_token_len, num_runs=20, num_warmup=5):
    device = "cuda"
    
    # 模拟文本 Prompt (比如 "USER: Please describe this image. \nASSISTANT:")
    # 假设文本长度固定为 50 个 Token
    text_len = 50
    dummy_text_embeds = torch.randn(1, text_len, 4096, dtype=torch.bfloat16, device=device)
    
    # 模拟边缘端传过来的视觉特征
    dummy_visual_features = torch.randn(1, edge_token_len, 1024, dtype=torch.bfloat16, device=device)
    
    # 阶段 A：投影器处理
    # 注意：MLP 输出是 [1, edge_token_len, 4096]
    # ADP 输出永远是 [1, 128, 4096]
    visual_embeds = projector(dummy_visual_features).to(torch.bfloat16)
    
    # 拼接最终喂给大模型的 Embeddings
    inputs_embeds = torch.cat([visual_embeds, dummy_text_embeds], dim=1)
    
    # 阶段 B：Warm-up (热身)，唤醒 GPU 的高频模式
    for _ in range(num_warmup):
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _ = llm(inputs_embeds=inputs_embeds)
    torch.cuda.synchronize()
    
    # 阶段 C：正式测速与显存监控
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # 仅仅执行一次前向传播，模拟首字延迟 (TTFT) 的巨量矩阵乘法开销
            _ = llm(inputs_embeds=inputs_embeds)
        torch.cuda.synchronize() # 必须等待 GPU 算完
    end_time = time.time()
    
    avg_latency = (end_time - start_time) / num_runs * 1000 # 转换为毫秒 (ms)
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3) # 转换为 GB
    
    # 打印格式化结果
    print(f"| {model_name:<10} | {edge_token_len:<15} | {visual_embeds.shape[1]:<16} | {avg_latency:>8.2f} ms | {peak_mem:>7.2f} GB |")
    return avg_latency, peak_mem

# ---------------------------------------------------------
# 3. 主程序
# ---------------------------------------------------------
def main():
    device = "cuda"
    llm_path = "./models/vicuna-7b" 
    
    print("🧠 1. 载入云端大模型 Vicuna-7B (BFloat16, 冻结)...")
    llm = AutoModelForCausalLM.from_pretrained(
        llm_path, 
        dtype=torch.bfloat16, 
        local_files_only=True
    ).to(device)
    llm.eval()

    print("🌉 2. 初始化投影器 (无需加载真实权重，随机即可，不影响测速)...")
    mlp_projector = LLaVAMLPProjector(in_dim=1024, out_dim=4096).to(device, dtype=torch.bfloat16).eval()
    adp_projector = SemanticResampler(in_dim=1024, llm_dim=4096, num_queries=128).to(device, dtype=torch.bfloat16).eval()

    # 模拟不同的网络环境导致边缘端截断/融合的 Token 数量
    # 576(满血), 400(良), 260(差), 130(极差)
    test_token_lengths = [130, 260, 400, 576]
    
    print("\n🚀 3. 开始端云协同系统级 Benchmark...")
    print("-" * 75)
    print(f"| {'架构类型':<10} | {'边缘端接收 Token数':<10} | {'最终喂入云端 Token数':<10} | {'云端首字延迟':<11} | {'峰值显存':<10} |")
    print("-" * 75)
    
    # 先测 MLP
    for length in test_token_lengths:
        measure_performance("LLaVA-MLP", mlp_projector, llm, length)
        
    print("-" * 75)
    
    # 再测 ADP
    for length in test_token_lengths:
        measure_performance("Prism-ADP", adp_projector, llm, length)
        
    print("-" * 75)
    print("🎉 Benchmark 测试完成！你可以直接把这些数据填进你的论文图表里了！")

if __name__ == "__main__":
    main()