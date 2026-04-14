import torch
import time
import random
import torch.nn as nn
from transformers import AutoModelForCausalLM, CLIPVisionModel

# 导入你的核心模块
from cloud.adp import SemanticResampler
from edge.cna import CNA_Allocator
from edge.tome.patch.clip import apply_patch_clip

# ---------------------------------------------------------
# 1. 模拟网络传输的物理学公式
# ---------------------------------------------------------
def simulate_network_latency(num_tokens, bandwidth_mbps):
    # 1个Token = 1024维 * 16位(bfloat16) = 16384 bits = 0.016384 Mb
    data_size_mb = num_tokens * 1024 * 16 / 1_000_000
    
    # 纯物理传输时间 (毫秒)
    transmission_time_ms = (data_size_mb / bandwidth_mbps) * 1000
    
    # 加入真实世界的基线延迟 (Ping) 和网络抖动 (Jitter)
    base_ping_ms = 20.0 
    jitter_ms = random.uniform(0, 10.0) 
    
    return transmission_time_ms + base_ping_ms + jitter_ms

# ---------------------------------------------------------
# 2. 官方 LLaVA 原装 MLP
# ---------------------------------------------------------
class LLaVAMLPProjector(nn.Module):
    def __init__(self, in_dim=1024, out_dim=4096):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))
    def forward(self, x): return self.mlp(x)

# ---------------------------------------------------------
# 3. 核心测速函数：端到端流水线
# ---------------------------------------------------------
def measure_e2e_pipeline(model_name, bandwidth, clip_model, allocator, projector, llm, is_baseline=False):
    device = "cuda"
    num_runs = 10 # 跑10次取平均
    
    # 假图片输入 [1, 3, 336, 336]
    dummy_pixel_values = torch.randn(1, 3, 336, 336, dtype=torch.bfloat16, device=device)
    dummy_text_embeds = torch.randn(1, 50, 4096, dtype=torch.bfloat16, device=device)

    # 变量初始化
    avg_t_edge, avg_t_net, avg_t_cloud = 0, 0, 0
    final_token_len = 0

    for i in range(num_runs + 5): # 前5次作为预热
        torch.cuda.empty_cache()
        
        # ================= 阶段 A: 边缘端计算 =================
        # ================= 阶段 A: 边缘端计算 =================
        start_edge = time.time()
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            if is_baseline:
                # 官方 Baseline: 不开启压缩，硬算 576 Token
                clip_model.r = 0  # ✅ 核心修复：用 0 告诉 ToMe "不丢弃任何 Token"
                edge_outputs = clip_model(dummy_pixel_values).last_hidden_state[:, 1:, :]
            else:
                # Prism 架构: CNA 动态分配 + ToMe 融合加速
                r_list = allocator.generate_r_list(h_norm=0.8, bandwidth_mbps=bandwidth)
                clip_model.r = r_list
                edge_outputs = clip_model(dummy_pixel_values).last_hidden_state[:, 1:, :]
        torch.cuda.synchronize()
        t_edge = (time.time() - start_edge) * 1000

        # ================= 阶段 B: 网络传输 =================
        transmitted_tokens = edge_outputs.shape[1]
        t_net = simulate_network_latency(transmitted_tokens, bandwidth)
        final_token_len = transmitted_tokens

        # ================= 阶段 C: 云端推理 =================
        start_cloud = time.time()
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            visual_embeds = projector(edge_outputs).to(torch.bfloat16)
            inputs_embeds = torch.cat([visual_embeds, dummy_text_embeds], dim=1)
            _ = llm(inputs_embeds=inputs_embeds)
        torch.cuda.synchronize()
        t_cloud = (time.time() - start_cloud) * 1000

        # 累加正式运行的时间
        if i >= 5:
            avg_t_edge += t_edge
            avg_t_net += t_net
            avg_t_cloud += t_cloud

    avg_t_edge /= num_runs
    avg_t_net /= num_runs
    avg_t_cloud /= num_runs
    t_total = avg_t_edge + avg_t_net + avg_t_cloud

    print(f"| {model_name:<10} | {bandwidth:>6.1f} Mbps | {final_token_len:>9} | {avg_t_edge:>9.1f} ms | {avg_t_net:>9.1f} ms | {avg_t_cloud:>9.1f} ms | {t_total:>10.1f} ms |")

# ---------------------------------------------------------
# 4. 主程序
# ---------------------------------------------------------
def main():
    device = "cuda"
    llm_path = "./models/vicuna-7b" 
    clip_path = "./models/clip-vit-large-patch14-336"
    
    print("🧠 1. 载入云端大模型 Vicuna-7B...")
    llm = AutoModelForCausalLM.from_pretrained(llm_path, dtype=torch.bfloat16, local_files_only=True).to(device).eval()

    print("📸 2. 载入边缘侧 CLIP 并注入 ToMe...")
    clip_model = CLIPVisionModel.from_pretrained(clip_path, dtype=torch.bfloat16, local_files_only=True).to(device).eval()
    apply_patch_clip(clip_model)
    allocator = CNA_Allocator(num_layers=24, total_tokens=576, max_drop=450)

    print("🌉 3. 初始化投影器...")
    mlp_projector = LLaVAMLPProjector(in_dim=1024, out_dim=4096).to(device, dtype=torch.bfloat16).eval()
    adp_projector = SemanticResampler(in_dim=1024, llm_dim=4096, num_queries=128).to(device, dtype=torch.bfloat16).eval()

    # 测试环境：极差(0.5) / 较差(2.0) / 良好(5.0) / 极速(10.0)
    test_bandwidths = [0.5, 2.0, 5.0, 10.0]
    
    print("\n🚀 4. 开始端到端系统级 Benchmark (物理模拟传输)")
    print("-" * 105)
    print(f"| {'架构类型':<10} | {'模拟带宽':<11} | {'传输Token':<9} | {'边缘侧耗时':<12} | {'网络传输耗时':<12} | {'云端推理耗时':<12} | {'端到端总延迟':<13} |")
    print("-" * 105)
    
    # 测 LLaVA Baseline (永远传 576个Token)
    for bw in test_bandwidths:
        measure_e2e_pipeline("LLaVA-Base", bw, clip_model, allocator, mlp_projector, llm, is_baseline=True)
        
    print("-" * 105)
    
    # 测 Prism-VLM (根据带宽动态压缩)
    for bw in test_bandwidths:
        measure_e2e_pipeline("Prism-ADP", bw, clip_model, allocator, adp_projector, llm, is_baseline=False)
        
    print("-" * 105)

if __name__ == "__main__":
    main()