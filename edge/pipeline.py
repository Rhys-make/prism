import torch
from PIL import Image
from torchvision import transforms
import os
import time
# 引入项目模块
from edge.cna import CNA_Allocator
# 将之前错误的导入替换为这句
from edge.tome.patch.clip import apply_patch_clip # 切换为 CLIP 专用补丁
# 建议使用 transformers 库来加载 CLIP
from transformers import CLIPVisionModel, CLIPImageProcessor

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ 当前计算硬件: {device.upper()}")

    print("="*65)
    print("🏭 [边缘端生产线] Prism-VLM (CLIP-Local 版) 启动")
    print("="*65)

    # ==========================================
    # 1. 动态带宽大脑配置 (适配 LLaVA-1.5 标准)
    # ==========================================
    # LLaVA 使用的 CLIP-ViT-L/14 在 336 分辨率下产生 (336/14)^2 = 576 个 Token
    total_tokens = 576 
    num_layers = 24
    allocator = CNA_Allocator(num_layers=num_layers, total_tokens=total_tokens, max_drop=450)
    
    current_bandwidth = 1.5 
    current_h_norm = 0.8
    r_list = allocator.generate_r_list(h_norm=current_h_norm, bandwidth_mbps=current_bandwidth)
    
    print(f"   - 📋 CNA 决策完成，准备在 {num_layers} 层中动态执行裁减")

    # ==========================================
    # 2. 从本地加载 CLIP 并注入 ToMe 补丁
    # ==========================================
    model_path = "./clip_model" # 👈 指向你本地的文件夹
    print(f"⚙️ 2. 正在从本地加载 CLIP 视觉底座: {model_path}")
    
    # 加载模型主体
    # 如果是 CPU，千万别用 float16，老老实实用 float32；如果是 GPU，就用 float16 狂飙
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    model = CLIPVisionModel.from_pretrained(model_path).to(device, dtype=dtype)

    # 注入 CLIP 专用补丁
    # 注意：CLIP 的结构与标准 ViT 略有不同，tome.patch.clip 会处理这些差异
    apply_patch_clip(model)
    # 将 CNA 的 r 指令下发到每一层
    # 对于 Transformers 库加载的 CLIP，层通常在 model.vision_model.encoder.layers
    model.r = r_list

    # ==========================================
    # 3. 图像预处理 (适配 LLaVA 的 336x336)
    # ==========================================
    print("🖼️ 3. 执行 LLaVA 标准预处理 (336x336)...")
    # 建议直接使用配套的 CLIPImageProcessor 保证归一化参数绝对对齐
    image_processor = CLIPImageProcessor.from_pretrained(model_path)
    
    img_path = "pic/dog.png" 
    if os.path.exists(img_path):
        img = Image.open(img_path).convert('RGB')
        inputs = image_processor(images=img, return_tensors="pt")
        input_pixel_values = inputs.pixel_values.to(device, dtype=dtype)
    else:
        print("   - ⚠️ 使用随机噪声模拟图像输入")
        input_pixel_values = torch.randn(1, 3, 336, 336, device=device, dtype=dtype)

    # ==========================================
    # 4. 特征提取与重构
    # ==========================================
  # ==========================================
    # 4. 执行推理与极限压缩 (带严谨测速)
    # ==========================================
    print("🗜️ 4. 正在提取压缩特征...")
    
    with torch.no_grad():
        # ♨️ 第一步：预热 (Warm-up)
        # 深度学习框架(尤其是用了 GPU)第一次运行时要分配显存和编译算子，会特别慢。
        # 我们先让它跑一次空转，把管道跑通。
        print("   - ♨️ 正在进行硬件预热...")
        _ = model(input_pixel_values)
        
        # ⏱️ 第二步：精准计时
        print("   - ⚡ 开始测试纯推理延迟...")
        start_time = time.perf_counter() # 使用高精度时钟
        
        outputs = model(input_pixel_values)
        
        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000

    features = outputs.last_hidden_state
    visual_features = features[:, 1:, :] 
    print(f"   - 🗜️ 压缩后张量形状: {visual_features.shape}")
    print(f"   - 🚀 [论文级数据] 边缘端纯推理延迟: {inference_time_ms:.2f} ms")
    
    payload_name = "edge_payload.pt"
    torch.save(visual_features.cpu(), payload_name)
if __name__ == "__main__":
    main()