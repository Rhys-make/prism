import torch
import os
import torch.nn as nn
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPVisionModel, CLIPImageProcessor

from edge.cna import CNA_Allocator
from edge.tome.patch.clip import apply_patch_clip

# ================= 1. 定义官方原装的直肠子 MLP =================
class LLaVAMLPProjector(nn.Module):
    def __init__(self, in_dim=1024, out_dim=4096):
        super().__init__()
        # 官方的 MLP2x 结构：两层线性网络 + GELU 激活
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, edge_features):
        return self.mlp(edge_features)

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    llm_path = "./models/vicuna-7b" 
    clip_path = "./models/clip-vit-large-patch14-336"
    projector_path = "./checkpoints/baseline_mlp_epoch_1.pth" # 👈 刚刚下载的官方权重
    image_path = "./pic/mide.png" 

    print("🧠 1. 载入云端 Vicuna-7B & 边缘侧 CLIP...")
    tokenizer = AutoTokenizer.from_pretrained(llm_path, local_files_only=True, use_fast=False)
    llm = AutoModelForCausalLM.from_pretrained(llm_path, dtype=torch.bfloat16, local_files_only=True).to(device).eval()
    
    processor = CLIPImageProcessor.from_pretrained(clip_path, local_files_only=True)
    clip_model = CLIPVisionModel.from_pretrained(clip_path, dtype=torch.bfloat16, local_files_only=True).to(device).eval()

    print("🧪 2. 注入 ToMe 补丁并激活 CNA...")
    apply_patch_clip(clip_model)
    allocator = CNA_Allocator(num_layers=24, total_tokens=576, max_drop=450)

    # ================= 2. 加载官方权重并“配钥匙” =================
    print("🌉 3. 载入【官方 LLaVA-1.5】原装 MLP 桥梁...")
    baseline_projector = LLaVAMLPProjector(in_dim=1024, out_dim=4096) 
    
    if os.path.exists(projector_path):
        raw_state_dict = torch.load(projector_path, map_location="cpu")
        clean_state_dict = {}
        # 把官方复杂的变量名 (model.mm_projector.x) 替换成我们简单的 (mlp.x)
        for k, v in raw_state_dict.items():
            new_key = k.replace("model.mm_projector.", "mlp.")
            clean_state_dict[new_key] = v
            
        baseline_projector.load_state_dict(clean_state_dict)
        baseline_projector = baseline_projector.to(device, dtype=torch.bfloat16).eval()
        print("✅ 官方 LLaVA 权重无缝加载成功！")
    else:
        print("❌ 找不到官方权重，请确认是否执行了 wget 命令！")
        return

    # ================= 3. 模拟弱网环境下的极限测试 =================
    print(f"\n🖼️ 边缘端处理图片: {image_path}")
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device, dtype=torch.bfloat16)

    # 🎲 模拟 0.5 Mbps 的恶劣网络！
    simulated_bandwidth =0.5
    print(f"📡 当前模拟带宽: {simulated_bandwidth} Mbps")
    
    r_list = allocator.generate_r_list(h_norm=0.8, bandwidth_mbps=simulated_bandwidth)
    clip_model.r = r_list 

    with torch.no_grad():
        # 1. 提取时加上 output_hidden_states=True
        edge_outputs = clip_model(pixel_values, output_hidden_states=True)
    
        # 2. 强行提取倒数第二层 (-2)
        visual_features = edge_outputs.hidden_states[-2][:, 1:, :]
        print(f"✂️ 边缘端压缩完成: 仅剩 {visual_features.shape[1]} 个 Token 传给云端")

        # 让官方 MLP 处理这堆残缺的 Token
        visual_embeds = baseline_projector(visual_features)

        prompt = "USER: Please describe this image in detail.\nASSISTANT:"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        text_embeds = llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        # 👇 [核心修改区开始] 手动构造全 1 的 attention_mask
        attention_mask = torch.ones(
            (inputs_embeds.shape[0], inputs_embeds.shape[1]), 
            dtype=torch.long, 
            device=device
        )
        # 👆 [核心修改区结束]

        output_ids = llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask, # 👇 [核心修改区]: 传入 attention_mask
            max_new_tokens=150,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print("\n" + "💥"*25)
    print("🤖 【官方 LLaVA Baseline】弱网输出结果:")
    print("-" * 50)
    print(generated_text)
    print("-" * 50)

if __name__ == "__main__":
    main()