import torch
import os
import random
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPVisionModel, CLIPImageProcessor

# 1. 导入你的核心模块
from cloud.adp import SemanticResampler
from edge.cna import CNA_Allocator
from edge.tome.patch.clip import apply_patch_clip

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # 路径配置
    llm_path = "./models/vicuna-7b" 
    clip_path = "./models/clip-vit-large-patch14-336"
    adp_ckpt_path = "./checkpoints/prism_adp_epoch_1.pth"
    image_path = "./pic/mide.png" 

    # ================= 1. 环境初始化 =================
    print("🧠 1. 载入云端 Vicuna-7B & 边缘侧 CLIP...")
    tokenizer = AutoTokenizer.from_pretrained(llm_path, local_files_only=True, use_fast=False)
    llm = AutoModelForCausalLM.from_pretrained(llm_path, dtype=torch.bfloat16, local_files_only=True).to(device).eval()
    
    processor = CLIPImageProcessor.from_pretrained(clip_path, local_files_only=True)
    clip_model = CLIPVisionModel.from_pretrained(clip_path, dtype=torch.bfloat16, local_files_only=True).to(device).eval()

    # ================= 2. 注入 Prism 核心技术 =================
    print("🧪 2. 注入 ToMe 补丁并激活 CNA 动态分配器...")
    # 核心补丁：让 CLIP 支持 Token 融合
    apply_patch_clip(clip_model)
    # 初始化 CNA：根据带宽决定每层压多少
    allocator = CNA_Allocator(num_layers=24, total_tokens=576, max_drop=450)

    print("🌉 3. 载入训练好的 SemanticResampler 权重...")
    adp = SemanticResampler(in_dim=1024, llm_dim=4096, num_queries=128) 
    adp.load_state_dict(torch.load(adp_ckpt_path, map_location="cpu"))
    adp = adp.to(device, dtype=torch.bfloat16).eval()

    # ================= 3. 模拟边缘端：压缩与提取 =================
    print(f"\n🖼️ 边缘端处理图片: {image_path}")
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device, dtype=torch.bfloat16)

    # 🎲 模拟真实网络环境 (0.5Mbps 极慢网络)
    simulated_bandwidth = 1000
    print(f"📡 当前模拟带宽: {simulated_bandwidth} Mbps (极低带宽模式)")
    
    # CNA 计算策略
    r_list = allocator.generate_r_list(h_norm=0.8, bandwidth_mbps=simulated_bandwidth)
    clip_model.r = r_list # 将压缩指令下发给 CLIP

    with torch.no_grad():
        # 边缘端提取：开启 output_hidden_states 以获取所有中间层
        clip_outputs = clip_model(pixel_values, output_hidden_states=True)
        
        # hidden_states 是一个 tuple，包含 1 层 Embedding 和 24 层 Encoder 层的输出
        # 取 [-2] 即为倒数第二层 (第 23 层)
        edge_outputs = clip_outputs.hidden_states[-2]
        
        # 剔除 CLS Token
        visual_features = edge_outputs[:, 1:, :] 
        
        current_tokens = visual_features.shape[1]
        reduction_rate = (1 - current_tokens/576) * 100
        print(f"✂️ 边缘端压缩完成: Token 数量从 576 锐减至 {current_tokens} (压缩率: {reduction_rate:.1f}%)")

    # ================= 4. 模拟云端：推理与生成 =================
    # ================= 4. 模拟云端：推理与生成 =================
    print("🚀 云端接收特征并生成回答...")
    with torch.no_grad():
        # ADP 翻译对齐
        visual_embeds = adp(visual_features) 

        # 拼接对话模板
        prompt = "USER: Please describe this image in detail.\nASSISTANT:"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        text_embeds = llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        # 👇 [新增] 手动生成一个全为 1 的 attention_mask
        # 长度刚好等于 inputs_embeds 的序列长度
        attention_mask = torch.ones(
            (inputs_embeds.shape[0], inputs_embeds.shape[1]), 
            dtype=torch.long, 
            device=device
        )

        # LLM 生成
        output_ids = llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask, # 👇 [新增] 把掩码传给大模型
            max_new_tokens=150,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\n" + "✨"*25)
    print("🤖 Prism-VLM 端云协同输出:")
    print("-" * 50)
    print(generated_text)
    print("-" * 50)

if __name__ == "__main__":
    main()