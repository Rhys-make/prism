import os
import json
import random
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPVisionModel, CLIPImageProcessor
from torch.optim import AdamW
from tqdm import tqdm
import torch.multiprocessing
from datetime import datetime
# 🛡️ 修复 1：解决多进程读取数据时的内存死锁问题
torch.multiprocessing.set_sharing_strategy('file_system')

from cloud.adp import SemanticResampler
from edge.cna import CNA_Allocator
from edge.tome.patch.clip import apply_patch_clip

# ---------------------------------------------------------
# 1. 真实的边缘端大脑 (支持动态 CNA 带宽模拟)
# ---------------------------------------------------------
class RealEdgeEncoder(nn.Module):
    def __init__(self, model_path, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        self.model = CLIPVisionModel.from_pretrained(model_path).to(device, dtype=dtype)
        apply_patch_clip(self.model)
        self.allocator = CNA_Allocator(num_layers=24, total_tokens=576, max_drop=450)
        
    def forward(self, pixel_values):
        simulated_bandwidth = random.uniform(0.5, 5.0)
        r_list = self.allocator.generate_r_list(h_norm=0.8, bandwidth_mbps=simulated_bandwidth)
        self.model.r = r_list
        outputs = self.model(pixel_values, output_hidden_states=True)
        # 提取倒数第二层特征 (-2)
        visual_features = outputs.hidden_states[-2][:, 1:, :] 
        return visual_features

# ---------------------------------------------------------
# 2. 适配 336x336 高清分辨率的数据集解析
# ---------------------------------------------------------
class LLavaPretrainDataset(Dataset):
    def __init__(self, img_dir, ann_file, tokenizer, image_processor, max_length=128):
        self.img_dir = img_dir
        with open(ann_file, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_file = item.get('image', None)
        img_path = os.path.join(self.img_dir, image_file)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (336, 336), (0, 0, 0)) 
            
        pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        conversations = item['conversations']
        human_text = conversations[0]['value'] 
        gpt_text = conversations[1]['value']   
        
        full_text = human_text + " " + gpt_text + self.tokenizer.eos_token
        tokens = self.tokenizer(full_text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        input_ids = tokens.input_ids.squeeze(0)
        attention_mask = tokens.attention_mask.squeeze(0)
        
        prompt_tokens = self.tokenizer(human_text + " ", return_tensors="pt").input_ids.squeeze(0)
        prompt_len = len(prompt_tokens)
        
        labels = input_ids.clone()
        labels[:prompt_len] = -100 
        labels[input_ids == self.tokenizer.pad_token_id] = -100 

        return pixel_values, input_ids, attention_mask, labels

# ---------------------------------------------------------
# 3. A800 端云协同炼丹主回路
# ---------------------------------------------------------
def train():
    device = "cuda"
    print("🚀 启动 Prism-VLM 端云协同训练引擎 (修复版: 恒定 Mask + 防死锁)")

    # 路径配置
    llm_path = "./models/vicuna-7b"
    clip_path = "./models/clip-vit-large-patch14-336"
    img_dir = "./data/llava_pretrain" 
    ann_file = "./data/llava_pretrain/blip_laion_cc_sbu_558k.json"
    
    batch_size = 4
    accumulation_steps = 16 
    epochs = 1  
    num_queries = 128
    EDGE_OUT_DIM = 1024 

    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 加载 LLM
    tokenizer = AutoTokenizer.from_pretrained(llm_path, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    llm = AutoModelForCausalLM.from_pretrained(llm_path, dtype=torch.bfloat16, local_files_only=True).to(device)
    llm.eval()
    for param in llm.parameters(): param.requires_grad = False

    # 2. 加载 Edge Encoder
    edge_encoder = RealEdgeEncoder(model_path=clip_path, device=device, dtype=torch.bfloat16)
    edge_encoder.eval() 
    for param in edge_encoder.parameters(): param.requires_grad = False 

    # 3. 加载 ADP
    adp = SemanticResampler(in_dim=EDGE_OUT_DIM, llm_dim=4096, num_queries=num_queries).to(device)
    adp.train()
    optimizer = AdamW(adp.parameters(), lr=1e-4)

    # 4. 数据管道 (🛡️ 修复 2：降低 num_workers 防爆内存)
    image_processor = CLIPImageProcessor.from_pretrained(clip_path)
    dataset = LLavaPretrainDataset(img_dir, ann_file, tokenizer, image_processor=image_processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # 5. 正式点火
    print("🔥 跨越端云，全速炼丹！")
    optimizer.zero_grad() 
    
    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        total_loss = 0
        
        for batch_idx, (pixel_values, input_ids, attention_mask, labels) in enumerate(loop):
            pixel_values = pixel_values.to(device, dtype=torch.bfloat16)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device) # 🛡️ 别忘了把它放到 GPU 上！
            labels = labels.to(device)
            
            # --- 阶段 A: 边缘侧动态提取 ---
            with torch.no_grad(): 
                edge_features = edge_encoder(pixel_values) 

            # --- 阶段 B: 云端 ADP 重采样 ---
            visual_embeds = adp(edge_features.float()).to(torch.bfloat16)
            
            # --- 阶段 C: 拼接与掩码构造 (🛡️ 修复 3：ADP 的恒定对齐) ---
            text_embeds = llm.get_input_embeddings()(input_ids)
            combined_embeds = torch.cat([visual_embeds, text_embeds], dim=1) 
            
            # 视觉部分的 Mask 永远是恒定的 num_queries (128) 长度，且全为 1
            visual_attn_mask = torch.ones((pixel_values.shape[0], num_queries), dtype=torch.long, device=device)
            visual_labels = torch.full((pixel_values.shape[0], num_queries), -100, dtype=torch.long, device=device)
            
            combined_attention_mask = torch.cat([visual_attn_mask, attention_mask], dim=1)
            combined_labels = torch.cat([visual_labels, labels], dim=1)
            
            # 前向传播
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = llm(
                    inputs_embeds=combined_embeds, 
                    attention_mask=combined_attention_mask, # 👈 注入灵魂
                    labels=combined_labels
                )
                loss = outputs.loss / accumulation_steps
            
            loss.backward()
            
            if ((batch_idx + 1) % accumulation_steps == 0) or ((batch_idx + 1) == len(dataloader)):
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
            loop.set_postfix(loss=loss.item() * accumulation_steps)
            
        print(f"✅ Epoch {epoch+1} 结束，平均 Loss: {total_loss/len(dataloader):.4f}")
        # 假设这些是你脚本里已经定义好的变量
        current_stage = "stage1" # 跑 Stage 2 时改成 "stage2"
        num_queries = 128
        current_epoch = 1
# ==========================================

# 1. 创建专门的 Checkpoint 文件夹（如果不存在会自动创建）
        save_dir = "./checkpoints/prism_adp"
        os.makedirs(save_dir, exist_ok=True)

# 2. 生成当前时间戳 (例如: 20260413_2015)
        time_str = datetime.now().strftime("%Y%m%d_%H%M")

# 3. 拼接极具辨识度的动态文件名
# 格式: 架构名_阶段_Query数量_Epoch_时间戳.pth
        save_name = f"adp_{current_stage}_q{num_queries}_ep{current_epoch}_{time_str}.pth"
        final_path = os.path.join(save_dir, save_name)

# 4. 安全保存权重
# 假设你的模型实例叫 semantic_resampler
        torch.save(semantic_resampler.state_dict(), final_path)

        print(f"\n✅ [防覆盖保护触发] ADP 权重已安全落盘至: {final_path}")
        
        

if __name__ == "__main__":
    train()