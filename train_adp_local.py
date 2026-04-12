import os
import json
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchvision import transforms
from torch.optim import AdamW
from tqdm import tqdm

# 引入你的 ADP 模型 (请确保 adp.py 在 cloud 文件夹下)
from cloud.adp import SemanticResampler

# ---------------------------------------------------------
# 1. 数据集定义 (COCO Captions - CPU 兼容版)
# -----------------------------------------------------------
class COCODataset(Dataset):
    def __init__(self, img_dir, ann_file, tokenizer, transform=None, max_length=64):
        self.img_dir = img_dir
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        self.annotations = data['annotations']
        self.image_dict = {img['id']: img['file_name'] for img in data['images']}
        
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_id = ann['image_id']
        caption = ann['caption']
        
        img_filename = self.image_dict[image_id]
        img_path = os.path.join(self.img_dir, img_filename)
        
        # 【免下载 6GB 图片黑科技】：如果找不到真实图片，直接生成一张黑图
        # 这样我们可以纯粹利用真实 JSON 里的复杂英文句子来测试大模型
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)

        prompt_text = "<image>\nDescribe the image. "
        full_text = prompt_text + caption + self.tokenizer.eos_token
        
        tokens = self.tokenizer(full_text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        input_ids = tokens.input_ids.squeeze(0)
        attention_mask = tokens.attention_mask.squeeze(0)
        
        prompt_tokens = self.tokenizer(prompt_text, return_tensors="pt").input_ids.squeeze(0)
        prompt_len = len(prompt_tokens)
        
        labels = input_ids.clone()
        labels[:prompt_len] = -100 
        labels[input_ids == self.tokenizer.pad_token_id] = -100 

        return image, input_ids, attention_mask, labels

# ---------------------------------------------------------
# 2. 核心训练逻辑 (强制 CPU & FP32 慢速安全模式)
# ---------------------------------------------------------
def train():
    # 【修改 1】：强制绑定 CPU，绝不调用 GPU，防止电源过载断电
    device = "cpu"
    print(f"🐌 已强制切换为 {device} 安全排雷模式！风扇可能会转，但绝对不会关机。")

    local_llm_path = "./tllm" 
    img_dir = "./data/val2014"
    ann_file = "./data/annotations/captions_val2014.json"
    batch_size = 2 # CPU 算力有限，Batch Size 保持最小
    epochs = 1     # 排雷跑 1 个 Epoch 足够了
    num_queries = 128
    
    print("🧠 1. 加载 TinyLlama (冻结状态，标准 FP32 精度)...")
    tokenizer = AutoTokenizer.from_pretrained(local_llm_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 【修改 2】：移除 torch_dtype=torch.float16，使用 CPU 最喜欢的 float32
    llm = AutoModelForCausalLM.from_pretrained(local_llm_path, torch_dtype=torch.float32, local_files_only=True).to(device)
    llm.eval()
    for param in llm.parameters(): param.requires_grad = False

    print("🌉 2. 加载 ADP 重采样器 (可训练状态)...")
    adp = SemanticResampler(in_dim=1024, llm_dim=2048, num_queries=num_queries).to(device)
    adp.train()
    optimizer = AdamW(adp.parameters(), lr=1e-4)

    print("📚 3. 准备真实 COCO 文本数据集...")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = COCODataset(img_dir, ann_file, tokenizer, transform=transform)
    subset = torch.utils.data.Subset(dataset, range(100)) # CPU 跑得慢，只抽 100 条数据验证即可
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True)

    print("🔥 4. 开始纯 CPU 安全计算... (请耐心等待，大概需要几十秒出一个结果)")
    
    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        total_loss = 0
        
        for batch_idx, (images, input_ids, attention_mask, labels) in enumerate(loop):
            # 将所有张量移至 CPU
            images = images.to(device)
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            B = images.shape[0]
            # 【修改 3】：模拟特征同样使用标准的 float32，移除 dtype=torch.float16
            dummy_edge_features = torch.randn(B, 212, 1024, device=device)
            
            # 【修改 4】：移除 ADP 输出结果后的 .half() 转换
            visual_embeds = adp(dummy_edge_features) 
            
            text_embeds = llm.get_input_embeddings()(input_ids)
            combined_embeds = torch.cat([visual_embeds, text_embeds], dim=1) 
            
            visual_labels = torch.full((B, num_queries), -100, dtype=torch.long, device=device)
            combined_labels = torch.cat([visual_labels, labels], dim=1)
            
            outputs = llm(inputs_embeds=combined_embeds, labels=combined_labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        print(f"✅ Epoch {epoch+1} 平均 Loss: {total_loss/len(dataloader):.4f}")

    print("🎉 5. 本地排雷完成，无任何 Shape 或 OOM 报错！")

if __name__ == "__main__":
    train()