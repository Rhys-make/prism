import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor

print("="*70)
print("☁️ [云端大脑] 端边云协同架构 - LLM 特征拼接与推理引擎")
print("="*70)

# ==========================================
# 0. 环境与模型初始化
# ==========================================
# 你的服务器应该有 GPU，所以把设备设为 cuda
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "llava-hf/llava-1.5-7b-hf" # 替换为你在服务器上的 LLaVA 路径

print(f"⏳ 正在加载语言模型与调味师 (MLP Projector)...")
processor = AutoProcessor.from_pretrained(model_path)
# 只加载模型，不需要传图片，所以我们主要用它的 projector 和 language_model
model = LlavaForConditionalGeneration.from_pretrained(
    model_path, 
    torch_dtype=torch.float16, # 使用 fp16 节省显存并加速
    low_cpu_mem_usage=True
).to(device)
model.eval()

# ==========================================
# 1. 接收边缘端的载荷
# ==========================================
payload_path = "edge_payload.pt" # 你从本地电脑传到服务器的那个 100KB 的文件
print(f"📥 正在接收边缘端压缩特征: {payload_path}")

# 载入特征，形状预期为 [1, 幸存Token数, 1024]
edge_features = torch.load(payload_path).to(device) 
num_surviving_tokens = edge_features.shape[1]
print(f"   - 收到 {num_surviving_tokens} 个被 CNA 高度浓缩的视觉特征块")

# ==========================================
# 2. 过多模态投影层 (The Projector)
# ==========================================
print(f"🔄 正在通过 MLP 翻译为 LLM 母语 (1024 -> 4096 维)...")
# 这一步极其关键：在云端完成维度膨胀，避免占用边缘网络带宽
with torch.no_grad():
    projected_vision = model.multi_modal_projector(edge_features) 
# 现在 projected_vision 的形状是 [1, 幸存Token数, 4096]

# ==========================================
# 3. 处理文本流 (Text Processing)
# ==========================================
# 构造符合 LLaVA 格式的 Prompt
prompt = "USER: <image>\n请详细描述一下这张图片。 ASSISTANT:"
print(f"📝 解析文本 Prompt: '{prompt}'")

# 将文本转为 ID，注意这里没有传 images，只传 text
text_inputs = processor(text=prompt, return_tensors="pt").to(device)
input_ids = text_inputs.input_ids
attention_mask = text_inputs.attention_mask

# ==========================================
# 4. 🔪 核心外科手术：张量精准拼接
# ==========================================
print("✂️ 正在执行特征序列手术...")

# 1. 把文本 ID 变成 4096 维的 Embedding
text_embeds = model.get_input_embeddings()(input_ids) # [1, 序列长度, 4096]

# 2. 找到 <image> 占位符在哪里
image_token_id = model.config.image_token_index # 对于 LLaVA，通常是 32000
image_positions = torch.where(input_ids == image_token_id)
image_idx = image_positions[1][0].item() # 获取占位符的具体位置索引

# 3. 像三明治一样把视觉特征夹进去
# 丢弃原来的 <image> token (长度为1)，替换为我们的 projected_vision (长度为 num_surviving_tokens)
embeds_before = text_embeds[:, :image_idx, :]
embeds_after = text_embeds[:, image_idx+1:, :]

final_inputs_embeds = torch.cat([
    embeds_before, 
    projected_vision, 
    embeds_after
], dim=1)

# ⚠️ 进阶细节：注意力掩码 (Attention Mask) 也要同步拼接！
# 告诉 LLM 哪些是有效数据，哪些是 Padding
mask_before = attention_mask[:, :image_idx]
mask_after = attention_mask[:, image_idx+1:]
# 我们新加入的视觉 Token 全是有效数据，所以用 ones
vision_mask = torch.ones((1, num_surviving_tokens), dtype=attention_mask.dtype, device=device)

final_attention_mask = torch.cat([
    mask_before, 
    vision_mask, 
    mask_after
], dim=1)

# ==========================================
# 5. 大脑推理引擎启动
# ==========================================
print("🧠 序列拼接完成，正在唤醒大语言模型生成回答...")

with torch.no_grad():
    # 绕过 LLaVA 的外壳，直接调用核心的 language_model
    # 喂给它我们亲手拼好的 inputs_embeds
    output_ids = model.language_model.generate(
        inputs_embeds=final_inputs_embeds,
        attention_mask=final_attention_mask,
        max_new_tokens=150,
        do_sample=False # 设为 False 保证输出稳定，便于做实验对比
    )

# 纯文本解码
generated_text = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("="*70)
print("🎯 [云端推理结果]:")
print(generated_text)
print("="*70)