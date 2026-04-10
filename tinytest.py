import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 【修改点 1】引入全新的语义重采样器 (假设你的 adp.py 放在 cloud 文件夹下)
# 如果你的 adp.py 就在根目录，直接写: from adp import SemanticResampler
from cloud.adp import SemanticResampler 

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("="*65)
    print("🧪 [本地联调测试 V2] Prism 端云协同流水线 (Cross-Attention 版)")
    print("="*65)

    # 1. 读取边缘端生成的特征 (数量 N 是波动的)
    try:
        edge_payload = torch.load("edge_payload.pt", weights_only=True).to(device)
        print(f"📥 1. 读取边缘端特征成功，输入形状: {edge_payload.shape}")
    except FileNotFoundError:
        print("❌ 找不到 edge_payload.pt，请先运行 python -m edge.pipeline")
        return

    # 2. 挂载全新的 ADP 桥梁
    # 【修改点 2】初始化 128 个 Query 的重采样器
    adp = SemanticResampler(in_dim=1024, llm_dim=2048, num_queries=128).to(device)
    adp.eval() # 联调测试阶段，关闭 dropout 等训练特性
    
    # 将边缘端特征通过 ADP，输出固定长度 (转成半精度 fp16 适配 LLM)
    visual_embeds = adp(edge_payload.float()).half() 
    print(f"🔄 2. 经过全新 ADP 语义重采样，输出形状恒定为: {visual_embeds.shape}")

    # 3. 离线加载 TinyLlama
    local_model_path = "./tllm"
    print("🧠 3. 正在离线加载 TinyLlama (1.1B) ...")
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
    llm = AutoModelForCausalLM.from_pretrained(local_model_path, torch_dtype=torch.float16, local_files_only=True).to(device)

    # 4. 构造文本提示词
    prompt = "<image>\nDescribe the image."
    text_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    text_embeds = llm.get_input_embeddings()(text_ids)

    # 5. 精确拼接 (视觉的 128 个 Token + 文本的 Token)
    # 【修改点 3】这里不再受边缘端压缩率影响，视觉长度永远是 128
    combined_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
    print(f"✂️ 4. 特征拼接完成，送入大模型的最终形状: {combined_embeds.shape}")

    # 6. 大模型推理
    print("🚀 5. 启动推理 (预期输出乱码，因全新 ADP 的 128 个 Query 尚未训练)...")
    outputs = llm.generate(
        inputs_embeds=combined_embeds,
        max_new_tokens=20,
        pad_token_id=tokenizer.eos_token_id
    )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print("="*65)
    print("🎉 [架构升级联调成功] 模型已成功吃下恒定重采样特征并输出：\n")
    print(">>>", result)
    print("="*65)

if __name__ == "__main__":
    main()