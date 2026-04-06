import torch
from typing import Tuple, Optional
# 导入 Hugging Face CLIP 库中原生的 Encoder 层和 Attention 层，我们稍后要“劫持”并继承它们
from transformers.models.clip.modeling_clip import CLIPEncoderLayer, CLIPAttention

# 导入 ToMe 核心算法：二分图软匹配、来源追踪、加权平均合并
from edge.tome.merge import bipartite_soft_matching, merge_source, merge_wavg
# 导入 ToMe 工具：解析 r（裁减数量）的分布
from edge.tome.utils import parse_r

# =====================================================================
# 1. 魔改 Hugging Face 的 CLIPAttention (纯原生 PyTorch，不依赖私有函数)
# =====================================================================
class ToMeCLIPAttention(CLIPAttention):
    """
    💡 功能举例：
    原生 Attention 只计算注意力。而这个魔改版做了两件额外的事：
    1. 接收 `size` 参数（也就是 Token 代表的原始像素大小）。如果一个 Token 是由 3 个小 Token 合并来的，
       它的 size 就是 3，在算注意力时，我们会给它加权（Proportional Attention），让模型多“看”它几眼。
    2. 提取 `metric`（这里用的是 Key 矩阵的平均值）。返回给外层，告诉外层“哪些 Token 长得像”，方便后续合并。
    """
    def forward(
        self,
        hidden_states: torch.Tensor,                                  # 输入特征张量 [Batch, 序列长度, 特征维度]
        attention_mask: Optional[torch.Tensor] = None,                # 填充遮罩（屏蔽无效 Token）
        causal_attention_mask: Optional[torch.Tensor] = None,         # 因果遮罩（在纯视觉模型中通常为 None）
        output_attentions: Optional[bool] = False,                    # 是否需要返回注意力权重矩阵
        size: torch.Tensor = None,                                    # 🚀 [ToMe 新增] 当前每个 Token 包含的原始 Token 数量
        **kwargs                                                      # 吸收并兼容不同 HF 版本传来的多余参数
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        # 获取 Batch 大小(bsz)，当前 Token 数量(tgt_len)，特征维度(embed_dim)
        bsz, tgt_len, embed_dim = hidden_states.size()
        
        # 1. 生成 Q, K, V 矩阵
        # Query 矩阵，并乘以缩放因子 scale (通常是 1 / sqrt(head_dim))，防止 Softmax 梯度消失
        query_states = self.q_proj(hidden_states) * self.scale
        # Key 矩阵，用于被 Query 查询
        key_states = self.k_proj(hidden_states)
        # Value 矩阵，真正包含信息内容的矩阵
        value_states = self.v_proj(hidden_states)
        
        # 2. 准备多头注意力 (Multi-Head) 的张量形状变换
        # 将 Q 矩阵拆分成多头形状：[bsz, tgt_len, 头数, 每头维度]，然后把“头数”移到第 1 维 -> [bsz, num_heads, tgt_len, head_dim]
        query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        # 同理，拆分 K 矩阵。注意 -1 表示自动推导序列长度
        key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # 同理，拆分 V 矩阵
        value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 提取 K 矩阵的序列长度 (源序列长度)
        src_len = key_states.size(2)
        
        # 把前两个维度 (bsz 和 num_heads) 压平在一起，变成 3D 张量，这是为了使用高效的 torch.bmm (批量矩阵乘法)
        query_states = query_states.reshape(bsz * self.num_heads, tgt_len, self.head_dim)
        key_states = key_states.reshape(bsz * self.num_heads, src_len, self.head_dim)
        value_states = value_states.reshape(bsz * self.num_heads, src_len, self.head_dim)
        
        # 3. 计算原始的注意力得分矩阵
        # Q 乘以 K 的转置：[bsz*heads, tgt_len, head_dim] x [bsz*heads, head_dim, src_len] -> [bsz*heads, tgt_len, src_len]
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        # 把压平的维度还原回来，变成 4D 张量：[bsz, num_heads, tgt_len, src_len]
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        
        # 4. 🚀 [ToMe 核心：比例注意力 Proportional Attention]
        # 如果当前 Token 是被合并过的，它代表的信息量更大，需要人为增加它的 Attention 权重
        if size is not None:
            # 取对数，防止合并数量过多导致权重爆炸
            s = size.log()
            # 兼容处理：确保维度正确。如果是 3D 的 [bsz, src_len, 1]，去掉最后一维
            if s.dim() == 3: 
                s = s.squeeze(-1)
            # 将大小权重 s 广播 (unsqueeze) 成 4D 形状，加到注意力权重上
            attn_weights = attn_weights + s.unsqueeze(1).unsqueeze(2)
            
        # 5. 加上 Hugging Face 传进来的各种 Mask (如果有的话)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        if causal_attention_mask is not None:
            attn_weights = attn_weights + causal_attention_mask
            
        # 再次压平维度，准备进行 Softmax 计算
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        # 对最后一个维度 (src_len) 做 Softmax，使每一行的注意力概率总和为 1
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        
        # 如果模型配置需要返回注意力图 (常用于可视化)，则保存一份
        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None
            
        # 6. 计算最终的注意力输出
        # 注意力权重 乘以 Value 矩阵：[bsz*heads, tgt_len, src_len] x [bsz*heads, src_len, head_dim]
        attn_output = torch.bmm(attn_weights, value_states)
        # 还原回 4D 形状
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        # 将头数维度移回第 2 维，并把多头特征拼接 (reshape) 回完整的 embed_dim 维度
        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
        # 通过最后的线性输出层 O 矩阵
        attn_output = self.out_proj(attn_output)
        
        # 7. 🚀 [ToMe 核心：提取特征匹配度量标准 Metric]
        # 为什么用 Key 矩阵？因为在自注意力中，Key 描述了 Token "包含什么特征"。
        # 我们把 Key 矩阵的多头维度求平均值，作为这个 Token 的“特征画像”，返回给外部做二分图匹配
        metric = key_states.view(bsz, self.num_heads, src_len, self.head_dim).mean(dim=1)
        
        # 返回：注意力计算结果，注意力权重图，特征匹配度量
        return attn_output, attn_weights_reshaped, metric

# =====================================================================
# 2. 魔改 Hugging Face 的 CLIPEncoderLayer
# =====================================================================
class ToMeCLIPEncoderLayer(CLIPEncoderLayer):
    """
    💡 功能举例：
    这是 ViT 的一层积木。原生只会做：输入 -> Attention -> MLP -> 输出。
    魔改版做的是：输入 -> Attention (拿到 Metric) -> 触发二分图匹配 -> 砍掉 r 个最相似的 Token (Merge) -> MLP -> 输出。
    例如：这层 r=10，输入 197 个 Token。匹配后发现 Token_A 和 Token_B 相似度 0.99，就把它俩融合成一个新的 Token_AB。
    最终这层输出的张量长度就变成了 187 个。
    """
    def forward(
        self,
        hidden_states: torch.Tensor,                                  # 当前层的输入 Token 序列
        attention_mask: Optional[torch.Tensor] = None,                # 遮罩
        causal_attention_mask: Optional[torch.Tensor] = None,         # 因果遮罩
        output_attentions: Optional[bool] = False,                    # 是否输出注意力
        **kwargs                                                      # 兼容多余参数
    ):
        # 🚀 [防御机制]：应对某些 HF 版本，如果传进来的是个元组 (tuple)，提取第一个真正的张量
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        # 保存残差连接 (Residual) 的备份
        residual = hidden_states
        # 过第一个 LayerNorm (Pre-Norm 架构)
        hidden_states = self.layer_norm1(hidden_states)
        
        # 取出全局配置中记录的每个 Token 的代表大小 size。如果不开启 prop_attn 则设为 None
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        
        # 调用我们刚刚魔改的 self_attn (ToMeCLIPAttention)，把 size 透传进去
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            size=attn_size,
            **kwargs, 
        )
        
        # attn_outputs[0] 是注意力计算的特征结果
        hidden_states = attn_outputs[0]
        # attn_outputs[2] 是我们在 Attention 里私带出来的 Key 矩阵特征 (metric)
        metric = attn_outputs[2]
        
        # 加上残差连接
        hidden_states = residual + hidden_states
        
        # 🚀 [ToMe 核心：动态 Token 合并]
        # 从全局指令大脑 `r_list` 中，弹出当前这一层应该砍掉的 Token 数量 `r`
        r = self._tome_info["r"].pop(0) if len(self._tome_info["r"]) > 0 else 0
        
        # 如果大脑指令说这层需要砍人 (r > 0)
        if r > 0:
            # 1. 寻找配对：使用二分图软匹配算法，根据 metric 找到 r 对最相似的 Token 组合
            # class_token=True 表示第 0 个 [CLS] 永远被屏蔽，绝对不会被合并！
            merge, _ = bipartite_soft_matching(
                metric, r, self._tome_info["class_token"], self._tome_info["distill_token"]
            )
            # 如果开启了追踪 (通常用于可视化查看最后剩下的 Token 是由哪些原始像素合并来的)
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(merge, hidden_states, self._tome_info["source"])
            
            # 2. 物理融合：把找到的配对 Token，根据它们的 size 权重，进行加权平均 (Weighted Average)
            # 同时更新全局的 size 列表 (比如原来 size 分别是 1 和 2，合并后新 Token 的 size 变成 3)
            hidden_states, self._tome_info["size"] = merge_wavg(merge, hidden_states, self._tome_info["size"])

        # 保存 MLP 之前的残差备份
        residual = hidden_states
        # 过第二个 LayerNorm
        hidden_states = self.layer_norm2(hidden_states)
        # 过多层感知机 (MLP / FFN)，进行特征升维再降维
        hidden_states = self.mlp(hidden_states)
        # 加上第二次残差连接
        hidden_states = residual + hidden_states
        
        # 按照 HF 的规范，组装返回值 (打包成元组)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_outputs[1],)
            
        # 🚀 [自适应返回机制]：根据探针测试的结果，决定是返回纯张量还是元组。彻底解决版本兼容报错！
        if not output_attentions and not self._tome_info.get("returns_tuple", True):
            return outputs[0]  # HF 新版逻辑：如果不要 attention，直接给你个张量
        return outputs         # HF 老版逻辑：永远返回元组

# =====================================================================
# 3. 封装装配车间 & 柔性探针检测 (Safe Probe)
# =====================================================================
def make_tome_class_hf(transformer_class):
    """
    💡 功能举例：
    这是一个闭包（工厂函数）。如果你传入一个 `CLIPVisionModel` 类，
    它会动态创建一个子类 `ToMeCLIPVisionModel`，里面偷偷塞进一个 `_setup_tome` 后勤函数。
    每次前向传播前，它都会把你在外部设置的 `model.r` 列表，装填进弹匣 `_tome_info["r"]` 里。
    """
    class ToMeCLIPVisionModel(transformer_class):
        def _setup_tome(self):
            # 兼容独立的 CLIPVision 模型和作为 LLaVA 内部组件的 CLIP 模型
            layers = self.vision_model.encoder.layers if hasattr(self, "vision_model") else self.encoder.layers
            # 如果你传的是个明确的层级指令列表 (比如长度为 24 的 r_list)
            if isinstance(self.r, list):
                # 校验列表长度必须等于模型层数
                assert len(self.r) == len(layers), f"r_list 长度({len(self.r)})必须等于网络层数({len(layers)})"
                self._tome_info["r"] = self.r.copy()
            else:
                # 如果你只传了一个数字 r (比如 10)，平均分配到每一层
                self._tome_info["r"] = parse_r(len(layers), self.r)
            
            # 重置每张图片前向传播时的局部状态
            self._tome_info["size"] = None
            self._tome_info["source"] = None

        def forward(self, *args, **kwargs):
            # 每次调用模型时，先偷偷执行装填指令
            self._setup_tome()
            # 然后再去调用 HF 原生的 forward 逻辑
            return super().forward(*args, **kwargs)
            
    return ToMeCLIPVisionModel 

def apply_patch_clip(model, trace_source=False, prop_attn=True):
    """
    💡 功能举例：
    核心手术刀！直接暴露给用户的接口。
    例如 `apply_patch_clip(lmm_model.vision_tower)`。
    它会遍历模型，把模型肚子里的原生 `CLIPEncoderLayer` 全部强行替换成我们写的 `ToMeCLIPEncoderLayer`。
    """
    # 获取模型的层叠列表，方便取第一层来做探针测试
    layers = model.vision_model.encoder.layers if hasattr(model, "vision_model") else model.encoder.layers
    first_layer = layers[0]
    
    # 获取这台电脑的隐层维度、所用的设备 (CPU/CUDA) 和数据类型 (fp16/fp32)
    hidden_size = first_layer.self_attn.q_proj.in_features
    device = next(first_layer.parameters()).device
    dtype = next(first_layer.parameters()).dtype
    
    # 🚀 1. 柔性探针 (Safe Probe)：极其温柔地试探当前 HF 版本的返回值脾气
    # 构造一个形状为 [1(bsz), 1(seq), hidden_size] 的假数据
    dummy_in = torch.randn(1, 1, hidden_size, device=device, dtype=dtype)
    # 默认兜底：假设它返回元组
    returns_tuple = True 
    
    # 关掉梯度图，进行盲测
    with torch.no_grad():
        try:
            # 尝试一：兼容旧版本，带上一个 None 作为 attention_mask
            dummy_out = first_layer(dummy_in, None)
            returns_tuple = isinstance(dummy_out, tuple)
        except Exception:
            try:
                # 尝试二：兼容新版本，带上两个 None 作为 attention_mask 和 causal_mask
                dummy_out = first_layer(dummy_in, None, None)
                returns_tuple = isinstance(dummy_out, tuple)
            except Exception:
                # 如果依然报错，直接忽略，依靠兜底值继续执行，防止程序阻断
                pass 

    # 2. 动态替换最高层的主干模型类
    ToMeCLIPVisionModel = make_tome_class_hf(model.__class__)
    model.__class__ = ToMeCLIPVisionModel
    model.r = 0 # 初始化默认的裁减数量为 0
    
    # 在模型对象内部挂载一个全局配置字典，供所有子层共享读取
    model._tome_info = {
        "r": model.r, "size": None, "source": None, "trace_source": trace_source,
        "prop_attn": prop_attn, "class_token": True, "distill_token": False,
        "returns_tuple": returns_tuple # 核心：将刚才探针测出来的结果存入字典，让下面的 Encoder 知道该返回什么！
    }

    # 3. 递归遍历模型内部的所有子模块
    for module in model.modules():
        # 发现原生的 EncoderLayer，替换为我们的魔改版
        if isinstance(module, CLIPEncoderLayer):
            module.__class__ = ToMeCLIPEncoderLayer
            # 将全局配置字典的指针塞给它，让它能读取到实时的 r
            module._tome_info = model._tome_info
        # 发现原生的 Attention，替换为我们的魔改版
        elif isinstance(module, CLIPAttention):
            module.__class__ = ToMeCLIPAttention