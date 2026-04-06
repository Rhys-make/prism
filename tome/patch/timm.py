# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------


from typing import Tuple # 导入 Tuple 类型，用来告诉 Python 某个函数会一次性返回多个值（打包成元组）。

import torch # 导入深度学习老大哥 PyTorch，所有矩阵运算都靠它。
from timm.models.vision_transformer import Attention, Block, VisionTransformer 
# 从 `timm`（一个超全的开源视觉模型库）里，把原版 ViT 的三大核心零件拆下来：注意力层 (Attention)、网络基础块 (Block)、以及整辆车 (VisionTransformer)。

from tome.merge import bipartite_soft_matching, merge_source, merge_wavg 
# 导入我们之前已经拆解过的、你自己写的 ToMe 三大神器：匹配找对象的、记族谱账本的、加权融合特征的。
from tome.utils import parse_r 
# 导入用来解析“每一层到底要裁员多少个 Token”的计划表工具。


# =====================================================================
# 1. 改装零件 A：支持合并的 Transformer Block (ToMeBlock)
# =====================================================================
class ToMeBlock(Block): 
    # 继承原版的 Block，我们要在这个零件内部动手脚。
    """
    Modifications (修改点):
     - Apply ToMe between the attention and mlp blocks (在注意力机制和前馈神经网络之间，强行插入 ToMe 压缩算法)
     - Compute and propogate token size and potentially the token sources. (计算并向后传递 Token 的合并体积 size，视情况传递溯源账本)
    """

    def _drop_path1(self, x):
        # 这是一个防过拟合的技巧（随机丢弃一些神经元连接）。这里写这么绕，单纯是为了兼容不同版本 timm 库的命名习惯。
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        # 同上，处理第二个残差分支的防过拟合兼容性。
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 这是数据真正流经这个零件时的加工流水线。
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        
        # 【拿体型数据】：从模型的共享黑板 (_tome_info) 上看一下 Token 的体型 size。如果没有开启比例注意力，就不用拿了。
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        
        # 【过注意力层】：先给数据做个归一化(norm1)，然后连同体型(attn_size)一起送进魔改后的注意力层。
        # 出来两个东西：x_attn 是交换过信息的特征，metric 是我们等会儿用来找朋友打分的“颜值指标”（它其实就是 Key 的平均值）。
        x_attn, metric = self.attn(self.norm1(x), attn_size)
        
        # 【第一次残差相加】：把原特征 x 和算完注意力的特征 x_attn 加起来。
        x = x + self._drop_path1(x_attn)

        # 【查计划表】：看看这一层我们需要砍掉多少个 Token？用 pop(0) 拿出列表里的第一个任务指标，拿完就删掉这个任务。
        r = self._tome_info["r"].pop(0)
        
        if r > 0:
            # 如果指标大于 0，说明要在这一层启动合并机制了！
            
            # 1. 算名单：调用你熟悉的函数，传入 metric 打分，找出谁和谁合并。
            merge, _ = bipartite_soft_matching(
                metric,
                r,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
            )
            
            # 2. 记账本：如果外界要求记录族谱溯源
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, x, self._tome_info["source"]
                )
                
            # 3. 物理执行：调用加权融合。不仅 x 的长度变短了，而且共享黑板上的 size 也更新变胖了！
            x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])

        # 【过全连接层】：把已经变短的 x 送进归一化(norm2) 和 多层感知机(mlp)。
        # 然后做第二次残差相加，大功告成，输出更短的 Tensor！
        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        return x

# =====================================================================
# 2. 改装零件 B：支持比例关注的 Attention 层 (ToMeAttention)
# =====================================================================
class ToMeAttention(Attention): 
    # 继承原生的 Attention 层做魔改
    """
    Modifications (修改点):
     - Apply proportional attention (应用比例注意力，处理“大块头”的权重问题)
     - Return the mean of k over heads from attention (从注意力里顺手把 Key 取个平均抽出来，当做打分指标返回)
    """

    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        B, N, C = x.shape # 获取 Batch大小、Token数量、特征维度
        
        # 算出 Query, Key, Value，并把多头注意力的维度拆分开来。
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        
        # 解包，把 Q, K, V 单独拿出来
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  

        # 算基础注意力分数：Q 和 K 的转置做矩阵乘法，再乘以缩放因子。
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # 【魔改亮点：比例注意力】
        if size is not None:
            # 如果 Token 体型不一样大，给大块头的注意力分数加一个自然对数 log() 作为奖励。
            # 这样保证合并后的“超级 Token”在接下来的计算中不吃亏。
            attn = attn + size.log()[:, None, None, :, 0]

        # 用 Softmax 把分数变成 0~1 的概率分布。
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) # 随机丢弃防过拟合

        # 拿着算好的概率分布去乘 Value，得到最终的注意力输出特征。
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x) # 过一个线性层
        x = self.proj_drop(x) # 再次防过拟合

        # 【核心操作】：把特征 x 返回的同时，顺手把 Key 沿着多头维度(dim=1)求了个平均，一起返回。
        # 这个返回的 K 就是上面 ToMeBlock 里用来找朋友的 metric！
        return x, k.mean(1)


# =====================================================================
# =====================================================================
# 3. 造车工厂：给模型裹上糖衣 (make_tome_class) - 【已植入动态接口】
# =====================================================================
def make_tome_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        
        def _setup_tome(self):
            """独立出来的后勤部长：专门负责把动态清单装填进中控台"""
            if isinstance(self.r, list):
                assert len(self.r) == len(self.blocks), f"动态 r 列表的长度({len(self.r)})必须等于网络层数({len(self.blocks)})"
                self._tome_info["r"] = self.r.copy()
            elif callable(self.r):
                self._tome_info["r"] = self.r(len(self.blocks))
            else:
                self._tome_info["r"] = parse_r(len(self.blocks), self.r)

            self._tome_info["size"] = None
            self._tome_info["source"] = None

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            # 走正门进，先装弹，再执行
            self._setup_tome()
            return super().forward(*args, **kwdargs)

        def forward_features(self, *args, **kwdargs) -> torch.Tensor:
            # 走侧门进（特征提取），也必须先装弹！
            self._setup_tome()
            return super().forward_features(*args, **kwdargs)
            
    return ToMeVisionTransformer


# =====================================================================
# 4. 终极主刀医生：一键换心脏 (apply_patch)
# =====================================================================
def apply_patch(
    model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True
):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.
    """
    # 1. 用上面的工厂，造一个继承了当前模型类型的新类
    ToMeVisionTransformer = make_tome_class(model.__class__)

    # 【黑客行为 1：李代桃僵】
    # 强行把当前模型的底层类型指针，替换成我们刚刚捏造的新类。
    model.__class__ = ToMeVisionTransformer
    
    # 给模型身上挂上一个共享数据字典 _tome_info，就像在车上装了一个中央控制台。
    model.r = 0
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.cls_token is not None,
        "distill_token": False, # 初始化特殊 Token 为 False
    }

    # 如果原模型有蒸馏 Token，就在中控台上标记一下。
    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    # 【黑客行为 2：深入敌后洗脑】
    # 遍历这辆车（模型）里成百上千的所有小零件。
    for module in model.modules():
        if isinstance(module, Block):
            # 只要遇到原生的 Transformer Block，直接把它的核心类换成我们的改装件 ToMeBlock
            module.__class__ = ToMeBlock
            # 给它连上中央控制台的数据线
            module._tome_info = model._tome_info
        elif isinstance(module, Attention):
            # 只要遇到原生的 Attention 层，强行换成我们的改装件 ToMeAttention
            module.__class__ = ToMeAttention
