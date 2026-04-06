# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import math # 导入数学库，后面需要用到 math.inf（无穷大）来处理特殊的 Token
from typing import Callable, Tuple # 导入类型提示，指明函数返回的是两个可调用函数组成的元组

import torch # 导入 PyTorch 深度学习框架

def do_nothing(x, mode=None):
    # 定义一个“占位”函数。当算出来不需要减少任何 Token 时，就调用这个函数。
    # 它的作用就是：输入什么张量 x，就原封不动地返回什么张量 x，绝对不浪费算力。
    return x

def bipartite_soft_matching(
    metric: torch.Tensor, # 输入用于打分的特征张量，形状为 (B, N, C)
    r: int,               # 我们希望在这一层削减掉的 Token 数量
    class_token: bool = False,  # 模型是否带有一个用于分类的 [CLS] Token
    distill_token: bool = False, # 模型是否带有一个用于蒸馏的 [DISTILL] Token
) -> Tuple[Callable, Callable]: # 最终返回制造好的两个函数：merge 和 unmerge
    """
    Applies ToMe with a balanced matching set (50%, 50%).
    # 应用 ToMe 算法，将 Token 平均分成两半 (50% 和 50%) 进行二分匹配。

    Input size is [batch, tokens, channels].
    # 明确了输入的维度形状是 [B, N, C]。

    r indicates the number of tokens to remove (max 50% of tokens).
    # r 代表你要移除的 token 数量。因为是两两合并，所以最多只能砍掉总数的一半。

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.
    # 额外的参数：说明是否包含分类 token 或蒸馏 token。

    When enabled, the class token and distillation tokens won't get merged.
    # 核心规则：一旦开启这两个参数，这些特殊用途的 Token 就相当于拿到了“免死金牌”，绝对不会被合并。
    """
    
    protected = 0 # 初始化受保护的 Token 数量计数器为 0
    if class_token:
        protected += 1 # 如果有 [CLS] Token，受保护名额加 1
    if distill_token:
        protected += 1 # 如果还有蒸馏 Token，受保护名额再加 1

    # We can only reduce by a maximum of 50% tokens
    # 我们最多只能减少 50% 的 Token
    t = metric.shape[1] # 获取输入张量的第 1 个维度的大小，也就是当前层总共有多少个 Token (N)
    
    # 重新修正 r 的值。拿总 Token 数减去受保护的数量后，除以 2（向下取整）。
    # 这步是为了防止你传入的 r 太大，导致越界（比如总共才 10 个可合并的，你非要合并 8 个，这是做不到的，最多 5 个）。
    r = min(r, (t - protected) // 2)

    if r <= 0:
        # 如果算下来需要合并的数量小于或等于 0（说明压根不需要压缩）
        # 直接返回上面定义好的“什么都不做”的占位函数，跳过一切复杂运算
        return do_nothing, do_nothing

    with torch.no_grad(): 
        # 开启无梯度上下文。
        # 极其关键的一步！接下来的计算纯粹是为了“找朋友(算相似度找索引)”，这个寻找过程不需要进行反向传播更新权重。
        # 加了这句能省下海量的显存和计算时间。
        
        # 将特征张量沿着最后一个维度 (C 维度) 计算 L2 范数，并让自己除以这个范数。
        # 这就是特征的“L2归一化”。归一化之后，两个向量的点乘结果就严丝合缝地等于它们的“余弦相似度”。
        metric = metric / metric.norm(dim=-1, keepdim=True)
        
        # 【核心暴力拆分】利用 Python 的数组切片语法，把所有 Token 按空间位置分家：
        # ::2 表示从头开始每隔一个取一个，拿到索引 0, 2, 4... 的 Token，归入 a 组
        # 1::2 表示从索引1开始每隔一个取一个，拿到索引 1, 3, 5... 的 Token，归入 b 组
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        
        # 计算 a 组和 b 组所有成员互相之间的相似度得分。
        # b.transpose(-1, -2) 是把 b 的最后两个维度对调，把 (B, N/2, C) 变成 (B, C, N/2)
        # @ 符号是矩阵乘法。a 乘以翻转后的 b，得到 scores 张量。
        # scores 的形状是 (B, N/2, N/2)，记录了 a 组每一个 Token 与 b 组每一个 Token 的相似度。
        scores = a @ b.transpose(-1, -2)

        if class_token:
            # 如果有分类 Token，因为它一开始在索引 0，所以它被分到了 a 组的第 0 个位置。
            # 这里把 a 组第 0 个 Token 对 b 组所有 Token 的相似度强行改成“负无穷大 (-math.inf)”。
            # 这样一来，在后面寻找最大相似度时，它永远选不上，也就永远不会被合并。
            scores[..., 0, :] = -math.inf
            
        if distill_token:
            # 如果有蒸馏 Token，因为它一开始在索引 1，所以它被分到了 b 组的第 0 个位置。
            # 这里把 a 组所有 Token 对 b 组第 0 个 Token 的相似度改成“负无穷大”。
            # 这样保证没有任何 a 组的 Token 敢合并到它身上去污染它的特征。
            scores[..., :, 0] = -math.inf

        # 沿着 b 组所在的维度（最后一个维度 dim=-1）寻找最大值。
        # node_max 记录的是 a 组每个 Token 找到的最大相似度具体分数是多少（比如 0.95）。
        # node_idx 记录的是产生这个最大分数的 b 组 Token 的索引（比如在 b 组的第 5 号）。
        node_max, node_idx = scores.max(dim=-1)
        
        # 把 a 组的 Token 按照刚才找到的最高相似度分数，从高到低 (descending=True) 排个序。
        # argsort 返回的是排序后的索引。排在最前面说明“我和 b 组的某个兄弟长得极其像，简直是冗余”。
        # [..., None] 的作用是在张量最后硬塞一个空维度进去，变成 (B, N/2, 1)，是为了给后面的 gather 铺路。
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        # 根据上面的排序，把 a 组的 Token 索引无情地切成两段：
        # unm_idx (Unmerged): 取从第 r 个开始到最后排在后面的索引。这些是长得不太像的，准许“保留”不合并。
        unm_idx = edge_idx[..., r:, :]  
        # src_idx (Merged): 取排在最前面的 r 个索引。这些是长得极度相似的冗余品，注定要作为源头被“合并（牺牲）”掉。
        src_idx = edge_idx[..., :r, :]  
        
        # 【顺藤摸瓜找目标】
        # 我们已经知道 a 组里哪些 Token (src_idx) 要牺牲了，现在要去 node_idx 这个字典里，
        # 查出它们对应的 b 组“接盘侠”是谁。
        # gather 的作用：按照 src_idx 给出的序号，去 node_idx 里面把目标地点的索引掏出来，存进 dst_idx。
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            # Sort to ensure the class token is at the start
            # 如果存在分类 Token，我们要确保它在保留列表 (unm_idx) 里依然排在最前面。
            # sort(dim=1)[0] 就是沿着长度方向重新排一次序，因为分类 Token 原本索引就是 0，排完序肯定回到第一位。
            unm_idx = unm_idx.sort(dim=1)[0]

    # =====================================================================
    # 真正的合并执行函数。由于上面是在 no_grad 下找索引，那些索引会被保存在闭包里。
    # 当网络前向传播时，会实际调用这个 merge 函数来融合带梯度的真实特征。
    # =====================================================================
    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        # 把真实输入的特征 x (比如 Attention 计算后的结果) 也按照同样的奇偶规则切分成 src(a组) 和 dst(b组)。
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        
        # 拿到 src 张量的形状：n(Batch大小)，t1(一半的Token数量)，c(特征维度 768)
        n, t1, c = src.shape
        
        # 利用 gather 函数，根据之前算好的“幸存者”索引 unm_idx，
        # 把它扩展到 768 维 (expand)，然后从 src 张量里把幸存 Token 的完整 768 维特征安全地提取出来。
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        
        # 同样的方法，根据“牺牲者”索引 src_idx，把注定要被合并的 Token 特征提取出来准备发车。
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        
        # 【物理融合发生在这里！】
        # scatter_reduce 的作用像发快递：拿着提取出的牺牲者特征 (src)，按照目标地址簿 (dst_idx)，
        # 强行投递到目标群体 (dst) 对应的位置里。
        # reduce=mode (默认是 mean)：如果原来那里有特征了，就把它俩的值加起来求一个平均数，完成“融合”！
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            # 如果有蒸馏 Token，为了保证对齐，要把那几个特殊的 Token 单独切出来拼在最前面，
            # 再把幸存者(unm)和吸收了别人的目标者(dst)拼在后面。
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            # 正常情况：直接把保留下来的 a 组 Token (unm) 和 已经被合并加持过的 b 组 Token (dst) 拼接成一个新的长序列。
            # 这个新序列的长度正好变短了 r 个，完成瘦身，送入下一层！
            return torch.cat([unm, dst], dim=1)

    # =====================================================================
    # 反合并执行函数。只有在做语义分割等需要逐像素输出的任务时才需要用到，
    # 它的任务是把前面变短的 Token 序列，重新“注水”撑回原来的长度。
    # =====================================================================
    def unmerge(x: torch.Tensor) -> torch.Tensor:
        # 获取当初幸存者的数量
        unm_len = unm_idx.shape[1]
        
        # 把经历了整个网络深层变换的压缩张量 x，按长度一分为二：
        # 前半截当初是幸存者演变的，后半截当初是目标接盘侠演变的。
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape # 获取各个维度的形状

        # 利用 gather 函数，根据当初合并的字典 dst_idx，
        # 从已经被深层网络加工过的目标接盘侠 (dst) 里面，把特征克隆 (复制) 一份出来，当作是当初牺牲者的“复活体”。
        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        # 制造一个全新的、全都是零的张量 out，它的大小正好等于当初未经压缩时的总 Token 数量。
        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        # 把经历过风雨的接盘侠 dst 原封不动地放回属于 b 组的所有奇数位置
        out[..., 1::2, :] = dst
        
        # 用 scatter_ 发快递：把幸存者 unm 按照它当初在 a 组的索引位置 (乘2恢复成偶数索引)，精准地填回偶数空坑里。
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        
        # 继续用 scatter_ 发快递：把克隆复活的牺牲者 src，填回它当初在 a 组的偶数位置里。
        # 此时，196 个坑全部填满！只不过当初被合并的两个位置，现在拿到了完全一样的一份融合特征。
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        # 返回注水还原后的完整大张量
        return out

    # 工厂生产完毕，把 merge 和 unmerge 这两个定制好的闭包函数扔给外部调用。
    return merge, unmerge



def kth_bipartite_soft_matching(
    metric: torch.Tensor, k: int
) -> Tuple[Callable, Callable]:
    """
    应用 ToMe，将所有 Token 分为两个集合：(每隔 k 个挑出来的目标集合, 剩下的所有源集合)。
    如果初始有 n 个 token，经过这个函数处理后，Token 数量会直接变成 n // k。

    输入张量的形状是 [batch, tokens, channels] (即 B, N, C)。
    注意：官方源码注释里的 'z' 其实就是参数里的 'k'。
    当 k = 2 时，它的效果等同于前面那个基础版的 r = 50% 时的 bipartite_soft_matching。
    """
    if k <= 1:
        # 如果 k<=1，说明不压缩，直接返回占位函数
        return do_nothing, do_nothing

    # ==========================================================
    # 定义核心的暴力切分逻辑
    # ==========================================================
    def split(x):
        # 1. 裁剪尾巴，确保 Token 总数能被 k 整除
        t_rnd = (x.shape[1] // k) * k
        
        # 2. 核心形变 (View)：把一维的 Token 序列，折叠成二维的矩阵块
        """
        举例说明 view 操作:
        假设你有 6 个 Token，k=3。这意味着你要把 Token 砍掉 2/3。
        原始 x 形状: (B, 6, C)。我们暂时忽略 B 和 C，只看 Token 序列: [T0, T1, T2, T3, T4, T5]
        view(-1, k, ...) 之后，相当于把它们分成了 2 个块，每个块 3 个 Token:
        块 1: [T0, T1, T2]
        块 2: [T3, T4, T5]
        """
        x = x[:, :t_rnd, :].view(x.shape[0], -1, k, x.shape[2])
        
        # 3. 分家：把每个块的前 k-1 个 Token 划给 a 组（献祭组），最后一个划给 b 组（保留组）
        """
        接上面的例子:
        x[:, :, : (k - 1), :] 取每个块的前 2 个 (索引0和1)：
        a 组拿到了: [T0, T1] 和 [T3, T4]。
        通过 view 展平后，a 组 (要被合并的 src) 变成了: [T0, T1, T3, T4]
        
        x[:, :, (k - 1), :] 取每个块的最后 1 个 (索引2)：
        b 组 (要保留的 dst) 拿到了: [T2] 和 [T5]。
        """
        a, b = (
            x[:, :, : (k - 1), :].contiguous().view(x.shape[0], -1, x.shape[-1]),
            x[:, :, (k - 1), :],
        )
        # a 是源头 (要消失的)，b 是目标 (要留下并吸收别人的)
        return a, b

    # ==========================================================
    # 找朋友阶段 (无梯度计算)
    # ==========================================================
    with torch.no_grad():
        # L2 归一化，为计算余弦相似度做准备
        metric = metric / metric.norm(dim=-1, keepdim=True)
        
        # 调用上面的 split 函数，把用于打分的特征切成 a 和 b
        a, b = split(metric)
        
        # r 记录了到底有多少个 Token 即将被献祭 (合并掉)
        r = a.shape[1] 
        
        # 计算全局相似度矩阵
        # a 里面的每一个 Token，都要去和 b 里面的每一个 Token 算一下相似度
        scores = a @ b.transpose(-1, -2)

        # 为 a 组的每一个 Token，在 b 组里找一个最像的接盘侠
        # 注意：这里和基础版不同，它不需要排序 argsort！
        # 因为在 kth 版本里，a 组全军覆没，统统都要被合并，所以根本不需要挑出“谁留下谁走”，只要找到去哪就行了。
        _, dst_idx = scores.max(dim=-1)
        
        # 增加一个空维度，方便后面 gather 取值
        dst_idx = dst_idx[..., None]

    # ==========================================================
    # 物理合并执行函数
    # ==========================================================
    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        # 把真实带有梯度的特征 x 切分成源 (src) 和目标 (dst)
        src, dst = split(x)
        n, _, c = src.shape
        
        # 极其暴力的合并：
        # 把 src 里的所有 Token，按照 dst_idx 的地址，全部强行砸进 dst 里面！
        # 如果好几个 src 的 Token 都觉得同一个 dst 最像，它们就会在那个 dst 身上累加并求平均。
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        # 直接返回 dst。
        # 极其狠毒，a 组 (src) 的形体彻底消失，只剩下吸饱了别人特征的 b 组 (dst)。
        # 此时 Token 数量已经变成了原来的 1/k。
        return dst

    # ==========================================================
    # 反合并执行函数 (用于密集预测任务恢复分辨率)
    # ==========================================================
    def unmerge(x: torch.Tensor) -> torch.Tensor:
        n, _, c = x.shape
        # 此时传进来的 x，就是经过了深层网络洗礼的 dst
        dst = x

        # 克隆技术：根据当初的合并地图 dst_idx，从进化后的 dst 身上，
        # 把属于牺牲者的那份特征硬生生地“拔”出来，作为复活体 src。
        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c)).to(x.dtype)

        # 重新打包成刚才折叠的块状结构
        # src 变回每个块有 k-1 个 Token 的形状
        src = src.view(n, -1, (k - 1), c)
        # dst 变回每个块有 1 个 Token 的形状
        dst = dst.view(n, -1, 1, c)

        # 拼图游戏：把块内的前 k-1 个 (src) 和最后 1 个 (dst) 重新拼起来
        out = torch.cat([src, dst], dim=-2)
        
        # 把块状结构彻底展平，恢复成最初的 [B, N, C] 的一维序列外观
        out = out.contiguous().view(n, -1, c)

        return out

    return merge, unmerge


def random_bipartite_soft_matching(
    metric: torch.Tensor, r: int
) -> Tuple[Callable, Callable]:
    """
    应用 ToMe，将所有 Token 分为两组：(随机抽取的 r 个 Token，剩下的所有 Token)。
    输入形状是 [batch, tokens, channels] (也就是 B, N, C)。
    
    这个函数会在这一层精准地砍掉 r 个 Token。
    """
    if r <= 0:
        return do_nothing, do_nothing

    # ==========================================================
    # 找朋友阶段：随机分组与匹配 (无梯度计算)
    # ==========================================================
    with torch.no_grad():
        # 获取 B (Batch 大小) 和 N (当前 Token 总数)
        B, N, _ = metric.shape
        
        # 【神仙操作：如何为每个 Batch 独立生成不重复的随机索引？】
        # torch.rand 生成 0~1 的随机小数。
        # argsort 将这些随机小数排序，返回的就是一个打乱的 0 ~ N-1 的随机索引序列！
        """
        举例说明 argsort 随机打乱法:
        假设 N=5。
        torch.rand 生成了: [0.8, 0.2, 0.9, 0.1, 0.5]
        argsort 排序后得到索引: [3, 1, 4, 0, 2]
        我们就完美得到了一串随机的、且不重复的 Token 序号！
        """
        rand_idx = torch.rand(B, N, 1, device=metric.device).argsort(dim=1)

        # 暴力切分：
        # a_idx 拿走排在最前面的 r 个随机索引。它们是被选中的“牺牲者” (源集合 src)
        a_idx = rand_idx[:, :r, :]
        # b_idx 拿走剩下的 N-r 个随机索引。它们是“幸存者” (目标集合 dst)
        b_idx = rand_idx[:, r:, :]

        # 定义一个内部闭包函数，用来根据刚才算好的随机索引，把真实的特征提取出来
        def split(x):
            # 获取特征维度 C (比如 768)
            C = x.shape[-1]
            # 查字典：根据 a_idx 提供的位置，把要牺牲的 r 个 Token 的完整 768 维特征掏出来
            a = x.gather(dim=1, index=a_idx.expand(B, r, C))
            # 查字典：根据 b_idx 提供的位置，把幸存的 N-r 个 Token 的特征掏出来
            b = x.gather(dim=1, index=b_idx.expand(B, N - r, C))
            return a, b

        # 1. 对用于打分的 metric 进行 L2 归一化
        metric = metric / metric.norm(dim=-1, keepdim=True)
        # 2. 调用上面的 split 函数，把特征正式切分成 a 和 b 两组
        a, b = split(metric)
        
        # 3. 算相似度矩阵：a 里面的这 r 个倒霉蛋，分别去看看 b 里面的幸存者谁和自己最像
        scores = a @ b.transpose(-1, -2)

        # 4. 找最大匹配：对于 a 中的每个 Token，找出 b 中最像它的那个 Token 的索引
        _, dst_idx = scores.max(dim=-1)
        # 加上一个空维度，变成 (B, r, 1)，准备后续操作
        dst_idx = dst_idx[..., None]

    # ==========================================================
    # 物理合并执行函数
    # ==========================================================
    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        # 此时的 x 是带有梯度的真实网络中间层特征
        # 切分成 src (那 r 个要消失的) 和 dst (剩下 N-r 个要留下的)
        src, dst = split(x)
        C = src.shape[-1]
        
        # 发快递 (物理融合)：
        # 拿着 src 的特征，按照前面算好的匹配地址簿 dst_idx，强行投递到 dst 的对应位置里。
        # 如果两人撞车了 (多个 src 觉得同一个 dst 最像)，就把特征加起来求平均 (mode="mean")。
        dst = dst.scatter_reduce(-2, dst_idx.expand(B, r, C), src, reduce=mode)

        # 直接返回融合后的 dst。此时 Token 总数从 N 变成了 N-r。
        return dst

    # ==========================================================
    # 反合并执行函数 (用于密集预测任务，如分割，需要恢复原图分辨率)
    # ==========================================================
    def unmerge(x: torch.Tensor) -> torch.Tensor:
        C = x.shape[-1]
        
        # 此时传进来的 x，是经历了后续 Transformer 层的 dst (数量为 N-r)
        dst = x
        
        # 克隆复活术：
        # 回忆起当初这 r 个牺牲者是和 dst 里的谁合并的 (依据 dst_idx)，
        # 直接从进化后的 dst 身上，把特征拷贝一份出来，作为牺牲者的复活体 src。
        src = dst.gather(dim=-2, index=dst_idx.expand(B, r, C))

        # 制造一个全新的、全都是零的张量 out，大小和最初没压缩前一模一样 (B, N, C)
        out = torch.zeros(B, N, C, device=x.device, dtype=x.dtype)

        # 【原位填坑】：这是这个版本独有的精妙操作！
        # 因为一开始大家是随机打乱的，现在必须按原本的位置填回去。
        
        # 发快递：把复活的 src 按照当初记录的随机位置字典 a_idx，精准塞回它最开始的坑位里。
        out.scatter_(dim=-2, index=a_idx.expand(B, r, C), src=src)
        
        # 发快递：把幸存的 dst 按照当初记录的随机位置字典 b_idx，塞回它原本的坑位里。
        out.scatter_(dim=-2, index=b_idx.expand(B, N - r, C), src=dst)

        # 返回完整长度的张量，神不知鬼不觉，就像从来没有打乱过一样！
        return out

    return merge, unmerge


def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    # 核心功能：调用你传进来的 merge 函数，但不是简单粗暴地求平均，
    # 而是根据每个 Token 内部包含的“原始 Token 数量 (size)”来进行【加权平均】。
    # 最终返回融合后的新特征张量 x，以及更新后的 size 张量。
    """
    
    if size is None:
        # 如果是第 1 层，还没有 size 参数传进来，我们就凭空初始化一个。
        # x[..., 0, None] 的作用是巧妙地把形状从 (B, N, C) 变成 (B, N, 1)。
        # torch.ones_like 制造一个全为 1 的张量。
        # 意思是：初始状态下，这 N 个 Token，每个人的体型/权重都是 1。
        size = torch.ones_like(x[..., 0, None])

    # ==========================================================
    # 举例说明加权平均逻辑：
    # 假设此时要合并 Token1 (特征为 10，体型为 3) 和 Token2 (特征为 20，体型为 1)
    # ==========================================================
    
    # 1. 放大特征：用真实特征乘以它的体型大小
    # Token1 变成 10 * 3 = 30；Token2 变成 20 * 1 = 20
    # 然后调用 merge(mode="sum") 把它们【相加】 (而不是求平均)
    # 相加结果：30 + 20 = 50
    x = merge(x * size, mode="sum")
    
    # 2. 放大体型：把两个 Token 的体型也【相加】
    # 相加结果：3 + 1 = 4
    # 此时新的 Token 体型变成了 4
    size = merge(size, mode="sum")

    # 3. 归一化求平均：把相加后的总特征，除以相加后的总体型
    # 结果：50 / 4 = 12.5 
    # (注意：如果是普通的直接求平均，结果会是 (10+20)/2 = 15，显然加权平均 12.5 更偏向于体型大的 Token1，这才是合理的！)
    x = x / size
    
    # 最终返回加权平均后的新特征，以及更新后的体型清单，传给下一层继续用
    return x, size


def merge_source(
    merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    # 核心功能：用于溯源追踪。
    # source 实际上是一个“邻接矩阵 (Adjacency Matrix)”，它记录了最终存活的 Token 和最初始的 Token 之间的血缘关系。
    
    x is used to find out how many tokens there are in case the source is None.
    # 这里的特征 x 其实不参与计算，仅仅是为了在第一层时，让我们能通过 x.shape 知道总共有多少个 Token。
    """
    
    if source is None:
        # 如果是第一层，建立初始的“账本”。
        n, t, _ = x.shape  # n是Batch, t是最初始的Token数量 (比如 196)
        
        # torch.eye(t) 会生成一个对角线为 1、其余为 0 的 t * t 单位矩阵。
        # [None, ...].expand(n, t, t) 是把它扩展到 Batch 维度，变成 (B, 196, 196)。
        """
        账本初始状态 (假设 t=3):
        [[1, 0, 0],  <- 当前第 0 个 Token 说：我是由初始的第 0 个 Patch 变来的
         [0, 1, 0],  <- 当前第 1 个 Token 说：我是由初始的第 1 个 Patch 变来的
         [0, 0, 1]]  <- 当前第 2 个 Token 说：我是由初始的第 2 个 Patch 变来的
        """
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    # ==========================================================
    # 【神来之笔】：用 mode="amax" (最大值池化) 来合并账本
    # ==========================================================
    # 如果 Token_0 和 Token_1 发生了合并，我们需要更新账本。
    # 为什么要用 amax (取最大值)？
    # Token_0 的账本记录是 [1, 0, 0]
    # Token_1 的账本记录是 [0, 1, 0]
    # 拿着这两个向量求最大值，结果就是 [1, 1, 0]！
    # 这完美地表示：合并后的新 Token，体内同时流淌着初始第 0 个和第 1 个 Patch 的血液！
    source = merge(source, mode="amax")
    
    # 将更新后的溯源矩阵传给下一层。
    # 等到了第 12 层，这个 source 矩阵的形状可能会变成 (B, 20, 196)
    # 你只要拿出来一看，就知道最后剩下的 20 个 Token，每一个究竟吞噬了原图上的哪些区域。
    return source