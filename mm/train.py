from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader, random_split
from transformers import get_cosine_schedule_with_warmup

from .builder import MMConfig, build_model
from .collator import SimpleCollator
from .dataset import CompressedFeatureDataset


# -----------------------------
# 训练入口
# -----------------------------
#
# 这部分负责：
# - 解析训练参数
# - 构建模型
# - 加载离线压缩特征数据集
# - 划分训练 / 验证集
# - 执行梯度累积、优化器更新、学习率调度、保存 checkpoint
#
# 这里默认 stage1 训练是冻结 LLM 和 vision tower，只训练 projector / ADP。


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_name_or_path", type=str, required=True, help="TinyLlama 等语言模型路径。")
    parser.add_argument("--vision_name_or_path", type=str, required=True, help="CLIP vision tower 路径。")
    parser.add_argument("--projector_type", type=str, default="linear", choices=["linear", "mlp", "adp"], help="视觉特征到 LLM 空间的映射模块类型。")
    parser.add_argument("--data_path", type=str, required=True, help="离线压缩特征目录或单个 .pt 文件路径。")
    parser.add_argument("--batch_size", type=int, default=1, help="训练 batch size。")
    parser.add_argument("--lr", type=float, default=2e-5, help="学习率。")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数。")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="权重衰减。")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="warmup 占总步数比例。")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader worker 数。")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="输出目录。")
    parser.add_argument("--save_steps", type=int, default=1000, help="每隔多少个更新步保存一次 checkpoint。")
    parser.add_argument("--eval_ratio", type=float, default=0.01, help="验证集比例。")
    parser.add_argument("--freeze_llm", action="store_true", default=True, help="是否冻结 LLM。默认冻结。")
    parser.add_argument("--freeze_vision", action="store_true", default=True, help="是否冻结 vision tower。默认冻结。")
    parser.add_argument("--use_tome", action="store_true", default=True, help="是否启用 ToMe vision patch。默认启用。")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数。")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪阈值。")
    parser.add_argument("--num_queries", type=int, default=128, help="ADP 中 query token 数量。")
    parser.add_argument("--mlp_depth", type=int, default=2, help="MLP projector 深度。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    return parser.parse_args()


def load_dataset(path: str, tokenizer):
    """加载离线压缩数据集。

    支持两种输入：
    - 单个 .pt 文件
    - 包含多个 .pt 样本文件的目录
    """
    if os.path.isdir(path):
        shards = []
        for fn in sorted(os.listdir(path)):
            if fn.endswith(".pt"):
                shards.append(CompressedFeatureDataset(os.path.join(path, fn), tokenizer=tokenizer))
        if not shards:
            raise ValueError(f"{path} 下没有找到 .pt shard。")

        class _Concat(torch.utils.data.Dataset):
            def __init__(self, parts):
                self.parts = parts
                self.offsets = []
                s = 0
                for p in parts:
                    s += len(p)
                    self.offsets.append(s)

            def __len__(self):
                return self.offsets[-1]

            def __getitem__(self, idx):
                for pi, off in enumerate(self.offsets):
                    if idx < off:
                        prev = 0 if pi == 0 else self.offsets[pi - 1]
                        return self.parts[pi][idx - prev]
                raise IndexError(idx)

        return _Concat(shards)

    return CompressedFeatureDataset(path, tokenizer=tokenizer)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = MMConfig(
        llm_name_or_path=args.llm_name_or_path,
        vision_name_or_path=args.vision_name_or_path,
        projector_type=args.projector_type,
        num_queries=args.num_queries,
        mlp_depth=args.mlp_depth,
        freeze_llm=args.freeze_llm,
        freeze_vision=args.freeze_vision,
        use_tome=args.use_tome,
    )
    model = build_model(config)
    tokenizer = model.tokenizer
    collator = SimpleCollator(pad_token_id=tokenizer.pad_token_id)

    dataset = load_dataset(args.data_path, tokenizer=tokenizer)
    eval_len = max(1, int(len(dataset) * args.eval_ratio)) if len(dataset) > 1 else 0
    train_len = len(dataset) - eval_len
    if eval_len > 0 and train_len > 0:
        train_ds, eval_ds = random_split(
            dataset,
            [train_len, eval_len],
            generator=torch.Generator().manual_seed(args.seed),
        )
    else:
        train_ds, eval_ds = dataset, None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    eval_loader = (
        DataLoader(
            eval_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collator,
        )
        if eval_ds is not None
        else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optim = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_update_steps = max(1, (len(train_loader) * args.epochs) // max(1, args.gradient_accumulation_steps))
    warmup_steps = max(1, int(total_update_steps * args.warmup_ratio))
    sched = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump({**vars(args), **asdict(config)}, f, indent=2, ensure_ascii=False)

    global_step = 0
    best_eval = float("inf")
    for epoch in range(args.epochs):
        running_loss = 0.0
        optim.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / max(1, args.gradient_accumulation_steps)
            loss.backward()
            running_loss += loss.item()

            should_step = (step + 1) % max(1, args.gradient_accumulation_steps) == 0 or (step + 1) == len(train_loader)
            if should_step:
                # 在真正更新参数前做一次梯度裁剪，避免梯度爆炸。
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 10 == 0:
                    print(f"epoch={epoch} step={global_step} loss={running_loss:.4f}")
                running_loss = 0.0

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    ckpt = os.path.join(args.output_dir, f"checkpoint-{global_step}.pt")
                    torch.save({"model": model.state_dict(), "step": global_step}, ckpt)

        if eval_loader is not None:
            model.eval()
            losses = []
            with torch.no_grad():
                for batch in eval_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    out = model(**batch)
                    losses.append(out.loss.item())
            eval_loss = float(sum(losses) / max(1, len(losses)))
            print(f"[eval] epoch={epoch} loss={eval_loss:.4f}")
            if eval_loss < best_eval:
                best_eval = eval_loss
                torch.save({"model": model.state_dict(), "step": global_step}, os.path.join(args.output_dir, "best.pt"))
            model.train()

    torch.save({"model": model.state_dict(), "step": global_step}, os.path.join(args.output_dir, "last.pt"))


if __name__ == "__main__":
    main()
