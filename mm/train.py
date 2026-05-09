from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from transformers import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm

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
# - 记录训练日志 / 验证日志
# - 显示进度条
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
    - 包含多个 .pt 样本文件的目录（支持递归搜索）
    """
    p = Path(path)
    if p.is_file() and p.suffix == ".pt":
        return CompressedFeatureDataset(path, tokenizer=tokenizer)

    if p.is_dir():
        pt_files = [str(x) for x in p.rglob("*.pt") if x.name not in {"best.pt", "last.pt"}]
        if not pt_files:
            raise ValueError(f"{path} 下没有找到 .pt shard。")

        # 递归收集到的 .pt 样本可以直接交给同一个 Dataset 处理。
        class _VirtualShardDataset(torch.utils.data.Dataset):
            def __init__(self, files, tokenizer):
                self.files = [Path(f) for f in files]
                self.tokenizer = tokenizer

            def __len__(self):
                return len(self.files)

            def __getitem__(self, idx):
                # 复用 CompressedFeatureDataset 对单个 .pt 的解析逻辑。
                return CompressedFeatureDataset(str(self.files[idx]), tokenizer=self.tokenizer)[0]

        return _VirtualShardDataset(pt_files, tokenizer=tokenizer)

    raise ValueError(f"无效的数据路径: {path}")


def _log_jsonl(path: str, record: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


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
    train_log_path = os.path.join(args.output_dir, "train_log.jsonl")
    eval_log_path = os.path.join(args.output_dir, "eval_log.jsonl")
    with open(os.path.join(args.output_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump({**vars(args), **asdict(config)}, f, indent=2, ensure_ascii=False)

    global_step = 0
    best_eval = float("inf")
    for epoch in range(args.epochs):
        running_loss = 0.0
        optim.zero_grad(set_to_none=True)

        progress = tqdm(total=len(train_loader), desc=f"Train {epoch + 1}/{args.epochs}", dynamic_ncols=True)
        for step, batch in enumerate(train_loader):
            # 只保留模型 forward 真正需要的字段，避免把统计字段传进 LLM。
            model_batch = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch.get("labels").to(device) if batch.get("labels") is not None else None,
            }
            if "pixel_values" in batch:
                model_batch["pixel_values"] = batch["pixel_values"].to(device)
            if "compressed_features" in batch:
                model_batch["compressed_features"] = batch["compressed_features"].to(device)
            if "compressed_attention_mask" in batch:
                model_batch["compressed_attention_mask"] = batch["compressed_attention_mask"].to(device)

            out = model(**model_batch)
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

                lr = sched.get_last_lr()[0] if hasattr(sched, "get_last_lr") else args.lr
                _log_jsonl(
                    train_log_path,
                    {
                        "type": "train",
                        "epoch": epoch,
                        "step": global_step,
                        "loss": float(running_loss),
                        "lr": float(lr),
                    },
                )

                if global_step % 10 == 0:
                    progress.set_postfix(loss=f"{running_loss:.4f}", step=global_step)
                running_loss = 0.0

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    ckpt = os.path.join(args.output_dir, f"checkpoint-{global_step}.pt")
                    torch.save({"model": model.state_dict(), "step": global_step}, ckpt)

            progress.update(1)
            progress.set_postfix(step=global_step)
        progress.close()

        if eval_loader is not None:
            model.eval()
            losses = []
            eval_progress = tqdm(total=len(eval_loader), desc=f"Eval {epoch + 1}/{args.epochs}", dynamic_ncols=True)
            with torch.no_grad():
                for batch in eval_loader:
                    model_batch = {
                        "input_ids": batch["input_ids"].to(device),
                        "attention_mask": batch["attention_mask"].to(device),
                        "labels": batch.get("labels").to(device) if batch.get("labels") is not None else None,
                    }
                    if "pixel_values" in batch:
                        model_batch["pixel_values"] = batch["pixel_values"].to(device)
                    if "compressed_features" in batch:
                        model_batch["compressed_features"] = batch["compressed_features"].to(device)
                    if "compressed_attention_mask" in batch:
                        model_batch["compressed_attention_mask"] = batch["compressed_attention_mask"].to(device)

                    out = model(**model_batch)
                    losses.append(out.loss.item())
                    eval_progress.update(1)
            eval_progress.close()
            eval_loss = float(sum(losses) / max(1, len(losses)))
            print(f"[eval] epoch={epoch} loss={eval_loss:.4f}")
            _log_jsonl(
                eval_log_path,
                {
                    "type": "eval",
                    "epoch": epoch,
                    "step": global_step,
                    "eval_loss": float(eval_loss),
                },
            )
            if eval_loss < best_eval:
                best_eval = eval_loss
                torch.save({"model": model.state_dict(), "step": global_step}, os.path.join(args.output_dir, "best.pt"))
            model.train()

    torch.save({"model": model.state_dict(), "step": global_step}, os.path.join(args.output_dir, "last.pt"))


if __name__ == "__main__":
    main()
