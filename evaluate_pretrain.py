from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from tqdm.auto import tqdm

from mm.builder import MMConfig, build_model
from mm.collator import SimpleCollator
from mm.train import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate projector checkpoints on the held-out LLaVA pretrain split."
    )
    parser.add_argument("--llm_name_or_path", type=str, required=True)
    parser.add_argument("--vision_name_or_path", type=str, required=True)
    parser.add_argument(
        "--projector_type",
        type=str,
        required=True,
        choices=["linear", "mlp", "perceiver", "source_packer"],
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True, help="训练输出的 best.pt 或 last.pt。")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--eval_ratio", type=float, default=None, help="兼容旧参数；设置后覆盖 test_ratio。")
    parser.add_argument(
        "--split_mode",
        type=str,
        default="random",
        choices=["random", "tail"],
        help="random 与训练脚本的 random_split(seed) 一致；tail 表示按顺序取最后 test_ratio。",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_queries", type=int, default=128)
    parser.add_argument("--mlp_depth", type=int, default=2)
    parser.add_argument("--freeze_llm", action="store_true", default=True)
    parser.add_argument("--freeze_vision", action="store_true", default=True)
    parser.add_argument("--use_tome", action="store_true", default=True)
    parser.add_argument("--max_eval_samples", type=int, default=0, help="调试用；0 表示评测完整验证集。")
    parser.add_argument("--compute_rouge_l", action="store_true", help="开启生成式 ROUGE-L 答案相似度评估。")
    parser.add_argument("--rouge_max_samples", type=int, default=1000, help="ROUGE-L 最多生成多少条；0 表示完整验证集。")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="生成式评估时的最大新 token 数。")
    parser.add_argument("--output_path", type=str, default=None, help="可选：保存 JSON 指标结果。")
    return parser.parse_args()


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    model_batch = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "labels": batch["labels"].to(device) if batch.get("labels") is not None else None,
    }
    for key in [
        "pixel_values",
        "compressed_features",
        "compressed_attention_mask",
        "token_centers",
        "token_sizes",
    ]:
        if key in batch:
            model_batch[key] = batch[key].to(device)
    return model_batch


def _load_projector_checkpoint(model, checkpoint_path: str, expected_projector_type: str):
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "projector" in payload:
        ckpt_projector_type = payload.get("projector_type")
        if ckpt_projector_type is not None and ckpt_projector_type != expected_projector_type:
            raise ValueError(
                f"checkpoint projector_type={ckpt_projector_type}, "
                f"但当前参数 projector_type={expected_projector_type}"
            )
        state_dict = payload["projector"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise ValueError(f"无法识别的 checkpoint 格式: {checkpoint_path}")
    model.projector.load_state_dict(state_dict, strict=True)


def _build_eval_dataset(dataset, ratio: float, split_mode: str, seed: int):
    if not 0 < ratio < 1:
        raise ValueError(f"test/eval ratio 必须在 (0, 1) 内，当前为 {ratio}")
    eval_len = max(1, int(len(dataset) * ratio))
    train_len = len(dataset) - eval_len
    if train_len <= 0:
        raise ValueError(f"数据集太小，无法按 ratio={ratio} 划分: len={len(dataset)}")

    if split_mode == "random":
        _, eval_ds = random_split(
            dataset,
            [train_len, eval_len],
            generator=torch.Generator().manual_seed(seed),
        )
    elif split_mode == "tail":
        eval_ds = Subset(dataset, range(train_len, len(dataset)))
    else:
        raise ValueError(f"未知 split_mode: {split_mode}")
    return eval_ds, train_len, eval_len


def _empty_stats() -> Dict[str, float]:
    return {
        "loss_sum": 0.0,
        "correct_tokens": 0.0,
        "total_tokens": 0.0,
        "num_samples": 0.0,
        "rouge_l_sum": 0.0,
        "rouge_l_count": 0.0,
    }


def _finalize_stats(stats: Dict[str, float]) -> Dict[str, float]:
    total_tokens = int(stats["total_tokens"])
    correct_tokens = int(stats["correct_tokens"])
    eval_loss = stats["loss_sum"] / max(1, total_tokens)
    ppl = math.exp(eval_loss) if eval_loss < 50 else float("inf")
    token_accuracy = correct_tokens / max(1, total_tokens)
    return {
        "eval_loss": float(eval_loss),
        "ppl": float(ppl),
        "token_accuracy": float(token_accuracy),
        "rouge_l": float(stats["rouge_l_sum"] / stats["rouge_l_count"]) if stats["rouge_l_count"] > 0 else None,
        "rouge_l_count": int(stats["rouge_l_count"]),
        "correct_tokens": correct_tokens,
        "total_tokens": total_tokens,
        "num_samples": int(stats["num_samples"]),
    }


def _retain_key(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def _rouge_tokens(text: str) -> list[str]:
    text = " ".join(text.lower().strip().split())
    if not text:
        return []
    tokens = text.split()
    if len(tokens) == 1:
        return list(text)
    return tokens


def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def _rouge_l_f1(prediction: str, reference: str) -> float:
    pred_tokens = _rouge_tokens(prediction)
    ref_tokens = _rouge_tokens(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_len(pred_tokens, ref_tokens)
    precision = lcs / max(1, len(pred_tokens))
    recall = lcs / max(1, len(ref_tokens))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _slice_batch_row(batch: Dict[str, Any], row_idx: int, batch_size: int) -> Dict[str, Any]:
    sample = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
            sample[key] = value[row_idx : row_idx + 1]
        else:
            sample[key] = value
    return sample


def _extract_prompt_and_reference(tokenizer, sample: Dict[str, Any], device: torch.device):
    input_ids = sample["input_ids"][0]
    attention_mask = sample["attention_mask"][0]
    labels = sample["labels"][0]
    text_len = int(attention_mask.sum().item())
    input_ids = input_ids[:text_len]
    labels = labels[:text_len]
    answer_pos = torch.nonzero(labels.ne(-100), as_tuple=False).flatten()
    if answer_pos.numel() == 0:
        return None

    prompt_len = int(answer_pos[0].item())
    prompt_ids = input_ids[:prompt_len].unsqueeze(0).to(device)
    prompt_attention_mask = torch.ones_like(prompt_ids, device=device)
    reference_ids = labels[answer_pos]
    reference_text = tokenizer.decode(reference_ids.tolist(), skip_special_tokens=True).strip()
    if not reference_text:
        return None
    return prompt_ids, prompt_attention_mask, reference_text


def _build_generation_inputs(model, sample: Dict[str, Any], prompt_ids: torch.Tensor, prompt_attention_mask: torch.Tensor):
    text_embeds = model.llm.get_input_embeddings()(prompt_ids)
    projector_dtype = model._projector_dtype()
    device = text_embeds.device

    image_attention_mask = None
    if "compressed_features" in sample:
        image_tokens = sample["compressed_features"].to(device, dtype=projector_dtype)
        if image_tokens.ndim == 2:
            image_tokens = image_tokens.unsqueeze(0)

        compressed_attention_mask = sample.get("compressed_attention_mask")
        if compressed_attention_mask is not None:
            compressed_attention_mask = compressed_attention_mask.to(device)
            if compressed_attention_mask.ndim == 1:
                compressed_attention_mask = compressed_attention_mask.unsqueeze(0)

        if model.projector_type == "perceiver":
            image_embeds = model.projector(image_tokens, attention_mask=compressed_attention_mask)
        elif model.projector_type == "source_packer":
            token_centers = sample.get("token_centers")
            token_sizes = sample.get("token_sizes")
            if token_centers is not None:
                token_centers = token_centers.to(device)
                if token_centers.ndim == 2:
                    token_centers = token_centers.unsqueeze(0)
            if token_sizes is not None:
                token_sizes = token_sizes.to(device)
                if token_sizes.ndim == 1:
                    token_sizes = token_sizes.unsqueeze(0)
            image_embeds = model.projector(
                image_tokens,
                attention_mask=compressed_attention_mask,
                token_centers=token_centers,
                token_sizes=token_sizes,
            )
        elif image_tokens.shape[-1] != model.config.hidden_size:
            image_embeds = model.projector(image_tokens)
            image_attention_mask = compressed_attention_mask
        else:
            image_embeds = image_tokens
            image_attention_mask = compressed_attention_mask
    elif "pixel_values" in sample:
        image_embeds = model.encode_images(sample["pixel_values"].to(device))
    else:
        image_embeds = None

    if image_embeds is None:
        return text_embeds, prompt_attention_mask

    image_embeds = image_embeds.to(dtype=text_embeds.dtype)
    inputs_embeds, attention_mask, _ = model._merge_text_and_image_embeddings(
        text_embeds=text_embeds,
        image_embeds=image_embeds,
        attention_mask=prompt_attention_mask,
        labels=None,
        image_attention_mask=image_attention_mask,
    )
    return inputs_embeds, attention_mask


@torch.no_grad()
def _generate_one(model, tokenizer, sample: Dict[str, Any], device: torch.device, max_new_tokens: int):
    extracted = _extract_prompt_and_reference(tokenizer, sample, device)
    if extracted is None:
        return None
    prompt_ids, prompt_attention_mask, reference_text = extracted
    inputs_embeds, attention_mask = _build_generation_inputs(model, sample, prompt_ids, prompt_attention_mask)
    input_len = inputs_embeds.shape[1]
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_token_id = tokenizer.eos_token_id
    dummy_input_ids = torch.full(
        (inputs_embeds.shape[0], input_len),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )

    try:
        generated = model.llm.generate(
            input_ids=dummy_input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
    except (TypeError, ValueError):
        generated = model.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )

    if generated.shape[1] > input_len:
        generated_ids = generated[0, input_len:]
    else:
        generated_ids = generated[0]
    prediction_text = tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True).strip()
    return prediction_text, reference_text, _rouge_l_f1(prediction_text, reference_text)


@torch.no_grad()
def evaluate(
    model,
    dataloader: DataLoader,
    device: torch.device,
    max_eval_samples: int = 0,
    compute_rouge_l: bool = False,
    rouge_max_samples: int = 1000,
    max_new_tokens: int = 64,
) -> Dict[str, Any]:
    model.eval()
    overall = _empty_stats()
    by_retain_ratio: Dict[str, Dict[str, float]] = {}
    seen_samples = 0
    rouge_seen = 0

    progress = tqdm(total=len(dataloader), desc="Eval pretrain", dynamic_ncols=True)
    for batch in dataloader:
        batch_size = int(batch["input_ids"].shape[0])
        if max_eval_samples > 0 and seen_samples >= max_eval_samples:
            break

        if max_eval_samples > 0 and seen_samples + batch_size > max_eval_samples:
            keep = max_eval_samples - seen_samples
            for key, value in list(batch.items()):
                if isinstance(value, torch.Tensor) and value.shape[0] == batch_size:
                    batch[key] = value[:keep]
            batch_size = keep

        model_batch = _move_batch_to_device(batch, device)
        out = model(**model_batch)

        text_labels = model_batch["labels"]
        image_len = out.logits.shape[1] - text_labels.shape[1]
        if image_len < 0:
            raise ValueError(
                f"logits length ({out.logits.shape[1]}) 小于 text label length ({text_labels.shape[1]})"
            )
        image_labels = torch.full(
            (text_labels.shape[0], image_len),
            -100,
            dtype=text_labels.dtype,
            device=text_labels.device,
        )
        full_labels = torch.cat([image_labels, text_labels], dim=1)

        shift_logits = out.logits[:, :-1, :].float()
        shift_labels = full_labels[:, 1:]
        token_losses = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape(shift_labels.shape)
        valid_mask = shift_labels.ne(-100)
        predictions = shift_logits.argmax(dim=-1)
        correct_mask = predictions.eq(shift_labels) & valid_mask

        loss_sum = float(token_losses[valid_mask].sum().item())
        correct = int(correct_mask.sum().item())
        total = int(valid_mask.sum().item())

        overall["loss_sum"] += loss_sum
        overall["correct_tokens"] += correct
        overall["total_tokens"] += total
        overall["num_samples"] += batch_size

        if "retain_ratio" in batch:
            retain_ratios = batch["retain_ratio"].detach().cpu().tolist()
            for i, retain_ratio in enumerate(retain_ratios):
                key = _retain_key(retain_ratio)
                stats = by_retain_ratio.setdefault(key, _empty_stats())
                row_mask = valid_mask[i]
                stats["loss_sum"] += float(token_losses[i][row_mask].sum().item())
                stats["correct_tokens"] += int(correct_mask[i].sum().item())
                stats["total_tokens"] += int(row_mask.sum().item())
                stats["num_samples"] += 1

        if compute_rouge_l and (rouge_max_samples <= 0 or rouge_seen < rouge_max_samples):
            for i in range(batch_size):
                if rouge_max_samples > 0 and rouge_seen >= rouge_max_samples:
                    break
                sample = _slice_batch_row(batch, i, batch_size)
                generated = _generate_one(model, model.tokenizer, sample, device, max_new_tokens=max_new_tokens)
                if generated is None:
                    continue
                _, _, rouge_l = generated
                overall["rouge_l_sum"] += float(rouge_l)
                overall["rouge_l_count"] += 1
                if "retain_ratio" in batch:
                    retain_ratio = batch["retain_ratio"][i].item()
                    key = _retain_key(retain_ratio)
                    stats = by_retain_ratio.setdefault(key, _empty_stats())
                    stats["rouge_l_sum"] += float(rouge_l)
                    stats["rouge_l_count"] += 1
                rouge_seen += 1

        seen_samples += batch_size
        progress.update(1)
        progress.set_postfix(loss=f"{overall['loss_sum'] / max(1, overall['total_tokens']):.4f}")

    progress.close()
    return {
        "overall": _finalize_stats(overall),
        "by_retain_ratio": {
            key: _finalize_stats(stats)
            for key, stats in sorted(by_retain_ratio.items(), key=lambda item: item[0])
        },
    }


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
    _load_projector_checkpoint(model, args.checkpoint_path, args.projector_type)

    tokenizer = model.tokenizer
    collator = SimpleCollator(pad_token_id=tokenizer.pad_token_id)
    dataset = load_dataset(args.data_path, tokenizer=tokenizer)
    split_ratio = args.eval_ratio if args.eval_ratio is not None else args.test_ratio
    eval_ds, train_len, eval_len = _build_eval_dataset(dataset, split_ratio, args.split_mode, args.seed)
    print(f"[split] mode={args.split_mode} train={train_len} eval={eval_len} ratio={split_ratio}")

    dataloader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    metrics = evaluate(
        model,
        dataloader,
        device,
        max_eval_samples=args.max_eval_samples,
        compute_rouge_l=args.compute_rouge_l,
        rouge_max_samples=args.rouge_max_samples,
        max_new_tokens=args.max_new_tokens,
    )
    result = {
        "projector_type": args.projector_type,
        "checkpoint_path": str(Path(args.checkpoint_path)),
        "data_path": str(Path(args.data_path)),
        "split_mode": args.split_mode,
        "test_ratio": float(split_ratio),
        "seed": int(args.seed),
        "metrics": metrics,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
