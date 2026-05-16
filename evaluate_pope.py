from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mm.builder import MMConfig, build_model
from mm.collator import SimpleCollator
from mm.train import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate projector checkpoints on compressed POPE data.")
    parser.add_argument("--llm_name_or_path", type=str, required=True)
    parser.add_argument("--vision_name_or_path", type=str, required=True)
    parser.add_argument(
        "--projector_type",
        type=str,
        required=True,
        choices=["linear", "mlp", "perceiver", "source_packer"],
    )
    parser.add_argument("--data_path", type=str, required=True, help="Compressed POPE directory or manifest path.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="训练输出的 best.pt 或 last.pt。")
    parser.add_argument("--batch_size", type=int, default=1, help="POPE 生成评估建议保持 1。")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--num_queries", type=int, default=128)
    parser.add_argument("--mlp_depth", type=int, default=2)
    parser.add_argument("--freeze_llm", action="store_true", default=True)
    parser.add_argument("--freeze_vision", action="store_true", default=True)
    parser.add_argument("--use_tome", action="store_true", default=True)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--max_eval_samples", type=int, default=0, help="调试用；0 表示评测完整 POPE。")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--save_predictions", type=str, default=None)
    return parser.parse_args()


def _load_projector_checkpoint(model, checkpoint_path: str, expected_projector_type: str):
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "projector" in payload:
        ckpt_projector_type = payload.get("projector_type")
        if ckpt_projector_type is not None and ckpt_projector_type != expected_projector_type:
            raise ValueError(
                f"checkpoint projector_type={ckpt_projector_type}, "
                f"but current projector_type={expected_projector_type}"
            )
        state_dict = payload["projector"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    model.projector.load_state_dict(state_dict, strict=True)


def _slice_batch_row(batch: Dict[str, Any], row_idx: int, batch_size: int) -> Dict[str, Any]:
    sample = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
            sample[key] = value[row_idx : row_idx + 1]
        else:
            sample[key] = value
    return sample


def _extract_prompt_and_label(tokenizer, sample: Dict[str, Any], device: torch.device):
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
    label_ids = labels[answer_pos]
    label_text = tokenizer.decode(label_ids.tolist(), skip_special_tokens=True).strip().lower()
    label = _parse_yes_no(label_text)
    if label is None:
        return None
    return prompt_ids, prompt_attention_mask, label, label_text


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


def _parse_yes_no(text: str) -> Optional[str]:
    clean = text.strip().lower()
    if not clean:
        return None
    first = clean.replace(".", " ").replace(",", " ").replace(":", " ").replace(";", " ").split()[0]
    if first in {"yes", "yeah", "yep"}:
        return "yes"
    if first in {"no", "not", "nope"}:
        return "no"
    if clean.startswith("yes"):
        return "yes"
    if clean.startswith("no"):
        return "no"
    return None


@torch.no_grad()
def _generate_one(model, tokenizer, sample: Dict[str, Any], device: torch.device, max_new_tokens: int):
    extracted = _extract_prompt_and_label(tokenizer, sample, device)
    if extracted is None:
        return None
    prompt_ids, prompt_attention_mask, label, label_text = extracted
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
    generated = model.llm.generate(
        input_ids=dummy_input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
    )
    generated_ids = generated[0, input_len:] if generated.shape[1] > input_len else generated[0]
    prediction_text = tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True).strip()
    prediction = _parse_yes_no(prediction_text)
    return {
        "label": label,
        "label_text": label_text,
        "prediction": prediction,
        "prediction_text": prediction_text,
    }


def _empty_stats() -> Dict[str, float]:
    return {
        "tp": 0.0,
        "tn": 0.0,
        "fp": 0.0,
        "fn": 0.0,
        "invalid": 0.0,
        "total": 0.0,
        "yes_predictions": 0.0,
    }


def _update_stats(stats: Dict[str, float], label: str, prediction: Optional[str]):
    stats["total"] += 1
    if prediction == "yes":
        stats["yes_predictions"] += 1
    if prediction not in {"yes", "no"}:
        stats["invalid"] += 1
        prediction = "no"

    if label == "yes" and prediction == "yes":
        stats["tp"] += 1
    elif label == "no" and prediction == "no":
        stats["tn"] += 1
    elif label == "no" and prediction == "yes":
        stats["fp"] += 1
    elif label == "yes" and prediction == "no":
        stats["fn"] += 1


def _finalize(stats: Dict[str, float]) -> Dict[str, float]:
    tp, tn, fp, fn = stats["tp"], stats["tn"], stats["fp"], stats["fn"]
    total = max(1.0, stats["total"])
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    specificity = tn / max(1.0, tn + fp)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "accuracy": float((tp + tn) / total),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fp / max(1.0, fp + tn)),
        "false_negative_rate": float(fn / max(1.0, fn + tp)),
        "specificity": float(specificity),
        "yes_ratio": float(stats["yes_predictions"] / total),
        "invalid_ratio": float(stats["invalid"] / total),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "invalid": int(stats["invalid"]),
        "total": int(stats["total"]),
    }


def _retain_key(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


@torch.no_grad()
def evaluate(model, dataloader: DataLoader, device: torch.device, max_new_tokens: int, max_eval_samples: int = 0):
    model.eval()
    overall = _empty_stats()
    by_retain_ratio: Dict[str, Dict[str, float]] = {}
    predictions = []
    seen = 0

    progress = tqdm(total=len(dataloader), desc="Eval POPE", dynamic_ncols=True)
    for batch in dataloader:
        batch_size = int(batch["input_ids"].shape[0])
        if max_eval_samples > 0 and seen >= max_eval_samples:
            break

        for i in range(batch_size):
            if max_eval_samples > 0 and seen >= max_eval_samples:
                break
            sample = _slice_batch_row(batch, i, batch_size)
            result = _generate_one(model, model.tokenizer, sample, device, max_new_tokens=max_new_tokens)
            if result is None:
                seen += 1
                continue

            retain_ratio = batch["retain_ratio"][i].item() if "retain_ratio" in batch else None
            _update_stats(overall, result["label"], result["prediction"])
            if retain_ratio is not None:
                key = _retain_key(retain_ratio)
                _update_stats(by_retain_ratio.setdefault(key, _empty_stats()), result["label"], result["prediction"])

            predictions.append(
                {
                    "index": seen,
                    "retain_ratio": retain_ratio,
                    **result,
                }
            )
            seen += 1

        progress.update(1)
        progress.set_postfix(acc=f"{_finalize(overall)['accuracy']:.4f}")

    progress.close()
    return {
        "overall": _finalize(overall),
        "by_retain_ratio": {
            key: _finalize(stats)
            for key, stats in sorted(by_retain_ratio.items(), key=lambda item: item[0])
        },
        "predictions": predictions,
    }


def main():
    args = parse_args()
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
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    metrics = evaluate(model, dataloader, device, args.max_new_tokens, args.max_eval_samples)
    predictions = metrics.pop("predictions")
    result = {
        "projector_type": args.projector_type,
        "checkpoint_path": str(Path(args.checkpoint_path)),
        "data_path": str(Path(args.data_path)),
        "metrics": metrics,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[saved] {output_path}")

    if args.save_predictions:
        pred_path = Path(args.save_predictions)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pred_path, "w", encoding="utf-8") as f:
            for row in predictions:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[saved] {pred_path}")


if __name__ == "__main__":
    main()
