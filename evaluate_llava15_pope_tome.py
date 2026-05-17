from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

from edge.cna import CNA_Allocator
from edge.tome.patch.clip import apply_patch_clip


DEFAULT_RETAIN_RATIOS = [1.0, 0.8, 0.6, 0.4, 0.2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate native LLaVA-1.5 on POPE with and without ToMe vision-token compression. "
            "The cloud side uses LLaVA's original projector and LLM; no trained prism projector is loaded."
        )
    )
    parser.add_argument("--model_name_or_path", type=str, required=True, help="LLaVA-1.5 HF model path.")
    parser.add_argument("--data_path", type=str, required=True, help="POPE annotation JSON/JSONL path.")
    parser.add_argument("--image_folder", type=str, required=True, help="Folder containing POPE images.")
    parser.add_argument("--retain_ratios", type=float, nargs="+", default=DEFAULT_RETAIN_RATIOS)
    parser.add_argument("--output_path", type=str, default="outputs/llava15_pope_tome_eval.json")
    parser.add_argument("--save_predictions", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype. Use float16 on A800 unless you have a reason not to.",
    )
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--max_samples", type=int, default=0, help="Debug only; 0 means all samples.")
    parser.add_argument("--num_layers", type=int, default=24)
    parser.add_argument("--total_tokens", type=int, default=576)
    parser.add_argument("--max_drop", type=int, default=575)
    parser.add_argument(
        "--question_suffix",
        type=str,
        default="Please answer yes or no.",
        help="Appended to POPE questions unless the question already contains 'yes or no'.",
    )
    parser.add_argument(
        "--candidate_prefix",
        type=str,
        default=" ",
        help="Prefix used when scoring candidate answers. A leading space matches LLaMA tokenization better.",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Warmup forwards per retain ratio.")
    return parser.parse_args()


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _sync_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_pope_dataset(data_path: str) -> List[Dict[str, Any]]:
    with open(data_path, "r", encoding="utf-8") as f:
        raw = f.read()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in raw.splitlines() if line.strip()]

    if isinstance(data, dict):
        for key in ["data", "annotations", "samples"]:
            if isinstance(data.get(key), list):
                data = data[key]
                break

    if not isinstance(data, list):
        raise ValueError("POPE data_path must contain a JSON list, JSONL rows, or a dict with data/annotations/samples.")
    return data


def resolve_image_path(image_folder: str, sample: Dict[str, Any]) -> str:
    image_file = sample.get("image") or sample.get("image_path") or sample.get("file_name")
    if image_file is None and sample.get("image_id") is not None:
        image_id = sample.get("image_id")
        try:
            image_file = f"COCO_val2014_{int(image_id):012d}.jpg"
        except (TypeError, ValueError):
            image_file = str(image_id)
    if image_file is None:
        raise ValueError(f"Sample is missing image path/id: {sample}")

    image_file = str(image_file)
    candidates = []
    if Path(image_file).is_absolute():
        candidates.append(Path(image_file))
    else:
        candidates.append(Path(image_folder) / image_file)
        candidates.append(Path(image_folder) / Path(image_file).name)

    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[0])


def normalize_question(sample: Dict[str, Any], question_suffix: str) -> str:
    question = sample.get("text") or sample.get("question") or sample.get("prompt")
    if question is None:
        raise ValueError(f"POPE sample is missing question/text/prompt: {sample}")
    question = str(question).strip()
    if question_suffix and "yes or no" not in question.lower():
        question = f"{question}\n{question_suffix.strip()}"
    return question


def normalize_label(sample: Dict[str, Any]) -> str:
    label = sample.get("label") or sample.get("answer")
    if label is None:
        raise ValueError(f"POPE sample is missing label/answer: {sample}")
    label = str(label).strip().lower()
    if label.startswith("yes"):
        return "yes"
    if label.startswith("no"):
        return "no"
    raise ValueError(f"POPE label must be yes/no, got: {label!r}")


def build_exact_r_list(drop_tokens: int, allocator: CNA_Allocator) -> List[int]:
    drop_tokens = int(max(0, min(drop_tokens, allocator.total_tokens - 1)))
    if drop_tokens == 0:
        return [0] * allocator.num_layers

    weights = [float(w) for w in allocator.weights]
    raw = [drop_tokens * w for w in weights]
    r_list = [int(math.floor(v)) for v in raw]
    remainder = drop_tokens - sum(r_list)

    if remainder > 0:
        order = sorted(range(len(raw)), key=lambda i: raw[i] - r_list[i], reverse=True)
        for i in order[:remainder]:
            r_list[i] += 1
    return r_list


def set_vision_retain_ratio(vision_tower, retain_ratio: float, allocator: CNA_Allocator) -> Dict[str, Any]:
    retain_ratio = float(retain_ratio)
    target_keep_tokens = int(round(allocator.total_tokens * retain_ratio))
    target_keep_tokens = max(1, min(allocator.total_tokens, target_keep_tokens))
    drop_tokens = allocator.total_tokens - target_keep_tokens
    r_list = build_exact_r_list(drop_tokens, allocator)
    if hasattr(vision_tower, "r"):
        vision_tower.r = r_list
    return {
        "target_keep_tokens": target_keep_tokens,
        "target_drop_tokens": drop_tokens,
        "r_list": r_list,
    }


def get_vision_tower(model):
    if hasattr(model, "vision_tower"):
        return model.vision_tower
    if hasattr(model, "model") and hasattr(model.model, "vision_tower"):
        return model.model.vision_tower
    raise AttributeError("Could not find LLaVA vision_tower on the model.")


def get_language_model(model):
    if hasattr(model, "language_model"):
        return model.language_model
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model
    raise AttributeError("Could not find LLaVA language_model on the model.")


def get_input_embedding_layer(model):
    try:
        return model.get_input_embeddings()
    except Exception:
        return get_language_model(model).get_input_embeddings()


def _select_vision_features(model, vision_outputs):
    feature_layer = getattr(model.config, "vision_feature_layer", -2)
    select_strategy = getattr(model.config, "vision_feature_select_strategy", "default")
    hidden_states = vision_outputs.hidden_states

    if isinstance(feature_layer, int):
        selected = hidden_states[feature_layer]
    elif isinstance(feature_layer, Iterable):
        selected = torch.cat([hidden_states[int(i)] for i in feature_layer], dim=-1)
    else:
        raise ValueError(f"Unsupported vision_feature_layer: {feature_layer}")

    if select_strategy == "default":
        selected = selected[:, 1:]
    elif select_strategy == "full":
        pass
    else:
        raise ValueError(f"Unsupported vision_feature_select_strategy: {select_strategy}")
    return selected


@torch.no_grad()
def build_image_embeds(
    model,
    processor,
    image_path: str,
    retain_ratio: float,
    allocator: CNA_Allocator,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    vision_tower = get_vision_tower(model)
    tome_info = set_vision_retain_ratio(vision_tower, retain_ratio, allocator)

    image = Image.open(image_path).convert("RGB")
    pixel_values = processor.image_processor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device=device, dtype=dtype)

    _sync_device(device)
    vision_start = time.perf_counter()
    vision_outputs = vision_tower(pixel_values, output_hidden_states=True)
    _sync_device(device)
    vision_latency_ms = (time.perf_counter() - vision_start) * 1000

    selected = _select_vision_features(model, vision_outputs)
    actual_keep_tokens = int(selected.shape[1])

    _sync_device(device)
    projector_start = time.perf_counter()
    image_embeds = model.multi_modal_projector(selected)
    _sync_device(device)
    projector_latency_ms = (time.perf_counter() - projector_start) * 1000

    hidden_size = int(selected.shape[-1])
    bytes_fp16 = actual_keep_tokens * hidden_size * 2
    bytes_int8 = actual_keep_tokens * hidden_size + actual_keep_tokens * 2

    return {
        "image_embeds": image_embeds,
        "actual_keep_tokens": actual_keep_tokens,
        "target_keep_tokens": int(tome_info["target_keep_tokens"]),
        "target_drop_tokens": int(tome_info["target_drop_tokens"]),
        "vision_latency_ms": vision_latency_ms,
        "projector_latency_ms": projector_latency_ms,
        "feature_bytes_fp16": bytes_fp16,
        "feature_bytes_int8": bytes_int8,
    }


def build_prompt(question: str) -> str:
    return f"USER: <image>\n{question} ASSISTANT:"


def build_prefix_embeds(model, tokenizer, prompt: str, image_embeds: torch.Tensor, device: torch.device):
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    image_token_id = int(getattr(model.config, "image_token_index", tokenizer.convert_tokens_to_ids("<image>")))
    image_positions = torch.nonzero(input_ids[0].eq(image_token_id), as_tuple=False).flatten()
    if image_positions.numel() == 0:
        raise ValueError(
            "Tokenizer did not produce an <image> token. Check that the model is a LLaVA tokenizer "
            "and the prompt contains '<image>'."
        )

    start = int(image_positions[0].item())
    end = int(image_positions[-1].item()) + 1
    expected = torch.arange(start, end, device=image_positions.device)
    if image_positions.numel() != expected.numel() or not torch.equal(image_positions, expected):
        raise ValueError("<image> token positions are not contiguous; cannot safely replace them.")

    text_embeds = get_input_embedding_layer(model)(input_ids)
    embeds_before = text_embeds[:, :start]
    embeds_after = text_embeds[:, end:]
    mask_before = attention_mask[:, :start]
    mask_after = attention_mask[:, end:]
    image_mask = torch.ones(
        (input_ids.shape[0], image_embeds.shape[1]),
        dtype=attention_mask.dtype,
        device=device,
    )

    prefix_embeds = torch.cat([embeds_before, image_embeds.to(dtype=text_embeds.dtype), embeds_after], dim=1)
    prefix_attention_mask = torch.cat([mask_before, image_mask, mask_after], dim=1)
    return prefix_embeds, prefix_attention_mask


def tokenize_candidate(tokenizer, candidate: str, device: torch.device) -> torch.Tensor:
    ids = tokenizer(candidate, add_special_tokens=False).input_ids
    if len(ids) > 0 and isinstance(ids[0], list):
        ids = ids[0]
    if not ids:
        raise ValueError(f"Candidate produced no tokens: {candidate!r}")
    return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)


@torch.no_grad()
def score_yes_no(
    model,
    tokenizer,
    prefix_embeds: torch.Tensor,
    prefix_attention_mask: torch.Tensor,
    candidate_prefix: str,
    device: torch.device,
) -> Dict[str, Any]:
    language_model = get_language_model(model)
    scores: Dict[str, float] = {}
    scoring_latency_ms = 0.0

    for label in ["yes", "no"]:
        candidate_ids = tokenize_candidate(tokenizer, f"{candidate_prefix}{label}", device)
        candidate_embeds = get_input_embedding_layer(model)(candidate_ids)
        inputs_embeds = torch.cat([prefix_embeds, candidate_embeds], dim=1)
        candidate_attention = torch.ones_like(candidate_ids, device=device)
        attention_mask = torch.cat([prefix_attention_mask, candidate_attention], dim=1)
        labels = torch.full(inputs_embeds.shape[:2], -100, dtype=torch.long, device=device)
        labels[:, -candidate_ids.shape[1] :] = candidate_ids

        _sync_device(device)
        start = time.perf_counter()
        out = language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        _sync_device(device)
        scoring_latency_ms += (time.perf_counter() - start) * 1000
        scores[label] = -float(out.loss.item())

    prediction = "yes" if scores["yes"] >= scores["no"] else "no"
    confidence = abs(scores["yes"] - scores["no"])
    return {
        "prediction": prediction,
        "yes_score": scores["yes"],
        "no_score": scores["no"],
        "confidence": confidence,
        "scoring_latency_ms": scoring_latency_ms,
    }


def empty_stats() -> Dict[str, float]:
    return {
        "tp": 0.0,
        "tn": 0.0,
        "fp": 0.0,
        "fn": 0.0,
        "total": 0.0,
        "yes_predictions": 0.0,
        "target_keep_tokens_sum": 0.0,
        "actual_keep_tokens_sum": 0.0,
        "feature_bytes_fp16_sum": 0.0,
        "feature_bytes_int8_sum": 0.0,
        "vision_latency_ms_sum": 0.0,
        "projector_latency_ms_sum": 0.0,
        "scoring_latency_ms_sum": 0.0,
        "total_latency_ms_sum": 0.0,
        "confidence_sum": 0.0,
    }


def update_stats(stats: Dict[str, float], label: str, result: Dict[str, Any]):
    prediction = result["prediction"]
    stats["total"] += 1
    stats["yes_predictions"] += 1 if prediction == "yes" else 0
    stats["target_keep_tokens_sum"] += float(result["target_keep_tokens"])
    stats["actual_keep_tokens_sum"] += float(result["actual_keep_tokens"])
    stats["feature_bytes_fp16_sum"] += float(result["feature_bytes_fp16"])
    stats["feature_bytes_int8_sum"] += float(result["feature_bytes_int8"])
    stats["vision_latency_ms_sum"] += float(result["vision_latency_ms"])
    stats["projector_latency_ms_sum"] += float(result["projector_latency_ms"])
    stats["scoring_latency_ms_sum"] += float(result["scoring_latency_ms"])
    stats["total_latency_ms_sum"] += float(result["total_latency_ms"])
    stats["confidence_sum"] += float(result["confidence"])

    if label == "yes" and prediction == "yes":
        stats["tp"] += 1
    elif label == "no" and prediction == "no":
        stats["tn"] += 1
    elif label == "no" and prediction == "yes":
        stats["fp"] += 1
    elif label == "yes" and prediction == "no":
        stats["fn"] += 1


def finalize_stats(stats: Dict[str, float], total_tokens: int) -> Dict[str, float]:
    tp, tn, fp, fn = stats["tp"], stats["tn"], stats["fp"], stats["fn"]
    total = max(1.0, stats["total"])
    accuracy = (tp + tn) / total
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    specificity = tn / max(1.0, tn + fp)
    fpr = fp / max(1.0, fp + tn)
    fnr = fn / max(1.0, fn + tp)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "specificity": specificity,
        "yes_ratio": stats["yes_predictions"] / total,
        "avg_target_keep_tokens": stats["target_keep_tokens_sum"] / total,
        "avg_actual_keep_tokens": stats["actual_keep_tokens_sum"] / total,
        "avg_actual_retain_ratio": stats["actual_keep_tokens_sum"] / total / total_tokens,
        "avg_feature_bytes_fp16": stats["feature_bytes_fp16_sum"] / total,
        "avg_feature_bytes_int8": stats["feature_bytes_int8_sum"] / total,
        "vision_latency_ms": stats["vision_latency_ms_sum"] / total,
        "projector_latency_ms": stats["projector_latency_ms_sum"] / total,
        "scoring_latency_ms": stats["scoring_latency_ms_sum"] / total,
        "total_latency_ms": stats["total_latency_ms_sum"] / total,
        "avg_confidence": stats["confidence_sum"] / total,
        "samples_per_second": 1000.0 / max(1e-12, stats["total_latency_ms_sum"] / total),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "total": int(stats["total"]),
    }


def evaluate_ratio(
    model,
    processor,
    samples: List[Dict[str, Any]],
    args: argparse.Namespace,
    retain_ratio: float,
    allocator: CNA_Allocator,
    device: torch.device,
    dtype: torch.dtype,
    pred_fh,
) -> Dict[str, Any]:
    stats = empty_stats()
    tokenizer = processor.tokenizer
    desc = f"POPE retain={retain_ratio:.2f}"

    iterator = samples
    if args.max_samples and args.max_samples > 0:
        iterator = samples[: args.max_samples]

    for index, sample in enumerate(tqdm(iterator, desc=desc)):
        question = normalize_question(sample, args.question_suffix)
        label = normalize_label(sample)
        image_path = resolve_image_path(args.image_folder, sample)
        prompt = build_prompt(question)

        _sync_device(device)
        total_start = time.perf_counter()
        image_result = build_image_embeds(
            model=model,
            processor=processor,
            image_path=image_path,
            retain_ratio=retain_ratio,
            allocator=allocator,
            device=device,
            dtype=dtype,
        )
        prefix_embeds, prefix_attention_mask = build_prefix_embeds(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            image_embeds=image_result["image_embeds"],
            device=device,
        )
        score_result = score_yes_no(
            model=model,
            tokenizer=tokenizer,
            prefix_embeds=prefix_embeds,
            prefix_attention_mask=prefix_attention_mask,
            candidate_prefix=args.candidate_prefix,
            device=device,
        )
        _sync_device(device)
        total_latency_ms = (time.perf_counter() - total_start) * 1000

        result = {
            **{k: v for k, v in image_result.items() if k != "image_embeds"},
            **score_result,
            "total_latency_ms": total_latency_ms,
        }
        update_stats(stats, label, result)

        if pred_fh is not None:
            pred_fh.write(
                json.dumps(
                    {
                        "index": index,
                        "retain_ratio": retain_ratio,
                        "image": sample.get("image") or sample.get("image_path") or sample.get("file_name"),
                        "question": question,
                        "label": label,
                        "prediction": result["prediction"],
                        "yes_score": result["yes_score"],
                        "no_score": result["no_score"],
                        "confidence": result["confidence"],
                        "actual_keep_tokens": result["actual_keep_tokens"],
                        "feature_bytes_int8": result["feature_bytes_int8"],
                        "feature_bytes_fp16": result["feature_bytes_fp16"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return finalize_stats(stats, args.total_tokens)


def warmup_model(model, processor, samples, args, retain_ratio, allocator, device, dtype):
    if args.warmup <= 0 or len(samples) == 0:
        return
    tokenizer = processor.tokenizer
    sample = samples[0]
    question = normalize_question(sample, args.question_suffix)
    image_path = resolve_image_path(args.image_folder, sample)
    prompt = build_prompt(question)
    for _ in range(args.warmup):
        image_result = build_image_embeds(model, processor, image_path, retain_ratio, allocator, device, dtype)
        prefix_embeds, prefix_attention_mask = build_prefix_embeds(
            model, tokenizer, prompt, image_result["image_embeds"], device
        )
        _ = score_yes_no(model, tokenizer, prefix_embeds, prefix_attention_mask, args.candidate_prefix, device)


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    dtype = _dtype_from_name(args.dtype)

    samples = load_pope_dataset(args.data_path)
    if args.max_samples and args.max_samples > 0:
        print(f"[INFO] max_samples={args.max_samples}, full dataset size={len(samples)}")
    print(f"[INFO] loading LLaVA model from {args.model_name_or_path}")
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, local_files_only=args.local_files_only)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()

    if processor.tokenizer.pad_token_id is None and processor.tokenizer.eos_token_id is not None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    allocator = CNA_Allocator(
        num_layers=args.num_layers,
        total_tokens=args.total_tokens,
        max_drop=args.max_drop,
    )

    ratios = [float(r) for r in args.retain_ratios]
    exact_baseline_ratios = [r for r in ratios if r >= 0.999]
    compressed_ratios = [r for r in ratios if r < 0.999]

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_fh = None
    if args.save_predictions:
        pred_path = Path(args.save_predictions)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_fh = open(pred_path, "w", encoding="utf-8")

    results: Dict[str, Any] = {}
    try:
        for ratio in exact_baseline_ratios:
            print(f"[INFO] evaluating exact no-compression baseline retain={ratio:.2f} before applying ToMe patch")
            warmup_model(model, processor, samples, args, ratio, allocator, device, dtype)
            results[f"{ratio:.2f}"] = evaluate_ratio(
                model, processor, samples, args, ratio, allocator, device, dtype, pred_fh
            )

        if compressed_ratios:
            print("[INFO] applying ToMe patch to LLaVA vision tower")
            apply_patch_clip(get_vision_tower(model), trace_source=False)
            for ratio in compressed_ratios:
                warmup_model(model, processor, samples, args, ratio, allocator, device, dtype)
                results[f"{ratio:.2f}"] = evaluate_ratio(
                    model, processor, samples, args, ratio, allocator, device, dtype, pred_fh
                )
    finally:
        if pred_fh is not None:
            pred_fh.close()

    payload = {
        "model_name_or_path": args.model_name_or_path,
        "data_path": args.data_path,
        "image_folder": args.image_folder,
        "eval_mode": "yes_no_likelihood",
        "retain_ratios": ratios,
        "metrics_by_retain_ratio": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[OK] saved metrics to {output_path}")


if __name__ == "__main__":
    main()
