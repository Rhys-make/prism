from __future__ import annotations

import argparse
import json
import time
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor, AutoTokenizer, CLIPImageProcessor, LlavaForConditionalGeneration


DEFAULT_BUDGET_SPECS = [
    "0.8:472986",
    "0.6:354996",
    "0.4:235980",
    "0.2:117990",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate native LLaVA-1.5 on POPE with compressed-image transmission baselines. "
            "The edge side JPEG/WebP-encodes the image under a byte budget; the cloud side decodes "
            "the image and runs the original LLaVA vision tower, projector, and LLM."
        )
    )
    parser.add_argument("--model_name_or_path", type=str, required=True, help="LLaVA-1.5 HF model path.")
    parser.add_argument("--data_path", type=str, required=True, help="POPE annotation JSON/JSONL path.")
    parser.add_argument("--image_folder", type=str, required=True, help="Folder containing POPE images.")
    parser.add_argument(
        "--codecs",
        type=str,
        nargs="+",
        default=["jpeg", "webp"],
        choices=["jpeg", "webp"],
        help="Compressed image codecs to evaluate.",
    )
    parser.add_argument(
        "--budget_specs",
        type=str,
        nargs="+",
        default=DEFAULT_BUDGET_SPECS,
        help="Budget specs in label:bytes format, e.g. 0.6:354996.",
    )
    parser.add_argument("--include_raw", action="store_true", help="Also evaluate original image files.")
    parser.add_argument("--output_path", type=str, default="outputs/llava15_image_compression/pope_eval.json")
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
    parser.add_argument("--warmup", type=int, default=1, help="Warmup forwards before measured evaluation.")
    parser.add_argument("--jpeg_min_quality", type=int, default=1)
    parser.add_argument("--jpeg_max_quality", type=int, default=95)
    parser.add_argument("--webp_min_quality", type=int, default=1)
    parser.add_argument("--webp_max_quality", type=int, default=95)
    parser.add_argument(
        "--network_mbps",
        type=float,
        nargs="+",
        default=[5.0, 10.0, 50.0, 100.0],
        help="Bandwidth values used to estimate network and end-to-end latency.",
    )
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


def load_llava_processor(model_name_or_path: str, local_files_only: bool):
    try:
        return AutoProcessor.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
            use_fast=False,
        )
    except Exception as exc:
        print(f"[WARN] AutoProcessor failed, fallback to slow tokenizer: {exc}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
            use_fast=False,
        )
        image_processor = CLIPImageProcessor.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        )
        return SimpleNamespace(tokenizer=tokenizer, image_processor=image_processor)


def parse_budget_specs(specs: List[str]) -> List[Tuple[str, int]]:
    budgets: List[Tuple[str, int]] = []
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"Budget spec must be label:bytes, got {spec!r}")
        label, value = spec.split(":", 1)
        label = label.strip()
        budget = int(value.strip())
        if not label:
            raise ValueError(f"Budget label is empty in {spec!r}")
        if budget <= 0:
            raise ValueError(f"Budget must be positive in {spec!r}")
        budgets.append((label, budget))
    return budgets


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


def encode_jpeg(image: Image.Image, quality: int) -> bytes:
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=int(quality), optimize=True)
    return buf.getvalue()


def encode_webp(image: Image.Image, quality: int) -> bytes:
    buf = BytesIO()
    image.save(buf, format="WEBP", quality=int(quality), method=6)
    return buf.getvalue()


def encode_image_at_quality(image: Image.Image, codec: str, quality: int) -> bytes:
    if codec == "jpeg":
        return encode_jpeg(image, quality)
    if codec == "webp":
        return encode_webp(image, quality)
    raise ValueError(f"Unsupported codec: {codec}")


def compress_to_budget(
    image: Image.Image,
    codec: str,
    target_bytes: int,
    min_quality: int,
    max_quality: int,
) -> Dict[str, Any]:
    """Return highest-quality compressed bytes not exceeding target when possible."""
    image = image.convert("RGB")
    low = int(min_quality)
    high = int(max_quality)
    if low > high:
        raise ValueError(f"min_quality={low} must be <= max_quality={high}")

    start = time.perf_counter()
    min_bytes = encode_image_at_quality(image, codec, low)
    if len(min_bytes) > target_bytes:
        return {
            "encoded_bytes": min_bytes,
            "quality": low,
            "edge_encode_latency_ms": (time.perf_counter() - start) * 1000,
            "over_budget": True,
        }

    max_bytes = encode_image_at_quality(image, codec, high)
    if len(max_bytes) <= target_bytes:
        return {
            "encoded_bytes": max_bytes,
            "quality": high,
            "edge_encode_latency_ms": (time.perf_counter() - start) * 1000,
            "over_budget": False,
        }

    best_quality = low
    best_bytes = min_bytes
    left, right = low, high
    while left <= right:
        mid = (left + right) // 2
        candidate = encode_image_at_quality(image, codec, mid)
        if len(candidate) <= target_bytes:
            best_quality = mid
            best_bytes = candidate
            left = mid + 1
        else:
            right = mid - 1

    return {
        "encoded_bytes": best_bytes,
        "quality": best_quality,
        "edge_encode_latency_ms": (time.perf_counter() - start) * 1000,
        "over_budget": False,
    }


def load_raw_image_bytes(image_path: str) -> Dict[str, Any]:
    start = time.perf_counter()
    data = Path(image_path).read_bytes()
    return {
        "encoded_bytes": data,
        "quality": None,
        "edge_encode_latency_ms": (time.perf_counter() - start) * 1000,
        "over_budget": False,
    }


def decode_image_bytes(encoded_bytes: bytes) -> Tuple[Image.Image, float]:
    start = time.perf_counter()
    with Image.open(BytesIO(encoded_bytes)) as image:
        decoded = image.convert("RGB")
    return decoded, (time.perf_counter() - start) * 1000


@torch.no_grad()
def build_image_embeds_from_image(
    model,
    processor,
    image: Image.Image,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    vision_tower = get_vision_tower(model)
    pixel_values = processor.image_processor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device=device, dtype=dtype)

    _sync_device(device)
    vision_start = time.perf_counter()
    vision_outputs = vision_tower(pixel_values, output_hidden_states=True)
    _sync_device(device)
    vision_latency_ms = (time.perf_counter() - vision_start) * 1000

    selected = _select_vision_features(model, vision_outputs)
    actual_tokens = int(selected.shape[1])

    _sync_device(device)
    projector_start = time.perf_counter()
    image_embeds = model.multi_modal_projector(selected)
    _sync_device(device)
    projector_latency_ms = (time.perf_counter() - projector_start) * 1000

    return {
        "image_embeds": image_embeds,
        "actual_tokens": actual_tokens,
        "vision_latency_ms": vision_latency_ms,
        "projector_latency_ms": projector_latency_ms,
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
        "transmitted_bytes_sum": 0.0,
        "quality_sum": 0.0,
        "quality_count": 0.0,
        "over_budget": 0.0,
        "actual_tokens_sum": 0.0,
        "edge_encode_latency_ms_sum": 0.0,
        "cloud_decode_latency_ms_sum": 0.0,
        "vision_latency_ms_sum": 0.0,
        "projector_latency_ms_sum": 0.0,
        "scoring_latency_ms_sum": 0.0,
        "cloud_latency_ms_sum": 0.0,
        "compute_latency_ms_sum": 0.0,
        "confidence_sum": 0.0,
    }


def update_stats(stats: Dict[str, float], label: str, result: Dict[str, Any]):
    prediction = result["prediction"]
    stats["total"] += 1
    stats["yes_predictions"] += 1 if prediction == "yes" else 0
    stats["transmitted_bytes_sum"] += float(result["transmitted_bytes"])
    stats["over_budget"] += 1 if result.get("over_budget") else 0
    if result.get("quality") is not None:
        stats["quality_sum"] += float(result["quality"])
        stats["quality_count"] += 1
    stats["actual_tokens_sum"] += float(result["actual_tokens"])
    stats["edge_encode_latency_ms_sum"] += float(result["edge_encode_latency_ms"])
    stats["cloud_decode_latency_ms_sum"] += float(result["cloud_decode_latency_ms"])
    stats["vision_latency_ms_sum"] += float(result["vision_latency_ms"])
    stats["projector_latency_ms_sum"] += float(result["projector_latency_ms"])
    stats["scoring_latency_ms_sum"] += float(result["scoring_latency_ms"])
    stats["cloud_latency_ms_sum"] += float(result["cloud_latency_ms"])
    stats["compute_latency_ms_sum"] += float(result["compute_latency_ms"])
    stats["confidence_sum"] += float(result["confidence"])

    if label == "yes" and prediction == "yes":
        stats["tp"] += 1
    elif label == "no" and prediction == "no":
        stats["tn"] += 1
    elif label == "no" and prediction == "yes":
        stats["fp"] += 1
    elif label == "yes" and prediction == "no":
        stats["fn"] += 1


def finalize_stats(stats: Dict[str, float], network_mbps: List[float]) -> Dict[str, Any]:
    tp, tn, fp, fn = stats["tp"], stats["tn"], stats["fp"], stats["fn"]
    total = max(1.0, stats["total"])
    accuracy = (tp + tn) / total
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)

    avg_bytes = stats["transmitted_bytes_sum"] / total
    compute_latency_ms = stats["compute_latency_ms_sum"] / total
    network_estimates = {}
    for mbps in network_mbps:
        network_ms = avg_bytes * 8.0 / (float(mbps) * 1_000_000.0) * 1000.0
        key = f"{float(mbps):g}mbps"
        network_estimates[key] = {
            "network_latency_ms": network_ms,
            "estimated_end_to_end_latency_ms": compute_latency_ms + network_ms,
        }

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fp / max(1.0, fp + tn)),
        "false_negative_rate": float(fn / max(1.0, fn + tp)),
        "specificity": float(tn / max(1.0, tn + fp)),
        "yes_ratio": float(stats["yes_predictions"] / total),
        "avg_transmitted_bytes": float(avg_bytes),
        "avg_quality": float(stats["quality_sum"] / stats["quality_count"]) if stats["quality_count"] > 0 else None,
        "over_budget_ratio": float(stats["over_budget"] / total),
        "avg_actual_tokens": float(stats["actual_tokens_sum"] / total),
        "edge_encode_latency_ms": float(stats["edge_encode_latency_ms_sum"] / total),
        "cloud_decode_latency_ms": float(stats["cloud_decode_latency_ms_sum"] / total),
        "vision_latency_ms": float(stats["vision_latency_ms_sum"] / total),
        "projector_latency_ms": float(stats["projector_latency_ms_sum"] / total),
        "scoring_latency_ms": float(stats["scoring_latency_ms_sum"] / total),
        "cloud_latency_ms": float(stats["cloud_latency_ms_sum"] / total),
        "compute_latency_ms": float(compute_latency_ms),
        "avg_confidence": float(stats["confidence_sum"] / total),
        "network_estimates": network_estimates,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "total": int(stats["total"]),
    }


@torch.no_grad()
def evaluate_encoded_image(
    model,
    processor,
    image_payload: Dict[str, Any],
    prompt: str,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    _sync_device(device)
    cloud_start = time.perf_counter()
    decoded_image, decode_latency_ms = decode_image_bytes(image_payload["encoded_bytes"])
    image_result = build_image_embeds_from_image(model, processor, decoded_image, device, dtype)
    prefix_embeds, prefix_attention_mask = build_prefix_embeds(
        model=model,
        tokenizer=processor.tokenizer,
        prompt=prompt,
        image_embeds=image_result["image_embeds"],
        device=device,
    )
    score_result = score_yes_no(
        model=model,
        tokenizer=processor.tokenizer,
        prefix_embeds=prefix_embeds,
        prefix_attention_mask=prefix_attention_mask,
        candidate_prefix=args.candidate_prefix,
        device=device,
    )
    _sync_device(device)
    cloud_latency_ms = (time.perf_counter() - cloud_start) * 1000

    return {
        "prediction": score_result["prediction"],
        "yes_score": score_result["yes_score"],
        "no_score": score_result["no_score"],
        "confidence": score_result["confidence"],
        "transmitted_bytes": len(image_payload["encoded_bytes"]),
        "quality": image_payload.get("quality"),
        "over_budget": bool(image_payload.get("over_budget", False)),
        "actual_tokens": image_result["actual_tokens"],
        "edge_encode_latency_ms": float(image_payload["edge_encode_latency_ms"]),
        "cloud_decode_latency_ms": decode_latency_ms,
        "vision_latency_ms": image_result["vision_latency_ms"],
        "projector_latency_ms": image_result["projector_latency_ms"],
        "scoring_latency_ms": score_result["scoring_latency_ms"],
        "cloud_latency_ms": cloud_latency_ms,
        "compute_latency_ms": float(image_payload["edge_encode_latency_ms"]) + cloud_latency_ms,
    }


def make_image_payload(
    image_path: str,
    codec: str,
    budget_bytes: Optional[int],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if codec == "raw":
        return load_raw_image_bytes(image_path)

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        if codec == "jpeg":
            return compress_to_budget(
                image=image,
                codec=codec,
                target_bytes=int(budget_bytes),
                min_quality=args.jpeg_min_quality,
                max_quality=args.jpeg_max_quality,
            )
        if codec == "webp":
            return compress_to_budget(
                image=image,
                codec=codec,
                target_bytes=int(budget_bytes),
                min_quality=args.webp_min_quality,
                max_quality=args.webp_max_quality,
            )
    raise ValueError(f"Unsupported codec: {codec}")


def evaluate_setting(
    model,
    processor,
    samples: List[Dict[str, Any]],
    args: argparse.Namespace,
    setting_name: str,
    codec: str,
    budget_label: Optional[str],
    budget_bytes: Optional[int],
    device: torch.device,
    dtype: torch.dtype,
    pred_fh,
) -> Dict[str, Any]:
    stats = empty_stats()
    iterator = samples
    if args.max_samples and args.max_samples > 0:
        iterator = samples[: args.max_samples]

    desc = setting_name
    for index, sample in enumerate(tqdm(iterator, desc=desc)):
        question = normalize_question(sample, args.question_suffix)
        label = normalize_label(sample)
        image_path = resolve_image_path(args.image_folder, sample)
        prompt = build_prompt(question)

        image_payload = make_image_payload(image_path, codec, budget_bytes, args)
        result = evaluate_encoded_image(model, processor, image_payload, prompt, args, device, dtype)
        update_stats(stats, label, result)

        if pred_fh is not None:
            pred_fh.write(
                json.dumps(
                    {
                        "index": index,
                        "setting": setting_name,
                        "codec": codec,
                        "budget_label": budget_label,
                        "budget_bytes": budget_bytes,
                        "image": sample.get("image") or sample.get("image_path") or sample.get("file_name"),
                        "question": question,
                        "label": label,
                        "prediction": result["prediction"],
                        "yes_score": result["yes_score"],
                        "no_score": result["no_score"],
                        "confidence": result["confidence"],
                        "transmitted_bytes": result["transmitted_bytes"],
                        "quality": result["quality"],
                        "over_budget": result["over_budget"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    metrics = finalize_stats(stats, args.network_mbps)
    metrics["codec"] = codec
    metrics["budget_label"] = budget_label
    metrics["target_budget_bytes"] = budget_bytes
    return metrics


def warmup_model(model, processor, samples, args, device, dtype):
    if args.warmup <= 0 or len(samples) == 0:
        return
    sample = samples[0]
    question = normalize_question(sample, args.question_suffix)
    image_path = resolve_image_path(args.image_folder, sample)
    prompt = build_prompt(question)
    image_payload = load_raw_image_bytes(image_path)
    for _ in range(args.warmup):
        _ = evaluate_encoded_image(model, processor, image_payload, prompt, args, device, dtype)


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    dtype = _dtype_from_name(args.dtype)
    budgets = parse_budget_specs(args.budget_specs)

    samples = load_pope_dataset(args.data_path)
    if args.max_samples and args.max_samples > 0:
        print(f"[INFO] max_samples={args.max_samples}, full dataset size={len(samples)}")
    print(f"[INFO] loading LLaVA model from {args.model_name_or_path}")
    processor = load_llava_processor(args.model_name_or_path, local_files_only=args.local_files_only)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()

    if processor.tokenizer.pad_token_id is None and processor.tokenizer.eos_token_id is not None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pred_fh = None
    if args.save_predictions:
        pred_path = Path(args.save_predictions)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_fh = open(pred_path, "w", encoding="utf-8")

    warmup_model(model, processor, samples, args, device, dtype)

    results: Dict[str, Any] = {}
    try:
        if args.include_raw:
            results["raw"] = evaluate_setting(
                model=model,
                processor=processor,
                samples=samples,
                args=args,
                setting_name="raw",
                codec="raw",
                budget_label=None,
                budget_bytes=None,
                device=device,
                dtype=dtype,
                pred_fh=pred_fh,
            )

        for codec in args.codecs:
            for budget_label, budget_bytes in budgets:
                setting_name = f"{codec}_{budget_label}"
                results[setting_name] = evaluate_setting(
                    model=model,
                    processor=processor,
                    samples=samples,
                    args=args,
                    setting_name=setting_name,
                    codec=codec,
                    budget_label=budget_label,
                    budget_bytes=budget_bytes,
                    device=device,
                    dtype=dtype,
                    pred_fh=pred_fh,
                )
    finally:
        if pred_fh is not None:
            pred_fh.close()

    payload = {
        "model_name_or_path": args.model_name_or_path,
        "data_path": args.data_path,
        "image_folder": args.image_folder,
        "eval_mode": "yes_no_likelihood",
        "budget_specs": [{"label": label, "bytes": budget} for label, budget in budgets],
        "codecs": args.codecs,
        "include_raw": args.include_raw,
        "network_mbps": args.network_mbps,
        "metrics_by_setting": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[OK] saved metrics to {output_path}")


if __name__ == "__main__":
    main()