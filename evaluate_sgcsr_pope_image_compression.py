from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor, LlavaForConditionalGeneration

from train_sgcsr import (
    SGCSRCompressedDataset,
    SGCSRCollator,
    dtype_from_name,
    get_input_embedding_layer,
    get_language_model,
    get_vision_tower,
    merge_text_and_visual_tokens,
    select_vision_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a compressed-image POPE baseline. The edge sends JPEG/WebP bytes; "
            "the cloud decodes the image and runs the full native LLaVA vision tower, "
            "projector, and LLM yes/no likelihood scoring."
        )
    )
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True, help="Compressed POPE feature directory or manifest.")
    parser.add_argument("--image_folder", type=str, required=True, help="Image folder used by the compressed manifest.")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--save_predictions", type=str, default=None)
    parser.add_argument(
        "--split_indices_path",
        type=str,
        default=None,
        help="Optional split_indices.json saved by train_sgcsr_pope_adapt.py.",
    )
    parser.add_argument("--split_name", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--max_eval_samples", type=int, default=0, help="Debug only; 0 evaluates all selected samples.")
    parser.add_argument(
        "--conversation_mode",
        type=str,
        default="first",
        choices=["first", "all", "full"],
        help="Must match the saved split. Formal POPE uses first.",
    )
    parser.add_argument("--max_text_length", type=int, default=0)
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
        help="Prefix before yes/no candidates. A leading space matches supervised LLaVA answer tokenization.",
    )
    parser.add_argument(
        "--compression_format",
        type=str,
        default="jpeg",
        choices=["jpeg", "webp"],
        help="Image codec used for the edge-to-cloud image baseline.",
    )
    parser.add_argument(
        "--compression_mode",
        type=str,
        default="byte_matched",
        choices=["byte_matched", "fixed_quality"],
        help=(
            "byte_matched searches the highest image quality whose byte size is no larger than "
            "the corresponding ToMe feature payload budget; fixed_quality uses --image_quality."
        ),
    )
    parser.add_argument("--image_quality", type=int, default=75, help="Quality for --compression_mode fixed_quality.")
    parser.add_argument("--min_quality", type=int, default=1, help="Minimum quality for byte-matched search.")
    parser.add_argument("--max_quality", type=int, default=95, help="Maximum quality for byte-matched search.")
    parser.add_argument(
        "--jpeg_subsampling",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="JPEG chroma subsampling. 2 is 4:2:0 and usually gives the smallest payload.",
    )
    parser.add_argument(
        "--metadata_bytes",
        type=int,
        default=0,
        help="Optional extra bytes added to the feature-payload budget to model manifest/header metadata.",
    )
    parser.add_argument("--feature_dim", type=int, default=1024, help="Compressed CLIP feature dimension.")
    parser.add_argument("--teacher_tokens", type=int, default=576, help="Original LLaVA-1.5 visual token count.")
    parser.add_argument(
        "--bandwidth_mbps",
        type=float,
        nargs="*",
        default=[5.0, 10.0, 20.0, 50.0],
        help="Bandwidths used to report transfer and end-to-end latency estimates.",
    )
    parser.add_argument("--allow_missing_source", action="store_true", help="Debug only.")
    return parser.parse_args()


def _sync_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _retain_key(value: Any) -> str:
    return f"{float(value):.2f}"


def _parse_yes_no(text: str) -> Optional[str]:
    clean = text.strip().lower()
    if not clean:
        return None
    first = clean.replace(".", " ").replace(",", " ").replace(":", " ").replace(";", " ").split()[0]
    if first in {"yes", "yeah", "yep"} or clean.startswith("yes"):
        return "yes"
    if first in {"no", "not", "nope"} or clean.startswith("no"):
        return "no"
    return None


def _load_saved_split_indices(
    path: str,
    split_name: str,
    dataset_len: int,
    args: argparse.Namespace,
) -> Tuple[List[int], Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("data_path") and str(payload["data_path"]) != str(args.data_path):
        raise ValueError(
            f"Split data_path mismatch: split={payload['data_path']} current={args.data_path}. "
            "Use the split_indices.json generated for this POPE compressed dataset."
        )
    if payload.get("conversation_mode") and payload["conversation_mode"] != args.conversation_mode:
        raise ValueError(
            f"Split conversation_mode mismatch: split={payload['conversation_mode']} current={args.conversation_mode}."
        )
    expected_suffix = str(payload.get("question_suffix", args.question_suffix))
    if expected_suffix != str(args.question_suffix):
        raise ValueError(
            f"Split question_suffix mismatch: split={expected_suffix!r} current={args.question_suffix!r}."
        )
    expected_max_text_length = int(payload.get("max_text_length", 0))
    if expected_max_text_length != int(args.max_text_length):
        raise ValueError(
            f"Split max_text_length mismatch: split={expected_max_text_length} current={args.max_text_length}."
        )
    if int(payload.get("max_samples", 0)) != 0:
        raise ValueError(
            "This split_indices.json was generated from a debug subset with max_samples="
            f"{payload.get('max_samples')}. Do not use it for formal evaluation."
        )
    if split_name not in payload:
        raise ValueError(f"Split file {path} does not contain split '{split_name}'.")
    indices = [int(idx) for idx in payload[split_name]]
    if not indices:
        raise ValueError(f"Split '{split_name}' in {path} is empty.")
    if max(indices) >= dataset_len:
        raise ValueError(
            f"Split index {max(indices)} is out of range for dataset length {dataset_len}. "
            "Check --data_path, --conversation_mode, and --max_text_length."
        )
    return indices, payload


def _feature_payload_budget_bytes(sample: Dict[str, Any], args: argparse.Namespace) -> int:
    """Estimate the transmitted ToMe feature payload for byte-matched image compression.

    The compressed dataset loader returns dequantized float features, so this
    reconstructs the actual serialized int8 payload size from the valid token
    count and the known binary storage format:

      features.int8        N * feature_dim * 1 byte
      feature_scales.fp16  N * 2 bytes
      source_indices.i16   teacher_tokens * 2 bytes
      source_offsets.i32   (N + 1) * 4 bytes
    """
    attention_mask = sample["compressed_attention_mask"]
    if attention_mask.ndim == 2:
        valid_tokens = int(attention_mask[0].sum().item())
    else:
        valid_tokens = int(attention_mask.sum().item())
    return int(
        valid_tokens * int(args.feature_dim)
        + valid_tokens * 2
        + int(args.teacher_tokens) * 2
        + (valid_tokens + 1) * 4
        + int(args.metadata_bytes)
    )


def _save_image_to_bytes(image: Image.Image, fmt: str, quality: int, jpeg_subsampling: int) -> bytes:
    buffer = io.BytesIO()
    if fmt == "jpeg":
        image.save(
            buffer,
            format="JPEG",
            quality=int(quality),
            optimize=True,
            subsampling=int(jpeg_subsampling),
        )
    elif fmt == "webp":
        image.save(buffer, format="WEBP", quality=int(quality), method=6)
    else:
        raise ValueError(f"Unsupported compression format: {fmt}")
    return buffer.getvalue()


def _compress_image(
    image: Image.Image,
    *,
    target_bytes: Optional[int],
    args: argparse.Namespace,
) -> Tuple[bytes, int, float, float]:
    fmt = args.compression_format

    if args.compression_mode == "fixed_quality":
        quality = max(1, min(100, int(args.image_quality)))
        start = time.perf_counter()
        payload = _save_image_to_bytes(image, fmt, quality, args.jpeg_subsampling)
        return payload, quality, (time.perf_counter() - start) * 1000.0, 0.0

    if target_bytes is None or target_bytes <= 0:
        raise ValueError("byte_matched compression requires a positive target byte budget.")

    search_start = time.perf_counter()
    min_q = max(1, min(100, int(args.min_quality)))
    max_q = max(1, min(100, int(args.max_quality)))
    if min_q > max_q:
        min_q, max_q = max_q, min_q

    min_payload = _save_image_to_bytes(image, fmt, min_q, args.jpeg_subsampling)
    if len(min_payload) > target_bytes:
        search_ms = (time.perf_counter() - search_start) * 1000.0
        final_start = time.perf_counter()
        final_payload = _save_image_to_bytes(image, fmt, min_q, args.jpeg_subsampling)
        final_ms = (time.perf_counter() - final_start) * 1000.0
        return final_payload, min_q, final_ms, search_ms

    max_payload = _save_image_to_bytes(image, fmt, max_q, args.jpeg_subsampling)
    if len(max_payload) <= target_bytes:
        search_ms = (time.perf_counter() - search_start) * 1000.0
        final_start = time.perf_counter()
        final_payload = _save_image_to_bytes(image, fmt, max_q, args.jpeg_subsampling)
        final_ms = (time.perf_counter() - final_start) * 1000.0
        return final_payload, max_q, final_ms, search_ms

    best_q = min_q
    lo, hi = min_q, max_q
    while lo <= hi:
        mid = (lo + hi) // 2
        payload = _save_image_to_bytes(image, fmt, mid, args.jpeg_subsampling)
        if len(payload) <= target_bytes:
            best_q = mid
            lo = mid + 1
        else:
            hi = mid - 1
    search_ms = (time.perf_counter() - search_start) * 1000.0
    final_start = time.perf_counter()
    final_payload = _save_image_to_bytes(image, fmt, best_q, args.jpeg_subsampling)
    final_ms = (time.perf_counter() - final_start) * 1000.0
    return final_payload, best_q, final_ms, search_ms


def _decode_image(payload: bytes) -> Tuple[Image.Image, float]:
    start = time.perf_counter()
    image = Image.open(io.BytesIO(payload)).convert("RGB")
    image.load()
    return image, (time.perf_counter() - start) * 1000.0


@torch.no_grad()
def _project_decoded_image(
    model,
    image_processor,
    image: Image.Image,
    device: torch.device,
    model_dtype: torch.dtype,
) -> Tuple[torch.Tensor, float, float]:
    preprocess_start = time.perf_counter()
    pixel_values = image_processor(images=[image], return_tensors="pt").pixel_values.to(
        device=device,
        dtype=model_dtype,
    )
    _sync_device(device)
    preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0
    vision_tower = get_vision_tower(model)
    _sync_device(device)
    start = time.perf_counter()
    vision_outputs = vision_tower(pixel_values, output_hidden_states=True)
    clip_tokens = select_vision_features(model, vision_outputs)
    visual_tokens = model.multi_modal_projector(clip_tokens)
    _sync_device(device)
    return visual_tokens, preprocess_ms, (time.perf_counter() - start) * 1000.0


def _extract_prompt_and_label(
    tokenizer,
    sample: Dict[str, Any],
    device: torch.device,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, str, str]]:
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
    answer_ids = labels[answer_pos]
    label_text = tokenizer.decode(answer_ids.tolist(), skip_special_tokens=True).strip()
    label = _parse_yes_no(label_text)
    if label is None:
        return None
    return prompt_ids, prompt_attention_mask, label, label_text


def _tokenize_candidate(tokenizer, text: str, device: torch.device) -> torch.Tensor:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if not ids:
        raise ValueError(f"Candidate text produced no tokens: {text!r}")
    return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)


@torch.no_grad()
def _score_yes_no(
    model,
    tokenizer,
    sample: Dict[str, Any],
    visual_tokens: torch.Tensor,
    device: torch.device,
    candidate_prefix: str,
) -> Optional[Dict[str, Any]]:
    extracted = _extract_prompt_and_label(tokenizer, sample, device)
    if extracted is None:
        return None
    prompt_ids, prompt_attention_mask, label, label_text = extracted
    prompt_labels = torch.full_like(prompt_ids, -100)
    prefix_embeds, prefix_attention_mask, _ = merge_text_and_visual_tokens(
        model=model,
        input_ids=prompt_ids,
        attention_mask=prompt_attention_mask,
        labels=prompt_labels,
        visual_tokens=visual_tokens,
    )

    scores: Dict[str, float] = {}
    scoring_latency_ms = 0.0
    language_model = get_language_model(model)
    input_embeddings = get_input_embedding_layer(model)
    for candidate in ["yes", "no"]:
        candidate_ids = _tokenize_candidate(tokenizer, f"{candidate_prefix}{candidate}", device)
        candidate_embeds = input_embeddings(candidate_ids)
        inputs_embeds = torch.cat([prefix_embeds, candidate_embeds], dim=1)
        candidate_attention = torch.ones_like(candidate_ids, device=device)
        attention_mask = torch.cat([prefix_attention_mask, candidate_attention], dim=1)
        labels = torch.full(inputs_embeds.shape[:2], -100, dtype=torch.long, device=device)
        labels[:, -candidate_ids.shape[1] :] = candidate_ids

        _sync_device(device)
        start = time.perf_counter()
        out = language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        _sync_device(device)
        scoring_latency_ms += (time.perf_counter() - start) * 1000.0
        scores[candidate] = -float(out.loss.item())

    prediction = "yes" if scores["yes"] >= scores["no"] else "no"
    return {
        "label": label,
        "label_text": label_text,
        "prediction": prediction,
        "yes_score": scores["yes"],
        "no_score": scores["no"],
        "confidence": abs(scores["yes"] - scores["no"]),
        "scoring_latency_ms": scoring_latency_ms,
    }


def _empty_stats() -> Dict[str, float]:
    return {
        "tp": 0.0,
        "tn": 0.0,
        "fp": 0.0,
        "fn": 0.0,
        "invalid": 0.0,
        "total": 0.0,
        "confidence_sum": 0.0,
        "payload_bytes_sum": 0.0,
        "target_bytes_sum": 0.0,
        "budget_utilization_sum": 0.0,
        "budgeted_count": 0.0,
        "over_budget_count": 0.0,
        "under_80pct_budget_count": 0.0,
        "under_50pct_budget_count": 0.0,
        "quality_sum": 0.0,
        "encode_latency_ms_sum": 0.0,
        "quality_search_latency_ms_sum": 0.0,
        "dynamic_encode_latency_ms_sum": 0.0,
        "decode_latency_ms_sum": 0.0,
        "image_preprocess_latency_ms_sum": 0.0,
        "vision_projector_latency_ms_sum": 0.0,
        "scoring_latency_ms_sum": 0.0,
        "cloud_latency_ms_sum": 0.0,
        "edge_latency_ms_sum": 0.0,
        "latency_count": 0.0,
    }


def _update_stats(
    stats: Dict[str, float],
    *,
    result: Optional[Dict[str, Any]],
    payload_bytes: int,
    target_bytes: Optional[int],
    quality: int,
    encode_latency_ms: float,
    quality_search_latency_ms: float,
    decode_latency_ms: float,
    image_preprocess_latency_ms: float,
    vision_projector_latency_ms: float,
):
    stats["total"] += 1
    stats["payload_bytes_sum"] += int(payload_bytes)
    stats["target_bytes_sum"] += int(target_bytes or 0)
    if target_bytes is not None and int(target_bytes) > 0:
        utilization = float(payload_bytes) / float(target_bytes)
        stats["budget_utilization_sum"] += utilization
        stats["budgeted_count"] += 1
        if int(payload_bytes) > int(target_bytes):
            stats["over_budget_count"] += 1
        if utilization < 0.8:
            stats["under_80pct_budget_count"] += 1
        if utilization < 0.5:
            stats["under_50pct_budget_count"] += 1
    stats["quality_sum"] += int(quality)
    stats["encode_latency_ms_sum"] += float(encode_latency_ms)
    stats["quality_search_latency_ms_sum"] += float(quality_search_latency_ms)
    stats["dynamic_encode_latency_ms_sum"] += float(encode_latency_ms) + float(quality_search_latency_ms)
    stats["decode_latency_ms_sum"] += float(decode_latency_ms)
    stats["image_preprocess_latency_ms_sum"] += float(image_preprocess_latency_ms)
    stats["vision_projector_latency_ms_sum"] += float(vision_projector_latency_ms)
    stats["edge_latency_ms_sum"] += float(encode_latency_ms)
    stats["cloud_latency_ms_sum"] += (
        float(decode_latency_ms)
        + float(image_preprocess_latency_ms)
        + float(vision_projector_latency_ms)
    )
    stats["latency_count"] += 1

    if result is None:
        stats["invalid"] += 1
        return

    label = result["label"]
    prediction = result["prediction"]
    if label == "yes" and prediction == "yes":
        stats["tp"] += 1
    elif label == "no" and prediction == "no":
        stats["tn"] += 1
    elif label == "no" and prediction == "yes":
        stats["fp"] += 1
    elif label == "yes" and prediction == "no":
        stats["fn"] += 1
    else:
        stats["invalid"] += 1

    scoring_ms = float(result.get("scoring_latency_ms", 0.0))
    stats["scoring_latency_ms_sum"] += scoring_ms
    stats["cloud_latency_ms_sum"] += scoring_ms
    stats["confidence_sum"] += float(result.get("confidence", 0.0))


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _finalize_stats(stats: Dict[str, float], bandwidth_mbps: Sequence[float]) -> Dict[str, Any]:
    tp = int(stats["tp"])
    tn = int(stats["tn"])
    fp = int(stats["fp"])
    fn = int(stats["fn"])
    invalid = int(stats["invalid"])
    total = int(stats["total"])
    valid = max(1, total - invalid)
    latency_count = max(1.0, float(stats["latency_count"]))
    budgeted_count = max(1.0, float(stats["budgeted_count"]))
    payload_bytes = stats["payload_bytes_sum"] / latency_count
    edge_ms = stats["edge_latency_ms_sum"] / latency_count
    cloud_ms = stats["cloud_latency_ms_sum"] / latency_count

    transfer_ms = {
        f"{float(mbps):g}Mbps": float(payload_bytes * 8.0 / (float(mbps) * 1_000_000.0) * 1000.0)
        for mbps in bandwidth_mbps
        if float(mbps) > 0
    }
    e2e_ms = {key: float(edge_ms + value + cloud_ms) for key, value in transfer_ms.items()}

    return {
        "accuracy": _safe_div(tp + tn, valid),
        "precision": _safe_div(tp, tp + fp),
        "recall": _safe_div(tp, tp + fn),
        "f1": _safe_div(2 * tp, 2 * tp + fp + fn),
        "false_positive_rate": _safe_div(fp, fp + tn),
        "false_negative_rate": _safe_div(fn, fn + tp),
        "specificity": _safe_div(tn, tn + fp),
        "yes_ratio": _safe_div(tp + fp, valid),
        "invalid_ratio": _safe_div(invalid, total),
        "payload_bytes": float(payload_bytes),
        "target_bytes": (
            float(stats["target_bytes_sum"] / budgeted_count)
            if stats["budgeted_count"] > 0
            else None
        ),
        "budget_utilization": (
            float(stats["budget_utilization_sum"] / budgeted_count)
            if stats["budgeted_count"] > 0
            else None
        ),
        "over_budget_ratio": (
            float(stats["over_budget_count"] / budgeted_count)
            if stats["budgeted_count"] > 0
            else None
        ),
        "under_80pct_budget_ratio": (
            float(stats["under_80pct_budget_count"] / budgeted_count)
            if stats["budgeted_count"] > 0
            else None
        ),
        "under_50pct_budget_ratio": (
            float(stats["under_50pct_budget_count"] / budgeted_count)
            if stats["budgeted_count"] > 0
            else None
        ),
        "compression_quality": float(stats["quality_sum"] / latency_count),
        "edge_encode_latency_ms": float(edge_ms),
        "quality_search_latency_ms": float(stats["quality_search_latency_ms_sum"] / latency_count),
        "edge_dynamic_encode_latency_ms": float(stats["dynamic_encode_latency_ms_sum"] / latency_count),
        "decode_latency_ms": float(stats["decode_latency_ms_sum"] / latency_count),
        "image_preprocess_latency_ms": float(stats["image_preprocess_latency_ms_sum"] / latency_count),
        "vision_projector_latency_ms": float(stats["vision_projector_latency_ms_sum"] / latency_count),
        "scoring_latency_ms": float(stats["scoring_latency_ms_sum"] / latency_count),
        "cloud_latency_ms": float(cloud_ms),
        "transfer_latency_ms": transfer_ms,
        "end_to_end_latency_ms": e2e_ms,
        "avg_confidence": float(stats["confidence_sum"] / max(1, tp + tn + fp + fn)),
        "samples_per_second_cloud": float(1000.0 * latency_count / max(1e-9, stats["cloud_latency_ms_sum"])),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "invalid": invalid,
        "total": total,
    }


def main() -> int:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("--num_workers must be non-negative")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_dtype = dtype_from_name(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    image_processor = CLIPImageProcessor.from_pretrained(
        args.model_name_or_path,
        local_files_only=args.local_files_only,
    )
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print(f"[INFO] loaded model on {device} dtype={model_dtype}", flush=True)

    dataset = SGCSRCompressedDataset(
        data_path=args.data_path,
        image_folder=args.image_folder,
        tokenizer=tokenizer,
        max_samples=0,
        allow_missing_source=args.allow_missing_source,
        seed=42,
        conversation_mode=args.conversation_mode,
        max_text_length=args.max_text_length,
        image_token_id=int(getattr(model.config, "image_token_index", 32000)),
        question_suffix=args.question_suffix,
    )

    split_payload = None
    split_mode = None
    if args.split_indices_path:
        split_indices, split_payload = _load_saved_split_indices(
            path=args.split_indices_path,
            split_name=args.split_name,
            dataset_len=len(dataset),
            args=args,
        )
        dataset = Subset(dataset, split_indices)
        split_mode = f"saved_split_indices:{args.split_name}"
        print(
            f"[INFO] using saved POPE split {args.split_name} from {args.split_indices_path}; "
            f"samples={len(split_indices)}",
            flush=True,
        )

    collator = SGCSRCollator(pad_token_id=tokenizer.pad_token_id)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )

    overall = _empty_stats()
    by_retain: Dict[str, Dict[str, float]] = {}

    predictions_f = None
    if args.save_predictions:
        Path(args.save_predictions).parent.mkdir(parents=True, exist_ok=True)
        predictions_f = open(args.save_predictions, "w", encoding="utf-8")

    seen = 0
    progress = tqdm(dataloader, desc="POPE compressed-image baseline", dynamic_ncols=True)
    try:
        for sample in progress:
            if args.max_eval_samples > 0 and seen >= args.max_eval_samples:
                break
            seen += 1
            retain = _retain_key(sample["retain_ratio"][0].item())
            target_bytes = (
                _feature_payload_budget_bytes(sample, args)
                if args.compression_mode == "byte_matched"
                else None
            )

            image = Image.open(sample["image_paths"][0]).convert("RGB")
            payload, quality, encode_ms, quality_search_ms = _compress_image(
                image,
                target_bytes=target_bytes,
                args=args,
            )
            decoded_image, decode_ms = _decode_image(payload)
            visual_tokens, image_preprocess_ms, vision_projector_ms = _project_decoded_image(
                model=model,
                image_processor=image_processor,
                image=decoded_image,
                device=device,
                model_dtype=model_dtype,
            )
            result = _score_yes_no(
                model=model,
                tokenizer=tokenizer,
                sample=sample,
                visual_tokens=visual_tokens,
                device=device,
                candidate_prefix=args.candidate_prefix,
            )

            for stats in [
                overall,
                by_retain.setdefault(retain, _empty_stats()),
            ]:
                _update_stats(
                    stats,
                    result=result,
                    payload_bytes=len(payload),
                    target_bytes=target_bytes,
                    quality=quality,
                    encode_latency_ms=encode_ms,
                    quality_search_latency_ms=quality_search_ms,
                    decode_latency_ms=decode_ms,
                    image_preprocess_latency_ms=image_preprocess_ms,
                    vision_projector_latency_ms=vision_projector_ms,
                )

            if predictions_f is not None:
                budget_utilization = (
                    float(len(payload)) / float(target_bytes)
                    if target_bytes is not None and int(target_bytes) > 0
                    else None
                )
                predictions_f.write(
                    json.dumps(
                        {
                            "index": seen - 1,
                            "retain_ratio": retain,
                            "image_path": sample["image_paths"][0],
                            "label": None if result is None else result["label"],
                            "prediction": None if result is None else result["prediction"],
                            "yes_score": None if result is None else result["yes_score"],
                            "no_score": None if result is None else result["no_score"],
                            "confidence": None if result is None else result["confidence"],
                            "payload_bytes": len(payload),
                            "target_bytes": target_bytes,
                            "budget_utilization": budget_utilization,
                            "over_budget": (
                                bool(len(payload) > int(target_bytes))
                                if target_bytes is not None and int(target_bytes) > 0
                                else None
                            ),
                            "compression_quality": quality,
                            "encode_latency_ms": encode_ms,
                            "quality_search_latency_ms": quality_search_ms,
                            "dynamic_encode_latency_ms": encode_ms + quality_search_ms,
                            "decode_latency_ms": decode_ms,
                            "image_preprocess_latency_ms": image_preprocess_ms,
                            "vision_projector_latency_ms": vision_projector_ms,
                            "scoring_latency_ms": None if result is None else result["scoring_latency_ms"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            progress.set_postfix(sample=seen, retain=retain, q=quality, bytes=len(payload))
    finally:
        if predictions_f is not None:
            predictions_f.close()

    finalized = {
        "overall": _finalize_stats(overall, args.bandwidth_mbps),
        "by_retain_ratio": {
            retain: _finalize_stats(stats, args.bandwidth_mbps)
            for retain, stats in sorted(by_retain.items(), key=lambda kv: float(kv[0]))
        },
    }

    result_payload = {
        "model_name_or_path": args.model_name_or_path,
        "data_path": args.data_path,
        "image_folder": args.image_folder,
        "split_mode": split_mode,
        "split_indices_path": args.split_indices_path,
        "split_name": args.split_name if args.split_indices_path else None,
        "split_summary": None if split_payload is None else split_payload.get("split_summary"),
        "eval_task": "pope_yes_no_likelihood_compressed_image_baseline",
        "conversation_mode": args.conversation_mode,
        "question_suffix": args.question_suffix,
        "candidate_prefix": args.candidate_prefix,
        "num_samples": seen,
        "compression": {
            "format": args.compression_format,
            "mode": args.compression_mode,
            "fixed_quality": args.image_quality if args.compression_mode == "fixed_quality" else None,
            "min_quality": args.min_quality,
            "max_quality": args.max_quality,
            "jpeg_subsampling": args.jpeg_subsampling if args.compression_format == "jpeg" else None,
            "metadata_bytes": args.metadata_bytes,
        },
        "feature_payload_budget_note": {
            "features": "N * feature_dim int8 bytes",
            "feature_scales": "N fp16 scales",
            "source_indices": "teacher_tokens int16 entries",
            "source_offsets": "N+1 int32 entries",
            "feature_dim": args.feature_dim,
            "teacher_tokens": args.teacher_tokens,
        },
        "metrics_note": {
            "payload_bytes": "actual compressed image bytes transmitted from edge to cloud",
            "target_bytes": "estimated ToMe feature payload budget used only in byte_matched mode",
            "budget_utilization": "payload_bytes / target_bytes for byte_matched mode; values below 1 mean the compressed image used less than the feature budget",
            "over_budget_ratio": "fraction of samples whose compressed image exceeded the target byte budget, usually because even minimum quality was too large",
            "under_80pct_budget_ratio": "fraction of budgeted samples using less than 80% of the target byte budget",
            "under_50pct_budget_ratio": "fraction of budgeted samples using less than 50% of the target byte budget",
            "edge_encode_latency_ms": "single final JPEG/WebP encode time on edge; CLIP/ToMe is not run for this baseline",
            "quality_search_latency_ms": "extra multi-encode search cost used to find the byte-matched quality; not included in default end_to_end_latency_ms",
            "edge_dynamic_encode_latency_ms": "edge_encode_latency_ms + quality_search_latency_ms if quality search is performed online",
            "image_preprocess_latency_ms": "cloud-side LLaVA image preprocessing and transfer to model device",
            "cloud_latency_ms": "image decode + image preprocessing + cloud LLaVA vision tower/projector + yes/no likelihood scoring",
            "end_to_end_latency_ms": "edge encode + payload transfer under assumed bandwidth + cloud latency",
        },
        "metrics": finalized,
    }

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(finalized["overall"], ensure_ascii=False, indent=2), flush=True)
    print(f"[DONE] saved to {args.output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
