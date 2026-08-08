from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor, LlavaForConditionalGeneration

from mm.semantic_reconstructor import SourceGuidedCompactSemanticReconstructor
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


METHODS = ("no_tome_no_sgcsr", "tome_no_sgcsr", "tome_sgcsr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate SGCSR zero-shot transfer on compressed POPE data. "
            "All three groups use the same POPE samples: no-ToMe/no-SGCSR, "
            "ToMe/no-SGCSR, and ToMe/SGCSR."
        )
    )
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Local LLaVA-1.5 HF model path.")
    parser.add_argument("--data_path", type=str, required=True, help="Compressed POPE feature directory or manifest.")
    parser.add_argument("--image_folder", type=str, required=True, help="POPE image folder for the no-ToMe baseline.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="SGCSR checkpoint, e.g. best.pt.")
    parser.add_argument("--output_path", type=str, default="outputs/sgcsr_pope_eval.json")
    parser.add_argument("--save_predictions", type=str, default=None)
    parser.add_argument(
        "--split_indices_path",
        type=str,
        default=None,
        help="Optional split_indices.json saved by train_sgcsr_pope_adapt.py.",
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which saved POPE split to evaluate when --split_indices_path is set.",
    )
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument(
        "--reconstructor_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--local_topk",
        type=int,
        default=None,
        help=(
            "Override the checkpoint local top-k for evaluation. "
            "Use 8/16/32/64 for ablations, or 0 to disable top-k. "
            "When omitted, the value stored in the checkpoint is used."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Keep at 1; visual lengths differ across methods.")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--max_eval_samples", type=int, default=0, help="Debug only; 0 evaluates all POPE samples.")
    parser.add_argument(
        "--conversation_mode",
        type=str,
        default="first",
        choices=["first", "all", "full"],
        help="Must match the dataset expansion used by the saved split. Formal POPE uses 'first'.",
    )
    parser.add_argument(
        "--max_text_length",
        type=int,
        default=0,
        help="Must match train_sgcsr_pope_adapt.py when evaluating a saved split.",
    )
    parser.add_argument(
        "--candidate_prefix",
        type=str,
        default=" ",
        help="Prefix before yes/no candidates. A leading space matches LLaVA supervised answer tokenization.",
    )
    parser.add_argument(
        "--question_suffix",
        type=str,
        default="Please answer yes or no.",
        help="Appended to POPE questions unless the question already contains 'yes or no'.",
    )
    parser.add_argument("--allow_missing_source", action="store_true", help="Debug only; formal SGCSR eval needs source maps.")
    return parser.parse_args()


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
    expected_max_text_length = int(payload.get("max_text_length", 0))
    if expected_max_text_length != int(args.max_text_length):
        raise ValueError(
            f"Split max_text_length mismatch: split={expected_max_text_length} current={args.max_text_length}."
        )
    expected_suffix = str(payload.get("question_suffix", args.question_suffix))
    if expected_suffix != str(args.question_suffix):
        raise ValueError(
            f"Split question_suffix mismatch: split={expected_suffix!r} current={args.question_suffix!r}."
        )
    if int(payload.get("max_samples", 0)) != 0:
        raise ValueError(
            "This split_indices.json was generated from a debug subset with max_samples="
            f"{payload.get('max_samples')}. Do not use it for formal POPE evaluation."
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


def _empty_stats() -> Dict[str, float]:
    return {
        "tp": 0.0,
        "tn": 0.0,
        "fp": 0.0,
        "fn": 0.0,
        "invalid": 0.0,
        "total": 0.0,
        "yes_predictions": 0.0,
        "projector_latency_ms_sum": 0.0,
        "sgcsr_latency_ms_sum": 0.0,
        "scoring_latency_ms_sum": 0.0,
        "cloud_latency_ms_sum": 0.0,
        "latency_count": 0.0,
        "confidence_sum": 0.0,
    }


def _update_stats(
    stats: Dict[str, float],
    *,
    label: str,
    prediction: Optional[str],
    projector_latency_ms: float,
    sgcsr_latency_ms: float,
    scoring_latency_ms: float,
    cloud_latency_ms: float,
    confidence: float,
):
    stats["total"] += 1
    stats["projector_latency_ms_sum"] += float(projector_latency_ms)
    stats["sgcsr_latency_ms_sum"] += float(sgcsr_latency_ms)
    stats["scoring_latency_ms_sum"] += float(scoring_latency_ms)
    stats["cloud_latency_ms_sum"] += float(cloud_latency_ms)
    stats["latency_count"] += 1
    stats["confidence_sum"] += float(confidence)

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


def _finalize(stats: Dict[str, float]) -> Dict[str, Any]:
    tp, tn, fp, fn = stats["tp"], stats["tn"], stats["fp"], stats["fn"]
    total = max(1.0, stats["total"])
    latency_count = max(1.0, stats["latency_count"])
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    specificity = tn / max(1.0, tn + fp)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    cloud_latency_ms = stats["cloud_latency_ms_sum"] / latency_count
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
        "projector_latency_ms": float(stats["projector_latency_ms_sum"] / latency_count),
        "sgcsr_latency_ms": float(stats["sgcsr_latency_ms_sum"] / latency_count),
        "scoring_latency_ms": float(stats["scoring_latency_ms_sum"] / latency_count),
        "cloud_latency_ms": float(cloud_latency_ms),
        "samples_per_second": float(1000.0 / cloud_latency_ms) if cloud_latency_ms > 0 else None,
        "avg_confidence": float(stats["confidence_sum"] / total),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "invalid": int(stats["invalid"]),
        "total": int(stats["total"]),
    }


def _load_reconstructor(
    checkpoint_path: str,
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
    local_topk_override: Optional[int] = None,
) -> SourceGuidedCompactSemanticReconstructor:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or "reconstructor" not in payload:
        raise ValueError(f"Unsupported SGCSR checkpoint format: {checkpoint_path}")
    ckpt_args = payload.get("args", {})
    checkpoint_local_topk = int(ckpt_args.get("local_topk", 0))
    local_topk = checkpoint_local_topk if local_topk_override is None else int(local_topk_override)
    if local_topk < 0:
        raise ValueError(f"local_topk must be non-negative, got {local_topk}")
    reconstructor = SourceGuidedCompactSemanticReconstructor(
        dim=hidden_size,
        num_queries=int(ckpt_args.get("num_queries", 144)),
        depth=int(ckpt_args.get("depth", 2)),
        heads=int(ckpt_args.get("heads", 8)),
        dim_head=int(ckpt_args.get("dim_head", 128)),
        ff_mult=int(ckpt_args.get("ff_mult", 2)),
        dropout=float(ckpt_args.get("dropout", 0.0)),
        local_topk=local_topk,
        local_radius=float(ckpt_args.get("local_radius", 0.0)),
    ).to(device=device, dtype=dtype)
    reconstructor.load_state_dict(payload["reconstructor"], strict=True)
    reconstructor.eval()
    return reconstructor


def _image_to_clip_tokens(
    model,
    image_processor,
    image_path: str,
    device: torch.device,
    model_dtype: torch.dtype,
) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    pixel_values = image_processor(images=[image], return_tensors="pt").pixel_values.to(device=device, dtype=model_dtype)
    with torch.no_grad():
        vision_outputs = get_vision_tower(model)(pixel_values, output_hidden_states=True)
    return select_vision_features(model, vision_outputs)


def _project_no_tome_visual(
    model,
    image_processor,
    image_path: str,
    device: torch.device,
    model_dtype: torch.dtype,
) -> Tuple[torch.Tensor, float]:
    clip_tokens = _image_to_clip_tokens(model, image_processor, image_path, device, model_dtype)
    _sync_device(device)
    start = time.perf_counter()
    with torch.no_grad():
        visual_tokens = model.multi_modal_projector(clip_tokens)
    _sync_device(device)
    return visual_tokens, (time.perf_counter() - start) * 1000.0


def _project_tome_visual(model, sample: Dict[str, Any], device: torch.device, model_dtype: torch.dtype) -> Tuple[torch.Tensor, float]:
    compressed_features = sample["compressed_features"].to(device=device, dtype=model_dtype)
    _sync_device(device)
    start = time.perf_counter()
    with torch.no_grad():
        visual_tokens = model.multi_modal_projector(compressed_features)
    _sync_device(device)
    return visual_tokens, (time.perf_counter() - start) * 1000.0


def _run_sgcsr(
    reconstructor,
    visual_tokens: torch.Tensor,
    sample: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, float]:
    token_centers = sample["token_centers"].to(device=device, dtype=dtype)
    token_sizes = sample["token_sizes"].to(device=device, dtype=dtype)
    retain_ratio = sample["retain_ratio"].to(device=device, dtype=dtype)
    attention_mask = sample["compressed_attention_mask"].to(device=device)
    _sync_device(device)
    start = time.perf_counter()
    with torch.no_grad():
        reconstructed = reconstructor(
            visual_embeddings=visual_tokens.to(dtype=dtype),
            token_centers=token_centers,
            token_sizes=token_sizes,
            retain_ratio=retain_ratio,
            attention_mask=attention_mask,
        )
    _sync_device(device)
    return reconstructed, (time.perf_counter() - start) * 1000.0


def _maybe_append_question_suffix(question: str, question_suffix: str) -> str:
    question = question.strip()
    if question_suffix and "yes or no" not in question.lower():
        question = f"{question}\n{question_suffix.strip()}"
    return question


def _decode_prompt_question(tokenizer, prompt_ids: torch.Tensor) -> str:
    prompt_text = tokenizer.decode(prompt_ids.tolist(), skip_special_tokens=False)
    if "USER:" in prompt_text:
        prompt_text = prompt_text.split("USER:", 1)[1]
    if "ASSISTANT:" in prompt_text:
        prompt_text = prompt_text.split("ASSISTANT:", 1)[0]
    prompt_text = prompt_text.replace("<image>", "").strip()
    return prompt_text


def _extract_prompt_and_label(tokenizer, sample: Dict[str, Any], device: torch.device, question_suffix: str):
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
    raw_prompt_ids = input_ids[:prompt_len]
    question = _decode_prompt_question(tokenizer, raw_prompt_ids)
    question = _maybe_append_question_suffix(question, question_suffix)
    prompt = f"USER: <image>\n{question} ASSISTANT:"
    prompt_ids = torch.tensor(tokenizer(prompt, add_special_tokens=True).input_ids, dtype=torch.long).unsqueeze(0).to(device)
    prompt_attention_mask = torch.ones_like(prompt_ids, device=device)
    label_ids = labels[answer_pos]
    label_text = tokenizer.decode(label_ids.tolist(), skip_special_tokens=True).strip().lower()
    label = _parse_yes_no(label_text)
    if label is None:
        return None
    return prompt_ids, prompt_attention_mask, label, label_text


def _tokenize_candidate(tokenizer, text: str, device: torch.device) -> torch.Tensor:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) > 0 and isinstance(ids[0], list):
        ids = ids[0]
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
    question_suffix: str,
) -> Optional[Dict[str, Any]]:
    extracted = _extract_prompt_and_label(tokenizer, sample, device, question_suffix)
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


def _evaluate_one_method(
    *,
    model,
    tokenizer,
    sample: Dict[str, Any],
    visual_tokens: torch.Tensor,
    projector_latency_ms: float,
    sgcsr_latency_ms: float,
    device: torch.device,
    candidate_prefix: str,
    question_suffix: str,
) -> Optional[Dict[str, Any]]:
    _sync_device(device)
    score_path_start = time.perf_counter()
    scored = _score_yes_no(model, tokenizer, sample, visual_tokens, device, candidate_prefix, question_suffix)
    _sync_device(device)
    score_path_latency_ms = (time.perf_counter() - score_path_start) * 1000.0
    cloud_latency_ms = projector_latency_ms + sgcsr_latency_ms
    if scored is None:
        return None
    cloud_latency_ms += score_path_latency_ms
    scored.update(
        {
            "projector_latency_ms": projector_latency_ms,
            "sgcsr_latency_ms": sgcsr_latency_ms,
            "cloud_latency_ms": cloud_latency_ms,
            "score_path_latency_ms": score_path_latency_ms,
        }
    )
    return scored


def main() -> int:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("evaluate_sgcsr_pope.py requires --batch_size 1 to keep variable visual lengths exact.")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_dtype = dtype_from_name(args.dtype)
    reconstructor_dtype = model_dtype if args.reconstructor_dtype == "auto" else dtype_from_name(args.reconstructor_dtype)

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
    print(f"[INFO] loaded model on {device} with dtype={model_dtype}", flush=True)

    hidden_size = int(get_language_model(model).config.hidden_size)
    reconstructor = _load_reconstructor(
        checkpoint_path=args.checkpoint_path,
        hidden_size=hidden_size,
        device=device,
        dtype=reconstructor_dtype,
        local_topk_override=args.local_topk,
    )
    effective_local_topk = int(reconstructor.layers[0].local_topk)
    effective_local_radius = float(reconstructor.layers[0].local_radius)
    print(
        f"[INFO] loaded SGCSR checkpoint: {args.checkpoint_path} "
        f"local_topk={effective_local_topk} local_radius={effective_local_radius}",
        flush=True,
    )

    dataset = SGCSRCompressedDataset(
        data_path=args.data_path,
        image_folder=args.image_folder,
        tokenizer=tokenizer,
        max_samples=args.max_eval_samples,
        allow_missing_source=args.allow_missing_source,
        seed=42,
        conversation_mode=args.conversation_mode,
        max_text_length=args.max_text_length,
    )
    split_payload = None
    split_mode = None
    if args.split_indices_path:
        if args.max_eval_samples > 0:
            raise ValueError("--max_eval_samples cannot be combined with --split_indices_path for formal evaluation.")
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
    print(f"[INFO] POPE samples={len(dataset)} data_path={args.data_path}", flush=True)

    metrics = {method: {"overall": _empty_stats(), "by_retain_ratio": {}} for method in METHODS}
    predictions_f = None
    if args.save_predictions:
        Path(args.save_predictions).parent.mkdir(parents=True, exist_ok=True)
        predictions_f = open(args.save_predictions, "w", encoding="utf-8")

    seen = 0
    progress = tqdm(dataloader, desc="SGCSR POPE eval", dynamic_ncols=True)
    try:
        for sample in progress:
            if args.max_eval_samples > 0 and seen >= args.max_eval_samples:
                break
            retain = _retain_key(sample["retain_ratio"][0].item())

            no_tome_visual, no_tome_projector_ms = _project_no_tome_visual(
                model=model,
                image_processor=image_processor,
                image_path=sample["image_paths"][0],
                device=device,
                model_dtype=model_dtype,
            )
            tome_visual, tome_projector_ms = _project_tome_visual(model, sample, device, model_dtype)
            sgcsr_visual, sgcsr_ms = _run_sgcsr(
                reconstructor=reconstructor,
                visual_tokens=tome_visual,
                sample=sample,
                device=device,
                dtype=reconstructor_dtype,
            )

            payloads = {
                "no_tome_no_sgcsr": (no_tome_visual, no_tome_projector_ms, 0.0),
                "tome_no_sgcsr": (tome_visual, tome_projector_ms, 0.0),
                "tome_sgcsr": (sgcsr_visual, tome_projector_ms, sgcsr_ms),
            }

            for method, (visual_tokens, projector_ms, sgcsr_latency_ms) in payloads.items():
                result = _evaluate_one_method(
                    model=model,
                    tokenizer=tokenizer,
                    sample=sample,
                    visual_tokens=visual_tokens,
                    projector_latency_ms=projector_ms,
                    sgcsr_latency_ms=sgcsr_latency_ms,
                    device=device,
                    candidate_prefix=args.candidate_prefix,
                    question_suffix=args.question_suffix,
                )
                if result is None:
                    continue

                method_stats = metrics[method]
                by_retain = method_stats["by_retain_ratio"].setdefault(retain, _empty_stats())
                for stats in [method_stats["overall"], by_retain]:
                    _update_stats(
                        stats,
                        label=result["label"],
                        prediction=result["prediction"],
                        projector_latency_ms=result["projector_latency_ms"],
                        sgcsr_latency_ms=result["sgcsr_latency_ms"],
                        scoring_latency_ms=result["scoring_latency_ms"],
                        cloud_latency_ms=result["cloud_latency_ms"],
                        confidence=result["confidence"],
                    )

                if predictions_f is not None:
                    predictions_f.write(
                        json.dumps(
                            {
                                "index": seen,
                                "method": method,
                                "retain_ratio": retain,
                                "label": result["label"],
                                "label_text": result["label_text"],
                                "prediction": result["prediction"],
                                "yes_score": result["yes_score"],
                                "no_score": result["no_score"],
                                "confidence": result["confidence"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            seen += 1
            progress.set_postfix(sample=seen, retain=retain)
    finally:
        if predictions_f is not None:
            predictions_f.close()

    finalized = {}
    for method, method_stats in metrics.items():
        finalized[method] = {
            "overall": _finalize(method_stats["overall"]),
            "by_retain_ratio": {
                retain: _finalize(stats)
                for retain, stats in sorted(method_stats["by_retain_ratio"].items(), key=lambda kv: float(kv[0]))
            },
        }

    result = {
        "model_name_or_path": args.model_name_or_path,
        "checkpoint_path": args.checkpoint_path,
        "data_path": args.data_path,
        "split_mode": split_mode,
        "split_indices_path": args.split_indices_path,
        "split_name": args.split_name if args.split_indices_path else None,
        "split_summary": split_payload.get("split_summary") if split_payload else None,
        "eval_task": "pope_yes_no_likelihood",
        "candidate_prefix": args.candidate_prefix,
        "question_suffix": args.question_suffix,
        "local_attention": {
            "local_topk": effective_local_topk,
            "local_radius": effective_local_radius,
            "local_topk_override": args.local_topk,
        },
        "metrics_note": {
            "cloud_latency_ms": "projector + optional SGCSR + yes/no likelihood scoring; edge vision time is excluded",
            "no_tome_no_sgcsr": "uses the same POPE sample image but recomputes no-ToMe CLIP features before the frozen LLaVA projector",
        },
        "num_samples": seen,
        "metrics": finalized,
    }

    if args.output_path:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
