from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor, LlavaForConditionalGeneration

from mm.semantic_reconstructor import SourceGuidedCompactSemanticReconstructor, pool_teacher_visual_tokens
from train_sgcsr import (
    SGCSRCompressedDataset,
    SGCSRCollator,
    build_stratified_train_val_test_split,
    build_teacher_visual_embeddings,
    dtype_from_name,
    get_input_embedding_layer,
    get_language_model,
    merge_text_and_visual_tokens,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate feature-space similarity between SGCSR reconstructed tokens and "
            "no-ToMe LLaVA visual tokens pooled from 576 tokens to the SGCSR compact grid."
        )
    )
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Local LLaVA-1.5 HF model path.")
    parser.add_argument("--data_path", type=str, required=True, help="Compressed feature directory, .pt file, or manifest.")
    parser.add_argument("--image_folder", type=str, default=None, help="Image root for no-ToMe teacher features.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="SGCSR checkpoint, e.g. best.pt.")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save JSON metrics.")
    parser.add_argument("--save_per_sample", type=str, default=None, help="Optional JSONL path for per-sample metrics.")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument(
        "--reconstructor_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument(
        "--split_indices_path",
        type=str,
        default=None,
        help="Optional split_indices.json saved by train_sgcsr.py or train_sgcsr_pope_adapt.py.",
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which saved split to evaluate when --split_indices_path is set.",
    )
    parser.add_argument(
        "--eval_mode",
        type=str,
        default="all",
        choices=["all", "stratified"],
        help=(
            "Used only without --split_indices_path. 'all' evaluates every compressed sample; "
            "'stratified' rebuilds the train_sgcsr.py-style held-out split using --test_ratio."
        ),
    )
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Held-out ratio for --eval_mode stratified.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_eval_samples", type=int, default=0, help="Debug cap after split selection; 0 means all.")
    parser.add_argument(
        "--conversation_mode",
        type=str,
        default="first",
        choices=["first", "all", "full"],
        help="Must match the saved split if --split_indices_path is used.",
    )
    parser.add_argument(
        "--max_text_length",
        type=int,
        default=0,
        help="Must match the saved split if --split_indices_path is used.",
    )
    parser.add_argument("--allow_missing_source", action="store_true", help="Debug only; formal SGCSR eval needs source maps.")
    parser.add_argument(
        "--low_cosine_threshold",
        type=float,
        default=0.5,
        help="Token cosine below this value is counted as low-similarity.",
    )
    parser.add_argument("--hist_bins", type=int, default=400, help="Histogram bins for approximate cosine percentiles.")
    parser.add_argument(
        "--kl_temperature",
        type=float,
        default=1.0,
        help="Temperature used when converting hidden features to distributions for KL divergence.",
    )
    parser.add_argument(
        "--compute_behavior_metrics",
        action="store_true",
        help="Also compare LLM yes/no likelihood behavior for full teacher, pooled compact teacher, and SGCSR tokens.",
    )
    parser.add_argument(
        "--candidate_prefix",
        type=str,
        default=" ",
        help="Prefix before yes/no candidates when --compute_behavior_metrics is enabled.",
    )
    parser.add_argument(
        "--question_suffix",
        type=str,
        default="Please answer yes or no.",
        help="Suffix appended to questions unless the prompt already asks for yes/no.",
    )
    parser.add_argument(
        "--behavior_temperature",
        type=float,
        default=1.0,
        help="Temperature for two-class yes/no probability and KL metrics.",
    )
    parser.add_argument(
        "--sgcsr_output_scale",
        type=float,
        default=1.0,
        help="Inference-only scale applied to SGCSR output tokens before feature and behavior metrics.",
    )
    return parser.parse_args()


def _sync_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _retain_key(value: Any) -> str:
    return f"{float(value):.2f}"


def _load_reconstructor(
    checkpoint_path: str,
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[SourceGuidedCompactSemanticReconstructor, Dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or "reconstructor" not in payload:
        raise ValueError(f"Unsupported SGCSR checkpoint format: {checkpoint_path}")
    ckpt_args = payload.get("args", {})
    reconstructor = SourceGuidedCompactSemanticReconstructor(
        dim=hidden_size,
        num_queries=int(ckpt_args.get("num_queries", 144)),
        depth=int(ckpt_args.get("depth", 2)),
        heads=int(ckpt_args.get("heads", 8)),
        dim_head=int(ckpt_args.get("dim_head", 128)),
        ff_mult=int(ckpt_args.get("ff_mult", 2)),
        dropout=float(ckpt_args.get("dropout", 0.0)),
        local_topk=int(ckpt_args.get("local_topk", 0)),
        local_radius=float(ckpt_args.get("local_radius", 0.0)),
    ).to(device=device, dtype=dtype)
    reconstructor.load_state_dict(payload["reconstructor"], strict=True)
    reconstructor.eval()
    return reconstructor, ckpt_args


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
            "Use the split_indices.json generated for this compressed dataset."
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


def _empty_stats(hist_bins: int) -> Dict[str, Any]:
    return {
        "num_samples": 0,
        "num_tokens": 0,
        "num_features": 0,
        "cosine_sum": 0.0,
        "sq_error_sum": 0.0,
        "abs_error_sum": 0.0,
        "teacher_sq_sum": 0.0,
        "normalized_sq_error_sum": 0.0,
        "normalized_abs_error_sum": 0.0,
        "normalized_teacher_sq_sum": 0.0,
        "student_norm_sum": 0.0,
        "teacher_norm_sum": 0.0,
        "norm_ratio_sum": 0.0,
        "low_cosine_count": 0,
        "token_relative_l2_sum": 0.0,
        "element_relative_abs_sum": 0.0,
        "feature_kl_t2s_sum": 0.0,
        "feature_kl_s2t_sum": 0.0,
        "feature_kl_sym_sum": 0.0,
        "position_mse_sum": 0.0,
        "position_mse_max": 0.0,
        "position_match_distance_sum": 0.0,
        "position_exact_match_count": 0,
        "position_within_one_count": 0,
        "position_count": 0,
        "hist": torch.zeros(hist_bins, dtype=torch.float64),
    }


def _factor_grid(num_tokens: int) -> Tuple[int, int]:
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}")
    height = int(math.sqrt(num_tokens))
    while height > 1 and num_tokens % height != 0:
        height -= 1
    return height, num_tokens // height


def _grid_coords(num_tokens: int, device: torch.device) -> torch.Tensor:
    height, width = _factor_grid(num_tokens)
    ys = torch.arange(height, dtype=torch.float32, device=device)
    xs = torch.arange(width, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1)


def _standardized_softmax(features: torch.Tensor, temperature: float) -> torch.Tensor:
    temperature = max(float(temperature), 1e-6)
    features = features.float()
    centered = features - features.mean(dim=-1, keepdim=True)
    scaled = centered / features.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return F.softmax(scaled / temperature, dim=-1).clamp_min(1e-12)


def _feature_kl(
    student: torch.Tensor,
    teacher: torch.Tensor,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    teacher_prob = _standardized_softmax(teacher, temperature)
    student_prob = _standardized_softmax(student, temperature)
    kl_t2s = (teacher_prob * (teacher_prob.log() - student_prob.log())).sum(dim=-1)
    kl_s2t = (student_prob * (student_prob.log() - teacher_prob.log())).sum(dim=-1)
    return kl_t2s, kl_s2t, 0.5 * (kl_t2s + kl_s2t)


def _position_metrics(student: torch.Tensor, teacher: torch.Tensor) -> Dict[str, torch.Tensor]:
    if student.shape != teacher.shape:
        raise ValueError(f"student and teacher shapes differ: {tuple(student.shape)} vs {tuple(teacher.shape)}")
    if student.ndim == 2:
        student = student.unsqueeze(0)
        teacher = teacher.unsqueeze(0)
    if student.ndim != 3:
        raise ValueError(f"student and teacher must be [B, K, D] or [K, D], got {tuple(student.shape)}")

    bsz, num_tokens, _ = student.shape
    device = student.device
    position_mse = (student.float() - teacher.float()).pow(2).mean(dim=-1)

    coords = _grid_coords(num_tokens, device=device)
    base_index = torch.arange(num_tokens, device=device)
    base_coords = coords[base_index].unsqueeze(0).expand(bsz, -1, -1)

    student_unit = F.normalize(student.float(), p=2, dim=-1)
    teacher_unit = F.normalize(teacher.float(), p=2, dim=-1)
    similarity = torch.bmm(student_unit, teacher_unit.transpose(1, 2))
    nearest_index = similarity.argmax(dim=-1)
    nearest_coords = coords[nearest_index]

    match_distance = (base_coords - nearest_coords).pow(2).sum(dim=-1).sqrt()
    exact_match = nearest_index.eq(base_index.unsqueeze(0))
    within_one = match_distance.le(1.0)
    return {
        "position_mse": position_mse,
        "match_distance": match_distance,
        "exact_match": exact_match,
        "within_one": within_one,
    }


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
    return prompt_text.replace("<image>", "").strip()


def _extract_prompt_and_label(
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    question_suffix: str,
):
    input_ids = input_ids[0]
    attention_mask = attention_mask[0]
    labels = labels[0]
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


def _scores_to_probs(scores: Dict[str, float], temperature: float) -> Dict[str, float]:
    temperature = max(float(temperature), 1e-6)
    logits = torch.tensor([scores["yes"], scores["no"]], dtype=torch.float32) / temperature
    probs = torch.softmax(logits, dim=0)
    log_probs = torch.log_softmax(logits, dim=0)
    return {
        "yes_prob": float(probs[0].item()),
        "no_prob": float(probs[1].item()),
        "yes_logprob": float(log_probs[0].item()),
        "no_logprob": float(log_probs[1].item()),
    }


@torch.no_grad()
def _score_yes_no_behavior(
    model,
    tokenizer,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    visual_tokens: torch.Tensor,
    device: torch.device,
    candidate_prefix: str,
    question_suffix: str,
    temperature: float,
) -> Optional[Dict[str, Any]]:
    extracted = _extract_prompt_and_label(tokenizer, input_ids, attention_mask, labels, device, question_suffix)
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

    language_model = get_language_model(model)
    input_embeddings = get_input_embedding_layer(model)
    scores: Dict[str, float] = {}
    for candidate in ["yes", "no"]:
        candidate_ids = _tokenize_candidate(tokenizer, f"{candidate_prefix}{candidate}", device)
        candidate_embeds = input_embeddings(candidate_ids)
        inputs_embeds = torch.cat([prefix_embeds, candidate_embeds], dim=1)
        candidate_attention = torch.ones_like(candidate_ids, device=device)
        full_attention_mask = torch.cat([prefix_attention_mask, candidate_attention], dim=1)
        candidate_labels = torch.full(inputs_embeds.shape[:2], -100, dtype=torch.long, device=device)
        candidate_labels[:, -candidate_ids.shape[1] :] = candidate_ids
        out = language_model(inputs_embeds=inputs_embeds, attention_mask=full_attention_mask, labels=candidate_labels)
        scores[candidate] = -float(out.loss.item())

    prediction = "yes" if scores["yes"] >= scores["no"] else "no"
    prob_payload = _scores_to_probs(scores, temperature=temperature)
    target_logprob = prob_payload[f"{label}_logprob"]
    return {
        "label": label,
        "label_text": label_text,
        "prediction": prediction,
        "yes_score": scores["yes"],
        "no_score": scores["no"],
        "confidence": abs(scores["yes"] - scores["no"]),
        "target_logprob": target_logprob,
        **prob_payload,
    }


def _empty_behavior_method_stats() -> Dict[str, float]:
    return {
        "total": 0.0,
        "correct": 0.0,
        "yes_predictions": 0.0,
        "tp": 0.0,
        "tn": 0.0,
        "fp": 0.0,
        "fn": 0.0,
        "confidence_sum": 0.0,
        "target_logprob_sum": 0.0,
    }


def _empty_behavior_pair_stats() -> Dict[str, float]:
    return {
        "count": 0.0,
        "agreement": 0.0,
        "ref_yes_cmp_no": 0.0,
        "ref_no_cmp_yes": 0.0,
        "yes_prob_abs_gap_sum": 0.0,
        "confidence_abs_gap_sum": 0.0,
        "target_logprob_abs_gap_sum": 0.0,
        "target_logprob_signed_gap_sum": 0.0,
        "kl_ref_to_cmp_sum": 0.0,
        "kl_cmp_to_ref_sum": 0.0,
        "kl_sym_sum": 0.0,
    }


def _empty_behavior_stats() -> Dict[str, Dict[str, Dict[str, float]]]:
    return {"methods": {}, "pairs": {}}


def _update_behavior_method_stats(stats: Dict[str, float], result: Dict[str, Any]) -> None:
    label = result["label"]
    prediction = result["prediction"]
    stats["total"] += 1
    stats["correct"] += 1 if prediction == label else 0
    stats["yes_predictions"] += 1 if prediction == "yes" else 0
    stats["confidence_sum"] += float(result["confidence"])
    stats["target_logprob_sum"] += float(result["target_logprob"])
    if label == "yes" and prediction == "yes":
        stats["tp"] += 1
    elif label == "no" and prediction == "no":
        stats["tn"] += 1
    elif label == "no" and prediction == "yes":
        stats["fp"] += 1
    elif label == "yes" and prediction == "no":
        stats["fn"] += 1


def _two_class_kl(ref: Dict[str, Any], cmp: Dict[str, Any]) -> Tuple[float, float]:
    ref_probs = torch.tensor([ref["yes_prob"], ref["no_prob"]], dtype=torch.float64).clamp_min(1e-12)
    cmp_probs = torch.tensor([cmp["yes_prob"], cmp["no_prob"]], dtype=torch.float64).clamp_min(1e-12)
    kl_ref_to_cmp = float((ref_probs * (ref_probs.log() - cmp_probs.log())).sum().item())
    kl_cmp_to_ref = float((cmp_probs * (cmp_probs.log() - ref_probs.log())).sum().item())
    return kl_ref_to_cmp, kl_cmp_to_ref


def _update_behavior_pair_stats(stats: Dict[str, float], ref: Dict[str, Any], cmp: Dict[str, Any]) -> None:
    stats["count"] += 1
    ref_pred = ref["prediction"]
    cmp_pred = cmp["prediction"]
    stats["agreement"] += 1 if ref_pred == cmp_pred else 0
    stats["ref_yes_cmp_no"] += 1 if ref_pred == "yes" and cmp_pred == "no" else 0
    stats["ref_no_cmp_yes"] += 1 if ref_pred == "no" and cmp_pred == "yes" else 0
    stats["yes_prob_abs_gap_sum"] += abs(float(cmp["yes_prob"]) - float(ref["yes_prob"]))
    stats["confidence_abs_gap_sum"] += abs(float(cmp["confidence"]) - float(ref["confidence"]))
    target_gap = float(cmp["target_logprob"]) - float(ref["target_logprob"])
    stats["target_logprob_signed_gap_sum"] += target_gap
    stats["target_logprob_abs_gap_sum"] += abs(target_gap)
    kl_ref_to_cmp, kl_cmp_to_ref = _two_class_kl(ref, cmp)
    stats["kl_ref_to_cmp_sum"] += kl_ref_to_cmp
    stats["kl_cmp_to_ref_sum"] += kl_cmp_to_ref
    stats["kl_sym_sum"] += 0.5 * (kl_ref_to_cmp + kl_cmp_to_ref)


def _finalize_behavior_method_stats(stats: Dict[str, float]) -> Dict[str, Any]:
    total = max(1.0, float(stats["total"]))
    tp = float(stats["tp"])
    tn = float(stats["tn"])
    fp = float(stats["fp"])
    fn = float(stats["fn"])
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    specificity = tn / max(1.0, tn + fp)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "total": int(stats["total"]),
        "accuracy": float(stats["correct"] / total),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fp / max(1.0, fp + tn)),
        "false_negative_rate": float(fn / max(1.0, fn + tp)),
        "specificity": float(specificity),
        "yes_ratio": float(stats["yes_predictions"] / total),
        "avg_confidence": float(stats["confidence_sum"] / total),
        "avg_target_logprob": float(stats["target_logprob_sum"] / total),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def _finalize_behavior_pair_stats(stats: Dict[str, float]) -> Dict[str, Any]:
    count = max(1.0, float(stats["count"]))
    return {
        "count": int(stats["count"]),
        "answer_agreement": float(stats["agreement"] / count),
        "answer_disagreement": float(1.0 - stats["agreement"] / count),
        "ref_yes_cmp_no_ratio": float(stats["ref_yes_cmp_no"] / count),
        "ref_no_cmp_yes_ratio": float(stats["ref_no_cmp_yes"] / count),
        "yes_prob_mae": float(stats["yes_prob_abs_gap_sum"] / count),
        "confidence_mae": float(stats["confidence_abs_gap_sum"] / count),
        "target_logprob_mae": float(stats["target_logprob_abs_gap_sum"] / count),
        "target_logprob_signed_gap": float(stats["target_logprob_signed_gap_sum"] / count),
        "kl_ref_to_cmp": float(stats["kl_ref_to_cmp_sum"] / count),
        "kl_cmp_to_ref": float(stats["kl_cmp_to_ref_sum"] / count),
        "kl_sym": float(stats["kl_sym_sum"] / count),
    }


def _finalize_behavior_stats(stats: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Any]:
    return {
        "methods": {
            name: _finalize_behavior_method_stats(method_stats)
            for name, method_stats in sorted(stats["methods"].items())
        },
        "pairs": {
            name: _finalize_behavior_pair_stats(pair_stats)
            for name, pair_stats in sorted(stats["pairs"].items())
        },
    }


def _compact_behavior_payload(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"methods": {}, "pairs": {}}
    for name, result in results.items():
        payload["methods"][name] = {
            "label": result["label"],
            "prediction": result["prediction"],
            "yes_score": result["yes_score"],
            "no_score": result["no_score"],
            "yes_prob": result["yes_prob"],
            "no_prob": result["no_prob"],
            "confidence": result["confidence"],
            "target_logprob": result["target_logprob"],
        }
    for pair_name, ref_name, cmp_name in [
        ("teacher_full_vs_teacher_compact", "teacher_full", "teacher_compact"),
        ("teacher_full_vs_sgcsr", "teacher_full", "sgcsr"),
        ("teacher_compact_vs_sgcsr", "teacher_compact", "sgcsr"),
    ]:
        if ref_name not in results or cmp_name not in results:
            continue
        ref = results[ref_name]
        cmp = results[cmp_name]
        kl_ref_to_cmp, kl_cmp_to_ref = _two_class_kl(ref, cmp)
        payload["pairs"][pair_name] = {
            "answer_agreement": ref["prediction"] == cmp["prediction"],
            "yes_prob_abs_gap": abs(float(cmp["yes_prob"]) - float(ref["yes_prob"])),
            "target_logprob_gap": float(cmp["target_logprob"]) - float(ref["target_logprob"]),
            "kl_ref_to_cmp": kl_ref_to_cmp,
            "kl_cmp_to_ref": kl_cmp_to_ref,
            "kl_sym": 0.5 * (kl_ref_to_cmp + kl_cmp_to_ref),
        }
    return payload


def _update_behavior_stats(
    stats: Dict[str, Dict[str, Dict[str, float]]],
    results: Dict[str, Dict[str, Any]],
) -> None:
    for method_name, result in results.items():
        method_stats = stats["methods"].setdefault(method_name, _empty_behavior_method_stats())
        _update_behavior_method_stats(method_stats, result)
    for pair_name, ref_name, cmp_name in [
        ("teacher_full_vs_teacher_compact", "teacher_full", "teacher_compact"),
        ("teacher_full_vs_sgcsr", "teacher_full", "sgcsr"),
        ("teacher_compact_vs_sgcsr", "teacher_compact", "sgcsr"),
    ]:
        if ref_name not in results or cmp_name not in results:
            continue
        pair_stats = stats["pairs"].setdefault(pair_name, _empty_behavior_pair_stats())
        _update_behavior_pair_stats(pair_stats, results[ref_name], results[cmp_name])


def _update_stats(
    stats: Dict[str, Any],
    *,
    student: torch.Tensor,
    teacher: torch.Tensor,
    cosine: torch.Tensor,
    low_cosine_threshold: float,
    kl_temperature: float,
):
    if student.shape != teacher.shape:
        raise ValueError(f"student and teacher shapes differ: {tuple(student.shape)} vs {tuple(teacher.shape)}")

    student_f = student.float()
    teacher_f = teacher.float()
    cosine_f = cosine.float().clamp(-1.0, 1.0)

    diff = student_f - teacher_f
    student_norm = student_f.norm(dim=-1)
    teacher_norm = teacher_f.norm(dim=-1)
    norm_ratio = student_norm / teacher_norm.clamp_min(1e-8)
    student_unit = F.normalize(student_f, p=2, dim=-1)
    teacher_unit = F.normalize(teacher_f, p=2, dim=-1)
    normalized_diff = student_unit - teacher_unit

    num_samples = int(student_f.shape[0])
    num_tokens = int(student_f.shape[0] * student_f.shape[1])
    num_features = int(student_f.numel())

    stats["num_samples"] += num_samples
    stats["num_tokens"] += num_tokens
    stats["num_features"] += num_features
    stats["cosine_sum"] += float(cosine_f.sum().item())
    stats["sq_error_sum"] += float(diff.pow(2).sum().item())
    stats["abs_error_sum"] += float(diff.abs().sum().item())
    stats["teacher_sq_sum"] += float(teacher_f.pow(2).sum().item())
    stats["normalized_sq_error_sum"] += float(normalized_diff.pow(2).sum().item())
    stats["normalized_abs_error_sum"] += float(normalized_diff.abs().sum().item())
    stats["normalized_teacher_sq_sum"] += float(teacher_unit.pow(2).sum().item())
    stats["student_norm_sum"] += float(student_norm.sum().item())
    stats["teacher_norm_sum"] += float(teacher_norm.sum().item())
    stats["norm_ratio_sum"] += float(norm_ratio.sum().item())
    stats["token_relative_l2_sum"] += float((diff.norm(dim=-1) / teacher_norm.clamp_min(1e-8)).sum().item())
    stats["element_relative_abs_sum"] += float((diff.abs() / teacher_f.abs().clamp_min(1e-8)).sum().item())
    kl_t2s, kl_s2t, kl_sym = _feature_kl(student_f, teacher_f, temperature=kl_temperature)
    stats["feature_kl_t2s_sum"] += float(kl_t2s.sum().item())
    stats["feature_kl_s2t_sum"] += float(kl_s2t.sum().item())
    stats["feature_kl_sym_sum"] += float(kl_sym.sum().item())
    position = _position_metrics(student_f, teacher_f)
    position_count = int(position["position_mse"].numel())
    stats["position_mse_sum"] += float(position["position_mse"].sum().item())
    stats["position_mse_max"] = max(float(stats["position_mse_max"]), float(position["position_mse"].max().item()))
    stats["position_match_distance_sum"] += float(position["match_distance"].sum().item())
    stats["position_exact_match_count"] += int(position["exact_match"].sum().item())
    stats["position_within_one_count"] += int(position["within_one"].sum().item())
    stats["position_count"] += position_count
    stats["low_cosine_count"] += int(cosine_f.lt(float(low_cosine_threshold)).sum().item())

    hist = torch.histc(cosine_f.detach().cpu(), bins=stats["hist"].numel(), min=-1.0, max=1.0)
    stats["hist"] += hist.to(dtype=torch.float64)


def _hist_percentile(hist: torch.Tensor, q: float) -> float:
    total = float(hist.sum().item())
    if total <= 0:
        return float("nan")
    target = q * total
    cumulative = torch.cumsum(hist, dim=0)
    hits = torch.nonzero(cumulative >= target, as_tuple=False)
    idx = int(hits[0].item()) if hits.numel() else int(hist.numel()) - 1
    idx = max(0, min(idx, int(hist.numel()) - 1))
    bin_width = 2.0 / float(hist.numel())
    return -1.0 + (idx + 0.5) * bin_width


def _finalize_stats(stats: Dict[str, Any], low_cosine_threshold: float) -> Dict[str, Any]:
    num_tokens = max(1, int(stats["num_tokens"]))
    num_features = max(1, int(stats["num_features"]))
    teacher_sq_sum = max(1e-12, float(stats["teacher_sq_sum"]))
    normalized_teacher_sq_sum = max(1e-12, float(stats["normalized_teacher_sq_sum"]))
    position_count = max(1, int(stats["position_count"]))
    hist = stats["hist"]
    normalized_token_l2_sq = float(stats["normalized_sq_error_sum"] / num_tokens)
    mse = float(stats["sq_error_sum"] / num_features)
    position_mse = float(stats["position_mse_sum"] / position_count)
    return {
        "num_samples": int(stats["num_samples"]),
        "num_tokens": int(stats["num_tokens"]),
        "mean_cosine": float(stats["cosine_sum"] / num_tokens),
        "cosine_p10": float(_hist_percentile(hist, 0.10)),
        "cosine_p50": float(_hist_percentile(hist, 0.50)),
        "cosine_p90": float(_hist_percentile(hist, 0.90)),
        "low_cosine_threshold": float(low_cosine_threshold),
        "low_cosine_ratio": float(stats["low_cosine_count"] / num_tokens),
        "mse": mse,
        "rmse": float(mse ** 0.5),
        "relative_mse": float(stats["sq_error_sum"] / teacher_sq_sum),
        "mae": float(stats["abs_error_sum"] / num_features),
        "mre": float(stats["token_relative_l2_sum"] / num_tokens),
        "element_mre": float(stats["element_relative_abs_sum"] / num_features),
        "normalized_mse": float(stats["normalized_sq_error_sum"] / num_features),
        "normalized_relative_mse": float(stats["normalized_sq_error_sum"] / normalized_teacher_sq_sum),
        "normalized_mae": float(stats["normalized_abs_error_sum"] / num_features),
        "normalized_token_l2_sq": normalized_token_l2_sq,
        "normalized_token_l2": float(normalized_token_l2_sq ** 0.5),
        "feature_kl_t2s": float(stats["feature_kl_t2s_sum"] / num_tokens),
        "feature_kl_s2t": float(stats["feature_kl_s2t_sum"] / num_tokens),
        "feature_kl_sym": float(stats["feature_kl_sym_sum"] / num_tokens),
        "position_mse": position_mse,
        "position_rmse": float(position_mse ** 0.5),
        "position_mse_max": float(stats["position_mse_max"]),
        "position_match_distance": float(stats["position_match_distance_sum"] / position_count),
        "position_exact_match_ratio": float(stats["position_exact_match_count"] / position_count),
        "position_within_one_ratio": float(stats["position_within_one_count"] / position_count),
        "student_norm": float(stats["student_norm_sum"] / num_tokens),
        "teacher_norm": float(stats["teacher_norm_sum"] / num_tokens),
        "norm_ratio": float(stats["norm_ratio_sum"] / num_tokens),
    }


@torch.no_grad()
def _compute_batch_similarity(
    model,
    reconstructor: SourceGuidedCompactSemanticReconstructor,
    image_processor,
    batch: Dict[str, Any],
    device: torch.device,
    model_dtype: torch.dtype,
    reconstructor_dtype: torch.dtype,
    sgcsr_output_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    teacher_visual = build_teacher_visual_embeddings(
        model=model,
        image_processor=image_processor,
        image_paths=batch["image_paths"],
        device=device,
        model_dtype=model_dtype,
    )
    teacher_compact = pool_teacher_visual_tokens(
        teacher_visual,
        output_grid=reconstructor.grid_size,
    ).to(device=device, dtype=reconstructor_dtype)

    compressed_features = batch["compressed_features"].to(device=device, dtype=model_dtype)
    with torch.no_grad():
        compressed_visual = model.multi_modal_projector(compressed_features)
    compressed_visual = compressed_visual.to(dtype=reconstructor_dtype)

    reconstructed = reconstructor(
        visual_embeddings=compressed_visual,
        token_centers=batch["token_centers"].to(device=device, dtype=reconstructor_dtype),
        token_sizes=batch["token_sizes"].to(device=device, dtype=reconstructor_dtype),
        retain_ratio=batch["retain_ratio"].to(device=device, dtype=reconstructor_dtype),
        attention_mask=batch["compressed_attention_mask"].to(device=device),
    )
    if float(sgcsr_output_scale) != 1.0:
        reconstructed = reconstructed * float(sgcsr_output_scale)

    cosine = F.cosine_similarity(reconstructed.float(), teacher_compact.float(), dim=-1)
    return reconstructed, teacher_compact, teacher_visual, cosine


def _per_sample_payload(
    *,
    global_index: int,
    image_path: str,
    retain_ratio: float,
    student: torch.Tensor,
    teacher: torch.Tensor,
    cosine: torch.Tensor,
    low_cosine_threshold: float,
    kl_temperature: float,
) -> Dict[str, Any]:
    student_f = student.float()
    teacher_f = teacher.float()
    cosine_f = cosine.float().clamp(-1.0, 1.0)
    diff = student_f - teacher_f
    mse = float(diff.pow(2).mean().item())
    teacher_energy = float(teacher_f.pow(2).mean().item())
    student_unit = F.normalize(student_f, p=2, dim=-1)
    teacher_unit = F.normalize(teacher_f, p=2, dim=-1)
    normalized_diff = student_unit - teacher_unit
    normalized_mse = float(normalized_diff.pow(2).mean().item())
    normalized_teacher_energy = float(teacher_unit.pow(2).mean().item())
    normalized_token_l2_sq = float(normalized_diff.pow(2).sum(dim=-1).mean().item())
    teacher_norm = teacher_f.norm(dim=-1).clamp_min(1e-8)
    kl_t2s, kl_s2t, kl_sym = _feature_kl(student_f, teacher_f, temperature=kl_temperature)
    position = _position_metrics(student_f, teacher_f)
    return {
        "index": int(global_index),
        "image_path": image_path,
        "retain_ratio": float(retain_ratio),
        "mean_cosine": float(cosine_f.mean().item()),
        "cosine_p10": float(torch.quantile(cosine_f, 0.10).item()),
        "cosine_p50": float(torch.quantile(cosine_f, 0.50).item()),
        "cosine_p90": float(torch.quantile(cosine_f, 0.90).item()),
        "low_cosine_ratio": float(cosine_f.lt(float(low_cosine_threshold)).float().mean().item()),
        "mse": mse,
        "rmse": float(mse ** 0.5),
        "relative_mse": float(mse / max(teacher_energy, 1e-12)),
        "mae": float(diff.abs().mean().item()),
        "mre": float((diff.norm(dim=-1) / teacher_norm).mean().item()),
        "element_mre": float((diff.abs() / teacher_f.abs().clamp_min(1e-8)).mean().item()),
        "normalized_mse": normalized_mse,
        "normalized_relative_mse": float(normalized_mse / max(normalized_teacher_energy, 1e-12)),
        "normalized_mae": float(normalized_diff.abs().mean().item()),
        "normalized_token_l2_sq": normalized_token_l2_sq,
        "normalized_token_l2": float(normalized_token_l2_sq ** 0.5),
        "feature_kl_t2s": float(kl_t2s.mean().item()),
        "feature_kl_s2t": float(kl_s2t.mean().item()),
        "feature_kl_sym": float(kl_sym.mean().item()),
        "position_mse": float(position["position_mse"].mean().item()),
        "position_rmse": float(position["position_mse"].mean().item() ** 0.5),
        "position_mse_max": float(position["position_mse"].max().item()),
        "position_match_distance": float(position["match_distance"].mean().item()),
        "position_exact_match_ratio": float(position["exact_match"].float().mean().item()),
        "position_within_one_ratio": float(position["within_one"].float().mean().item()),
        "student_norm": float(student_f.norm(dim=-1).mean().item()),
        "teacher_norm": float(teacher_f.norm(dim=-1).mean().item()),
        "norm_ratio": float((student_f.norm(dim=-1) / teacher_f.norm(dim=-1).clamp_min(1e-8)).mean().item()),
    }


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.hist_bins <= 0:
        raise ValueError("--hist_bins must be positive")
    if args.kl_temperature <= 0:
        raise ValueError("--kl_temperature must be positive")
    if args.behavior_temperature <= 0:
        raise ValueError("--behavior_temperature must be positive")
    if args.sgcsr_output_scale <= 0:
        raise ValueError("--sgcsr_output_scale must be positive")

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
    reconstructor, ckpt_args = _load_reconstructor(
        checkpoint_path=args.checkpoint_path,
        hidden_size=hidden_size,
        device=device,
        dtype=reconstructor_dtype,
    )
    print(
        f"[INFO] loaded SGCSR checkpoint: {args.checkpoint_path}; "
        f"grid={reconstructor.grid_size} K={reconstructor.num_queries}",
        flush=True,
    )

    dataset = SGCSRCompressedDataset(
        data_path=args.data_path,
        image_folder=args.image_folder,
        tokenizer=tokenizer,
        max_samples=0,
        allow_missing_source=args.allow_missing_source,
        seed=args.seed,
        conversation_mode=args.conversation_mode,
        max_text_length=args.max_text_length,
        image_token_id=int(getattr(model.config, "image_token_index", 32000)),
    )

    split_payload: Optional[Dict[str, Any]] = None
    split_summary = None
    split_mode = args.eval_mode
    if args.split_indices_path:
        split_indices, split_payload = _load_saved_split_indices(
            path=args.split_indices_path,
            split_name=args.split_name,
            dataset_len=len(dataset),
            args=args,
        )
        eval_ds = Subset(dataset, split_indices)
        split_summary = split_payload.get("split_summary")
        split_mode = f"saved_split_indices:{args.split_name}"
        print(
            f"[INFO] using saved split {args.split_name} from {args.split_indices_path}; "
            f"samples={len(eval_ds)}",
            flush=True,
        )
    elif args.eval_mode == "stratified":
        _, eval_ds, _, split_summary = build_stratified_train_val_test_split(
            dataset=dataset,
            val_ratio=float(args.test_ratio),
            final_test_ratio=0.0,
            seed=args.seed,
        )
        if eval_ds is None:
            raise ValueError("No held-out split was created; check --test_ratio.")
        print(f"[INFO] using rebuilt stratified held-out split; samples={len(eval_ds)}", flush=True)
    else:
        eval_ds = dataset
        print(f"[INFO] using all compressed samples; samples={len(eval_ds)}", flush=True)

    collator = SGCSRCollator(pad_token_id=tokenizer.pad_token_id)
    dataloader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )

    overall = _empty_stats(args.hist_bins)
    by_retain: Dict[str, Dict[str, Any]] = {}
    behavior_overall = _empty_behavior_stats() if args.compute_behavior_metrics else None
    behavior_by_retain: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}

    sample_f = None
    if args.save_per_sample:
        Path(args.save_per_sample).parent.mkdir(parents=True, exist_ok=True)
        sample_f = open(args.save_per_sample, "w", encoding="utf-8")

    seen = 0
    progress = tqdm(dataloader, desc="SGCSR feature similarity", dynamic_ncols=True)
    try:
        for batch in progress:
            if args.max_eval_samples > 0 and seen >= args.max_eval_samples:
                break

            if args.max_eval_samples > 0:
                remaining = int(args.max_eval_samples) - seen
                if remaining < batch["compressed_features"].shape[0]:
                    keep = remaining
                    batch = {
                        key: (value[:keep] if isinstance(value, torch.Tensor) else value[:keep])
                        for key, value in batch.items()
                    }

            reconstructed, teacher_compact, teacher_visual, cosine = _compute_batch_similarity(
                model=model,
                reconstructor=reconstructor,
                image_processor=image_processor,
                batch=batch,
                device=device,
                model_dtype=model_dtype,
                reconstructor_dtype=reconstructor_dtype,
                sgcsr_output_scale=args.sgcsr_output_scale,
            )
            _sync_device(device)

            behavior_payloads: Dict[int, Dict[str, Any]] = {}
            if args.compute_behavior_metrics and behavior_overall is not None:
                for i in range(int(reconstructed.shape[0])):
                    text_input_ids = batch["input_ids"][i : i + 1].to(device)
                    text_attention_mask = batch["attention_mask"][i : i + 1].to(device)
                    text_labels = batch["labels"][i : i + 1].to(device)
                    method_visuals = {
                        "teacher_full": teacher_visual[i : i + 1],
                        "teacher_compact": teacher_compact[i : i + 1],
                        "sgcsr": reconstructed[i : i + 1],
                    }
                    behavior_results: Dict[str, Dict[str, Any]] = {}
                    for behavior_name, visual_tokens in method_visuals.items():
                        scored = _score_yes_no_behavior(
                            model=model,
                            tokenizer=tokenizer,
                            input_ids=text_input_ids,
                            attention_mask=text_attention_mask,
                            labels=text_labels,
                            visual_tokens=visual_tokens,
                            device=device,
                            candidate_prefix=args.candidate_prefix,
                            question_suffix=args.question_suffix,
                            temperature=args.behavior_temperature,
                        )
                        if scored is not None:
                            behavior_results[behavior_name] = scored
                    if behavior_results:
                        retain_for_behavior = _retain_key(batch["retain_ratio"][i].detach().cpu().item())
                        _update_behavior_stats(behavior_overall, behavior_results)
                        retain_behavior_stats = behavior_by_retain.setdefault(retain_for_behavior, _empty_behavior_stats())
                        _update_behavior_stats(retain_behavior_stats, behavior_results)
                        behavior_payloads[i] = _compact_behavior_payload(behavior_results)

            reconstructed_cpu = reconstructed.detach().cpu()
            teacher_cpu = teacher_compact.detach().cpu()
            cosine_cpu = cosine.detach().cpu()
            retain_values = batch["retain_ratio"].detach().cpu().flatten().tolist()

            _update_stats(
                overall,
                student=reconstructed_cpu,
                teacher=teacher_cpu,
                cosine=cosine_cpu,
                low_cosine_threshold=args.low_cosine_threshold,
                kl_temperature=args.kl_temperature,
            )

            for i, retain_value in enumerate(retain_values):
                retain = _retain_key(retain_value)
                stats = by_retain.setdefault(retain, _empty_stats(args.hist_bins))
                _update_stats(
                    stats,
                    student=reconstructed_cpu[i : i + 1],
                    teacher=teacher_cpu[i : i + 1],
                    cosine=cosine_cpu[i : i + 1],
                    low_cosine_threshold=args.low_cosine_threshold,
                    kl_temperature=args.kl_temperature,
                )
                if sample_f is not None:
                    sample_payload = _per_sample_payload(
                        global_index=seen + i,
                        image_path=batch["image_paths"][i],
                        retain_ratio=float(retain_value),
                        student=reconstructed_cpu[i],
                        teacher=teacher_cpu[i],
                        cosine=cosine_cpu[i],
                        low_cosine_threshold=args.low_cosine_threshold,
                        kl_temperature=args.kl_temperature,
                    )
                    if i in behavior_payloads:
                        sample_payload["behavior"] = behavior_payloads[i]
                    sample_f.write(
                        json.dumps(
                            sample_payload,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            seen += int(reconstructed_cpu.shape[0])
            progress.set_postfix(sample=seen, retain=_retain_key(retain_values[-1]))
    finally:
        if sample_f is not None:
            sample_f.close()

    finalized = {
        "overall": _finalize_stats(overall, args.low_cosine_threshold),
        "by_retain_ratio": {
            retain: _finalize_stats(stats, args.low_cosine_threshold)
            for retain, stats in sorted(by_retain.items(), key=lambda kv: float(kv[0]))
        },
    }
    finalized_behavior = None
    if behavior_overall is not None:
        finalized_behavior = {
            "overall": _finalize_behavior_stats(behavior_overall),
            "by_retain_ratio": {
                retain: _finalize_behavior_stats(stats)
                for retain, stats in sorted(behavior_by_retain.items(), key=lambda kv: float(kv[0]))
            },
        }

    result = {
        "model_name_or_path": args.model_name_or_path,
        "checkpoint_path": args.checkpoint_path,
        "data_path": args.data_path,
        "image_folder": args.image_folder,
        "split_mode": split_mode,
        "split_indices_path": args.split_indices_path,
        "split_name": args.split_name if args.split_indices_path else None,
        "conversation_mode": args.conversation_mode,
        "max_text_length": args.max_text_length,
        "test_ratio": args.test_ratio if args.eval_mode == "stratified" else None,
        "seed": args.seed,
        "split_summary": split_summary,
        "num_eval_samples": seen,
        "sgcsr_grid_size": list(reconstructor.grid_size),
        "sgcsr_num_queries": int(reconstructor.num_queries),
        "kl_temperature": float(args.kl_temperature),
        "compute_behavior_metrics": bool(args.compute_behavior_metrics),
        "behavior_temperature": float(args.behavior_temperature),
        "sgcsr_output_scale": float(args.sgcsr_output_scale),
        "checkpoint_args": ckpt_args,
        "metrics_note": {
            "teacher": "no-ToMe image -> CLIP vision tower -> frozen LLaVA projector -> 576 visual tokens -> pooled to SGCSR grid",
            "student": "ToMe compressed CLIP features -> frozen LLaVA projector -> SGCSR compact tokens",
            "sgcsr_output_scale": "inference-only multiplier applied to SGCSR output tokens before all student-vs-teacher feature and behavior metrics",
            "mean_cosine": "mean token-wise cosine similarity between student and pooled teacher compact tokens",
            "mse": "mean((student - teacher)^2) over all compact-token hidden dimensions",
            "rmse": "sqrt(mse)",
            "mae": "mean(abs(student - teacher)) over all compact-token hidden dimensions",
            "mre": "mean token-level relative L2 error: ||student_i - teacher_i||_2 / (||teacher_i||_2 + eps)",
            "element_mre": "mean element-wise relative absolute error: abs(student - teacher) / (abs(teacher) + eps); can be large when teacher elements are near zero",
            "relative_mse": "sum((student-teacher)^2) / sum(teacher^2)",
            "feature_kl_t2s": "KL(teacher || student) after per-token feature standardization and softmax over hidden dimensions",
            "feature_kl_s2t": "KL(student || teacher) after per-token feature standardization and softmax over hidden dimensions",
            "feature_kl_sym": "0.5 * (feature_kl_t2s + feature_kl_s2t)",
            "position_mse": "mean per-token MSE at the same compact-grid position",
            "position_rmse": "sqrt(position_mse)",
            "position_match_distance": "mean grid-cell distance from each student token position to its most cosine-similar teacher token position",
            "position_exact_match_ratio": "fraction of student tokens whose most similar teacher token is at the same compact-grid position",
            "position_within_one_ratio": "fraction of student tokens whose most similar teacher token is within one grid-cell distance",
            "normalized_mse": "MSE after L2-normalizing each student/teacher token along the hidden dimension",
            "normalized_relative_mse": "sum((normalize(student)-normalize(teacher))^2) / sum(normalize(teacher)^2)",
            "normalized_token_l2_sq": "mean squared L2 distance between L2-normalized student/teacher tokens; equals 2-2*cosine up to aggregation",
            "low_cosine_ratio": "fraction of compact tokens whose cosine is below --low_cosine_threshold",
        },
        "metrics": finalized,
    }
    if finalized_behavior is not None:
        result["behavior_metrics_note"] = {
            "teacher_full": "no-ToMe image -> CLIP vision tower -> frozen LLaVA projector -> 576 visual tokens; closest to normal LLaVA visual input",
            "teacher_compact": "teacher_full pooled from 576 tokens to the SGCSR compact grid; isolates compact-token-length effects",
            "sgcsr": "ToMe compressed CLIP features -> frozen LLaVA projector -> SGCSR compact tokens",
            "answer_agreement": "fraction of samples where two visual-token sources choose the same yes/no likelihood prediction",
            "yes_prob_mae": "mean absolute difference in two-class yes probability",
            "target_logprob_signed_gap": "mean log P_cmp(label) - log P_ref(label); negative means the compared method assigns lower probability to the ground-truth answer than the reference",
            "kl_ref_to_cmp": "two-class KL(ref yes/no distribution || compared yes/no distribution)",
            "kl_sym": "0.5 * (KL(ref||compared) + KL(compared||ref)) over yes/no distributions",
        }
        result["behavior_metrics"] = finalized_behavior

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(finalized["overall"], ensure_ascii=False, indent=2), flush=True)
    print(f"[DONE] saved to {args.output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())