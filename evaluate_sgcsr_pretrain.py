from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor, LlavaForConditionalGeneration

from mm.semantic_reconstructor import SourceGuidedCompactSemanticReconstructor
from train_sgcsr import (
    SGCSRCompressedDataset,
    SGCSRCollator,
    build_stratified_train_val_test_split,
    dtype_from_name,
    get_language_model,
    get_vision_tower,
    merge_text_and_visual_tokens,
    select_vision_features,
)


METHODS = ("no_tome_no_sgcsr", "tome_no_sgcsr", "tome_sgcsr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate first-level SGCSR pretrain metrics on the held-out split. "
            "The three groups use the same held-out manifest samples: no-ToMe/no-SGCSR, "
            "ToMe/no-SGCSR, and ToMe/SGCSR."
        )
    )
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image_folder", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="outputs/sgcsr_pretrain_eval.json")
    parser.add_argument("--save_predictions", type=str, default=None)
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
    parser.add_argument("--batch_size", type=int, default=1, help="Keep at 1; groups have different visual lengths.")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument(
        "--split_indices_path",
        type=str,
        default=None,
        help=(
            "Optional split_indices.json saved by train_sgcsr.py. "
            "When set, evaluation uses the requested split exactly instead of rebuilding a random split."
        ),
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split from --split_indices_path to evaluate.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--conversation_mode",
        type=str,
        default="first",
        choices=["first", "all", "full"],
        help=(
            "Must match the dataset expansion used by split_indices.json for exact index reuse. "
            "Use 'full' for the stage-2 mix665k split generated during SGCSR training."
        ),
    )
    parser.add_argument(
        "--max_text_length",
        type=int,
        default=0,
        help="Optional text token cap. Use the same value as training if split_indices.json was produced with truncation.",
    )
    parser.add_argument("--max_eval_samples", type=int, default=0, help="Debug only; 0 evaluates the full held-out split.")
    parser.add_argument(
        "--gen_max_samples_per_retain",
        type=int,
        default=250,
        help=(
            "Generation metrics are expensive. This is the max generated samples per retain bucket; "
            "0 means generate for every evaluated sample."
        ),
    )
    parser.add_argument(
        "--skip_generation_metrics",
        action="store_true",
        help="Skip generation-based ROUGE-L/CIDEr and only evaluate loss, perplexity, token accuracy, and latency.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=64)
    return parser.parse_args()


def _sync_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _retain_key(value: Any) -> str:
    return f"{float(value):.2f}"


def _empty_stats() -> Dict[str, float]:
    return {
        "loss_sum": 0.0,
        "correct_tokens": 0.0,
        "total_tokens": 0.0,
        "num_samples": 0.0,
        "cloud_latency_ms_sum": 0.0,
        "llm_latency_ms_sum": 0.0,
        "projector_latency_ms_sum": 0.0,
        "sgcsr_latency_ms_sum": 0.0,
        "latency_count": 0.0,
        "rouge_l_sum": 0.0,
        "cider_sum": 0.0,
        "generation_count": 0.0,
    }


def _update_stats(
    stats: Dict[str, float],
    *,
    loss_sum: float,
    correct_tokens: int,
    total_tokens: int,
    cloud_latency_ms: float,
    projector_latency_ms: float,
    llm_latency_ms: float,
    sgcsr_latency_ms: float,
):
    stats["loss_sum"] += float(loss_sum)
    stats["correct_tokens"] += int(correct_tokens)
    stats["total_tokens"] += int(total_tokens)
    stats["num_samples"] += 1
    stats["cloud_latency_ms_sum"] += float(cloud_latency_ms)
    stats["projector_latency_ms_sum"] += float(projector_latency_ms)
    stats["llm_latency_ms_sum"] += float(llm_latency_ms)
    stats["sgcsr_latency_ms_sum"] += float(sgcsr_latency_ms)
    stats["latency_count"] += 1


def _finalize_stats(stats: Dict[str, float]) -> Dict[str, Any]:
    total_tokens = int(stats["total_tokens"])
    correct_tokens = int(stats["correct_tokens"])
    eval_loss = stats["loss_sum"] / max(1, total_tokens)
    ppl = math.exp(eval_loss) if eval_loss < 50 else float("inf")
    latency_count = max(1.0, stats["latency_count"])
    generation_count = max(1.0, stats["generation_count"])
    return {
        "eval_loss": float(eval_loss),
        "ppl": float(ppl),
        "token_accuracy": float(correct_tokens / max(1, total_tokens)),
        "rouge_l": float(stats["rouge_l_sum"] / generation_count) if stats["generation_count"] > 0 else None,
        "cider": float(stats["cider_sum"] / generation_count) if stats["generation_count"] > 0 else None,
        "generation_count": int(stats["generation_count"]),
        "cloud_latency_ms": float(stats["cloud_latency_ms_sum"] / latency_count),
        "projector_latency_ms": float(stats["projector_latency_ms_sum"] / latency_count),
        "llm_latency_ms": float(stats["llm_latency_ms_sum"] / latency_count),
        "sgcsr_latency_ms": float(stats["sgcsr_latency_ms_sum"] / latency_count),
        "samples_per_second": float(1000.0 * latency_count / max(1e-9, stats["cloud_latency_ms_sum"])),
        "correct_tokens": correct_tokens,
        "total_tokens": total_tokens,
        "num_samples": int(stats["num_samples"]),
    }


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
            f"Split conversation_mode mismatch: split={payload['conversation_mode']} current={args.conversation_mode}. "
            "Pass the same --conversation_mode used during training."
        )
    if int(payload.get("max_samples", 0)) != 0:
        raise ValueError(
            "This split_indices.json was generated from a debug subset with max_samples="
            f"{payload.get('max_samples')}. Rebuild the same subset-aware evaluator before using it."
        )
    expected_max_text_length = int(payload.get("max_text_length", 0))
    if expected_max_text_length != int(args.max_text_length):
        raise ValueError(
            f"Split max_text_length mismatch: split={expected_max_text_length} current={args.max_text_length}. "
            "Pass the same --max_text_length used during training."
        )
    if split_name not in payload:
        raise ValueError(f"Split file {path} does not contain split '{split_name}'.")
    indices = [int(idx) for idx in payload[split_name]]
    if not indices:
        raise ValueError(f"Split '{split_name}' in {path} is empty.")
    max_index = max(indices)
    if max_index >= dataset_len:
        raise ValueError(
            f"Split index {max_index} is out of range for dataset length {dataset_len}. "
            "Check that --conversation_mode, --max_samples, and --data_path match training."
        )
    return indices, payload


def _rouge_tokens(text: str) -> List[str]:
    text = " ".join(text.lower().strip().split())
    if not text:
        return []
    tokens = text.split()
    if len(tokens) == 1:
        return list(text)
    return tokens


def _lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
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


def rouge_l_f1(prediction: str, reference: str) -> float:
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


def _ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _build_cider_df(references: Sequence[str]) -> Tuple[Dict[int, Counter], int]:
    dfs = {n: Counter() for n in range(1, 5)}
    for reference in references:
        tokens = _rouge_tokens(reference)
        for n in range(1, 5):
            dfs[n].update(set(_ngram_counts(tokens, n).keys()))
    return dfs, max(1, len(references))


def _tfidf_vector(counts: Counter, df: Counter, num_docs: int) -> Dict[Any, float]:
    total = sum(counts.values())
    if total <= 0:
        return {}
    vec = {}
    for gram, count in counts.items():
        tf = float(count) / float(total)
        idf = math.log((float(num_docs) + 1.0) / (float(df.get(gram, 0)) + 1.0))
        vec[gram] = tf * idf
    return vec


def _cosine_dict(a: Dict[Any, float], b: Dict[Any, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(value * b.get(key, 0.0) for key, value in a.items())
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def cider_single_ref(prediction: str, reference: str, dfs: Dict[int, Counter], num_docs: int) -> float:
    """Single-reference CIDEr-style score.

    LLaVA pretrain samples usually have one target answer/caption, so this uses
    the held-out references as the IDF corpus and computes the standard 1-4 gram
    TF-IDF cosine average, scaled by 10 like CIDEr.
    """
    pred_tokens = _rouge_tokens(prediction)
    ref_tokens = _rouge_tokens(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    scores = []
    for n in range(1, 5):
        pred_vec = _tfidf_vector(_ngram_counts(pred_tokens, n), dfs[n], num_docs)
        ref_vec = _tfidf_vector(_ngram_counts(ref_tokens, n), dfs[n], num_docs)
        scores.append(_cosine_dict(pred_vec, ref_vec))
    return 10.0 * sum(scores) / len(scores)


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
    vision_tower = get_vision_tower(model)
    with torch.no_grad():
        vision_outputs = vision_tower(pixel_values, output_hidden_states=True)
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


def _teacher_forced_metrics(
    model,
    sample: Dict[str, Any],
    visual_tokens: torch.Tensor,
    device: torch.device,
) -> Tuple[float, int, int, float]:
    input_ids = sample["input_ids"].to(device)
    attention_mask = sample["attention_mask"].to(device)
    labels = sample["labels"].to(device)
    inputs_embeds, merged_mask, merged_labels = merge_text_and_visual_tokens(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        visual_tokens=visual_tokens,
    )
    _sync_device(device)
    start = time.perf_counter()
    with torch.no_grad():
        out = get_language_model(model)(
            inputs_embeds=inputs_embeds,
            attention_mask=merged_mask,
            labels=merged_labels,
        )
    _sync_device(device)
    llm_latency_ms = (time.perf_counter() - start) * 1000.0

    shift_logits = out.logits[:, :-1, :].float()
    shift_labels = merged_labels[:, 1:]
    valid_mask = shift_labels.ne(-100)
    token_losses = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(shift_labels.shape)
    predictions = shift_logits.argmax(dim=-1)
    correct = int((predictions.eq(shift_labels) & valid_mask).sum().item())
    total = int(valid_mask.sum().item())
    loss_sum = float(token_losses[valid_mask].sum().item())
    return loss_sum, correct, total, llm_latency_ms


@torch.no_grad()
def _generate_one(
    model,
    tokenizer,
    sample: Dict[str, Any],
    visual_tokens: torch.Tensor,
    device: torch.device,
    max_new_tokens: int,
) -> Optional[Tuple[str, str]]:
    extracted = _extract_prompt_and_reference(tokenizer, sample, device)
    if extracted is None:
        return None
    prompt_ids, prompt_attention_mask, reference_text = extracted
    prompt_labels = torch.full_like(prompt_ids, -100)
    inputs_embeds, merged_mask, _ = merge_text_and_visual_tokens(
        model=model,
        input_ids=prompt_ids,
        attention_mask=prompt_attention_mask,
        labels=prompt_labels,
        visual_tokens=visual_tokens,
    )
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
        generated = get_language_model(model).generate(
            input_ids=dummy_input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=merged_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
    except (TypeError, ValueError):
        generated = get_language_model(model).generate(
            inputs_embeds=inputs_embeds,
            attention_mask=merged_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
    generated_ids = generated[0, input_len:] if generated.shape[1] > input_len else generated[0]
    prediction_text = tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True).strip()
    return prediction_text, reference_text


def _prepare_generation_references(
    dataloader: DataLoader,
    tokenizer,
    max_per_retain: int,
    max_eval_samples: int,
    target_retain_keys: Optional[Sequence[str]] = None,
) -> List[str]:
    refs: List[str] = []
    seen_by_retain: Dict[str, int] = {}
    target_keys = set(target_retain_keys or [])
    scanned = 0
    for sample in dataloader:
        if max_eval_samples > 0 and scanned >= max_eval_samples:
            break
        scanned += 1
        retain = _retain_key(sample["retain_ratio"][0].item())
        if max_per_retain > 0 and seen_by_retain.get(retain, 0) >= max_per_retain:
            continue
        extracted = _extract_prompt_and_reference(tokenizer, sample, torch.device("cpu"))
        if extracted is None:
            continue
        _, _, reference_text = extracted
        refs.append(reference_text)
        seen_by_retain[retain] = seen_by_retain.get(retain, 0) + 1
        if (
            max_per_retain > 0
            and target_keys
            and all(seen_by_retain.get(key, 0) >= max_per_retain for key in target_keys)
        ):
            break
    return refs


def _add_generation_metric(
    stats: Dict[str, float],
    prediction: str,
    reference: str,
    cider_df: Dict[int, Counter],
    cider_num_docs: int,
):
    stats["rouge_l_sum"] += rouge_l_f1(prediction, reference)
    stats["cider_sum"] += cider_single_ref(prediction, reference, cider_df, cider_num_docs)
    stats["generation_count"] += 1


def main() -> int:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("evaluate_sgcsr_pretrain.py requires --batch_size 1 to keep variable visual lengths exact.")

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
        max_samples=0,
        allow_missing_source=False,
        seed=args.seed,
        conversation_mode=args.conversation_mode,
        max_text_length=args.max_text_length,
        image_token_id=int(getattr(model.config, "image_token_index", 32000)),
    )
    split_payload: Optional[Dict[str, Any]] = None
    if args.split_indices_path:
        eval_indices, split_payload = _load_saved_split_indices(
            path=args.split_indices_path,
            split_name=args.split_name,
            dataset_len=len(dataset),
            args=args,
        )
        eval_ds = Subset(dataset, eval_indices)
        split_summary = split_payload.get("split_summary", {})
        split_mode = f"saved_split_indices:{args.split_name}"
        print(
            f"[INFO] using saved split {args.split_name} from {args.split_indices_path}; "
            f"samples={len(eval_ds)}",
            flush=True,
        )
    else:
        _, eval_ds, _, split_summary = build_stratified_train_val_test_split(
            dataset=dataset,
            val_ratio=float(args.test_ratio),
            final_test_ratio=0.0,
            seed=args.seed,
        )
        split_mode = "stratified_by_retain_ratio_val_fallback"
        if eval_ds is None:
            raise ValueError("No held-out split was created; check --test_ratio.")

    effective_eval_samples = len(eval_ds) if args.max_eval_samples <= 0 else min(len(eval_ds), args.max_eval_samples)
    retain_keys = sorted(split_summary.keys(), key=float) if split_summary else []
    print(
        f"[INFO] held-out samples={len(eval_ds)} effective_eval_samples={effective_eval_samples} "
        f"retain_buckets={retain_keys}",
        flush=True,
    )
    collator = SGCSRCollator(pad_token_id=tokenizer.pad_token_id)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )

    # Build the IDF corpus for single-reference CIDEr using the exact generation subset.
    # This is skipped for fast full-set loss/latency evaluation because generation
    # metrics require extra autoregressive decoding for all three comparison groups.
    if args.skip_generation_metrics:
        generation_refs = []
        cider_df, cider_num_docs = _build_cider_df(generation_refs)
        print("[INFO] skipped generation references and ROUGE-L/CIDEr.", flush=True)
    else:
        ref_loader = DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collator)
        print("[INFO] preparing generation references for ROUGE-L/CIDEr...", flush=True)
        generation_refs = _prepare_generation_references(
            ref_loader,
            tokenizer,
            args.gen_max_samples_per_retain,
            args.max_eval_samples,
            target_retain_keys=retain_keys,
        )
        cider_df, cider_num_docs = _build_cider_df(generation_refs)
        print(f"[INFO] generation reference count={len(generation_refs)}", flush=True)

    metrics = {method: {"overall": _empty_stats(), "by_retain_ratio": {}} for method in METHODS}
    predictions_f = None
    if args.save_predictions and not args.skip_generation_metrics:
        Path(args.save_predictions).parent.mkdir(parents=True, exist_ok=True)
        predictions_f = open(args.save_predictions, "w", encoding="utf-8")

    generation_seen_by_retain: Dict[str, int] = {}
    seen = 0
    progress = tqdm(eval_loader, desc="SGCSR pretrain eval", dynamic_ncols=True)
    try:
        for sample in progress:
            if args.max_eval_samples > 0 and seen >= args.max_eval_samples:
                break
            seen += 1
            retain = _retain_key(sample["retain_ratio"][0].item())
            do_generation = False
            if not args.skip_generation_metrics:
                do_generation = (
                    args.gen_max_samples_per_retain <= 0
                    or generation_seen_by_retain.get(retain, 0) < args.gen_max_samples_per_retain
                )

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

            method_payloads = {
                "no_tome_no_sgcsr": (no_tome_visual, no_tome_projector_ms, 0.0),
                "tome_no_sgcsr": (tome_visual, tome_projector_ms, 0.0),
                "tome_sgcsr": (sgcsr_visual, tome_projector_ms, sgcsr_ms),
            }

            for method, (visual_tokens, projector_ms, sgcsr_latency_ms) in method_payloads.items():
                loss_sum, correct, total, llm_ms = _teacher_forced_metrics(model, sample, visual_tokens, device)
                cloud_ms = projector_ms + sgcsr_latency_ms + llm_ms

                method_stats = metrics[method]
                by_retain = method_stats["by_retain_ratio"].setdefault(retain, _empty_stats())
                for stats in [method_stats["overall"], by_retain]:
                    _update_stats(
                        stats,
                        loss_sum=loss_sum,
                        correct_tokens=correct,
                        total_tokens=total,
                        cloud_latency_ms=cloud_ms,
                        projector_latency_ms=projector_ms,
                        llm_latency_ms=llm_ms,
                        sgcsr_latency_ms=sgcsr_latency_ms,
                    )

                if do_generation:
                    generated = _generate_one(
                        model=model,
                        tokenizer=tokenizer,
                        sample=sample,
                        visual_tokens=visual_tokens,
                        device=device,
                        max_new_tokens=args.max_new_tokens,
                    )
                    if generated is not None:
                        prediction, reference = generated
                        _add_generation_metric(method_stats["overall"], prediction, reference, cider_df, cider_num_docs)
                        _add_generation_metric(by_retain, prediction, reference, cider_df, cider_num_docs)
                        if predictions_f is not None:
                            predictions_f.write(
                                json.dumps(
                                    {
                                        "index": seen - 1,
                                        "method": method,
                                        "retain_ratio": retain,
                                        "prediction": prediction,
                                        "reference": reference,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )

            if do_generation:
                generation_seen_by_retain[retain] = generation_seen_by_retain.get(retain, 0) + 1

            progress.set_postfix(sample=seen, retain=retain)
    finally:
        if predictions_f is not None:
            predictions_f.close()

    finalized = {}
    for method, method_stats in metrics.items():
        finalized[method] = {
            "overall": _finalize_stats(method_stats["overall"]),
            "by_retain_ratio": {
                retain: _finalize_stats(stats)
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
        "conversation_mode": args.conversation_mode,
        "max_text_length": args.max_text_length,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "split_summary": split_summary,
        "local_attention": {
            "local_topk": effective_local_topk,
            "local_radius": effective_local_radius,
            "local_topk_override": args.local_topk,
        },
        "metrics_note": {
            "cider": "single-reference CIDEr-style TF-IDF 1-4 gram score over the generated subset",
            "cloud_latency_ms": "projector + optional SGCSR + LLM teacher-forced forward; edge vision time is excluded",
        },
        "generation_max_samples_per_retain": args.gen_max_samples_per_retain,
        "skip_generation_metrics": args.skip_generation_metrics,
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
