from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor, LlavaForConditionalGeneration, get_cosine_schedule_with_warmup

from mm.semantic_reconstructor import SourceGuidedCompactSemanticReconstructor
from train_sgcsr import (
    SGCSRCompressedDataset,
    SGCSRCollator,
    dtype_from_name,
    evaluate,
    forward_losses,
    get_language_model,
    load_reconstructor_checkpoint,
    log_jsonl,
    save_checkpoint,
    subset_indices,
    user_assistant_pairs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "POPE-domain adaptation for SGCSR. This script initializes from a "
            "Stage-2 SGCSR checkpoint, freezes the LLaVA backbone, and trains "
            "only the cloud-side semantic reconstructor on a POPE train split."
        )
    )
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Local LLaVA-1.5 HF model path.")
    parser.add_argument("--data_path", type=str, required=True, help="Compressed POPE feature directory or manifest.")
    parser.add_argument("--image_folder", type=str, required=True, help="POPE image folder for no-ToMe teacher features.")
    parser.add_argument("--output_dir", type=str, default="outputs/sgcsr_k144_pope_adapt_90_5_5")
    parser.add_argument(
        "--init_checkpoint_path",
        type=str,
        required=True,
        help="Stage-2 SGCSR checkpoint to continue from, e.g. outputs/.../best.pt.",
    )
    parser.add_argument(
        "--allow_checkpoint_config_mismatch",
        action="store_true",
        help="Allow intentionally loading a checkpoint with different SGCSR architecture/locality args.",
    )
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--num_queries", type=int, default=144)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim_head", type=int, default=128)
    parser.add_argument("--ff_mult", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--local_topk", type=int, default=16)
    parser.add_argument("--local_radius", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--final_test_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument(
        "--reconstructor_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--task_weight", type=float, default=1.0)
    parser.add_argument("--rec_weight", type=float, default=1.0)
    parser.add_argument("--rec_mse_weight", type=float, default=1.0)
    parser.add_argument("--rec_cosine_weight", type=float, default=0.1)
    parser.add_argument("--logit_weight", type=float, default=0.1)
    parser.add_argument("--logit_teacher_mode", type=str, default="compact", choices=["compact", "full"])
    parser.add_argument("--logit_temperature", type=float, default=2.0)
    parser.add_argument("--max_samples", type=int, default=0, help="Debug only; 0 means all samples.")
    parser.add_argument(
        "--question_suffix",
        type=str,
        default="Please answer yes or no.",
        help="Appended to POPE training questions, matching evaluate_sgcsr_pope.py.",
    )
    parser.add_argument(
        "--conversation_mode",
        type=str,
        default="first",
        choices=["first", "all", "full"],
        help="POPE is normally single-turn; keep 'first' for formal runs.",
    )
    parser.add_argument("--max_text_length", type=int, default=0)
    parser.add_argument("--save_every_epoch", action="store_true")
    parser.add_argument("--allow_missing_source", action="store_true", help="Debug only; formal SGCSR needs source maps.")
    return parser.parse_args()


def _payload_for_index(dataset: SGCSRCompressedDataset, idx: int) -> Dict[str, Any]:
    return dataset._item_payload(dataset.items[idx])


def _retain_key(dataset: SGCSRCompressedDataset, idx: int) -> str:
    return f"{dataset.get_retain_ratio(idx):.2f}"


def _label_key(payload: Dict[str, Any]) -> str:
    pairs = user_assistant_pairs(payload.get("conversations", []))
    if not pairs:
        return "unknown"
    answer = pairs[0][1].strip().lower()
    if answer.startswith("yes"):
        return "yes"
    if answer.startswith("no"):
        return "no"
    return "unknown"


def _image_group_key(payload: Dict[str, Any]) -> str:
    """Group by image identity so one image cannot leak across splits."""
    for key in ("image_id", "image", "image_path"):
        value = payload.get(key)
        if value is not None:
            return str(value)
    # Last-resort fallback keeps the script usable for malformed debug shards,
    # but formal POPE manifests should always contain an image identifier.
    return f"sample:{payload.get('sample_id', payload.get('index', 'unknown'))}"


def _split_group_lists(
    grouped_indices: Sequence[List[int]],
    val_ratio: float,
    test_ratio: float,
    generator: torch.Generator,
) -> Tuple[List[int], List[int], List[int], Dict[str, int]]:
    order = torch.randperm(len(grouped_indices), generator=generator).tolist()
    groups = [grouped_indices[i] for i in order]
    num_groups = len(groups)

    val_group_len = max(1, int(num_groups * val_ratio)) if val_ratio > 0 and num_groups > 1 else 0
    test_group_len = max(1, int(num_groups * test_ratio)) if test_ratio > 0 and num_groups > 1 else 0
    max_holdout_groups = max(0, num_groups - 1)
    while val_group_len + test_group_len > max_holdout_groups:
        if test_group_len >= val_group_len and test_group_len > 0:
            test_group_len -= 1
        elif val_group_len > 0:
            val_group_len -= 1
        else:
            break

    test_groups = groups[:test_group_len]
    val_groups = groups[test_group_len : test_group_len + val_group_len]
    train_groups = groups[test_group_len + val_group_len :]
    train = [idx for group in train_groups for idx in group]
    val = [idx for group in val_groups for idx in group]
    test = [idx for group in test_groups for idx in group]
    return train, val, test, {
        "base_groups": num_groups,
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "test_groups": len(test_groups),
    }


def build_pope_image_group_split(
    dataset: SGCSRCompressedDataset,
    val_ratio: float,
    final_test_ratio: float,
    seed: int,
) -> Tuple[Subset, Optional[Subset], Optional[Subset], Dict[str, Any]]:
    """Build a POPE split with image-level isolation and rough stratification.

    Groups are keyed by image id/path to avoid training and testing on different
    questions from the same image.  Groups are then split inside coarse
    retain-ratio/answer-label strata, preserving the POPE yes/no and compression
    distributions as much as the available image groups allow.
    """
    if not 0 <= val_ratio < 1:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    if not 0 <= final_test_ratio < 1:
        raise ValueError(f"final_test_ratio must be in [0, 1), got {final_test_ratio}")
    if val_ratio + final_test_ratio >= 1:
        raise ValueError("val_ratio + final_test_ratio must be < 1.")

    image_groups: Dict[str, List[int]] = {}
    for idx in range(len(dataset)):
        payload = _payload_for_index(dataset, idx)
        image_groups.setdefault(_image_group_key(payload), []).append(idx)

    strata: Dict[str, List[List[int]]] = {}
    for indices in image_groups.values():
        retain_votes = Counter(_retain_key(dataset, idx) for idx in indices)
        label_votes = Counter(_label_key(_payload_for_index(dataset, idx)) for idx in indices)
        retain = retain_votes.most_common(1)[0][0]
        label = label_votes.most_common(1)[0][0]
        strata.setdefault(f"{retain}|{label}", []).append(indices)

    generator = torch.Generator().manual_seed(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []
    stratum_summary: Dict[str, Dict[str, int]] = {}

    for stratum_key in sorted(strata.keys()):
        train, val, test, group_summary = _split_group_lists(
            grouped_indices=strata[stratum_key],
            val_ratio=val_ratio,
            test_ratio=final_test_ratio,
            generator=generator,
        )
        train_indices.extend(train)
        val_indices.extend(val)
        test_indices.extend(test)
        stratum_summary[stratum_key] = {
            "total": len(train) + len(val) + len(test),
            "train": len(train),
            "val": len(val),
            "test": len(test),
            **group_summary,
        }

    split_summary = _summarize_split(dataset, train_indices, val_indices, test_indices)
    split_summary["strata"] = stratum_summary
    split_summary["image_groups"] = {
        "total": len(image_groups),
        "train": len({_image_group_key(_payload_for_index(dataset, idx)) for idx in train_indices}),
        "val": len({_image_group_key(_payload_for_index(dataset, idx)) for idx in val_indices}),
        "test": len({_image_group_key(_payload_for_index(dataset, idx)) for idx in test_indices}),
    }
    _assert_disjoint_image_splits(dataset, train_indices, val_indices, test_indices)

    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices) if val_indices else None
    test_ds = Subset(dataset, test_indices) if test_indices else None
    return train_ds, val_ds, test_ds, split_summary


def _summarize_split(
    dataset: SGCSRCompressedDataset,
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    test_indices: Sequence[int],
) -> Dict[str, Any]:
    splits = {"train": list(train_indices), "val": list(val_indices), "test": list(test_indices)}
    summary: Dict[str, Any] = {"by_retain_ratio": {}, "by_label": {}}

    for split_name, indices in splits.items():
        for idx in indices:
            retain = _retain_key(dataset, idx)
            label = _label_key(_payload_for_index(dataset, idx))
            retain_bucket = summary["by_retain_ratio"].setdefault(retain, {"train": 0, "val": 0, "test": 0})
            retain_bucket[split_name] += 1
            label_bucket = summary["by_label"].setdefault(label, {"train": 0, "val": 0, "test": 0})
            label_bucket[split_name] += 1

    summary["total"] = {
        "train": len(train_indices),
        "val": len(val_indices),
        "test": len(test_indices),
        "all": len(train_indices) + len(val_indices) + len(test_indices),
    }
    return summary


def _image_group_keys_for_indices(dataset: SGCSRCompressedDataset, indices: Sequence[int]) -> set[str]:
    return {_image_group_key(_payload_for_index(dataset, idx)) for idx in indices}


def _assert_disjoint_image_splits(
    dataset: SGCSRCompressedDataset,
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    test_indices: Sequence[int],
) -> None:
    """Fail fast if one source image appears in more than one split."""
    train_groups = _image_group_keys_for_indices(dataset, train_indices)
    val_groups = _image_group_keys_for_indices(dataset, val_indices)
    test_groups = _image_group_keys_for_indices(dataset, test_indices)
    overlaps = {
        "train_val": sorted(train_groups & val_groups)[:5],
        "train_test": sorted(train_groups & test_groups)[:5],
        "val_test": sorted(val_groups & test_groups)[:5],
    }
    if any(overlaps.values()):
        raise ValueError(
            "POPE image-group split leakage detected. "
            f"Example overlapping image ids: {json.dumps(overlaps, ensure_ascii=False)}"
        )


def save_pope_split_indices(
    output_dir: str,
    args: argparse.Namespace,
    train_ds: Subset,
    val_ds: Optional[Subset],
    test_ds: Optional[Subset],
    split_summary: Dict[str, Any],
) -> None:
    payload = {
        "format": "sgcsr_pope_image_split_v1",
        "indices_are": "indices into SGCSRCompressedDataset after max_samples filtering and conversation expansion",
        "data_path": args.data_path,
        "image_folder": args.image_folder,
        "seed": int(args.seed),
        "conversation_mode": args.conversation_mode,
        "question_suffix": args.question_suffix,
        "max_samples": int(args.max_samples),
        "max_text_length": int(args.max_text_length),
        "val_ratio": float(args.val_ratio),
        "final_test_ratio": float(args.final_test_ratio),
        "split_mode": "pope_image_group_stratified_by_retain_and_label",
        "split_summary": split_summary,
        "train": subset_indices(train_ds),
        "val": subset_indices(val_ds),
        "test": subset_indices(test_ds),
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "split_indices.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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
    )
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    hidden_size = int(get_language_model(model).config.hidden_size)
    reconstructor = SourceGuidedCompactSemanticReconstructor(
        dim=hidden_size,
        num_queries=args.num_queries,
        depth=args.depth,
        heads=args.heads,
        dim_head=args.dim_head,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
        local_topk=args.local_topk,
        local_radius=args.local_radius,
    ).to(device=device, dtype=reconstructor_dtype)
    load_reconstructor_checkpoint(reconstructor, args.init_checkpoint_path, device, args)

    dataset = SGCSRCompressedDataset(
        data_path=args.data_path,
        image_folder=args.image_folder,
        tokenizer=tokenizer,
        max_samples=args.max_samples,
        allow_missing_source=args.allow_missing_source,
        seed=args.seed,
        conversation_mode=args.conversation_mode,
        max_text_length=args.max_text_length,
        image_token_id=int(getattr(model.config, "image_token_index", 32000)),
        question_suffix=args.question_suffix,
    )
    train_ds, val_ds, test_ds, split_summary = build_pope_image_group_split(
        dataset=dataset,
        val_ratio=args.val_ratio,
        final_test_ratio=args.final_test_ratio,
        seed=args.seed,
    )
    train_len = len(train_ds)
    val_len = len(val_ds) if val_ds is not None else 0
    test_len = len(test_ds) if test_ds is not None else 0

    collator = SGCSRCollator(pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collator)
        if val_ds is not None
        else None
    )
    test_loader = (
        DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collator)
        if test_ds is not None
        else None
    )

    os.makedirs(args.output_dir, exist_ok=True)
    save_pope_split_indices(args.output_dir, args, train_ds, val_ds, test_ds, split_summary)
    with open(os.path.join(args.output_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                **vars(args),
                "train_samples": train_len,
                "val_samples": val_len,
                "test_samples": test_len,
                "split_mode": "pope_image_group_stratified_by_retain_and_label",
                "split_summary": split_summary,
                "hidden_size": hidden_size,
                "grid_size": list(reconstructor.grid_size),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    optimizer = torch.optim.AdamW(reconstructor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_update_steps = max(
        1,
        math.ceil((len(train_loader) * args.epochs) / max(1, args.gradient_accumulation_steps)),
    )
    warmup_steps = max(1, int(total_update_steps * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    train_log_path = os.path.join(args.output_dir, "train_log.jsonl")
    val_log_path = os.path.join(args.output_dir, "val_log.jsonl")
    test_log_path = os.path.join(args.output_dir, "test_log.jsonl")
    print(f"[SGCSR-POPE] train={train_len} val={val_len} test={test_len} grid={reconstructor.grid_size} device={device}")
    print(f"[SGCSR-POPE] image-group split: {json.dumps(split_summary, ensure_ascii=False)}")
    print(f"[SGCSR-POPE] loss = {args.task_weight}*L_task + {args.rec_weight}*L_rec + {args.logit_weight}*L_logit")

    global_step = 0
    best_val = float("inf")
    for epoch in range(args.epochs):
        reconstructor.train()
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(train_loader, desc=f"POPE Train {epoch + 1}/{args.epochs}", dynamic_ncols=True)
        running = {"loss": 0.0, "task_loss": 0.0, "rec_loss": 0.0, "logit_loss": 0.0}

        for step, batch in enumerate(progress):
            losses = forward_losses(
                model=model,
                reconstructor=reconstructor,
                image_processor=image_processor,
                batch=batch,
                device=device,
                model_dtype=model_dtype,
                reconstructor_dtype=reconstructor_dtype,
                args=args,
                train=True,
            )
            loss = losses["loss"] / max(1, args.gradient_accumulation_steps)
            loss.backward()

            for key in running:
                running[key] += float(losses[key].item())

            should_step = (step + 1) % max(1, args.gradient_accumulation_steps) == 0 or (step + 1) == len(train_loader)
            if should_step:
                if args.max_grad_norm and args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(reconstructor.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            denom = step + 1
            progress.set_postfix(
                loss=running["loss"] / denom,
                task=running["task_loss"] / denom,
                rec=running["rec_loss"] / denom,
                logit=running["logit_loss"] / denom,
            )

        train_record = {
            "epoch": epoch + 1,
            "step": global_step,
            **{key: value / max(1, len(train_loader)) for key, value in running.items()},
        }
        log_jsonl(train_log_path, train_record)

        val_record = None
        if val_loader is not None:
            val_record = evaluate(
                model=model,
                reconstructor=reconstructor,
                image_processor=image_processor,
                eval_loader=val_loader,
                device=device,
                model_dtype=model_dtype,
                reconstructor_dtype=reconstructor_dtype,
                args=args,
                split_name="POPE Val",
            )
            val_record = {"epoch": epoch + 1, "step": global_step, **val_record}
            log_jsonl(val_log_path, val_record)
            if val_record["loss"] < best_val:
                best_val = float(val_record["loss"])
                save_checkpoint(reconstructor, args.output_dir, "best.pt", args, global_step, best_val)

        if args.save_every_epoch:
            save_checkpoint(
                reconstructor,
                args.output_dir,
                f"epoch_{epoch + 1}.pt",
                args,
                global_step,
                float(val_record["loss"]) if val_record is not None else float(train_record["loss"]),
            )
        print("[epoch]", json.dumps({"train": train_record, "val": val_record}, ensure_ascii=False))

    save_checkpoint(reconstructor, args.output_dir, "last.pt", args, global_step, best_val)

    test_record = None
    if test_loader is not None:
        best_path = os.path.join(args.output_dir, "best.pt")
        if os.path.exists(best_path):
            load_reconstructor_checkpoint(reconstructor, best_path, device, args)
        test_record = evaluate(
            model=model,
            reconstructor=reconstructor,
            image_processor=image_processor,
            eval_loader=test_loader,
            device=device,
            model_dtype=model_dtype,
            reconstructor_dtype=reconstructor_dtype,
            args=args,
            split_name="POPE Test",
        )
        test_record = {"step": global_step, "checkpoint": "best.pt" if os.path.exists(best_path) else "last.pt", **test_record}
        log_jsonl(test_log_path, test_record)
        with open(os.path.join(args.output_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(test_record, f, indent=2, ensure_ascii=False)
        print("[test]", json.dumps(test_record, ensure_ascii=False))

    print(f"[DONE] saved POPE-adapted SGCSR to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())