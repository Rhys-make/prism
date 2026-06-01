from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor, LlavaForConditionalGeneration, get_cosine_schedule_with_warmup

from mm.dataset import (
    _read_binary_tensor,
    compute_source_geometry,
    load_compressed_features_from_manifest,
    load_compressed_features_from_payload,
)
from mm.semantic_reconstructor import (
    SourceGuidedCompactSemanticReconstructor,
    compact_feature_distillation_loss,
    pool_teacher_visual_tokens,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Source-Guided Compact Semantic Reconstruction with feature distillation "
            "from no-ToMe LLaVA visual embeddings."
        )
    )
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Local LLaVA-1.5 HF model path.")
    parser.add_argument("--data_path", type=str, required=True, help="Compressed feature directory, .pt file, or manifest.jsonl.")
    parser.add_argument("--image_folder", type=str, default=None, help="Image root used when manifest records only contain image_id.")
    parser.add_argument("--output_dir", type=str, default="outputs/sgcsr_k144")
    parser.add_argument(
        "--init_checkpoint_path",
        type=str,
        default=None,
        help="Optional SGCSR checkpoint to initialize from before training, e.g. the pretrain-stage best.pt.",
    )
    parser.add_argument(
        "--allow_checkpoint_config_mismatch",
        action="store_true",
        help="Allow initializing from a checkpoint whose SGCSR architecture/locality args differ from current args.",
    )
    parser.add_argument("--local_files_only", action="store_true", help="Only load model/tokenizer files from local cache/path.")
    parser.add_argument("--num_queries", type=int, default=144, help="K reconstructed semantic tokens.")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim_head", type=int, default=128)
    parser.add_argument("--ff_mult", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--local_topk",
        type=int,
        default=0,
        help="Maximum source tokens per query after radius filtering. 0 means no top-k cap.",
    )
    parser.add_argument(
        "--local_radius",
        type=float,
        default=0.0,
        help=(
            "Normalized source-center radius for radius-topk attention. "
            "0 disables radius filtering; use with --local_topk for source-guided local reconstruction."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help=(
            "Backward-compatible validation ratio. Used only when --val_ratio is not set. "
            "For formal train/val/test runs, prefer --val_ratio and --final_test_ratio."
        ),
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=None,
        help="Validation ratio used for selecting best.pt. If omitted, falls back to --test_ratio.",
    )
    parser.add_argument(
        "--final_test_ratio",
        type=float,
        default=0.0,
        help="Held-out test ratio evaluated only after training. It is never used to select best.pt.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument(
        "--reconstructor_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="auto follows --dtype. Use bfloat16 on A800 for a good stability/memory tradeoff.",
    )
    parser.add_argument("--task_weight", type=float, default=1.0)
    parser.add_argument("--rec_weight", type=float, default=1.0)
    parser.add_argument("--rec_mse_weight", type=float, default=1.0)
    parser.add_argument("--rec_cosine_weight", type=float, default=0.1)
    parser.add_argument(
        "--logit_weight",
        type=float,
        default=0.0,
        help="Set >0 to enable no-ToMe teacher output distillation. Expensive.",
    )
    parser.add_argument(
        "--logit_teacher_mode",
        type=str,
        default="compact",
        choices=["compact", "full"],
        help=(
            "compact pools teacher visual tokens to K before logit distillation, "
            "so teacher/student answer positions match. full keeps 576 teacher "
            "tokens and is only for ablation."
        ),
    )
    parser.add_argument("--logit_temperature", type=float, default=1.0)
    parser.add_argument("--max_samples", type=int, default=0, help="Debug only; 0 means all samples.")
    parser.add_argument(
        "--max_text_length",
        type=int,
        default=0,
        help=(
            "Optional cap on text tokens before replacing <image> with visual tokens. "
            "0 disables truncation. Useful for long stage-2 multi-turn conversations."
        ),
    )
    parser.add_argument(
        "--conversation_mode",
        type=str,
        default="first",
        choices=["first", "all", "full"],
        help=(
            "How to use multi-turn conversations. 'first' keeps the original stage-1 behavior; "
            "'all' expands every user-assistant QA pair into a separate training example; "
            "'full' keeps the whole multi-turn conversation and supervises all assistant replies."
        ),
    )
    parser.add_argument("--save_every_epoch", action="store_true")
    parser.add_argument(
        "--allow_missing_source",
        action="store_true",
        help="Use fallback centers/sizes if source map is missing. Keep disabled for formal experiments.",
    )
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


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


def select_vision_features(model, vision_outputs) -> torch.Tensor:
    """Select the same CLIP layer that native LLaVA uses before its projector."""
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


def normalize_turns(conversations: Sequence[Dict[str, str]]) -> List[Tuple[str, str]]:
    turns: List[Tuple[str, str]] = []
    for item in conversations:
        role = str(item.get("from", "")).strip().lower()
        value = str(item.get("value", ""))
        if role in {"human", "user", "question"}:
            turns.append(("user", value))
        elif role in {"gpt", "assistant", "answer"}:
            turns.append(("assistant", value))
    return turns


def first_user_assistant_pair(conversations: Sequence[Dict[str, str]]) -> Tuple[str, str]:
    pairs = user_assistant_pairs(conversations)
    if pairs:
        return pairs[0]
    raise ValueError("Each sample needs at least one user turn followed by one assistant answer.")


def user_assistant_pairs(conversations: Sequence[Dict[str, str]]) -> List[Tuple[str, str]]:
    """Extract supervised QA pairs from a LLaVA-style multi-turn conversation.

    The stage-1 pretrain file is mostly single-turn, while LLaVA-1.5 stage-2
    data often contains several user/assistant rounds for one image.  We keep
    each round independent because the cloud module receives the same visual
    features, but the text supervision should cover every answer.
    """
    pairs: List[Tuple[str, str]] = []
    pending_user: Optional[str] = None
    for role, content in normalize_turns(conversations):
        if role == "user":
            pending_user = content.replace("<image>", "").strip()
        elif role == "assistant" and pending_user is not None:
            answer = content.strip()
            if pending_user and answer:
                pairs.append((pending_user, answer))
            pending_user = None
    return pairs


def append_question_suffix_to_conversations(
    conversations: Sequence[Dict[str, str]],
    question_suffix: str,
) -> List[Dict[str, str]]:
    """Append a task instruction to user turns without changing answers.

    POPE evaluation scores yes/no likelihood with an explicit suffix.  During
    POPE-domain adaptation we keep the supervised answer unchanged, but train
    with the same question wording used by the evaluator.
    """
    suffix = question_suffix.strip()
    if not suffix:
        return [dict(item) for item in conversations]

    updated: List[Dict[str, str]] = []
    for item in conversations:
        cloned = dict(item)
        role = str(cloned.get("from", "")).strip().lower()
        value = str(cloned.get("value", ""))
        lower_value = value.lower()
        if role in {"human", "user", "question"} and suffix.lower() not in lower_value and "yes or no" not in lower_value:
            value = value.rstrip()
            cloned["value"] = f"{value}\n{suffix}" if value else suffix
        updated.append(cloned)
    return updated


def build_llava_training_example(
    tokenizer,
    conversations: Sequence[Dict[str, str]],
    pair_index: int = 0,
) -> Dict[str, torch.Tensor]:
    """Build one LLaVA-style supervised example with assistant-only labels."""
    pairs = user_assistant_pairs(conversations)
    if not pairs:
        raise ValueError("Each sample needs at least one user turn followed by one assistant answer.")
    if pair_index < 0 or pair_index >= len(pairs):
        raise IndexError(f"pair_index={pair_index} out of range for {len(pairs)} QA pairs.")
    question, answer = pairs[pair_index]
    prefix = f"USER: <image>\n{question} ASSISTANT:"
    answer_text = " " + answer
    if tokenizer.eos_token and not answer_text.rstrip().endswith(tokenizer.eos_token):
        answer_text = answer_text + tokenizer.eos_token

    prefix_ids = tokenizer(prefix, add_special_tokens=True).input_ids
    answer_ids = tokenizer(answer_text, add_special_tokens=False).input_ids
    if not answer_ids:
        raise ValueError("Tokenized assistant answer is empty; cannot train on this sample.")

    input_ids = torch.tensor(prefix_ids + answer_ids, dtype=torch.long)
    labels = input_ids.clone()
    labels[: len(prefix_ids)] = -100
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def build_llava_full_conversation_example(
    tokenizer,
    conversations: Sequence[Dict[str, str]],
) -> Dict[str, torch.Tensor]:
    """Build a full multi-turn LLaVA example with labels on every assistant turn.

    This is the closest mode to LLaVA stage-2 instruction tuning: the first user
    turn contains the single <image> placeholder, later user turns provide text
    context, and every assistant answer contributes supervised tokens.
    """
    turns = normalize_turns(conversations)
    input_ids: List[int] = []
    labels: List[int] = []
    has_image = False
    waiting_for_assistant = False
    supervised_tokens = 0

    for role, content in turns:
        if role == "user":
            question = content.replace("<image>", "").strip()
            if not question:
                continue
            if not has_image:
                segment = f"USER: <image>\n{question} ASSISTANT:"
                has_image = True
            else:
                segment = f" USER: {question} ASSISTANT:"
            segment_ids = tokenizer(segment, add_special_tokens=(len(input_ids) == 0)).input_ids
            input_ids.extend(segment_ids)
            labels.extend([-100] * len(segment_ids))
            waiting_for_assistant = True
            continue

        if role == "assistant" and waiting_for_assistant:
            answer_text = " " + content.strip()
            if not answer_text.strip():
                waiting_for_assistant = False
                continue
            if tokenizer.eos_token and not answer_text.rstrip().endswith(tokenizer.eos_token):
                answer_text = answer_text + tokenizer.eos_token
            answer_ids = tokenizer(answer_text, add_special_tokens=False).input_ids
            if answer_ids:
                input_ids.extend(answer_ids)
                labels.extend(answer_ids)
                supervised_tokens += len(answer_ids)
            waiting_for_assistant = False

    if not has_image:
        raise ValueError("Full conversation example has no valid user turn for the <image> placeholder.")
    if supervised_tokens == 0:
        raise ValueError("Full conversation example has no supervised assistant tokens.")

    input_ids_t = torch.tensor(input_ids, dtype=torch.long)
    labels_t = torch.tensor(labels, dtype=torch.long)
    attention_mask_t = torch.ones_like(input_ids_t)
    return {"input_ids": input_ids_t, "attention_mask": attention_mask_t, "labels": labels_t}


def truncate_training_example(
    example: Dict[str, torch.Tensor],
    max_text_length: int,
    image_token_id: int,
) -> Dict[str, torch.Tensor]:
    """Right-truncate long text sequences while keeping the single image token.

    LLaVA stage-2 conversations can be much longer than pretrain captions.
    Right truncation preserves the leading <image> token and early dialogue
    context. We reject truncations that would remove all supervised tokens.
    """
    if max_text_length <= 0 or int(example["input_ids"].numel()) <= max_text_length:
        return example

    input_ids = example["input_ids"][:max_text_length].clone()
    attention_mask = example["attention_mask"][:max_text_length].clone()
    labels = example["labels"][:max_text_length].clone()
    if int(input_ids.eq(image_token_id).sum().item()) != 1:
        raise ValueError("Truncation removed or duplicated the <image> token; increase --max_text_length.")
    if not bool(labels.ne(-100).any()):
        raise ValueError("Truncation removed all supervised assistant tokens; increase --max_text_length.")
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def load_manifest_records(root_path: str) -> List[Tuple[Path, Dict[str, Any]]]:
    root = Path(root_path)
    manifest_paths = [root] if root.is_file() and root.name.endswith(".jsonl") else sorted(root.rglob("manifest.jsonl"))
    records: List[Tuple[Path, Dict[str, Any]]] = []
    for manifest_path in manifest_paths:
        shard_root = manifest_path.parent
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "meta" in record:
                    continue
                if "features" in record:
                    records.append((shard_root, record))
    return records


def resolve_image_path(image_folder: Optional[str], payload: Dict[str, Any]) -> str:
    image_path = payload.get("image_path")
    if image_path and Path(str(image_path)).exists():
        return str(image_path)

    image_id = payload.get("image_id") or payload.get("image")
    if image_id is None:
        raise ValueError("Sample is missing image_path/image_id; cannot run no-ToMe teacher.")
    if image_folder is None:
        raise ValueError("image_folder is required when samples do not contain an absolute image_path.")

    image_id = str(image_id)
    candidates = [
        Path(image_folder) / image_id,
        Path(image_folder) / Path(image_id).name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def fallback_source_geometry(num_tokens: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fallback only for debugging when source tracing was not saved."""
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    positions = torch.linspace(0, 1, steps=num_tokens)
    centers = torch.stack([positions, torch.full_like(positions, 0.5)], dim=-1)
    sizes = torch.full((num_tokens,), 1.0 / float(num_tokens), dtype=torch.float32)
    return centers, sizes


class SGCSRCompressedDataset(Dataset):
    """Compressed feature dataset for SGCSR distillation training."""

    def __init__(
        self,
        data_path: str,
        image_folder: Optional[str],
        tokenizer,
        max_samples: int = 0,
        allow_missing_source: bool = False,
        seed: int = 42,
        conversation_mode: str = "first",
        max_text_length: int = 0,
        image_token_id: int = 32000,
        question_suffix: str = "",
    ):
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.allow_missing_source = allow_missing_source
        self.max_text_length = max_text_length
        self.image_token_id = int(image_token_id)
        self.question_suffix = question_suffix
        self.conversation_mode = conversation_mode
        if self.conversation_mode not in {"first", "all", "full"}:
            raise ValueError(f"Unsupported conversation_mode: {self.conversation_mode}")

        root = Path(data_path)
        if root.is_file() and root.suffix == ".pt":
            base_items: List[Any] = [root]
        elif root.is_dir():
            pt_files = sorted([p for p in root.rglob("*.pt") if p.name not in {"best.pt", "last.pt"}])
            if pt_files:
                base_items = pt_files
            else:
                base_items = load_manifest_records(data_path)
        elif root.is_file() and root.name.endswith(".jsonl"):
            base_items = load_manifest_records(data_path)
        else:
            raise ValueError(f"Invalid compressed data path: {data_path}")

        if not base_items:
            raise ValueError(f"No compressed samples found under {data_path}")
        if max_samples and max_samples > 0 and max_samples < len(base_items):
            base_items = self._balanced_subset_by_retain_ratio(base_items, max_samples, seed)

        self.base_items = base_items
        self.items = self._expand_conversation_items(base_items)

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def _base_item_payload(base_item: Any) -> Dict[str, Any]:
        if isinstance(base_item, tuple):
            return base_item[1]
        return torch.load(base_item, map_location="cpu")

    @staticmethod
    def _unpack_item(item: Any) -> Tuple[Any, int, int]:
        if isinstance(item, dict) and "base_item" in item:
            return item["base_item"], int(item["pair_index"]), int(item["base_index"])
        return item, 0, -1

    @classmethod
    def _item_payload(cls, item: Any) -> Dict[str, Any]:
        base_item, _, _ = cls._unpack_item(item)
        return cls._base_item_payload(base_item)

    @classmethod
    def _item_retain_ratio(cls, item: Any) -> float:
        payload = cls._item_payload(item)

        retain_ratio = payload.get("actual_retain_ratio", payload.get("retain_ratio"))
        if retain_ratio is None:
            compressed_count = payload.get("compressed_token_count", payload.get("actual_keep_tokens"))
            if compressed_count is None:
                raise ValueError("Sample is missing retain_ratio and token-count metadata.")
            retain_ratio = float(compressed_count) / 576.0
        return float(retain_ratio)

    def _expand_conversation_items(self, base_items: Sequence[Any]) -> List[Dict[str, Any]]:
        expanded: List[Dict[str, Any]] = []
        for base_index, base_item in enumerate(base_items):
            payload = self._base_item_payload(base_item)
            pairs = user_assistant_pairs(payload.get("conversations", []))
            if not pairs:
                continue
            if self.conversation_mode == "all":
                pair_indices = range(len(pairs))
            elif self.conversation_mode == "full":
                pair_indices = [-1]
            else:
                pair_indices = range(1)
            for pair_index in pair_indices:
                expanded.append(
                    {
                        "base_item": base_item,
                        "base_index": base_index,
                        "pair_index": int(pair_index),
                    }
                )
        if not expanded:
            raise ValueError("No valid user-assistant QA pairs found in compressed samples.")
        return expanded

    @staticmethod
    def _base_item_retain_ratio(base_item: Any) -> float:
        if isinstance(base_item, tuple):
            payload = base_item[1]
        else:
            payload = torch.load(base_item, map_location="cpu")

        retain_ratio = payload.get("actual_retain_ratio", payload.get("retain_ratio"))
        if retain_ratio is None:
            compressed_count = payload.get("compressed_token_count", payload.get("actual_keep_tokens"))
            if compressed_count is None:
                raise ValueError("Sample is missing retain_ratio and token-count metadata.")
            retain_ratio = float(compressed_count) / 576.0
        return float(retain_ratio)

    @classmethod
    def _balanced_subset_by_retain_ratio(cls, items: Sequence[Any], max_samples: int, seed: int) -> List[Any]:
        groups: Dict[str, List[Any]] = {}
        for item in items:
            ratio_key = f"{cls._base_item_retain_ratio(item):.2f}"
            groups.setdefault(ratio_key, []).append(item)

        generator = torch.Generator().manual_seed(seed)
        selected: List[Any] = []
        ratio_keys = sorted(groups.keys(), key=float)
        base = max_samples // max(1, len(ratio_keys))
        remainder = max_samples % max(1, len(ratio_keys))

        for i, ratio_key in enumerate(ratio_keys):
            group = groups[ratio_key]
            want = base + (1 if i < remainder else 0)
            if want <= 0:
                continue
            order = torch.randperm(len(group), generator=generator).tolist()
            selected.extend(group[j] for j in order[: min(want, len(group))])

        return selected

    def get_retain_ratio(self, idx: int) -> float:
        """Read retain ratio metadata without loading the feature tensor when possible."""
        return self._item_retain_ratio(self.items[idx])

    def get_group_id(self, idx: int) -> int:
        """Return the compressed-image id used to keep expanded QA pairs in one split."""
        _, _, base_index = self._unpack_item(self.items[idx])
        return idx if base_index < 0 else base_index

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item, pair_index, _ = self._unpack_item(self.items[idx])
        if isinstance(item, tuple):
            shard_root, record = item
            payload = record
            compressed_features = load_compressed_features_from_manifest(record, shard_root)
            if record.get("source_encoding") == "csr_binary_patch_i16":
                source_indices = _read_binary_tensor(shard_root, record["source_indices"])
                source_offsets = _read_binary_tensor(shard_root, record["source_offsets"])
                token_centers, token_sizes = compute_source_geometry(
                    source_indices=source_indices,
                    source_offsets=source_offsets,
                    grid_shape=record.get("grid_shape", [24, 24]),
                )
            elif self.allow_missing_source:
                token_centers, token_sizes = fallback_source_geometry(int(compressed_features.shape[0]))
            else:
                raise ValueError("Missing ToMe source map. Regenerate data without --no_source for SGCSR training.")
        else:
            payload = torch.load(item, map_location="cpu")
            compressed_features = load_compressed_features_from_payload(payload)
            if payload.get("source_encoding") == "csr_binary_patch_i16":
                source_indices = payload["source_indices"]
                source_offsets = payload["source_offsets"]
                if not isinstance(source_indices, torch.Tensor):
                    source_indices = torch.as_tensor(source_indices)
                if not isinstance(source_offsets, torch.Tensor):
                    source_offsets = torch.as_tensor(source_offsets)
                token_centers, token_sizes = compute_source_geometry(
                    source_indices=source_indices,
                    source_offsets=source_offsets,
                    grid_shape=payload.get("grid_shape", [24, 24]),
                )
            elif self.allow_missing_source:
                token_centers, token_sizes = fallback_source_geometry(int(compressed_features.shape[0]))
            else:
                raise ValueError("Missing ToMe source map. Regenerate data without --no_source for SGCSR training.")

        conversations = payload.get("conversations", [])
        if self.question_suffix:
            conversations = append_question_suffix_to_conversations(conversations, self.question_suffix)

        if pair_index < 0:
            text = build_llava_full_conversation_example(self.tokenizer, conversations)
        else:
            text = build_llava_training_example(
                self.tokenizer,
                conversations,
                pair_index=pair_index,
            )
        text = truncate_training_example(text, self.max_text_length, self.image_token_id)
        image_path = resolve_image_path(self.image_folder, payload)
        retain_ratio = payload.get("actual_retain_ratio", payload.get("retain_ratio"))
        if retain_ratio is None:
            retain_ratio = float(compressed_features.shape[0]) / 576.0

        return {
            **text,
            "compressed_features": compressed_features.float(),
            "compressed_attention_mask": torch.ones(compressed_features.shape[0], dtype=torch.long),
            "token_centers": token_centers.float(),
            "token_sizes": token_sizes.float(),
            "retain_ratio": torch.tensor(float(retain_ratio), dtype=torch.float32),
            "image_path": image_path,
        }


class SGCSRCollator:
    def __init__(self, pad_token_id: int, label_pad_id: int = -100):
        self.pad_token_id = pad_token_id
        self.label_pad_id = label_pad_id

    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = pad_sequence([x["input_ids"] for x in instances], batch_first=True, padding_value=self.pad_token_id)
        attention_mask = pad_sequence([x["attention_mask"] for x in instances], batch_first=True, padding_value=0)
        labels = pad_sequence([x["labels"] for x in instances], batch_first=True, padding_value=self.label_pad_id)
        compressed_features = pad_sequence(
            [x["compressed_features"] for x in instances],
            batch_first=True,
            padding_value=0.0,
        )
        compressed_attention_mask = pad_sequence(
            [x["compressed_attention_mask"] for x in instances],
            batch_first=True,
            padding_value=0,
        )
        token_centers = pad_sequence([x["token_centers"] for x in instances], batch_first=True, padding_value=0.0)
        token_sizes = pad_sequence([x["token_sizes"] for x in instances], batch_first=True, padding_value=0.0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "compressed_features": compressed_features,
            "compressed_attention_mask": compressed_attention_mask,
            "token_centers": token_centers,
            "token_sizes": token_sizes,
            "retain_ratio": torch.stack([x["retain_ratio"] for x in instances]),
            "image_paths": [x["image_path"] for x in instances],
        }


def build_stratified_train_val_test_split(
    dataset: SGCSRCompressedDataset,
    val_ratio: float,
    final_test_ratio: float,
    seed: int,
) -> Tuple[Subset, Optional[Subset], Optional[Subset], Dict[str, Dict[str, int]]]:
    """Split each retain-ratio bucket into train/val/test.

    This guarantees that 0.2/0.4/0.6/0.8 compression levels are represented in
    train, validation, and final test with the same ratio when enough samples
    exist. Validation is used to select best.pt; final test is held out until
    the end and is never used for checkpoint selection.
    """
    if not 0 <= val_ratio < 1:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    if not 0 <= final_test_ratio < 1:
        raise ValueError(f"final_test_ratio must be in [0, 1), got {final_test_ratio}")
    if val_ratio + final_test_ratio >= 1:
        raise ValueError(
            "val_ratio + final_test_ratio must be < 1 so at least one train split remains; "
            f"got {val_ratio + final_test_ratio}"
        )

    groups: Dict[str, Dict[int, List[int]]] = {}
    for idx in range(len(dataset)):
        ratio_key = f"{dataset.get_retain_ratio(idx):.2f}"
        group_id = dataset.get_group_id(idx)
        groups.setdefault(ratio_key, {}).setdefault(group_id, []).append(idx)

    generator = torch.Generator().manual_seed(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []
    summary: Dict[str, Dict[str, int]] = {}

    for ratio_key in sorted(groups.keys(), key=float):
        grouped_indices = list(groups[ratio_key].values())
        order = torch.randperm(len(grouped_indices), generator=generator).tolist()
        shuffled_groups = [grouped_indices[i] for i in order]
        num_groups = len(shuffled_groups)

        if val_ratio > 0 and num_groups > 1:
            val_group_len = max(1, int(num_groups * val_ratio))
        else:
            val_group_len = 0
        if final_test_ratio > 0 and num_groups > 1:
            test_group_len = max(1, int(num_groups * final_test_ratio))
        else:
            test_group_len = 0

        # Tiny debug subsets may not be large enough for all three splits. Keep
        # at least one group for training, then shrink holdouts deterministically.
        max_holdout_groups = max(0, num_groups - 1)
        while val_group_len + test_group_len > max_holdout_groups:
            if test_group_len >= val_group_len and test_group_len > 0:
                test_group_len -= 1
            elif val_group_len > 0:
                val_group_len -= 1
            else:
                break

        test_groups = shuffled_groups[:test_group_len]
        val_groups = shuffled_groups[test_group_len : test_group_len + val_group_len]
        train_groups = shuffled_groups[test_group_len + val_group_len :]
        test_part = [idx for group in test_groups for idx in group]
        val_part = [idx for group in val_groups for idx in group]
        train_part = [idx for group in train_groups for idx in group]
        train_indices.extend(train_part)
        val_indices.extend(val_part)
        test_indices.extend(test_part)
        summary[ratio_key] = {
            "total": len(train_part) + len(val_part) + len(test_part),
            "train": len(train_part),
            "val": len(val_part),
            "test": len(test_part),
            "base_groups": len(shuffled_groups),
            "train_groups": len(train_groups),
            "val_groups": len(val_groups),
            "test_groups": len(test_groups),
        }

    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices) if val_indices else None
    test_ds = Subset(dataset, test_indices) if test_indices else None
    return train_ds, val_ds, test_ds, summary


def build_teacher_visual_embeddings(
    model,
    image_processor,
    image_paths: Sequence[str],
    device: torch.device,
    model_dtype: torch.dtype,
) -> torch.Tensor:
    """Run no-ToMe teacher vision tower and frozen LLaVA projector."""
    images = [Image.open(path).convert("RGB") for path in image_paths]
    pixel_values = image_processor(images=images, return_tensors="pt").pixel_values.to(device=device, dtype=model_dtype)
    vision_tower = get_vision_tower(model)
    with torch.no_grad():
        vision_outputs = vision_tower(pixel_values, output_hidden_states=True)
        teacher_clip_tokens = select_vision_features(model, vision_outputs)
        teacher_visual_embeddings = model.multi_modal_projector(teacher_clip_tokens)
    return teacher_visual_embeddings


def merge_text_and_visual_tokens(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    visual_tokens: torch.Tensor,
    label_pad_id: int = -100,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replace the <image> token in each sample with visual tokens."""
    embedding_layer = get_input_embedding_layer(model)
    image_token_id = int(getattr(model.config, "image_token_index", 32000))
    text_embeds = embedding_layer(input_ids)
    visual_tokens = visual_tokens.to(device=text_embeds.device, dtype=text_embeds.dtype)

    merged_embeds: List[torch.Tensor] = []
    merged_masks: List[torch.Tensor] = []
    merged_labels: List[torch.Tensor] = []
    for i in range(input_ids.shape[0]):
        valid_len = int(attention_mask[i].sum().item())
        ids_i = input_ids[i, :valid_len]
        embeds_i = text_embeds[i, :valid_len]
        labels_i = labels[i, :valid_len]
        image_positions = torch.nonzero(ids_i.eq(image_token_id), as_tuple=False).flatten()
        if image_positions.numel() != 1:
            raise ValueError(
                f"Expected exactly one <image> token per sample, got {int(image_positions.numel())}. "
                "Check tokenizer/model compatibility and prompt formatting."
            )
        pos = int(image_positions[0].item())

        image_embeds = visual_tokens[i]
        image_labels = torch.full((image_embeds.shape[0],), label_pad_id, dtype=labels.dtype, device=labels.device)
        image_mask = torch.ones((image_embeds.shape[0],), dtype=attention_mask.dtype, device=attention_mask.device)

        merged_embeds.append(torch.cat([embeds_i[:pos], image_embeds, embeds_i[pos + 1 :]], dim=0))
        merged_masks.append(torch.cat([attention_mask[i, :pos], image_mask, attention_mask[i, pos + 1 : valid_len]], dim=0))
        merged_labels.append(torch.cat([labels_i[:pos], image_labels, labels_i[pos + 1 :]], dim=0))

    return (
        pad_sequence(merged_embeds, batch_first=True, padding_value=0.0),
        pad_sequence(merged_masks, batch_first=True, padding_value=0),
        pad_sequence(merged_labels, batch_first=True, padding_value=label_pad_id),
    )


def shifted_answer_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Select logits that predict supervised answer tokens under causal LM shifting."""
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    if not bool(mask.any()):
        raise ValueError("No supervised answer tokens found for logit distillation.")
    return shifted_logits[mask]


def logit_distillation_loss(
    student_logits: torch.Tensor,
    student_labels: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """KL(student || teacher) on answer-token prediction positions.

    The caller should make teacher/student visual token lengths equal when
    possible. We still select logits by supervised answer labels rather than by
    absolute sequence indices, so the compared distributions are ordered by
    answer-token step.
    """
    student_selected = shifted_answer_logits(student_logits, student_labels)
    teacher_selected = shifted_answer_logits(teacher_logits, teacher_labels)
    if student_selected.shape[0] != teacher_selected.shape[0]:
        raise ValueError(
            "Student/teacher answer-token counts differ: "
            f"student={student_selected.shape[0]}, teacher={teacher_selected.shape[0]}"
        )

    temp = max(float(temperature), 1e-6)
    student_log_probs = F.log_softmax(student_selected.float() / temp, dim=-1)
    teacher_probs = F.softmax(teacher_selected.float() / temp, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temp * temp)


def forward_losses(
    model,
    reconstructor,
    image_processor,
    batch: Dict[str, Any],
    device: torch.device,
    model_dtype: torch.dtype,
    reconstructor_dtype: torch.dtype,
    args: argparse.Namespace,
    train: bool,
) -> Dict[str, torch.Tensor]:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    compressed_features = batch["compressed_features"].to(device=device, dtype=model_dtype)
    compressed_attention_mask = batch["compressed_attention_mask"].to(device)
    token_centers = batch["token_centers"].to(device=device, dtype=reconstructor_dtype)
    token_sizes = batch["token_sizes"].to(device=device, dtype=reconstructor_dtype)
    retain_ratio = batch["retain_ratio"].to(device=device, dtype=reconstructor_dtype)

    # Frozen LLaVA projector maps compressed CLIP features into LLM visual space.
    with torch.no_grad():
        compressed_visual_embeddings = model.multi_modal_projector(compressed_features)
    compressed_visual_embeddings = compressed_visual_embeddings.to(dtype=reconstructor_dtype)

    reconstructed_tokens = reconstructor(
        visual_embeddings=compressed_visual_embeddings,
        token_centers=token_centers,
        token_sizes=token_sizes,
        retain_ratio=retain_ratio,
        attention_mask=compressed_attention_mask,
    )

    student_embeds, student_mask, student_labels = merge_text_and_visual_tokens(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        visual_tokens=reconstructed_tokens,
    )
    student_out = get_language_model(model)(
        inputs_embeds=student_embeds,
        attention_mask=student_mask,
        labels=student_labels,
    )
    task_loss = student_out.loss

    teacher_visual_embeddings = build_teacher_visual_embeddings(
        model=model,
        image_processor=image_processor,
        image_paths=batch["image_paths"],
        device=device,
        model_dtype=model_dtype,
    )
    teacher_compact = pool_teacher_visual_tokens(
        teacher_visual_embeddings,
        output_grid=reconstructor.grid_size,
    ).to(dtype=reconstructor_dtype)
    rec_loss = compact_feature_distillation_loss(
        student_tokens=reconstructed_tokens,
        teacher_compact_tokens=teacher_compact,
        mse_weight=args.rec_mse_weight,
        cosine_weight=args.rec_cosine_weight,
    )

    logit_loss = torch.zeros((), device=device, dtype=task_loss.dtype)
    if args.logit_weight > 0:
        # For the default compact mode, the teacher uses the same K visual-token
        # slots as the student. This avoids absolute-position mismatch caused by
        # comparing a K-token student sequence with a 576-token teacher sequence.
        if args.logit_teacher_mode == "compact":
            teacher_logit_tokens = teacher_compact
        elif args.logit_teacher_mode == "full":
            teacher_logit_tokens = teacher_visual_embeddings
        else:
            raise ValueError(f"Unsupported logit_teacher_mode: {args.logit_teacher_mode}")

        with torch.no_grad():
            teacher_embeds, teacher_mask, teacher_labels = merge_text_and_visual_tokens(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                visual_tokens=teacher_logit_tokens,
            )
            teacher_out = get_language_model(model)(
                inputs_embeds=teacher_embeds,
                attention_mask=teacher_mask,
                labels=teacher_labels,
            )
        logit_loss = logit_distillation_loss(
            student_logits=student_out.logits,
            student_labels=student_labels,
            teacher_logits=teacher_out.logits,
            teacher_labels=teacher_labels,
            temperature=args.logit_temperature,
        ).to(dtype=task_loss.dtype)

    total_loss = args.task_weight * task_loss + args.rec_weight * rec_loss.to(dtype=task_loss.dtype)
    total_loss = total_loss + args.logit_weight * logit_loss
    return {
        "loss": total_loss,
        "task_loss": task_loss.detach(),
        "rec_loss": rec_loss.detach(),
        "logit_loss": logit_loss.detach(),
    }


def save_checkpoint(reconstructor, output_dir: str, name: str, args: argparse.Namespace, step: int, eval_loss: float):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    torch.save(
        {
            "model_type": "sgcsr",
            "step": step,
            "eval_loss": eval_loss,
            "args": vars(args),
            "grid_size": list(reconstructor.grid_size),
            "reconstructor": reconstructor.state_dict(),
        },
        path,
    )


def load_reconstructor_checkpoint(
    reconstructor,
    checkpoint_path: str,
    device: torch.device,
    args: argparse.Namespace,
):
    """Initialize SGCSR weights from a previous training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    config_keys = ["num_queries", "depth", "heads", "dim_head", "ff_mult", "local_topk", "local_radius"]
    mismatches = []
    for key in config_keys:
        if key not in saved_args:
            continue
        current_value = getattr(args, key)
        saved_value = saved_args[key]
        if isinstance(current_value, float) or isinstance(saved_value, float):
            same = abs(float(current_value) - float(saved_value)) < 1e-9
        else:
            same = current_value == saved_value
        if not same:
            mismatches.append(f"{key}: checkpoint={saved_value} current={current_value}")
    if mismatches and not args.allow_checkpoint_config_mismatch:
        raise ValueError(
            "SGCSR checkpoint config does not match current training args. "
            "This can silently change reconstructor behavior during stage-2 continuation. "
            "Pass matching args or use --allow_checkpoint_config_mismatch for an intentional ablation. "
            + "; ".join(mismatches)
        )

    state_dict = checkpoint.get("reconstructor", checkpoint)
    reconstructor.load_state_dict(state_dict, strict=True)
    step = checkpoint.get("step", None) if isinstance(checkpoint, dict) else None
    eval_loss = checkpoint.get("eval_loss", None) if isinstance(checkpoint, dict) else None
    print(f"[SGCSR] initialized from {checkpoint_path} step={step} eval_loss={eval_loss}")


def log_jsonl(path: str, record: Dict[str, Any]):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def subset_indices(subset: Optional[Subset]) -> List[int]:
    """Return JSON-serializable indices from a torch Subset."""
    if subset is None:
        return []
    return [int(idx) for idx in subset.indices]


def save_split_indices(
    output_dir: str,
    args: argparse.Namespace,
    train_ds: Subset,
    val_ds: Optional[Subset],
    test_ds: Optional[Subset],
    split_summary: Dict[str, Dict[str, int]],
    val_ratio: float,
    final_test_ratio: float,
) -> None:
    """Persist exact train/val/test membership for reproducible evaluation.

    The saved indices refer to SGCSRCompressedDataset after optional max_samples
    filtering and conversation expansion. This lets later evaluation scripts
    reuse the same uncontaminated final-test split instead of reconstructing it
    implicitly from seed and ratio.
    """
    payload = {
        "format": "sgcsr_split_indices_v1",
        "indices_are": "indices into SGCSRCompressedDataset after max_samples filtering and conversation expansion",
        "data_path": args.data_path,
        "image_folder": args.image_folder,
        "seed": int(args.seed),
        "conversation_mode": args.conversation_mode,
        "max_samples": int(args.max_samples),
        "max_text_length": int(args.max_text_length),
        "val_ratio": float(val_ratio),
        "final_test_ratio": float(final_test_ratio),
        "split_mode": "stratified_by_retain_ratio_train_val_test",
        "split_summary": split_summary,
        "train": subset_indices(train_ds),
        "val": subset_indices(val_ds),
        "test": subset_indices(test_ds),
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "split_indices.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))


def evaluate(
    model,
    reconstructor,
    image_processor,
    eval_loader,
    device,
    model_dtype,
    reconstructor_dtype,
    args,
    split_name: str = "Eval",
) -> Dict[str, float]:
    model.eval()
    reconstructor.eval()
    totals = {"loss": 0.0, "task_loss": 0.0, "rec_loss": 0.0, "logit_loss": 0.0, "count": 0.0}
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=split_name, dynamic_ncols=True):
            losses = forward_losses(
                model=model,
                reconstructor=reconstructor,
                image_processor=image_processor,
                batch=batch,
                device=device,
                model_dtype=model_dtype,
                reconstructor_dtype=reconstructor_dtype,
                args=args,
                train=False,
            )
            bsz = int(batch["input_ids"].shape[0])
            for key in ["loss", "task_loss", "rec_loss", "logit_loss"]:
                totals[key] += float(losses[key].item()) * bsz
            totals["count"] += bsz

    count = max(1.0, totals.pop("count"))
    return {key: value / count for key, value in totals.items()}


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
    if args.init_checkpoint_path:
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
    )
    val_ratio = float(args.test_ratio if args.val_ratio is None else args.val_ratio)
    final_test_ratio = float(args.final_test_ratio)
    train_ds, val_ds, test_ds, split_summary = build_stratified_train_val_test_split(
        dataset=dataset,
        val_ratio=val_ratio,
        final_test_ratio=final_test_ratio,
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
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collator,
        )
        if val_ds is not None
        else None
    )
    test_loader = (
        DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collator,
        )
        if test_ds is not None
        else None
    )

    os.makedirs(args.output_dir, exist_ok=True)
    save_split_indices(
        output_dir=args.output_dir,
        args=args,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        split_summary=split_summary,
        val_ratio=val_ratio,
        final_test_ratio=final_test_ratio,
    )
    with open(os.path.join(args.output_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                **vars(args),
                "train_samples": train_len,
                "val_samples": val_len,
                "test_samples": test_len,
                "effective_val_ratio": val_ratio,
                "effective_final_test_ratio": final_test_ratio,
                "split_mode": "stratified_by_retain_ratio_train_val_test",
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
    legacy_eval_log_path = os.path.join(args.output_dir, "eval_log.jsonl")
    test_log_path = os.path.join(args.output_dir, "test_log.jsonl")
    print(f"[SGCSR] train={train_len} val={val_len} test={test_len} grid={reconstructor.grid_size} device={device}")
    print(f"[SGCSR] stratified split: {json.dumps(split_summary, ensure_ascii=False)}")
    print(f"[SGCSR] loss = {args.task_weight}*L_task + {args.rec_weight}*L_rec + {args.logit_weight}*L_logit")

    global_step = 0
    best_val = float("inf")
    for epoch in range(args.epochs):
        reconstructor.train()
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(train_loader, desc=f"Train {epoch + 1}/{args.epochs}", dynamic_ncols=True)
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
                split_name="Val",
            )
            val_record = {"epoch": epoch + 1, "step": global_step, **val_record}
            log_jsonl(val_log_path, val_record)
            log_jsonl(legacy_eval_log_path, val_record)
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
            split_name="Test",
        )
        test_record = {"step": global_step, "checkpoint": "best.pt" if os.path.exists(best_path) else "last.pt", **test_record}
        log_jsonl(test_log_path, test_record)
        with open(os.path.join(args.output_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(test_record, f, indent=2, ensure_ascii=False)
        print("[test]", json.dumps(test_record, ensure_ascii=False))

    print(f"[DONE] saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

