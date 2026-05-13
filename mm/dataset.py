from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from .vision import VisionTowerWrapper


# -----------------------------
# 对话 / chat 格式化
# -----------------------------
#
# 这里的数据处理逻辑尽量严格地贴合 TinyLlama：
# - TinyLlama-1.1B-Chat-v1.0 自带 tokenizer chat template
# - 我们应该直接使用 tokenizer.apply_chat_template，而不是自己发明 prompt 格式
# - 训练时仍然需要 assistant-only 的 loss mask
#
# 下面的策略是：
# 1) 把原始 conversation 记录转换成统一的 role 序列
# 2) 使用 tokenizer.apply_chat_template 构造完全一致的 chat 文本 / token 序列
# 3) 构造 labels，使得只有 assistant 的 token 参与 loss
#
# 这比早期的简化版训练方式更接近正式 stage1 的 SFT 设定。


def _normalize_turns(conversations: Sequence[Dict[str, str]]) -> List[Tuple[str, str]]:
    """把原始 conversation 记录规范化为 (role, content) 形式。"""
    turns: List[Tuple[str, str]] = []
    for c in conversations:
        role = str(c.get("from", "")).strip().lower()
        value = str(c.get("value", ""))

        if role in {"human", "user", "question"}:
            turns.append(("user", value))
        elif role in {"gpt", "assistant", "answer"}:
            turns.append(("assistant", value))
        elif role in {"system"}:
            turns.append(("system", value))
        else:
            turns.append((role or "unknown", value))
    return turns


def _build_tinyllama_chat_example(tokenizer, turns: Sequence[Tuple[str, str]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """使用 TinyLlama 的 chat template 构造 input_ids / attention_mask / labels。"""
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("Tokenizer 没有 apply_chat_template，无法构造 TinyLlama 的正式 chat 输入。")

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer 必须提供 pad_token_id 或 eos_token_id。")
        tokenizer.pad_token = tokenizer.eos_token

    def _tokenize_rendered(text: str) -> List[int]:
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if len(ids) > 0 and isinstance(ids[0], list):
            ids = ids[0]
        return ids

    input_ids: List[int] = []
    labels: List[int] = []
    messages: List[Dict[str, str]] = []
    prev_rendered_ids: List[int] = []

    for role, content in turns:
        if role == "system":
            messages.append({"role": "system", "content": content})
            rendered_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            rendered_ids = _tokenize_rendered(rendered_text)
            if len(rendered_ids) < len(prev_rendered_ids) or rendered_ids[: len(prev_rendered_ids)] != prev_rendered_ids:
                raise ValueError("chat template 渲染不一致：添加 system turn 时前缀不匹配。")
            delta_ids = rendered_ids[len(prev_rendered_ids):]
            input_ids.extend(delta_ids)
            labels.extend([-100] * len(delta_ids))
            prev_rendered_ids = rendered_ids
            continue

        if role == "user":
            messages.append({"role": "user", "content": content})
            rendered_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            rendered_ids = _tokenize_rendered(rendered_text)
            if len(rendered_ids) < len(prev_rendered_ids) or rendered_ids[: len(prev_rendered_ids)] != prev_rendered_ids:
                raise ValueError("chat template 渲染不一致：添加 user turn 时前缀不匹配。")
            delta_ids = rendered_ids[len(prev_rendered_ids):]
            input_ids.extend(delta_ids)
            labels.extend([-100] * len(delta_ids))
            prev_rendered_ids = rendered_ids
            continue

        if role != "assistant":
            messages.append({"role": "user", "content": content})
            rendered_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            rendered_ids = _tokenize_rendered(rendered_text)
            if len(rendered_ids) < len(prev_rendered_ids) or rendered_ids[: len(prev_rendered_ids)] != prev_rendered_ids:
                raise ValueError("chat template 渲染不一致：添加未知 role turn 时前缀不匹配。")
            delta_ids = rendered_ids[len(prev_rendered_ids):]
            input_ids.extend(delta_ids)
            labels.extend([-100] * len(delta_ids))
            prev_rendered_ids = rendered_ids
            continue

        context_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        context_ids = _tokenize_rendered(context_text)
        if len(context_ids) < len(prev_rendered_ids) or context_ids[: len(prev_rendered_ids)] != prev_rendered_ids:
            raise ValueError("chat template 渲染不一致：准备 assistant 上下文时前缀不匹配。")

        context_delta = context_ids[len(prev_rendered_ids):]
        input_ids.extend(context_delta)
        labels.extend([-100] * len(context_delta))
        prev_rendered_ids = context_ids

        assistant_messages = messages + [{"role": "assistant", "content": content}]
        full_text = tokenizer.apply_chat_template(assistant_messages, tokenize=False, add_generation_prompt=False)
        full_ids = _tokenize_rendered(full_text)
        if len(full_ids) < len(context_ids) or full_ids[: len(context_ids)] != context_ids:
            raise ValueError("chat template 渲染不一致：添加 assistant 回复时前缀不匹配。")

        assistant_delta = full_ids[len(context_ids):]
        if len(assistant_delta) == 0:
            raise ValueError("assistant turn 产生了空 token 序列，无法监督空回复。")

        input_ids.extend(assistant_delta)
        labels.extend(assistant_delta)
        prev_rendered_ids = full_ids
        messages.append({"role": "assistant", "content": content})

    if len(input_ids) == 0:
        raise ValueError("conversation 中没有可监督的 assistant turn。")

    input_ids_t = torch.tensor(input_ids, dtype=torch.long)
    labels_t = torch.tensor(labels, dtype=torch.long)
    attention_mask_t = torch.ones_like(input_ids_t)
    return input_ids_t, attention_mask_t, labels_t


def load_compressed_features_from_payload(payload: Dict[str, Any]) -> torch.Tensor:
    """Load compressed visual features, dequantizing int8 payloads when needed."""
    if "compressed_features" in payload:
        compressed_features = payload["compressed_features"]
        if not isinstance(compressed_features, torch.Tensor):
            compressed_features = torch.as_tensor(compressed_features)
        return compressed_features

    if "compressed_features_q" in payload and "compressed_features_scale" in payload:
        q = payload["compressed_features_q"]
        scale = payload["compressed_features_scale"]
        if not isinstance(q, torch.Tensor):
            q = torch.as_tensor(q)
        if not isinstance(scale, torch.Tensor):
            scale = torch.as_tensor(scale)
        return q.float() * scale.float()

    raise KeyError("payload 中缺少 compressed_features 或 int8 量化特征字段。")


_DTYPE_MAP = {
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _read_binary_tensor(root: Path, spec: Dict[str, Any]) -> torch.Tensor:
    dtype = _DTYPE_MAP[str(spec["dtype"])]
    numel = int(spec["numel"])
    shape = tuple(int(x) for x in spec["shape"])
    offset = int(spec["offset"])
    nbytes = numel * torch.empty((), dtype=dtype).element_size()
    path = root / spec["path"]
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read(nbytes)
    if len(data) != nbytes:
        raise ValueError(f"读取 {path} 失败：期望 {nbytes} bytes，实际 {len(data)} bytes。")
    return torch.frombuffer(bytearray(data), dtype=dtype).clone().reshape(shape)


def load_compressed_features_from_manifest(record: Dict[str, Any], root: Path) -> torch.Tensor:
    storage = record.get("compressed_feature_storage")
    if storage == "int8_symmetric_per_token":
        q = _read_binary_tensor(root, record["features"])
        scale = _read_binary_tensor(root, record["feature_scales"])
        return q.float() * scale.float()
    if "features" in record:
        return _read_binary_tensor(root, record["features"])
    raise KeyError("manifest record 中缺少 features 字段。")


def _load_manifest_records(root_path: str) -> List[Tuple[Path, Dict[str, Any]]]:
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
                rec = json.loads(line)
                if "meta" in rec:
                    continue
                if "features" in rec:
                    records.append((shard_root, rec))
    return records


# -----------------------------
# 原始 JSON 数据集
# -----------------------------


class JsonConversationDataset(Dataset):
    """原始多模态 conversation 数据集。"""

    def __init__(self, data_path: str, image_folder: str, tokenizer, vision: VisionTowerWrapper):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        if not isinstance(self.data, list):
            raise ValueError("JSON 中应为样本列表。")
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.vision = vision

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]
        image_path = str(Path(self.image_folder) / sample["image"])
        image = Image.open(image_path).convert("RGB")
        inputs = self.vision.preprocess(image)
        pixel_values = inputs["pixel_values"].squeeze(0)

        turns = _normalize_turns(sample.get("conversations", []))
        input_ids, attention_mask, labels = _build_tinyllama_chat_example(self.tokenizer, turns)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
        }


# -----------------------------
# 离线压缩特征数据集
# -----------------------------


class CompressedFeatureDataset(Dataset):
    """离线压缩后的多模态训练数据集。"""

    def __init__(self, root_path: str, tokenizer):
        self.tokenizer = tokenizer
        root = Path(root_path)
        if root.is_file() and root.suffix == ".pt":
            self.samples = [root]
        elif root.is_dir():
            self.samples = sorted([p for p in root.rglob("*.pt") if p.name not in {"best.pt", "last.pt"}])
            if not self.samples:
                raise ValueError(f"{root_path} 下没有找到 .pt 样本文件")
        else:
            raise ValueError(f"无效的压缩数据路径: {root_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pt_path = self.samples[idx]
        payload = torch.load(pt_path, map_location="cpu")

        turns = _normalize_turns(payload.get("conversations", []))
        input_ids, attention_mask, labels = _build_tinyllama_chat_example(self.tokenizer, turns)

        compressed_features = load_compressed_features_from_payload(payload)

        # 保存一个与压缩特征等长的 mask，后续 collator 会做 padding。
        compressed_attention_mask = torch.ones(compressed_features.shape[0], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "compressed_features": compressed_features,
            "compressed_attention_mask": compressed_attention_mask,
            "retain_ratio": payload.get("retain_ratio"),
            "target_keep_tokens": payload.get("target_keep_tokens"),
            "drop_tokens": payload.get("drop_tokens"),
        }


class BinaryCompressedFeatureDataset(Dataset):
    """读取 compact bin + manifest.jsonl 格式的压缩特征数据集。"""

    def __init__(self, root_path: str, tokenizer):
        self.tokenizer = tokenizer
        self.records = _load_manifest_records(root_path)
        if not self.records:
            raise ValueError(f"{root_path} 下没有找到 bin manifest 样本记录")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        shard_root, record = self.records[idx]

        turns = _normalize_turns(record.get("conversations", []))
        input_ids, attention_mask, labels = _build_tinyllama_chat_example(self.tokenizer, turns)

        compressed_features = load_compressed_features_from_manifest(record, shard_root)
        compressed_attention_mask = torch.ones(compressed_features.shape[0], dtype=torch.long)

        sample = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "compressed_features": compressed_features,
            "compressed_attention_mask": compressed_attention_mask,
            "retain_ratio": record.get("retain_ratio"),
            "target_keep_tokens": record.get("target_keep_tokens"),
            "drop_tokens": record.get("drop_tokens"),
        }

        if record.get("source_encoding") == "csr_binary_patch_i16":
            sample["source_indices"] = _read_binary_tensor(shard_root, record["source_indices"])
            sample["source_offsets"] = _read_binary_tensor(shard_root, record["source_offsets"])
            sample["grid_shape"] = torch.tensor(record.get("grid_shape", [24, 24]), dtype=torch.long)

        return sample
