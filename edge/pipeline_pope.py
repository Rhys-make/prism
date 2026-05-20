import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModel

try:
    from edge.cna import CNA_Allocator
    from edge.tome.patch.clip import apply_patch_clip
except ModuleNotFoundError:
    import sys
    ROOT = Path(__file__).resolve().parent.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from edge.cna import CNA_Allocator
    from edge.tome.patch.clip import apply_patch_clip

DEFAULT_RETAIN_RATIOS = [0.2, 0.4, 0.6, 0.8]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed compressed visual-feature shards for POPE.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the POPE JSON dataset.")
    parser.add_argument("--image_folder", type=str, required=True, help="Root folder containing images.")
    parser.add_argument("--clip_model_path", type=str, required=True, help="Local CLIP-ViT-L/14-336 path.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save compressed shards.")
    parser.add_argument("--retain_ratios", type=float, nargs="+", default=DEFAULT_RETAIN_RATIOS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--num_layers", type=int, default=24)
    parser.add_argument("--total_tokens", type=int, default=576)
    parser.add_argument("--max_drop", type=int, default=450)
    parser.add_argument(
        "--vision_feature_layer",
        type=int,
        default=-2,
        help="CLIP hidden-state layer to save. LLaVA-1.5 uses -2 before the projector.",
    )
    parser.add_argument(
        "--feature_storage",
        type=str,
        default="int8",
        choices=["fp", "int8"],
        help="How to store compressed visual features. int8 uses symmetric per-token quantization.",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="bin",
        choices=["bin", "pt"],
        help="bin writes compact shard files plus manifest.jsonl; pt writes one payload file per sample.",
    )
    parser.add_argument(
        "--no_source",
        action="store_true",
        help="Disable ToMe source tracing. V3 spatial reassembly needs source tracing, so keep this off by default.",
    )
    return parser.parse_args()


def load_dataset(data_path: str) -> List[Dict]:
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
        raise ValueError("Expected the POPE dataset JSON to be a list of samples.")
    return data


def make_fixed_assignment(samples: List[Dict], retain_ratios: List[float], seed: int) -> List[float]:
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    assignments = [None] * len(samples)
    group_sizes = [len(samples) // len(retain_ratios)] * len(retain_ratios)
    for i in range(len(samples) % len(retain_ratios)):
        group_sizes[i] += 1
    cursor = 0
    for ratio, size in zip(retain_ratios, group_sizes):
        for idx in indices[cursor: cursor + size]:
            assignments[idx] = ratio
        cursor += size
    if any(v is None for v in assignments):
        raise RuntimeError("Failed to assign retain ratios to all samples.")
    return assignments


def resolve_image_path(image_folder: str, sample: Dict) -> Optional[str]:
    image_file = sample.get("image") or sample.get("image_path") or sample.get("file_name")
    if image_file is None and sample.get("image_id") is not None:
        image_id = sample.get("image_id")
        try:
            image_file = f"COCO_val2014_{int(image_id):012d}.jpg"
        except (TypeError, ValueError):
            image_file = str(image_id)
    if not image_file:
        return None

    image_file = str(image_file)
    candidates = []
    if os.path.isabs(image_file):
        candidates.append(image_file)
    else:
        candidates.append(os.path.join(image_folder, image_file))
        candidates.append(os.path.join(image_folder, os.path.basename(image_file)))

    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def normalize_pope_sample(sample: Dict) -> Dict:
    question = sample.get("text") or sample.get("question") or sample.get("prompt")
    label = sample.get("label") or sample.get("answer")
    if question is None:
        raise ValueError("POPE sample is missing question text.")
    if label is None:
        raise ValueError("POPE sample is missing label/answer.")

    question = str(question).strip()
    label = str(label).strip().lower()

    conversations = [
        {"from": "human", "value": question},
        {"from": "gpt", "value": label},
    ]
    normalized = dict(sample)
    normalized["text"] = question
    normalized["label"] = label
    normalized["conversations"] = conversations
    return normalized


def build_model(clip_model_path: str, device: str, trace_source: bool = True):
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = CLIPVisionModel.from_pretrained(clip_model_path).to(device=device, dtype=dtype)
    model.eval()
    apply_patch_clip(model, trace_source=trace_source)
    image_processor = CLIPImageProcessor.from_pretrained(clip_model_path)
    return model, image_processor, dtype


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


def build_sparse_source_payload(model, actual_keep_tokens: int) -> Dict[str, object]:
    source = getattr(model, "_tome_info", {}).get("source")
    if source is None:
        return {"source_encoding": "none"}

    source = source.detach().cpu()
    source = source[:, 1:, 1:]
    if source.shape[0] != 1:
        raise ValueError(f"pipeline currently expects batch size 1 for source tracing, got {source.shape[0]}")
    source = source.squeeze(0)
    if source.shape[0] != actual_keep_tokens:
        raise ValueError(
            f"source rows do not match compressed tokens: source={source.shape[0]} tokens={actual_keep_tokens}"
        )

    source_bool = source > 0
    indices: List[torch.Tensor] = []
    offsets = [0]
    for row in source_bool:
        idx = torch.nonzero(row, as_tuple=False).flatten().to(torch.int16)
        indices.append(idx)
        offsets.append(offsets[-1] + int(idx.numel()))

    if indices:
        source_indices = torch.cat(indices, dim=0)
    else:
        source_indices = torch.empty(0, dtype=torch.int16)

    return {
        "source_encoding": "csr_binary_patch_i16",
        "source_indices": source_indices,
        "source_offsets": torch.tensor(offsets, dtype=torch.int32),
        "grid_shape": torch.tensor([24, 24], dtype=torch.int16),
    }


def select_vision_features(outputs, feature_layer: int) -> torch.Tensor:
    """Select the CLIP feature layer that will be sent to the cloud projector."""
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise ValueError("CLIP outputs do not include hidden_states. Call the model with output_hidden_states=True.")
    hidden = hidden_states[int(feature_layer)]
    return hidden[:, 1:, :].contiguous()


def infer_compressed_features(
    model,
    image_processor,
    dtype,
    device: str,
    image_path: str,
    retain_ratio: float,
    allocator: CNA_Allocator,
    vision_feature_layer: int,
):
    img = Image.open(image_path).convert("RGB")
    inputs = image_processor(images=img, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device=device, dtype=dtype)

    total_tokens = allocator.total_tokens
    target_keep_tokens = int(round(total_tokens * retain_ratio))
    target_keep_tokens = max(1, min(total_tokens, target_keep_tokens))
    drop_tokens = total_tokens - target_keep_tokens

    r_list = build_exact_r_list(drop_tokens, allocator)
    model.r = r_list

    with torch.no_grad():
        _ = model(pixel_values, output_hidden_states=True)
        start_time = time.perf_counter()
        outputs = model(pixel_values, output_hidden_states=True)
        inference_ms = (time.perf_counter() - start_time) * 1000

    compressed = select_vision_features(outputs, vision_feature_layer).squeeze(0).cpu()
    actual_keep_tokens = int(compressed.shape[0])
    actual_drop_tokens = int(total_tokens - actual_keep_tokens)
    source_payload = build_sparse_source_payload(model, actual_keep_tokens)
    return (
        compressed,
        inference_ms,
        target_keep_tokens,
        drop_tokens,
        actual_keep_tokens,
        actual_drop_tokens,
        r_list,
        source_payload,
    )


def quantize_features_int8_per_token(features: torch.Tensor) -> Dict[str, object]:
    features_f = features.float()
    scale = features_f.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    quantized = torch.round(features_f / scale).clamp(-127, 127).to(torch.int8)
    return {
        "compressed_features_q": quantized,
        "compressed_features_scale": scale.to(torch.float16),
        "compressed_feature_storage": "int8_symmetric_per_token",
    }


def build_feature_payload(features: torch.Tensor, feature_storage: str) -> Dict:
    if feature_storage == "int8":
        return quantize_features_int8_per_token(features)
    return {
        "compressed_features": features,
        "compressed_feature_storage": f"fp_{str(features.dtype).replace('torch.', '')}",
    }


def append_tensor(path: str, tensor: torch.Tensor) -> Dict[str, object]:
    tensor = tensor.detach().cpu().contiguous()
    offset = os.path.getsize(path) if os.path.exists(path) else 0
    with open(path, "ab") as f:
        f.write(tensor.numpy().tobytes())
    return {
        "path": os.path.basename(path),
        "offset": offset,
        "numel": int(tensor.numel()),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "shape": list(tensor.shape),
    }


def write_binary_payload(shard_dir: str, feature_payload: Dict, source_payload: Dict) -> Dict:
    storage = feature_payload["compressed_feature_storage"]
    record: Dict = {"compressed_feature_storage": storage}

    if storage == "int8_symmetric_per_token":
        record["features"] = append_tensor(os.path.join(shard_dir, "features.int8.bin"), feature_payload["compressed_features_q"])
        record["feature_scales"] = append_tensor(
            os.path.join(shard_dir, "feature_scales.fp16.bin"),
            feature_payload["compressed_features_scale"],
        )
    else:
        record["features"] = append_tensor(os.path.join(shard_dir, "features.fp.bin"), feature_payload["compressed_features"])

    record["source_encoding"] = source_payload.get("source_encoding", "none")
    if record["source_encoding"] != "none":
        record["source_indices"] = append_tensor(
            os.path.join(shard_dir, "source_indices.i16.bin"),
            source_payload["source_indices"],
        )
        record["source_offsets"] = append_tensor(
            os.path.join(shard_dir, "source_offsets.i32.bin"),
            source_payload["source_offsets"],
        )
        record["grid_shape"] = [int(x) for x in source_payload["grid_shape"].tolist()]

    return record


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def append_jsonl(path: str, record: Dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_done_sample_ids(manifest_path: str) -> set[int]:
    done: set[int] = set()
    if not os.path.exists(manifest_path):
        return done
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or '"meta"' in line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict) and "sample_id" in rec:
                try:
                    done.add(int(rec["sample_id"]))
                except Exception:
                    continue
    return done


def count_existing_samples(shard_dir: str, manifest_path: str, output_format: str) -> int:
    if not os.path.exists(shard_dir):
        return 0
    if output_format == "pt":
        return sum(1 for n in os.listdir(shard_dir) if n.startswith("sample_") and n.endswith(".pt"))
    return len(load_done_sample_ids(manifest_path))


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Prism] Using device: {device}")

    samples = load_dataset(args.data_path)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    retain_ratios = args.retain_ratios
    if len(retain_ratios) != 4:
        raise ValueError("For now this script expects exactly 4 retain ratios, e.g. 0.2 0.4 0.6 0.8.")

    samples = [normalize_pope_sample(s) for s in samples]
    assignments = make_fixed_assignment(samples, retain_ratios, args.seed)
    model, image_processor, dtype = build_model(args.clip_model_path, device, trace_source=not args.no_source)
    allocator = CNA_Allocator(num_layers=args.num_layers, total_tokens=args.total_tokens, max_drop=args.max_drop)

    ensure_dir(args.output_dir)
    shard_dirs = {r: os.path.join(args.output_dir, f"retain{int(round(r * 100))}") for r in retain_ratios}
    for d in shard_dirs.values():
        ensure_dir(d)

    manifest_paths = {r: os.path.join(shard_dirs[r], "manifest.jsonl") for r in retain_ratios}
    bad_images_path = os.path.join(args.output_dir, "bad_images.txt")
    no_image_path = os.path.join(args.output_dir, "no_image_samples.txt")

    meta = {
        "source_data_path": args.data_path,
        "image_folder": args.image_folder,
        "clip_model_path": args.clip_model_path,
        "retain_ratios": retain_ratios,
        "seed": args.seed,
        "compression_method": "tome",
        "total_tokens": args.total_tokens,
        "max_drop": args.max_drop,
        "dataset": "pope",
        "vision_feature_layer": args.vision_feature_layer,
        "vision_feature_select_strategy": "default",
        "feature_storage": args.feature_storage,
        "output_format": args.output_format,
        "source_tracing": not args.no_source,
    }
    for ratio, path in manifest_paths.items():
        if not os.path.exists(path):
            append_jsonl(path, {"meta": {**meta, "retain_ratio": ratio}})

    for p in [bad_images_path, no_image_path]:
        if not os.path.exists(p):
            with open(p, "w", encoding="utf-8") as f:
                f.write("# logged by pipeline_pope.py\n")

    done_sample_ids = {r: load_done_sample_ids(manifest_paths[r]) for r in retain_ratios}
    counters = {r: count_existing_samples(shard_dirs[r], manifest_paths[r], args.output_format) for r in retain_ratios}

    skipped_missing = 0
    skipped_corrupt = 0
    skipped_other = 0
    skipped_no_image = 0
    resumed = 0

    for idx, (sample, retain_ratio) in enumerate(zip(samples, assignments)):
        image_path = resolve_image_path(args.image_folder, sample)
        if image_path is None:
            print(f"[Skip] No image field for sample_id={idx}")
            skipped_no_image += 1
            append_jsonl(no_image_path, {"sample_id": idx, "reason": "no image field", "sample": sample.get("question_id", sample.get("id", None))})
            continue

        if idx in done_sample_ids[retain_ratio]:
            resumed += 1
            continue

        if not os.path.exists(image_path):
            print(f"[Skip] Missing image: {image_path}")
            skipped_missing += 1
            append_jsonl(bad_images_path, {"sample_id": idx, "image_path": image_path, "reason": "missing file"})
            continue

        try:
            with Image.open(image_path) as im:
                im.verify()
        except Exception as e:
            print(f"[Skip] Corrupt image: {image_path} ({e})")
            skipped_corrupt += 1
            append_jsonl(bad_images_path, {"sample_id": idx, "image_path": image_path, "reason": f"corrupt image: {e}"})
            continue

        try:
            (
                compressed_features,
                inference_ms,
                target_keep_tokens,
                drop_tokens,
                actual_keep_tokens,
                actual_drop_tokens,
                r_list,
                source_payload,
            ) = infer_compressed_features(
                model=model,
                image_processor=image_processor,
                dtype=dtype,
                device=device,
                image_path=image_path,
                retain_ratio=retain_ratio,
                allocator=allocator,
                vision_feature_layer=args.vision_feature_layer,
            )
        except Exception as e:
            print(f"[Skip] Failed processing image: {image_path} ({e})")
            skipped_other += 1
            append_jsonl(bad_images_path, {"sample_id": idx, "image_path": image_path, "reason": f"processing error: {e}"})
            continue

        counters[retain_ratio] += 1
        feature_payload = build_feature_payload(compressed_features, args.feature_storage)
        if args.output_format == "pt":
            sample_path = os.path.join(shard_dirs[retain_ratio], f"sample_{counters[retain_ratio]:08d}.pt")
            storage_payload = {}
        else:
            sample_path = None
            storage_payload = write_binary_payload(shard_dirs[retain_ratio], feature_payload, source_payload)

        payload = {
            "sample_id": idx,
            "image_id": sample.get("image", f"sample_{idx}"),
            "image_path": image_path,
            "retain_ratio": retain_ratio,
            "target_keep_tokens": target_keep_tokens,
            "drop_tokens": drop_tokens,
            "actual_keep_tokens": actual_keep_tokens,
            "actual_drop_tokens": actual_drop_tokens,
            "actual_retain_ratio": actual_keep_tokens / float(args.total_tokens),
            "compression_method": "tome",
            "compression_seed": args.seed,
            "r_list": r_list,
            "compressed_token_count": int(compressed_features.shape[0]),
            "compressed_feature_storage": feature_payload["compressed_feature_storage"],
            "vision_feature_layer": args.vision_feature_layer,
            "vision_feature_select_strategy": "default",
            "conversations": sample.get("conversations", []),
            "inference_ms": inference_ms,
            "pope_sample": {
                "question_id": sample.get("question_id"),
                "text": sample.get("text"),
                "label": sample.get("label"),
            },
        }
        if args.output_format == "pt":
            payload.update(feature_payload)
            payload.update(source_payload)
            torch.save(payload, sample_path)

        manifest_record = {
            "sample_id": idx,
            "image_id": payload["image_id"],
            "image_path": payload["image_path"],
            "retain_ratio": retain_ratio,
            "target_keep_tokens": target_keep_tokens,
            "drop_tokens": drop_tokens,
            "actual_keep_tokens": actual_keep_tokens,
            "actual_drop_tokens": actual_drop_tokens,
            "actual_retain_ratio": actual_keep_tokens / float(args.total_tokens),
            "compressed_token_count": int(compressed_features.shape[0]),
            "compressed_feature_storage": payload["compressed_feature_storage"],
            "vision_feature_layer": args.vision_feature_layer,
            "vision_feature_select_strategy": "default",
            "inference_ms": inference_ms,
            "conversations": payload["conversations"],
            "pope_sample": payload["pope_sample"],
        }
        if sample_path is not None:
            manifest_record["sample_path"] = sample_path
        manifest_record.update(storage_payload)
        append_jsonl(manifest_paths[retain_ratio], manifest_record)

        if actual_keep_tokens != target_keep_tokens:
            print(
                f"[Warn] sample_id={idx} target_keep={target_keep_tokens} "
                f"actual_keep={actual_keep_tokens} r_sum={sum(r_list)}"
            )
        print(f"[{idx + 1}/{len(samples)}] ratio={retain_ratio:.1f} tokens={compressed_features.shape[0]} time={inference_ms:.2f}ms")

    print("Done.")
    print(f"Resumed/skipped existing samples: {resumed}")
    print(f"Skipped no-image samples: {skipped_no_image}")
    print(f"Skipped missing images: {skipped_missing}")
    print(f"Skipped corrupt images: {skipped_corrupt}")
    print(f"Skipped other errors: {skipped_other}")
    for r in retain_ratios:
        print(f"retain {r:.1f}: {counters[r]} samples -> {shard_dirs[r]}")


if __name__ == "__main__":
    main()
