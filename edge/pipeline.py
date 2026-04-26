import argparse
import json
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

DEFAULT_RETAIN_RATIOS = [0.8, 0.6, 0.4, 0.2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a fixed compressed visual-feature dataset for Prism.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the original JSON dataset.")
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
    return parser.parse_args()


def load_dataset(data_path: str) -> List[Dict]:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected the dataset JSON to be a list of samples.")
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
    image_file = sample.get("image")
    if not image_file:
        return None
    return os.path.join(image_folder, image_file)


def build_model(clip_model_path: str, device: str):
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = CLIPVisionModel.from_pretrained(clip_model_path).to(device=device, dtype=dtype)
    model.eval()
    apply_patch_clip(model)
    image_processor = CLIPImageProcessor.from_pretrained(clip_model_path)
    return model, image_processor, dtype


def infer_compressed_features(model, image_processor, dtype, device: str, image_path: str, retain_ratio: float, allocator: CNA_Allocator):
    img = Image.open(image_path).convert("RGB")
    inputs = image_processor(images=img, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device=device, dtype=dtype)

    total_tokens = allocator.total_tokens
    target_keep_tokens = int(round(total_tokens * retain_ratio))
    target_keep_tokens = max(1, min(total_tokens, target_keep_tokens))
    drop_tokens = total_tokens - target_keep_tokens

    h_norm = drop_tokens / float(allocator.max_drop)
    r_list = allocator.generate_r_list(h_norm=h_norm, bandwidth_mbps=1.0)
    model.r = r_list

    with torch.no_grad():
        _ = model(pixel_values)
        start_time = time.perf_counter()
        outputs = model(pixel_values)
        inference_ms = (time.perf_counter() - start_time) * 1000

    hidden = outputs.last_hidden_state
    compressed = hidden[:, 1:, :].contiguous().squeeze(0).cpu()
    return compressed, inference_ms, target_keep_tokens, drop_tokens, r_list


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


def count_existing_samples(shard_dir: str) -> int:
    if not os.path.exists(shard_dir):
        return 0
    return sum(1 for n in os.listdir(shard_dir) if n.startswith("sample_") and n.endswith(".pt"))


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Prism] Using device: {device}")

    samples = load_dataset(args.data_path)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    retain_ratios = args.retain_ratios
    if len(retain_ratios) != 4:
        raise ValueError("For now this script expects exactly 4 retain ratios, e.g. 0.8 0.6 0.4 0.2.")

    assignments = make_fixed_assignment(samples, retain_ratios, args.seed)
    model, image_processor, dtype = build_model(args.clip_model_path, device)
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
    }
    for ratio, path in manifest_paths.items():
        if not os.path.exists(path):
            append_jsonl(path, {"meta": {**meta, "retain_ratio": ratio}})

    for p in [bad_images_path, no_image_path]:
        if not os.path.exists(p):
            with open(p, "w", encoding="utf-8") as f:
                f.write("# logged by pipeline.py\n")

    done_sample_ids = {r: load_done_sample_ids(manifest_paths[r]) for r in retain_ratios}
    counters = {r: count_existing_samples(shard_dirs[r]) for r in retain_ratios}

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
            append_jsonl(no_image_path, {"sample_id": idx, "reason": "no image field", "sample": sample.get("id", None)})
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
            compressed_features, inference_ms, target_keep_tokens, drop_tokens, r_list = infer_compressed_features(
                model=model,
                image_processor=image_processor,
                dtype=dtype,
                device=device,
                image_path=image_path,
                retain_ratio=retain_ratio,
                allocator=allocator,
            )
        except Exception as e:
            print(f"[Skip] Failed processing image: {image_path} ({e})")
            skipped_other += 1
            append_jsonl(bad_images_path, {"sample_id": idx, "image_path": image_path, "reason": f"processing error: {e}"})
            continue

        counters[retain_ratio] += 1
        sample_path = os.path.join(shard_dirs[retain_ratio], f"sample_{counters[retain_ratio]:08d}.pt")
        payload = {
            "sample_id": idx,
            "image_id": sample.get("image", f"sample_{idx}"),
            "image_path": image_path,
            "retain_ratio": retain_ratio,
            "target_keep_tokens": target_keep_tokens,
            "drop_tokens": drop_tokens,
            "compression_method": "tome",
            "compression_seed": args.seed,
            "r_list": r_list,
            "compressed_token_count": int(compressed_features.shape[0]),
            "compressed_features": compressed_features,
            "conversations": sample.get("conversations", []),
            "inference_ms": inference_ms,
        }

        torch.save(payload, sample_path)
        append_jsonl(manifest_paths[retain_ratio], {
            "sample_path": sample_path,
            "sample_id": idx,
            "image_id": payload["image_id"],
            "retain_ratio": retain_ratio,
            "target_keep_tokens": target_keep_tokens,
            "drop_tokens": drop_tokens,
            "compressed_token_count": int(compressed_features.shape[0]),
            "inference_ms": inference_ms,
        })

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
