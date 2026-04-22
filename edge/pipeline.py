import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModel

from edge.cna import CNA_Allocator
from edge.tome.patch.clip import apply_patch_clip

DEFAULT_RETAIN_RATIOS = [0.8, 0.6, 0.4, 0.2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a fixed compressed visual-feature dataset for Prism."
    )
    parser.add_argument("--data_path", type=str, required=True, help="Path to the original JSON dataset.")
    parser.add_argument("--image_folder", type=str, required=True, help="Root folder containing images.")
    parser.add_argument("--clip_model_path", type=str, required=True, help="Local CLIP-ViT-L/14-336 path.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save compressed shards.")
    parser.add_argument(
        "--retain_ratios",
        type=float,
        nargs="+",
        default=DEFAULT_RETAIN_RATIOS,
        help="Token retention ratios to assign across the dataset.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for fixed sample assignment.")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu; default auto-detect.")
    parser.add_argument("--max_samples", type=int, default=-1, help="Optional limit for debugging.")
    parser.add_argument("--num_layers", type=int, default=24, help="Encoder layers for CNA allocator.")
    parser.add_argument("--total_tokens", type=int, default=576, help="Total visual tokens before compression.")
    parser.add_argument("--max_drop", type=int, default=450, help="Max token drop budget used by CNA.")
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
        for idx in indices[cursor : cursor + size]:
            assignments[idx] = ratio
        cursor += size

    if any(v is None for v in assignments):
        raise RuntimeError("Failed to assign retain ratios to all samples.")
    return assignments


def resolve_image_path(image_folder: str, sample: Dict) -> str:
    image_file = sample.get("image")
    if image_file is None:
        raise ValueError("Sample does not contain an 'image' field.")
    return os.path.join(image_folder, image_file)


def build_model(clip_model_path: str, device: str):
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = CLIPVisionModel.from_pretrained(clip_model_path).to(device=device, dtype=dtype)
    apply_patch_clip(model)
    image_processor = CLIPImageProcessor.from_pretrained(clip_model_path)
    return model, image_processor, dtype


def infer_compressed_features(
    model,
    image_processor,
    dtype,
    device: str,
    image_path: str,
    retain_ratio: float,
    allocator: CNA_Allocator,
):
    img = Image.open(image_path).convert("RGB")
    inputs = image_processor(images=img, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device=device, dtype=dtype)

    # 固定保留率 -> 固定目标 token 数 -> 固定 drop_tokens -> 固定 r_list
    total_tokens = allocator.total_tokens
    target_keep_tokens = int(round(total_tokens * retain_ratio))
    target_keep_tokens = max(1, min(total_tokens, target_keep_tokens))
    drop_tokens = total_tokens - target_keep_tokens

    # 由固定 drop_tokens 反推每层删除量
    h_norm = drop_tokens / float(allocator.max_drop)
    r_list = allocator.generate_r_list(h_norm=h_norm, bandwidth_mbps=1.0)
    model.r = r_list

    with torch.no_grad():
        _ = model(pixel_values)
        start_time = time.perf_counter()
        outputs = model(pixel_values)
        inference_ms = (time.perf_counter() - start_time) * 1000

    hidden = outputs.last_hidden_state
    compressed = hidden[:, 1:, :].contiguous()  # remove CLS token
    return compressed.squeeze(0).cpu(), inference_ms, target_keep_tokens, drop_tokens, r_list


def save_shards(output_dir: str, shards: Dict[float, List[Dict]], meta: Dict):
    os.makedirs(output_dir, exist_ok=True)
    for ratio, samples in shards.items():
        ratio_tag = int(round(ratio * 100))
        payload = {
            "meta": {
                **meta,
                "retain_ratio": ratio,
                "num_samples": len(samples),
            },
            "samples": samples,
        }
        out_path = os.path.join(output_dir, f"retain{ratio_tag}.pt")
        torch.save(payload, out_path)
        print(f"Saved {len(samples)} samples to {out_path}")


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

    allocator = CNA_Allocator(
        num_layers=args.num_layers,
        total_tokens=args.total_tokens,
        max_drop=args.max_drop,
    )

    grouped: Dict[float, List[Dict]] = {ratio: [] for ratio in retain_ratios}
    for idx, (sample, retain_ratio) in enumerate(zip(samples, assignments)):
        image_path = resolve_image_path(args.image_folder, sample)
        if not os.path.exists(image_path):
            print(f"[Skip] Missing image: {image_path}")
            continue

        compressed_features, inference_ms, target_keep_tokens, drop_tokens, r_list = infer_compressed_features(
            model=model,
            image_processor=image_processor,
            dtype=dtype,
            device=device,
            image_path=image_path,
            retain_ratio=retain_ratio,
            allocator=allocator,
        )

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
        grouped[retain_ratio].append(payload)
        print(
            f"[{idx + 1}/{len(samples)}] ratio={retain_ratio:.1f} "
            f"tokens={compressed_features.shape[0]} time={inference_ms:.2f}ms"
        )

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
    save_shards(args.output_dir, grouped, meta)


if __name__ == "__main__":
    main()
