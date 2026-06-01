import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image

try:
    from edge.cna import CNA_Allocator
    from edge.pipeline import (
        DEFAULT_RETAIN_RATIOS,
        append_jsonl,
        build_feature_payload,
        build_model,
        count_existing_samples,
        ensure_dir,
        infer_compressed_features,
        load_done_sample_ids,
        make_fixed_assignment,
        write_binary_payload,
    )
except ModuleNotFoundError:
    import sys

    ROOT = Path(__file__).resolve().parent.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from edge.cna import CNA_Allocator
    from edge.pipeline import (
        DEFAULT_RETAIN_RATIOS,
        append_jsonl,
        build_feature_payload,
        build_model,
        count_existing_samples,
        ensure_dir,
        infer_compressed_features,
        load_done_sample_ids,
        make_fixed_assignment,
        write_binary_payload,
    )


COMMON_IMAGE_SUBDIRS = [
    "",
    "coco",
    "coco/train2014",
    "coco/train2017",
    "coco/val2014",
    "gqa",
    "gqa/images",
    "ocr_vqa",
    "ocr_vqa/images",
    "textvqa",
    "textvqa/train_images",
    "textvqa/val_images",
    "vg",
    "vg/VG_100K",
    "vg/VG_100K_2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build fixed compressed visual-feature shards for LLaVA-1.5 stage-2 "
            "instruction data, e.g. llava_v1_5_mix665k.json."
        )
    )
    parser.add_argument("--data_path", type=str, required=True, help="Path to llava_v1_5_mix665k.json or JSONL.")
    parser.add_argument("--image_folder", type=str, required=True, help="Dataset root containing coco/gqa/ocr_vqa/textvqa/vg.")
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
        help="Disable ToMe source tracing. SGCSR needs source tracing, so keep this off for formal experiments.",
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
        raise ValueError("Expected JSON list, JSONL rows, or dict with data/annotations/samples.")
    return data


def validate_llava_stage2_sample(sample: Dict) -> bool:
    image_file = sample.get("image") or sample.get("image_path") or sample.get("file_name")
    conversations = sample.get("conversations")
    if not image_file:
        return False
    if not isinstance(conversations, list) or len(conversations) < 2:
        return False
    return True


def _append_unique(candidates: List[Path], path: Path):
    if path not in candidates:
        candidates.append(path)


def resolve_image_path(image_folder: str, sample: Dict) -> Optional[str]:
    image_file = sample.get("image") or sample.get("image_path") or sample.get("file_name")
    if not image_file:
        return None

    image_file = str(image_file)
    root = Path(image_folder)
    candidates: List[Path] = []
    image_path = Path(image_file)

    if image_path.is_absolute():
        _append_unique(candidates, image_path)
    else:
        _append_unique(candidates, root / image_file)
        _append_unique(candidates, root / image_path.name)
        for subdir in COMMON_IMAGE_SUBDIRS:
            base = root / subdir if subdir else root
            _append_unique(candidates, base / image_file)
            _append_unique(candidates, base / image_path.name)

    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[0]) if candidates else None


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

    assignments = make_fixed_assignment(samples, retain_ratios, args.seed)
    model, image_processor, dtype = build_model(args.clip_model_path, device, trace_source=not args.no_source)
    allocator = CNA_Allocator(num_layers=args.num_layers, total_tokens=args.total_tokens, max_drop=args.max_drop)

    ensure_dir(args.output_dir)
    shard_dirs = {r: os.path.join(args.output_dir, f"retain{int(round(r * 100))}") for r in retain_ratios}
    for shard_dir in shard_dirs.values():
        ensure_dir(shard_dir)

    manifest_paths = {r: os.path.join(shard_dirs[r], "manifest.jsonl") for r in retain_ratios}
    bad_images_path = os.path.join(args.output_dir, "bad_images.txt")
    no_image_path = os.path.join(args.output_dir, "no_image_samples.txt")
    invalid_samples_path = os.path.join(args.output_dir, "invalid_samples.txt")

    meta = {
        "source_data_path": args.data_path,
        "image_folder": args.image_folder,
        "clip_model_path": args.clip_model_path,
        "retain_ratios": retain_ratios,
        "seed": args.seed,
        "compression_method": "tome",
        "total_tokens": args.total_tokens,
        "max_drop": args.max_drop,
        "dataset": "llava_v1_5_mix665k",
        "vision_feature_layer": args.vision_feature_layer,
        "vision_feature_select_strategy": "default",
        "feature_storage": args.feature_storage,
        "output_format": args.output_format,
        "source_tracing": not args.no_source,
    }
    for ratio, path in manifest_paths.items():
        if not os.path.exists(path):
            append_jsonl(path, {"meta": {**meta, "retain_ratio": ratio}})

    for path in [bad_images_path, no_image_path, invalid_samples_path]:
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write("# logged by pipeline_llava_stage2.py\n")

    done_sample_ids = {r: load_done_sample_ids(manifest_paths[r]) for r in retain_ratios}
    counters = {r: count_existing_samples(shard_dirs[r], manifest_paths[r], args.output_format) for r in retain_ratios}

    skipped_invalid = 0
    skipped_no_image = 0
    skipped_missing = 0
    skipped_corrupt = 0
    skipped_other = 0
    resumed = 0

    for idx, (sample, retain_ratio) in enumerate(zip(samples, assignments)):
        if not validate_llava_stage2_sample(sample):
            skipped_invalid += 1
            append_jsonl(
                invalid_samples_path,
                {
                    "sample_id": idx,
                    "reason": "missing image or conversations",
                    "id": sample.get("id"),
                    "image": sample.get("image"),
                },
            )
            continue

        image_path = resolve_image_path(args.image_folder, sample)
        if image_path is None:
            print(f"[Skip] No image field for sample_id={idx}")
            skipped_no_image += 1
            append_jsonl(no_image_path, {"sample_id": idx, "reason": "no image field", "id": sample.get("id")})
            continue

        if idx in done_sample_ids[retain_ratio]:
            resumed += 1
            continue

        if not os.path.exists(image_path):
            print(f"[Skip] Missing image: {image_path}")
            skipped_missing += 1
            append_jsonl(
                bad_images_path,
                {
                    "sample_id": idx,
                    "image_path": image_path,
                    "image": sample.get("image"),
                    "reason": "missing file",
                },
            )
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
            "id": sample.get("id"),
            "image_id": sample.get("image", sample.get("id", f"sample_{idx}")),
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
        }
        if args.output_format == "pt":
            payload.update(feature_payload)
            payload.update(source_payload)
            torch.save(payload, sample_path)

        manifest_record = {
            "sample_id": idx,
            "id": payload["id"],
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
    print(f"Skipped invalid samples: {skipped_invalid}")
    print(f"Skipped no-image samples: {skipped_no_image}")
    print(f"Skipped missing images: {skipped_missing}")
    print(f"Skipped corrupt images: {skipped_corrupt}")
    print(f"Skipped other errors: {skipped_other}")
    for ratio in retain_ratios:
        print(f"retain {ratio:.1f}: {counters[ratio]} samples -> {shard_dirs[ratio]}")


if __name__ == "__main__":
    main()
