from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mm.builder import MMConfig, build_model
from mm.collator import SimpleCollator
from mm.dataset import CompressedFeatureDataset


def find_pt_files(data_path: Path) -> list[Path]:
    if data_path.is_file() and data_path.suffix == ".pt":
        return [data_path]
    if data_path.is_dir():
        return [p for p in data_path.rglob("*.pt") if p.name not in {"best.pt", "last.pt"}]
    return []


def build_fake_batch(model, projector_type: str, device: torch.device) -> dict[str, torch.Tensor]:
    """用随机张量构造一个最小可运行 batch，用于纯代码级检查。"""
    tokenizer = model.tokenizer

    # 构造一段最短的文本输入，确保语言侧能跑通。
    text = "Hello"
    tokens = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    input_ids = tokens.input_ids.to(device)
    attention_mask = tokens.attention_mask.to(device)
    labels = input_ids.clone()

    # 构造压缩视觉特征。
    # 这里不依赖真实 .pt 文件，只是验证模型 forward 和维度对齐是否正常。
    if projector_type == "perceiver":
        # perceiver 会读取变长 token 并输出固定 latent，所以这里给一个变长序列。
        compressed_features = torch.randn(1, 32, 1024, device=device)
        compressed_attention_mask = torch.ones(1, 32, dtype=torch.long, device=device)
    else:
        compressed_features = torch.randn(1, 32, 1024, device=device)
        compressed_attention_mask = torch.ones(1, 32, dtype=torch.long, device=device)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "compressed_features": compressed_features,
        "compressed_attention_mask": compressed_attention_mask,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Sanity check for stage1 training readiness.")
    parser.add_argument("--llm_name_or_path", type=str, required=True)
    parser.add_argument("--vision_name_or_path", type=str, required=True)
    parser.add_argument("--projector_type", type=str, default="linear", choices=["linear", "mlp", "perceiver"])
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--num_queries", type=int, default=128)
    parser.add_argument("--mlp_depth", type=int, default=2)
    parser.add_argument("--use_tome", action="store_true", default=True)
    parser.add_argument("--freeze_llm", action="store_true", default=True)
    parser.add_argument("--freeze_vision", action="store_true", default=True)
    parser.add_argument("--skip_data_check", action="store_true", help="跳过 .pt 数据存在性检查，只做模型和假 batch 检查。")
    parser.add_argument("--fake_batch_only", action="store_true", help="强制只使用随机假 batch，不读取真实 .pt。")
    args = parser.parse_args()

    if not args.skip_data_check and not args.fake_batch_only:
        if args.data_path is None:
            print("[FAIL] 你没有提供 data_path，也没有启用 skip_data_check / fake_batch_only")
            return 1

        data_path = Path(args.data_path)
        pt_files = find_pt_files(data_path)
        print(f"[OK] data_path: {data_path}")
        print(f"[OK] found {len(pt_files)} pt files")
        if not pt_files:
            print("[FAIL] no .pt files found")
            return 1
    else:
        print("[INFO] skip_data_check / fake_batch_only 已启用，将不检查真实 .pt 文件")

    config = MMConfig(
        llm_name_or_path=args.llm_name_or_path,
        vision_name_or_path=args.vision_name_or_path,
        projector_type=args.projector_type,
        num_queries=args.num_queries,
        mlp_depth=args.mlp_depth,
        freeze_llm=args.freeze_llm,
        freeze_vision=args.freeze_vision,
        use_tome=args.use_tome,
    )

    print("[INFO] building model...")
    model = build_model(config)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 先做一个纯代码级的假 batch forward，确保模型可以正常跑通。
    batch = build_fake_batch(model, args.projector_type, device)
    print("[OK] fake batch shapes:")
    for k, v in batch.items():
        print(f"    - {k}: shape={tuple(v.shape)} dtype={v.dtype}")

    with torch.no_grad():
        out = model(**batch)
    loss = getattr(out, "loss", None)
    print(f"[OK] forward pass succeeded, loss={float(loss):.4f}" if loss is not None else "[OK] forward pass succeeded")

    summary = {
        "projector_type": args.projector_type,
        "skip_data_check": args.skip_data_check,
        "fake_batch_only": args.fake_batch_only,
        "device": str(device),
        "batch_shapes": {k: list(v.shape) for k, v in batch.items()},
    }
    print("[SUMMARY]", json.dumps(summary, ensure_ascii=False))
    print("[READY] Code path looks good. When .pt files are available, you can start stage1 training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
