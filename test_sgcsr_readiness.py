from __future__ import annotations

import argparse
import json

import torch

from mm.semantic_reconstructor import (
    SourceGuidedCompactSemanticReconstructor,
    compact_feature_distillation_loss,
    pool_teacher_visual_tokens,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity check for Source-Guided Compact Semantic Reconstruction.")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_tokens", type=int, default=230, help="Padded compressed token length N.")
    parser.add_argument("--num_queries", type=int, default=144, help="Compact reconstructed token count K.")
    parser.add_argument("--dim", type=int, default=4096, help="LLaVA projected visual embedding dimension.")
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim_head", type=int, default=64)
    parser.add_argument("--local_topk", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = SourceGuidedCompactSemanticReconstructor(
        dim=args.dim,
        num_queries=args.num_queries,
        depth=args.depth,
        heads=args.heads,
        dim_head=args.dim_head,
        local_topk=args.local_topk,
    ).to(device)
    model.eval()

    # Fake projected compressed visual embeddings. In real training these come
    # from: compressed CLIP features [B,N,1024] -> frozen LLaVA projector.
    visual_embeddings = torch.randn(args.batch_size, args.num_tokens, args.dim, device=device)

    # Build a slightly padded batch to verify attention_mask logic.
    attention_mask = torch.ones(args.batch_size, args.num_tokens, dtype=torch.long, device=device)
    if args.batch_size > 1 and args.num_tokens > 8:
        attention_mask[1, -8:] = 0

    token_centers = torch.rand(args.batch_size, args.num_tokens, 2, device=device)
    token_sizes = torch.full((args.batch_size, args.num_tokens), 1.0 / 576.0, device=device)
    retain_ratio = torch.full((args.batch_size,), 0.4, device=device)

    with torch.no_grad():
        reconstructed = model(
            visual_embeddings=visual_embeddings,
            token_centers=token_centers,
            token_sizes=token_sizes,
            retain_ratio=retain_ratio,
            attention_mask=attention_mask,
        )

        # Fake no-ToMe teacher projected visual embeddings [B, 576, D].
        teacher_tokens = torch.randn(args.batch_size, 576, args.dim, device=device)
        teacher_compact = pool_teacher_visual_tokens(teacher_tokens, output_grid=model.grid_size)
        loss = compact_feature_distillation_loss(reconstructed, teacher_compact)

    if reconstructed.shape != (args.batch_size, args.num_queries, args.dim):
        raise RuntimeError(f"Unexpected reconstructed shape: {tuple(reconstructed.shape)}")
    if teacher_compact.shape != reconstructed.shape:
        raise RuntimeError(
            f"Teacher compact shape mismatch: teacher={tuple(teacher_compact.shape)}, "
            f"student={tuple(reconstructed.shape)}"
        )
    if not torch.isfinite(reconstructed).all():
        raise RuntimeError("Reconstructed tokens contain NaN or Inf.")
    if not torch.isfinite(loss):
        raise RuntimeError("Distillation loss is NaN or Inf.")

    summary = {
        "device": str(device),
        "input_shape": list(visual_embeddings.shape),
        "attention_mask_shape": list(attention_mask.shape),
        "query_grid": list(model.grid_size),
        "output_shape": list(reconstructed.shape),
        "teacher_compact_shape": list(teacher_compact.shape),
        "distill_loss": float(loss.item()),
    }
    print("[OK] SGCSR forward and compact teacher pooling succeeded.")
    print("[SUMMARY]", json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
