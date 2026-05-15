from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from .perceiver import PerceiverResampler
from .projector import build_projector
from .source_packer import SourceAwareTokenPackerLite
from .vision import VisionSpec, VisionTowerWrapper
if TYPE_CHECKING:
    from .builder import MMConfig
class PrismMultiModalModel(nn.Module):
    def __init__(self, config: "MMConfig"):
        super().__init__()
        self.config = config
        self.llm = AutoModelForCausalLM.from_pretrained(config.llm_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_name_or_path, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vision = VisionTowerWrapper(VisionSpec(config.vision_name_or_path, use_tome=config.use_tome))
        vision_dim = config.vision_hidden_size or getattr(self.vision.model.config, "hidden_size", None)
        if vision_dim is None:
            raise ValueError("无法推断 vision hidden size。")
        self.projector_type = config.projector_type
        if config.projector_type == "perceiver":
            self.projector = PerceiverResampler(
                dim=vision_dim,
                depth=max(2, config.mlp_depth),
                dim_head=64,
                heads=8,
                num_latents=config.num_queries,
                max_num_media=16,
                max_num_frames=None,
                ff_mult=4,
                out_dim=config.hidden_size,
            )
        elif config.projector_type == "source_packer":
            self.projector = SourceAwareTokenPackerLite(
                in_dim=vision_dim,
                out_dim=config.hidden_size,
                num_queries=config.num_queries,
                depth=max(1, config.mlp_depth),
                dim_head=64,
                heads=8,
                ff_mult=4,
                local_topk=8,
            )
        else:
            self.projector = build_projector(
                config.projector_type,
                in_dim=vision_dim,
                out_dim=config.hidden_size,
                depth=config.mlp_depth,
            )
        if config.freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False
        if config.freeze_vision:
            for p in self.vision.parameters():
                p.requires_grad = False
    @property
    def device(self):
        return next(self.parameters()).device
    def _projector_dtype(self) -> torch.dtype:
        return next(self.projector.parameters()).dtype
    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.vision(pixel_values)
        feats = feats[:, 1:, :]  # 去掉 CLS token
        feats = feats.to(dtype=self._projector_dtype())
        return self.projector(feats)
    def _merge_text_and_image_embeddings(
        self,
        text_embeds: torch.Tensor,
        image_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        image_attention_mask: Optional[torch.Tensor] = None,
    ):
        """把图像 token 前置到文本 token 前面，并同步对齐 mask 和 labels。"""
        bsz, img_len, _ = image_embeds.shape
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        if attention_mask is not None:
            if image_attention_mask is None:
                img_mask = torch.ones((bsz, img_len), dtype=attention_mask.dtype, device=attention_mask.device)
            else:
                img_mask = image_attention_mask.to(device=attention_mask.device, dtype=attention_mask.dtype)
                if img_mask.ndim == 1:
                    img_mask = img_mask.unsqueeze(0)
                if img_mask.shape != (bsz, img_len):
                    raise ValueError(
                        f"image_attention_mask shape must be {(bsz, img_len)}, got {tuple(img_mask.shape)}"
                    )
            attention_mask = torch.cat([img_mask, attention_mask], dim=1)
        if labels is not None:
            img_labels = torch.full((bsz, img_len), -100, dtype=labels.dtype, device=labels.device)
            labels = torch.cat([img_labels, labels], dim=1)
        return inputs_embeds, attention_mask, labels
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        compressed_features: Optional[torch.Tensor] = None,
        compressed_attention_mask: Optional[torch.Tensor] = None,
        token_centers: Optional[torch.Tensor] = None,
        token_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        projector_dtype = self._projector_dtype()
        if compressed_features is not None:
            image_attention_mask = None
            image_tokens = compressed_features.to(text_embeds.device, dtype=projector_dtype)
            if image_tokens.ndim == 2:
                image_tokens = image_tokens.unsqueeze(0)
            if compressed_attention_mask is not None:
                compressed_attention_mask = compressed_attention_mask.to(text_embeds.device)
                if compressed_attention_mask.ndim == 1:
                    compressed_attention_mask = compressed_attention_mask.unsqueeze(0)
            if self.projector_type == "perceiver":
                image_embeds = self.projector(image_tokens, attention_mask=compressed_attention_mask)
            elif self.projector_type == "source_packer":
                if token_centers is not None:
                    token_centers = token_centers.to(text_embeds.device)
                    if token_centers.ndim == 2:
                        token_centers = token_centers.unsqueeze(0)
                if token_sizes is not None:
                    token_sizes = token_sizes.to(text_embeds.device)
                    if token_sizes.ndim == 1:
                        token_sizes = token_sizes.unsqueeze(0)
                image_embeds = self.projector(
                    image_tokens,
                    attention_mask=compressed_attention_mask,
                    token_centers=token_centers,
                    token_sizes=token_sizes,
                )
            elif image_tokens.shape[-1] != self.config.hidden_size:
                image_embeds = self.projector(image_tokens)
                image_attention_mask = compressed_attention_mask
            else:
                image_embeds = image_tokens
                image_attention_mask = compressed_attention_mask
        elif pixel_values is not None:
            image_embeds = self.encode_images(pixel_values.to(text_embeds.device))
            image_attention_mask = None
        else:
            image_embeds = None
            image_attention_mask = None
        if image_embeds is not None:
            image_embeds = image_embeds.to(dtype=text_embeds.dtype)
            inputs_embeds, attention_mask, labels = self._merge_text_and_image_embeddings(
                text_embeds=text_embeds,
                image_embeds=image_embeds,
                attention_mask=attention_mask,
                labels=labels,
                image_attention_mask=image_attention_mask,
            )
        else:
            inputs_embeds = text_embeds
        return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, **kwargs)

