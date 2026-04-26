from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .adp import SemanticResampler
from .builder import MMConfig
from .projector import build_projector
from .vision import VisionSpec, VisionTowerWrapper


class PrismMultiModalModel(nn.Module):
    def __init__(self, config: MMConfig):
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
        if config.projector_type == "adp":
            self.projector = SemanticResampler(
                in_dim=vision_dim,
                hidden_size=config.hidden_size,
                num_queries=config.num_queries,
                num_heads=8,
                depth=max(2, config.mlp_depth),
                dropout=0.0,
                use_input_skip=True,
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

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.vision(pixel_values)
        feats = feats[:, 1:, :]  # 去掉 CLS token
        return self.projector(feats)

    def _merge_text_and_image_embeddings(
        self,
        text_embeds: torch.Tensor,
        image_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
    ):
        """把图像 token 前置到文本 token 前面，并同步对齐 mask 和 labels。"""
        bsz, img_len, hidden = image_embeds.shape
        _, txt_len, _ = text_embeds.shape

        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)

        if attention_mask is not None:
            img_mask = torch.ones((bsz, img_len), dtype=attention_mask.dtype, device=attention_mask.device)
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
        **kwargs,
    ):
        text_embeds = self.llm.get_input_embeddings()(input_ids)

        if compressed_features is not None:
            image_tokens = compressed_features.to(text_embeds.device)
            if image_tokens.ndim == 2:
                image_tokens = image_tokens.unsqueeze(0)

            if compressed_attention_mask is not None:
                compressed_attention_mask = compressed_attention_mask.to(text_embeds.device)
                if compressed_attention_mask.ndim == 1:
                    compressed_attention_mask = compressed_attention_mask.unsqueeze(0)
            # 如果压缩后的特征仍然处于 vision 维度，就先投影到 LLM hidden size。
            if image_tokens.shape[-1] != self.config.hidden_size:
                if self.projector_type == "adp":
                    image_embeds = self.projector(image_tokens, attention_mask=compressed_attention_mask)
                else:
                    image_embeds = self.projector(image_tokens)
            else:
                image_embeds = image_tokens
        elif pixel_values is not None:
            image_embeds = self.encode_images(pixel_values.to(text_embeds.device))
        else:
            image_embeds = None

        if image_embeds is not None:
            inputs_embeds, attention_mask, labels = self._merge_text_and_image_embeddings(
                text_embeds=text_embeds,
                image_embeds=image_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )
        else:
            inputs_embeds = text_embeds

        return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, **kwargs)
