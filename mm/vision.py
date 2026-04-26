from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import CLIPImageProcessor, CLIPVisionModel

from edge.tome.patch.clip import apply_patch_clip


@dataclass
class VisionSpec:
    vision_name_or_path: str
    use_tome: bool = True


class VisionTowerWrapper(torch.nn.Module):
    def __init__(self, spec: VisionSpec, device: Optional[str] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.spec = spec
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if self.device.startswith("cuda") else torch.float32)
        self.model = CLIPVisionModel.from_pretrained(spec.vision_name_or_path).to(self.device, dtype=self.dtype)
        self.processor = CLIPImageProcessor.from_pretrained(spec.vision_name_or_path)
        if spec.use_tome:
            apply_patch_clip(self.model)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values)
        return outputs.last_hidden_state

    def preprocess(self, image):
        return self.processor(images=image, return_tensors="pt")
