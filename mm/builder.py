from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from .modeling_mm import PrismMultiModalModel

ProjectorType = Literal["linear", "mlp", "adp"]


@dataclass
class MMConfig:
    llm_name_or_path: str
    vision_name_or_path: str
    projector_type: ProjectorType = "linear"
    hidden_size: int = 2048
    vision_hidden_size: Optional[int] = None
    num_queries: int = 128
    mlp_depth: int = 2
    freeze_llm: bool = True
    freeze_vision: bool = True
    use_tome: bool = True


def build_model(config: MMConfig) -> PrismMultiModalModel:
    """Build the TinyLlama multimodal skeleton."""
    return PrismMultiModalModel(config)
