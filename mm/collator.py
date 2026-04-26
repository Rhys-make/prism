from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import torch


@dataclass
class SimpleCollator:
    """把样本列表拼成 batch，并对变长序列做 padding。"""

    pad_token_id: int
    label_pad_id: int = -100

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [x["input_ids"] for x in instances]
        attention_mask = [x["attention_mask"] for x in instances]
        labels = [x.get("labels") for x in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

        if labels[0] is not None:
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.label_pad_id)
        else:
            labels = None

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if labels is not None:
            batch["labels"] = labels

        # 原始图像分支或离线压缩特征分支都会走这里。
        if "pixel_values" in instances[0]:
            pix = [x["pixel_values"] for x in instances]
            if pix[0].ndim == 2:
                batch["pixel_values"] = torch.stack(pix)
            else:
                batch["pixel_values"] = torch.stack([p if isinstance(p, torch.Tensor) else torch.as_tensor(p) for p in pix])

        if "compressed_features" in instances[0]:
            feats = [x["compressed_features"] for x in instances]
            feat_tensors = [f if isinstance(f, torch.Tensor) else torch.as_tensor(f) for f in feats]
            batch["compressed_features"] = torch.nn.utils.rnn.pad_sequence(feat_tensors, batch_first=True, padding_value=0.0)

            if "compressed_attention_mask" in instances[0]:
                am = [x["compressed_attention_mask"] for x in instances]
                am_tensors = [a if isinstance(a, torch.Tensor) else torch.as_tensor(a) for a in am]
                batch["compressed_attention_mask"] = torch.nn.utils.rnn.pad_sequence(am_tensors, batch_first=True, padding_value=0)

            for k in ["retain_ratio", "target_keep_tokens", "drop_tokens"]:
                if k in instances[0] and instances[0][k] is not None:
                    batch[k] = torch.tensor([x[k] for x in instances], dtype=torch.float32 if k == "retain_ratio" else torch.long)

        return batch
