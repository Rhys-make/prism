import os

import torch


SOURCE_PATH = "/data/wangwenbin/sunxuze/prism/prism_adp_epoch_1.pth"
TARGET_PATH = "/data/wangwenbin/sunxuze/prism/LLaVA/checkpoints/prism-stage1/mm_projector.bin"
PREFIX = "model.mm_projector."


def main():
    state = torch.load(SOURCE_PATH, map_location="cpu")
    state_dict = state.get("state_dict", state)

    converted_state_dict = {}
    for key, value in state_dict.items():
        new_key = key if key.startswith(PREFIX) else f"{PREFIX}{key}"
        converted_state_dict[new_key] = value

    os.makedirs(os.path.dirname(TARGET_PATH), exist_ok=True)
    torch.save(converted_state_dict, TARGET_PATH)

    print(f"Converted {len(converted_state_dict)} parameters.")
    print(f"Saved projector weights to: {TARGET_PATH}")


if __name__ == "__main__":
    main()
