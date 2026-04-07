#!/usr/bin/env python3
"""Download HuggingFace models."""
from pathlib import Path
from huggingface_hub import snapshot_download

# REPOS = [
#     "StarVLA/Qwen-FAST-Bridge-RT-1",
#     "StarVLA/Qwen-OFT-Bridge-RT-1",
#     "StarVLA/Qwen-GR00T-Bridge-RT-1",
#     "StarVLA/Qwen-PI-Bridge-RT-1",
#     "StarVLA/Qwen-GR00T-Bridge",
#     "StarVLA/Qwen3VL-OFT-Bridge-RT-1",
# ]
REPOS = [
    "StarVLA/Qwen-OFT-Bridge-RT-1",
]
BASE_DIR = Path("/inspire/hdd/global_user/gongjingjing-25039/zhdai/starVLA/playground/Pretrained_models")

for repo_id in REPOS:
    model_name = repo_id.split("/")[-1]
    print(f"\nDownloading: {repo_id}")
    try:
        local_path = snapshot_download(
            repo_id=repo_id,
            local_dir=BASE_DIR / model_name,
            local_dir_use_symlinks=False
        )
        print(f"✓ Downloaded to: {local_path}")
    except Exception as e:
        print(f"✗ Failed: {e}")
