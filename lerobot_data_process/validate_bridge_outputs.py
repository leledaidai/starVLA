from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def validate(dataset_dir: Path, cot_dir: Path) -> None:
    info = json.loads((dataset_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    episodes = _read_jsonl(dataset_dir / "meta" / "episodes.jsonl")
    tasks = _read_jsonl(dataset_dir / "meta" / "tasks.jsonl")
    manifest = _read_jsonl(cot_dir / "episode_manifest.jsonl")
    summary = json.loads((cot_dir / "summary.json").read_text(encoding="utf-8"))

    total_episodes = info["total_episodes"]
    if total_episodes != len(episodes):
        raise ValueError(f"Episode count mismatch: info={total_episodes}, episodes.jsonl={len(episodes)}")
    if total_episodes != len(manifest):
        raise ValueError(f"Manifest count mismatch: info={total_episodes}, manifest={len(manifest)}")

    train_start, train_end = [int(x) for x in info["splits"]["train"].split(":")]
    valid_start, valid_end = [int(x) for x in info["splits"]["valid"].split(":")]
    if train_start != 0:
        raise ValueError(f"Expected train split to start at 0, got {info['splits']['train']}")
    if valid_start != train_end:
        raise ValueError(f"Train/valid split boundary mismatch: {info['splits']}")
    if valid_end != total_episodes:
        raise ValueError(f"Expected valid split to end at total episodes, got {info['splits']['valid']}")

    task_indices = {row["task_index"] for row in tasks}
    manifest_task_indices = {row["task_index"] for row in manifest}
    if not manifest_task_indices.issubset(task_indices):
        missing = sorted(manifest_task_indices - task_indices)
        raise ValueError(f"Task indices referenced by manifest but missing in tasks.jsonl: {missing[:10]}")

    if summary["total_episodes"] != total_episodes:
        raise ValueError("summary.json total_episodes does not match dataset metadata")

    missing_cot_shards = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        cot_path = cot_dir / "records" / f"episode_{episode_index:06d}.jsonl"
        if not cot_path.exists():
            missing_cot_shards.append(episode_index)

    if missing_cot_shards:
        raise ValueError(f"Missing CoT shard files for episodes: {missing_cot_shards[:10]}")

    print("Validation passed.")
    print(f"episodes={total_episodes}, tasks={len(tasks)}, total_frames={info['total_frames']}")
    print(f"train={info['splits']['train']}, valid={info['splits']['valid']}")
    print(
        "cot matched frames="
        f"{summary['matched_frames']} / {summary['total_frames']}, "
        f"matched episodes={summary['matched_episodes']} / {summary['total_episodes']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate converted Bridge LeRobot dataset and CoT sidecar outputs.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/inspire/hdd/global_user/gongjingjing-25039/zhdai/datasets/bridge_orig_train_valid_lerobot"),
    )
    parser.add_argument(
        "--cot-dir",
        type=Path,
        default=Path("/inspire/hdd/global_user/gongjingjing-25039/zhdai/datasets/bridge_orig_train_valid_cot_index"),
    )
    args = parser.parse_args()
    validate(args.dataset_dir, args.cot_dir)


if __name__ == "__main__":
    main()
