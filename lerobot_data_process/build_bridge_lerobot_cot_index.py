from __future__ import annotations

import argparse
import json
import shutil
import sys
from functools import partial
from pathlib import Path
from typing import Any

from starVLA.dataloader.bridge_cot_sidecar import BridgeCotSourceDbReader, BridgeCotSidecarReader


DEFAULT_RLDS_ROOT = Path("/inspire/hdd/global_user/gongjingjing-25039/zhdai/openpi_dataset_bridge")
DEFAULT_SOURCE_DB_DIR = Path("/inspire/hdd/global_user/gongjingjing-25039/zhdai/datasets/bridge_orig_openpi_cot_source_db")
DEFAULT_DATASET_DIR = Path("/inspire/hdd/global_user/gongjingjing-25039/zhdai/datasets/bridge_orig_lerobot")
DEFAULT_OUTPUT_DIR = Path("/inspire/hdd/global_user/gongjingjing-25039/zhdai/datasets/bridge_orig_lerobot_cot_index")
ANY4LEROBOT_ROOT = Path("/inspire/hdd/global_user/gongjingjing-25039/zhdai/any4lerobot")
OPENX2LEROBOT_ROOT = ANY4LEROBOT_ROOT / "openx2lerobot"


def _decode_scalar(value: Any) -> Any:
    if hasattr(value, "item") and not isinstance(value, (bytes, str)):
        try:
            value = value.item()
        except ValueError:
            pass
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _import_tf_stack():
    import tensorflow as tf
    import tensorflow_datasets as tfds

    tf.config.set_visible_devices([], "GPU")
    return tf, tfds


def _import_openx_transform():
    for path in (ANY4LEROBOT_ROOT, OPENX2LEROBOT_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    from openx2lerobot.openx_rlds import transform_raw_dataset

    return transform_raw_dataset


def _load_dataset_metadata(dataset_dir: Path) -> tuple[dict[int, dict[str, Any]], dict[int, str], dict[str, Any]]:
    episodes: dict[int, dict[str, Any]] = {}
    with (dataset_dir / "meta/episodes.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            episodes[int(row["episode_index"])] = row

    tasks_by_index: dict[int, str] = {}
    with (dataset_dir / "meta/tasks.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            tasks_by_index[int(row["task_index"])] = str(row.get("task", "") or "")

    with (dataset_dir / "meta/info.json").open("r", encoding="utf-8") as f:
        info = json.load(f)
    return episodes, tasks_by_index, info


def build_dataset_sidecar(
    *,
    rlds_root: Path,
    source_db_dir: Path,
    dataset_dir: Path,
    output_dir: Path,
    splits: list[str],
    validate_against_dataset: bool,
    overwrite: bool,
) -> None:
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "records").mkdir(parents=True, exist_ok=True)

    dataset_name = dataset_dir.name
    episodes_meta, tasks_by_index, dataset_info = _load_dataset_metadata(dataset_dir)
    total_expected_episodes = int(dataset_info.get("total_episodes", -1))

    source_reader = BridgeCotSourceDbReader(source_db_dir)
    tf, tfds = _import_tf_stack()
    transform_raw_dataset = _import_openx_transform()
    builder = tfds.builder("bridge_orig", data_dir=str(rlds_root), version="1.0.0")

    summary = {
        "dataset_dir": str(dataset_dir),
        "source_db_dir": str(source_db_dir),
        "rlds_root": str(rlds_root),
        "dataset_name": dataset_name,
        "splits": list(splits),
        "total_episodes": 0,
        "total_frames": 0,
        "matched_frames": 0,
        "unmatched_frames": 0,
    }

    manifest_path = output_dir / "episode_manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()

    current_episode_index = 0
    with manifest_path.open("w", encoding="utf-8") as manifest_f:
        for split in splits:
            raw_dataset = (
                builder.as_dataset(split=split)
                .map(partial(transform_raw_dataset, dataset_name="bridge_orig"), num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
            )

            for episode in raw_dataset.as_numpy_iterator():
                if current_episode_index not in episodes_meta:
                    raise ValueError(
                        f"Replayed RLDS episode_index={current_episode_index} is missing from `{dataset_dir}`. "
                        "This indicates the LeRobot dataset was not generated with the expected openx_rlds ordering."
                    )

                traj = episode["steps"]
                observations = traj["observation"]
                task = _decode_scalar(traj["task"][0])
                source_file_path = _decode_scalar(episode["episode_metadata"]["file_path"])
                source_episode_id = int(_decode_scalar(episode["episode_metadata"]["episode_id"]))
                frame_count = int(traj["action"].shape[0])

                dataset_episode_row = episodes_meta[current_episode_index]
                expected_length = int(dataset_episode_row["length"])
                if frame_count != expected_length:
                    raise ValueError(
                        f"Frame count mismatch for episode {current_episode_index}: "
                        f"replayed={frame_count}, dataset={expected_length}."
                    )

                expected_tasks = dataset_episode_row.get("tasks", []) or []
                expected_task = str(expected_tasks[0] or "") if expected_tasks else ""
                if not expected_task:
                    task_index = dataset_episode_row.get("task_index", None)
                    if task_index is not None:
                        expected_task = tasks_by_index.get(int(task_index), "")
                if task != expected_task:
                    raise ValueError(
                        f"Task mismatch for episode {current_episode_index}: replay={task!r}, dataset={expected_task!r}."
                    )

                matched_frame_count = 0
                record_path = output_dir / "records" / f"episode_{current_episode_index:06d}.jsonl"
                with record_path.open("w", encoding="utf-8") as record_f:
                    for frame_index in range(frame_count):
                        source_record = source_reader.get_source_record(source_file_path, source_episode_id, frame_index)
                        has_cot = bool(source_record.get("has_cot", False))
                        if has_cot:
                            matched_frame_count += 1
                        row = {
                            "episode_index": current_episode_index,
                            "frame_index": frame_index,
                            "has_cot": has_cot,
                            "reasoning_text": str(source_record.get("reasoning_text", "") or ""),
                            "source_file_path": source_file_path,
                            "source_episode_id": source_episode_id,
                            "source_frame_index": frame_index,
                        }
                        for field_name in (
                            "task",
                            "plan",
                            "bboxes",
                            "subtask_reason",
                            "subtask",
                            "move_reason",
                            "move",
                            "gripper",
                        ):
                            row[field_name] = str(source_record.get(field_name, "") or "")
                        record_f.write(json.dumps(row, ensure_ascii=False) + "\n")

                manifest_row = {
                    "episode_index": current_episode_index,
                    "split": "valid" if split == "val" else split,
                    "length": frame_count,
                    "task": task,
                    "matched_frame_count": matched_frame_count,
                    "source_file_path": source_file_path,
                    "source_episode_id": source_episode_id,
                }
                manifest_f.write(json.dumps(manifest_row, ensure_ascii=False) + "\n")

                summary["total_episodes"] += 1
                summary["total_frames"] += frame_count
                summary["matched_frames"] += matched_frame_count
                summary["unmatched_frames"] += frame_count - matched_frame_count
                current_episode_index += 1

    if total_expected_episodes >= 0 and current_episode_index != total_expected_episodes:
        raise ValueError(
            f"Replayed total episodes mismatch for `{dataset_dir}`: "
            f"replayed={current_episode_index}, dataset={total_expected_episodes}."
        )

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if validate_against_dataset:
        BridgeCotSidecarReader(output_dir).validate_against_dataset(dataset_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a LeRobot-aligned Bridge CoT sidecar from the OpenPI source DB.")
    parser.add_argument("--rlds-root", type=Path, default=DEFAULT_RLDS_ROOT)
    parser.add_argument("--source-db-dir", type=Path, default=DEFAULT_SOURCE_DB_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--validate-against-dataset", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    build_dataset_sidecar(
        rlds_root=args.rlds_root,
        source_db_dir=args.source_db_dir,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        splits=list(args.splits),
        validate_against_dataset=bool(args.validate_against_dataset),
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    main()
