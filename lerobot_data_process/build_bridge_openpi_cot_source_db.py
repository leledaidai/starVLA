from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from starVLA.dataloader.bridge_cot_sidecar import derive_source_db_relpath, load_json_or_literal
from starVLA.dataloader.cot_formatter import SUPPORTED_COT_FIELDS


DEFAULT_REASONING_PATH = Path(
    "/inspire/hdd/global_user/daizihao-CZXS25110035/zhdai/hf_cache/hub/"
    "datasets--Embodied-CoT--embodied_features_bridge/snapshots/"
    "854ee59c7c76868d63fac37c33e0f031ed678014/embodied_features_bridge.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "/inspire/hdd/global_user/daizihao-CZXS25110035/zhdai/datasets/bridge_orig_openpi_cot_source_db"
)

IMPLICIT_COT_TAGS = [
    ("TASK:", "task"),
    ("PLAN:", "plan"),
    ("VISIBLE OBJECTS:", "bboxes"),
    ("SUBTASK REASONING:", "subtask_reason"),
    ("SUBTASK:", "subtask"),
    ("MOVE REASONING:", "move_reason"),
    ("MOVE:", "move"),
    ("GRIPPER POSITION:", "gripper"),
]


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _augment_reasoning_dict(episode_data: dict[str, Any], frame_idx: int, reasoning_dict: dict[str, Any]) -> dict[str, str]:
    augmented = {str(key): _normalize_text(value) for key, value in reasoning_dict.items()}

    features = episode_data.get("features", {}) or {}
    gripper_positions = features.get("gripper_position")
    if gripper_positions is not None and frame_idx < len(gripper_positions):
        future_positions: list[Any] = []
        for offset in range(5):
            pos_idx = frame_idx + offset
            if pos_idx < len(gripper_positions):
                future_positions.extend(gripper_positions[pos_idx])
            elif future_positions:
                future_positions.extend(future_positions[-2:])
        augmented["gripper"] = str(future_positions)
    else:
        augmented.setdefault("gripper", "")

    bboxes = features.get("bboxes")
    if bboxes is not None and frame_idx < len(bboxes):
        boxes = bboxes[frame_idx]
        if boxes:
            augmented["bboxes"] = ", ".join(f"{name} {box}" for _, name, box in boxes)
        else:
            augmented.setdefault("bboxes", "")
    else:
        augmented.setdefault("bboxes", "")

    for _, key in IMPLICIT_COT_TAGS:
        augmented.setdefault(key, "")
    return augmented


def _format_reasoning_dict(reasoning_dict: dict[str, str]) -> str:
    parts: list[str] = []
    for tag, key in IMPLICIT_COT_TAGS:
        value = reasoning_dict.get(key, "")
        if value:
            parts.append(f"{tag} {value}")
    return " ".join(parts)


def build_source_db(reasoning_json: Path, output_dir: Path) -> None:
    raw_data = load_json_or_literal(reasoning_json)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "reasoning_json": str(reasoning_json),
        "total_source_episodes": 0,
        "total_source_frames": 0,
        "matched_source_frames": 0,
        "unmatched_source_frames": 0,
        "splits": {},
    }

    source_manifest_path = output_dir / "source_manifest.jsonl"
    if source_manifest_path.exists():
        source_manifest_path.unlink()

    with source_manifest_path.open("w", encoding="utf-8") as manifest_f:
        for source_file_path in sorted(raw_data.keys()):
            split_name = "train" if "/train/" in source_file_path else ("valid" if "/val/" in source_file_path else "unknown")
            split_summary = summary["splits"].setdefault(
                split_name,
                {
                    "episodes": 0,
                    "frames": 0,
                    "matched_frames": 0,
                    "unmatched_frames": 0,
                },
            )
            for source_episode_id_str in sorted(raw_data[source_file_path].keys(), key=lambda x: int(x)):
                episode_data = raw_data[source_file_path][source_episode_id_str]
                source_episode_id = int(source_episode_id_str)
                reasoning_by_frame = (episode_data.get("reasoning") or {}) if isinstance(episode_data, dict) else {}
                all_frame_indices = sorted({int(key) for key in reasoning_by_frame.keys()})
                shard_relpath = derive_source_db_relpath(source_file_path, source_episode_id)
                shard_path = output_dir / shard_relpath
                shard_path.parent.mkdir(parents=True, exist_ok=True)

                matched_frame_count = 0
                total_frame_count = len(all_frame_indices)
                with shard_path.open("w", encoding="utf-8") as shard_f:
                    for frame_idx in all_frame_indices:
                        raw_reasoning = reasoning_by_frame.get(str(frame_idx))
                        if raw_reasoning:
                            augmented = _augment_reasoning_dict(episode_data, frame_idx, raw_reasoning)
                            reasoning_text = _format_reasoning_dict(augmented)
                            has_cot = bool(reasoning_text)
                            if has_cot:
                                matched_frame_count += 1
                        else:
                            augmented = {key: "" for key in SUPPORTED_COT_FIELDS}
                            reasoning_text = ""
                            has_cot = False

                        row = {
                            "split": split_name,
                            "source_file_path": source_file_path,
                            "source_episode_id": source_episode_id,
                            "source_frame_index": frame_idx,
                            "has_cot": has_cot,
                            "reasoning_text": reasoning_text,
                        }
                        for field_name in SUPPORTED_COT_FIELDS:
                            row[field_name] = str(augmented.get(field_name, "") or "")
                        shard_f.write(json.dumps(row, ensure_ascii=False) + "\n")

                manifest_row = {
                    "split": split_name,
                    "source_file_path": source_file_path,
                    "source_episode_id": source_episode_id,
                    "num_frames": total_frame_count,
                    "matched_frame_count": matched_frame_count,
                    "shard_relpath": shard_relpath,
                }
                manifest_f.write(json.dumps(manifest_row, ensure_ascii=False) + "\n")

                summary["total_source_episodes"] += 1
                summary["total_source_frames"] += total_frame_count
                summary["matched_source_frames"] += matched_frame_count
                summary["unmatched_source_frames"] += total_frame_count - matched_frame_count
                split_summary["episodes"] += 1
                split_summary["frames"] += total_frame_count
                split_summary["matched_frames"] += matched_frame_count
                split_summary["unmatched_frames"] += total_frame_count - matched_frame_count

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a normalized OpenPI Bridge CoT source DB.")
    parser.add_argument("--reasoning-json", type=Path, default=DEFAULT_REASONING_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    build_source_db(args.reasoning_json, args.output_dir)


if __name__ == "__main__":
    main()
