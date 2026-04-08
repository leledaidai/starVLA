from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def bridge_features(*, fps: int, video_codec: str) -> dict[str, Any]:
    video_info = {
        "video.fps": float(fps),
        "video.height": 256,
        "video.width": 256,
        "video.channels": 3,
        "video.codec": video_codec,
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "has_audio": False,
    }
    features: dict[str, Any] = {}
    for camera_idx in range(4):
        features[f"observation.images.image_{camera_idx}"] = {
            "dtype": "video",
            "shape": [256, 256, 3],
            "names": ["height", "width", "rgb"],
            "info": video_info,
        }
    features["observation.state"] = {
        "dtype": "float32",
        "shape": [8],
        "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"]},
    }
    features["action"] = {
        "dtype": "float32",
        "shape": [7],
        "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]},
    }
    features["timestamp"] = {"dtype": "float32", "shape": [1], "names": None}
    features["frame_index"] = {"dtype": "int64", "shape": [1], "names": None}
    features["episode_index"] = {"dtype": "int64", "shape": [1], "names": None}
    features["index"] = {"dtype": "int64", "shape": [1], "names": None}
    features["task_index"] = {"dtype": "int64", "shape": [1], "names": None}
    return features


def bridge_modality_json() -> dict[str, Any]:
    return {
        "state": {
            "x": {"start": 0, "end": 1},
            "y": {"start": 1, "end": 2},
            "z": {"start": 2, "end": 3},
            "roll": {"start": 3, "end": 4},
            "pitch": {"start": 4, "end": 5},
            "yaw": {"start": 5, "end": 6},
            "pad": {"start": 6, "end": 7},
            "gripper": {"start": 7, "end": 8},
        },
        "action": {
            "x": {"start": 0, "end": 1},
            "y": {"start": 1, "end": 2},
            "z": {"start": 2, "end": 3},
            "roll": {"start": 3, "end": 4},
            "pitch": {"start": 4, "end": 5},
            "yaw": {"start": 5, "end": 6},
            "gripper": {"start": 6, "end": 7},
        },
        "video": {
            "image_0": {"original_key": "observation.images.image_0"},
            "image_1": {"original_key": "observation.images.image_1"},
            "image_2": {"original_key": "observation.images.image_2"},
            "image_3": {"original_key": "observation.images.image_3"},
        },
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"},
        },
    }


@dataclass(frozen=True)
class EpisodeMeta:
    episode_index: int
    split: str
    task: str
    task_index: int
    length: int
    source_file_path: str
    source_episode_id: int

    def to_episodes_jsonl(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "tasks": [self.task],
            "length": self.length,
        }

    def to_manifest_row(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "split": self.split,
            "task": self.task,
            "task_index": self.task_index,
            "length": self.length,
            "source_file_path": self.source_file_path,
            "source_episode_id": self.source_episode_id,
        }


def build_info_json(
    *,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    total_videos: int,
    total_chunks: int,
    chunks_size: int,
    fps: int,
    video_codec: str,
    train_episode_count: int,
    valid_episode_count: int,
) -> dict[str, Any]:
    return {
        "codebase_version": "v2.0",
        "robot_type": "widowx",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_videos,
        "total_chunks": total_chunks,
        "chunks_size": chunks_size,
        "fps": fps,
        "splits": {
            "train": f"0:{train_episode_count}",
            "valid": f"{train_episode_count}:{train_episode_count + valid_episode_count}",
        },
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": bridge_features(fps=fps, video_codec=video_codec),
    }
