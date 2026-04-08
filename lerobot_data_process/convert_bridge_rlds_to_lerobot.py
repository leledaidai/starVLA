"""
python -m starVLA.lerobot_data_process.convert_bridge_rlds_to_lerobot --overwrite --video-codec av1 --video-encoder libsvtav1 --camera-workers 2 --video-threads 4 --tf-num-parallel-reads 4 --tf-private-threadpool-size 8 --tf-max-intra-op-parallelism 1
"""


from __future__ import annotations

import argparse
import ast
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Any

from starVLA.lerobot_data_process.cot_utils import build_cot_record
from starVLA.lerobot_data_process.lerobot_v2 import (
    EpisodeMeta,
    bridge_modality_json,
    build_info_json,
    ensure_dir,
    write_json,
    write_jsonl,
)


DEFAULT_RLDS_DATA_DIR = "/inspire/hdd/global_user/gongjingjing-25039/zhdai/openpi_dataset_bridge"
DEFAULT_REASONING_PATH = (
    "/inspire/hdd/global_user/gongjingjing-25039/zhdai/hf_cache/hub/"
    "datasets--Embodied-CoT--embodied_features_bridge/snapshots/"
    "854ee59c7c76868d63fac37c33e0f031ed678014/embodied_features_bridge.json"
)
DEFAULT_OUTPUT_DATASET_DIR = (
    "/inspire/hdd/global_user/gongjingjing-25039/zhdai/datasets/bridge_orig_train_valid_lerobot"
)
DEFAULT_OUTPUT_COT_DIR = (
    "/inspire/hdd/global_user/gongjingjing-25039/zhdai/datasets/bridge_orig_train_valid_cot_index"
)


@dataclass
class ConversionConfig:
    rlds_data_dir: Path
    reasoning_path: Path
    output_dataset_dir: Path
    output_cot_dir: Path
    fps: int = 5
    chunk_size: int = 1000
    overwrite: bool = False
    max_episodes_per_split: int | None = None
    video_codec: str = "av1"
    video_encoder: str | None = None
    video_preset: int = 8
    video_crf: int = 35
    video_threads: int = 4
    camera_workers: int = 2
    tf_num_parallel_reads: int = 4
    tf_private_threadpool_size: int = 8
    tf_max_intra_op_parallelism: int = 1


class BridgeDatasetConverter:
    def __init__(self, config: ConversionConfig):
        self.config = config
        self._task_to_index: OrderedDict[str, int] = OrderedDict()
        self._episodes: list[EpisodeMeta] = []
        self._cot_summary = {
            "total_episodes": 0,
            "total_frames": 0,
            "matched_episodes": 0,
            "unmatched_episodes": 0,
            "matched_frames": 0,
            "unmatched_frames": 0,
        }

    def convert(self) -> None:
        self._prepare_output_dirs()
        reasoning_dataset = self._load_reasoning_dataset()
        tfds = self._import_rlds_stack()

        split_plan = [("train", "train"), ("val", "valid")]
        global_frame_index = 0
        train_episode_count = 0
        valid_episode_count = 0

        for source_split, output_split in split_plan:
            builder = tfds.builder("bridge_orig", data_dir=str(self.config.rlds_data_dir), version="1.0.0")
            dataset = self._build_tf_dataset(builder=builder, split=source_split)

            split_episode_count = 0
            for trajectory in tfds.as_numpy(dataset):
                if self.config.max_episodes_per_split is not None and split_episode_count >= self.config.max_episodes_per_split:
                    break

                episode_index = len(self._episodes)
                episode_payload = self._convert_episode(
                    trajectory=trajectory,
                    episode_index=episode_index,
                    output_split=output_split,
                    global_frame_start=global_frame_index,
                    reasoning_dataset=reasoning_dataset,
                )
                self._episodes.append(episode_payload.meta)
                global_frame_index += episode_payload.frame_count
                split_episode_count += 1

            if output_split == "train":
                train_episode_count = split_episode_count
            else:
                valid_episode_count = split_episode_count

        self._write_metadata(train_episode_count=train_episode_count, valid_episode_count=valid_episode_count)

    def _prepare_output_dirs(self) -> None:
        for path in [self.config.output_dataset_dir, self.config.output_cot_dir]:
            if path.exists():
                if not self.config.overwrite:
                    raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")
                shutil.rmtree(path)
            ensure_dir(path)

        ensure_dir(self.config.output_dataset_dir / "data")
        ensure_dir(self.config.output_dataset_dir / "videos")
        ensure_dir(self.config.output_dataset_dir / "meta")
        ensure_dir(self.config.output_cot_dir / "records")

    def _load_reasoning_dataset(self) -> dict[str, dict[str, Any]]:
        if not self.config.reasoning_path.exists():
            raise FileNotFoundError(f"Reasoning dataset not found: {self.config.reasoning_path}")

        raw_text = self.config.reasoning_path.read_text(encoding="utf-8")
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            # Some local debug artifacts were stored as Python repr instead of strict JSON.
            return ast.literal_eval(raw_text)

    def _import_rlds_stack(self):
        try:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            import tensorflow as tf
            import tensorflow_datasets as tfds
        except ImportError as exc:
            raise ImportError(
                "Bridge RLDS conversion requires tensorflow and tensorflow_datasets. "
                "Use the OpenPI environment or install the missing packages."
            ) from exc

        tf.config.set_visible_devices([], "GPU")
        return tfds

    def _build_tf_dataset(self, *, builder: Any, split: str) -> Any:
        tf = self._import_tensorflow()
        tfds = self._import_tfds()

        read_config_kwargs = {
            "interleave_cycle_length": self.config.tf_num_parallel_reads,
            "num_parallel_calls_for_interleave_files": self.config.tf_num_parallel_reads,
            "num_parallel_calls_for_decode": self.config.tf_num_parallel_reads,
        }
        try:
            read_config = tfds.ReadConfig(**read_config_kwargs)
        except TypeError:
            read_config_kwargs.pop("num_parallel_calls_for_decode", None)
            read_config = tfds.ReadConfig(**read_config_kwargs)

        dataset = builder.as_dataset(split=split, read_config=read_config)
        options = tf.data.Options()
        threading_options = getattr(options, "threading", None)
        if threading_options is not None:
            threading_options.private_threadpool_size = self.config.tf_private_threadpool_size
            threading_options.max_intra_op_parallelism = self.config.tf_max_intra_op_parallelism
        else:
            options.experimental_threading.private_threadpool_size = self.config.tf_private_threadpool_size
            options.experimental_threading.max_intra_op_parallelism = self.config.tf_max_intra_op_parallelism
        return dataset.with_options(options)

    def _convert_episode(
        self,
        *,
        trajectory: dict[str, Any],
        episode_index: int,
        output_split: str,
        global_frame_start: int,
        reasoning_dataset: dict[str, dict[str, Any]],
    ) -> "_EpisodeConversion":
        np = self._import_numpy()
        pa, pq = self._import_arrow()

        episode_metadata = trajectory["episode_metadata"]
        steps = list(trajectory["steps"])
        if not steps:
            raise ValueError("Encountered an empty episode in Bridge RLDS.")

        file_path = self._decode_scalar(episode_metadata["file_path"])
        source_episode_id = int(self._decode_scalar(episode_metadata["episode_id"]))
        instruction = self._decode_scalar(steps[0]["language_instruction"])
        actions = np.asarray([step["action"] for step in steps], dtype=np.float32)
        states_raw = np.asarray([step["observation"]["state"] for step in steps], dtype=np.float32)
        states = self._normalize_states(states_raw, np=np)
        frame_count = int(actions.shape[0])

        episode_reasoning = reasoning_dataset.get(file_path, {}).get(str(source_episode_id))
        cot_rows: list[dict[str, Any]] = []
        matched_frame_count = 0
        for frame_index in range(frame_count):
            cot_record = build_cot_record(
                episode_index=episode_index,
                frame_index=frame_index,
                episode_data=episode_reasoning,
            )
            cot_rows.append(cot_record.to_dict())
            if cot_record.has_cot:
                matched_frame_count += 1

        self._cot_summary["total_episodes"] += 1
        self._cot_summary["total_frames"] += frame_count
        self._cot_summary["matched_frames"] += matched_frame_count
        self._cot_summary["unmatched_frames"] += frame_count - matched_frame_count
        if matched_frame_count > 0:
            self._cot_summary["matched_episodes"] += 1
        else:
            self._cot_summary["unmatched_episodes"] += 1

        task_index = self._get_or_create_task_index(instruction)
        timestamps = np.arange(frame_count, dtype=np.float32) / float(self.config.fps)
        frame_indices = np.arange(frame_count, dtype=np.int64)
        global_indices = np.arange(global_frame_start, global_frame_start + frame_count, dtype=np.int64)

        parquet_row = {
            "observation.state": [row.tolist() for row in states],
            "action": [row.tolist() for row in actions],
            "timestamp": [[float(value)] for value in timestamps.tolist()],
            "frame_index": [[int(value)] for value in frame_indices.tolist()],
            "episode_index": [[episode_index] for _ in range(frame_count)],
            "index": [[int(value)] for value in global_indices.tolist()],
            "task_index": [[task_index] for _ in range(frame_count)],
        }

        parquet_table = pa.table(parquet_row)
        chunk_dir = self.config.output_dataset_dir / "data" / f"chunk-{episode_index // self.config.chunk_size:03d}"
        ensure_dir(chunk_dir)
        parquet_path = chunk_dir / f"episode_{episode_index:06d}.parquet"
        pq.write_table(parquet_table, parquet_path)

        video_dir_chunk = self.config.output_dataset_dir / "videos" / f"chunk-{episode_index // self.config.chunk_size:03d}"
        camera_jobs: list[tuple[tuple[Any, ...], Path]] = []
        for camera_idx in range(4):
            camera_key = f"image_{camera_idx}"
            video_dir = video_dir_chunk / f"observation.images.{camera_key}"
            ensure_dir(video_dir)
            camera_jobs.append(
                (
                    tuple(step["observation"][camera_key] for step in steps),
                    video_dir / f"episode_{episode_index:06d}.mp4",
                )
            )

        max_workers = min(self.config.camera_workers, len(camera_jobs))
        if max_workers <= 1:
            for frames, output_path in camera_jobs:
                self._write_video(frames=frames, output_path=output_path)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self._write_video, frames=frames, output_path=output_path)
                    for frames, output_path in camera_jobs
                ]
                for future in futures:
                    future.result()

        cot_path = self.config.output_cot_dir / "records" / f"episode_{episode_index:06d}.jsonl"
        write_jsonl(cot_path, cot_rows)

        meta = EpisodeMeta(
            episode_index=episode_index,
            split=output_split,
            task=instruction,
            task_index=task_index,
            length=frame_count,
            source_file_path=file_path,
            source_episode_id=source_episode_id,
        )
        return _EpisodeConversion(meta=meta, frame_count=frame_count)

    def _write_metadata(self, *, train_episode_count: int, valid_episode_count: int) -> None:
        episodes_rows = [episode.to_episodes_jsonl() for episode in self._episodes]
        tasks_rows = [
            {"task_index": task_index, "task": task}
            for task, task_index in self._task_to_index.items()
        ]
        manifest_rows = [episode.to_manifest_row() for episode in self._episodes]

        total_episodes = len(self._episodes)
        total_frames = sum(episode.length for episode in self._episodes)
        total_chunks = (total_episodes + self.config.chunk_size - 1) // self.config.chunk_size
        info_json = build_info_json(
            total_episodes=total_episodes,
            total_frames=total_frames,
            total_tasks=len(self._task_to_index),
            total_videos=total_episodes * 4,
            total_chunks=total_chunks,
            chunks_size=self.config.chunk_size,
            fps=self.config.fps,
            video_codec=self.config.video_codec,
            train_episode_count=train_episode_count,
            valid_episode_count=valid_episode_count,
        )

        write_json(self.config.output_dataset_dir / "meta" / "info.json", info_json)
        write_json(self.config.output_dataset_dir / "meta" / "modality.json", bridge_modality_json())
        write_jsonl(self.config.output_dataset_dir / "meta" / "episodes.jsonl", episodes_rows)
        write_jsonl(self.config.output_dataset_dir / "meta" / "tasks.jsonl", tasks_rows)

        write_jsonl(self.config.output_cot_dir / "episode_manifest.jsonl", manifest_rows)
        write_json(
            self.config.output_cot_dir / "summary.json",
            {
                **self._cot_summary,
                "train_episode_count": train_episode_count,
                "valid_episode_count": valid_episode_count,
                "total_tasks": len(self._task_to_index),
            },
        )

    def _write_video(self, *, frames: Any, output_path: Path) -> None:
        np = self._import_numpy()
        try:
            import av
        except ImportError as exc:
            raise ImportError("Video writing requires PyAV (`av`).") from exc

        with av.open(str(output_path), mode="w") as container:
            stream_name = self.config.video_encoder or self.config.video_codec
            stream = container.add_stream(stream_name, rate=self.config.fps)
            stream.width = 256
            stream.height = 256
            stream.pix_fmt = "yuv420p"
            stream.thread_type = "FRAME"
            stream.options = self._video_encoder_options()

            for frame in frames:
                frame_array = self._decode_frame(frame, np=np)
                video_frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
                for packet in stream.encode(video_frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

    @staticmethod
    def _decode_frame(frame: Any, *, np: Any) -> Any:
        if isinstance(frame, bytes):
            try:
                from PIL import Image
            except ImportError as exc:
                raise ImportError("Decoding encoded RLDS images requires Pillow.") from exc

            import io

            return np.asarray(Image.open(io.BytesIO(frame)).convert("RGB"))
        return np.asarray(frame)

    def _video_encoder_options(self) -> dict[str, str]:
        encoder = (self.config.video_encoder or self.config.video_codec).lower()
        options = {"threads": str(max(1, self.config.video_threads))}
        if "aom" in encoder:
            options["cpu-used"] = str(self.config.video_preset)
            options["crf"] = str(self.config.video_crf)
            options["row-mt"] = "1"
        elif "svtav1" in encoder or encoder == "av1":
            options["preset"] = str(self.config.video_preset)
            options["crf"] = str(self.config.video_crf)
        return options

    @staticmethod
    def _normalize_states(states: Any, *, np: Any) -> Any:
        states = np.asarray(states, dtype=np.float32)
        if states.ndim != 2:
            raise ValueError(f"Expected states to be 2D, got shape {states.shape}")
        if states.shape[1] == 8:
            return states
        if states.shape[1] == 7:
            pad_column = np.zeros((states.shape[0], 1), dtype=np.float32)
            return np.concatenate([states[:, :6], pad_column, states[:, 6:7]], axis=1)
        raise ValueError(f"Unexpected Bridge state dimension {states.shape[1]}; expected 7 or 8")

    def _get_or_create_task_index(self, task: str) -> int:
        if task not in self._task_to_index:
            self._task_to_index[task] = len(self._task_to_index)
        return self._task_to_index[task]

    @staticmethod
    def _decode_scalar(value: Any) -> Any:
        if hasattr(value, "item") and not isinstance(value, (bytes, str)):
            try:
                value = value.item()
            except ValueError:
                pass
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return value

    @staticmethod
    def _import_numpy():
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError("Conversion requires numpy.") from exc
        return np

    @staticmethod
    def _import_tensorflow():
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError("Conversion requires tensorflow.") from exc
        return tf

    @staticmethod
    def _import_tfds():
        try:
            import tensorflow_datasets as tfds
        except ImportError as exc:
            raise ImportError("Conversion requires tensorflow_datasets.") from exc
        return tfds

    @staticmethod
    def _import_arrow():
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("Conversion requires pyarrow.") from exc
        return pa, pq


@dataclass(frozen=True)
class _EpisodeConversion:
    meta: EpisodeMeta
    frame_count: int


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert Bridge RLDS train+val to a StarVLA-compatible LeRobot dataset.")
    parser.add_argument("--rlds-data-dir", type=Path, default=Path(DEFAULT_RLDS_DATA_DIR))
    parser.add_argument("--reasoning-path", type=Path, default=Path(DEFAULT_REASONING_PATH))
    parser.add_argument("--output-dataset-dir", type=Path, default=Path(DEFAULT_OUTPUT_DATASET_DIR))
    parser.add_argument("--output-cot-dir", type=Path, default=Path(DEFAULT_OUTPUT_COT_DIR))
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--video-codec", type=str, default="av1")
    parser.add_argument("--video-encoder", type=str, default=None)
    parser.add_argument("--video-preset", type=int, default=8)
    parser.add_argument("--video-crf", type=int, default=35)
    parser.add_argument("--video-threads", type=int, default=4)
    parser.add_argument("--camera-workers", type=int, default=2)
    parser.add_argument("--tf-num-parallel-reads", type=int, default=4)
    parser.add_argument("--tf-private-threadpool-size", type=int, default=8)
    parser.add_argument("--tf-max-intra-op-parallelism", type=int, default=1)
    parser.add_argument("--max-episodes-per-split", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    config = ConversionConfig(
        rlds_data_dir=args.rlds_data_dir,
        reasoning_path=args.reasoning_path,
        output_dataset_dir=args.output_dataset_dir,
        output_cot_dir=args.output_cot_dir,
        fps=args.fps,
        chunk_size=args.chunk_size,
        overwrite=args.overwrite,
        max_episodes_per_split=args.max_episodes_per_split,
        video_codec=args.video_codec,
        video_encoder=args.video_encoder,
        video_preset=args.video_preset,
        video_crf=args.video_crf,
        video_threads=args.video_threads,
        camera_workers=args.camera_workers,
        tf_num_parallel_reads=args.tf_num_parallel_reads,
        tf_private_threadpool_size=args.tf_private_threadpool_size,
        tf_max_intra_op_parallelism=args.tf_max_intra_op_parallelism,
    )
    BridgeDatasetConverter(config).convert()


if __name__ == "__main__":
    main()
