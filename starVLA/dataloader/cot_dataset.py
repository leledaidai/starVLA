from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoProcessor

from starVLA.dataloader.bridge_cot_sidecar import BridgeCotSidecarReader, derive_cot_index_path, load_json_or_literal
from starVLA.dataloader.cot_formatter import (
    FIELD_TO_TAG,
    build_prefix_message,
    build_student_message,
    build_teacher_message,
    build_visible_cot_text_with_spans,
    parse_cot_fields,
    parse_thinking_tokens,
)
from starVLA.dataloader.gr00t_lerobot.datasets import (
    LE_ROBOT_EPISODE_FILENAME,
    LE_ROBOT_MODALITY_FILENAME,
    LE_ROBOT3_EPISODE_FILENAME,
    EmbodimentTag,
    LeRobotMixtureDataset,
    LeRobotSingleDataset,
    ModalityConfig,
    _build_stats_cache_config,
    _load_or_compute_statistics,
    _load_stats_cache,
    _normalize_action_mode,
    _normalize_action_mode_apply_keys,
    _normalize_action_mode_state_map,
)
from starVLA.dataloader.gr00t_lerobot.registry import (
    DATASET_NAMED_MIXTURES,
    ROBOT_TYPE_CONFIG_MAP,
    ROBOT_TYPE_TO_EMBODIMENT_TAG,
    EmbodimentTag as RegistryEmbodimentTag,
)
from starVLA.dataloader.gr00t_lerobot.schema import (
    DatasetMetadata,
    DatasetStatisticalValues,
    LeRobotModalityMetadata,
    LeRobotStateActionMetadata,
)

IGNORE_INDEX = -100
_LRU_EPISODE_CACHE_SIZE = 16


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _cot_losses_enabled(cot_cfg: Any) -> bool:
    enable_distill = bool(_cfg_get(cot_cfg, "enable_slot_distill_loss", True)) or bool(
        _cfg_get(cot_cfg, "enable_pool_distill_loss", True)
    )
    flags = (
        "enable_teacher_cot_loss",
        "enable_teacher_action_loss",
        "enable_student_action_loss",
        "enable_decoder_loss",
    )
    return enable_distill or any(bool(_cfg_get(cot_cfg, flag, False)) for flag in flags)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CotDatasetConfig:
    field_names: list[str]
    max_cot_length: int
    teacher_max_prompt_length: int
    student_max_prompt_length: int
    enable_teacher_cot_loss: bool
    enable_teacher_action_loss: bool
    enable_student_action_loss: bool
    enable_decoder_loss: bool
    enable_slot_distill_loss: bool
    enable_pool_distill_loss: bool
    thinking_token: str
    start_thinking_token: str
    end_thinking_token: str

    @property
    def num_latent_tokens(self) -> int:
        return len(self.field_names)

    @property
    def needs_teacher_inputs(self) -> bool:
        return self.enable_teacher_cot_loss or self.enable_teacher_action_loss or self.needs_distill_targets

    @property
    def needs_student_inputs(self) -> bool:
        return self.enable_student_action_loss or self.enable_decoder_loss or self.needs_distill_targets

    @property
    def needs_decoder_targets(self) -> bool:
        return self.enable_decoder_loss

    @property
    def needs_distill_targets(self) -> bool:
        return self.enable_slot_distill_loss or self.enable_pool_distill_loss

    @classmethod
    def from_cfg(cls, cot_cfg: Any) -> "CotDatasetConfig":
        field_specs = parse_cot_fields(cot_cfg)
        if not field_specs:
            raise ValueError("`cot.fields` must be non-empty when any CoT loss is enabled.")
        max_cot_length = int(_cfg_get(cot_cfg, "max_cot_length", 0) or 0)
        teacher_max_prompt_length = int(_cfg_get(cot_cfg, "teacher_max_prompt_length", 0) or 0)
        student_max_prompt_length = int(_cfg_get(cot_cfg, "student_max_prompt_length", 0) or 0)
        if max_cot_length <= 0:
            raise ValueError("`cot.max_cot_length` must be a positive integer.")
        if teacher_max_prompt_length <= 0:
            raise ValueError("`cot.teacher_max_prompt_length` must be a positive integer.")
        if student_max_prompt_length <= 0:
            raise ValueError("`cot.student_max_prompt_length` must be a positive integer.")
        return cls(
            field_names=[field.name for field in field_specs],
            max_cot_length=max_cot_length,
            teacher_max_prompt_length=teacher_max_prompt_length,
            student_max_prompt_length=student_max_prompt_length,
            enable_teacher_cot_loss=bool(_cfg_get(cot_cfg, "enable_teacher_cot_loss", False)),
            enable_teacher_action_loss=bool(_cfg_get(cot_cfg, "enable_teacher_action_loss", False)),
            enable_student_action_loss=bool(_cfg_get(cot_cfg, "enable_student_action_loss", False)),
            enable_decoder_loss=bool(_cfg_get(cot_cfg, "enable_decoder_loss", False)),
            enable_slot_distill_loss=bool(_cfg_get(cot_cfg, "enable_slot_distill_loss", True)),
            enable_pool_distill_loss=bool(_cfg_get(cot_cfg, "enable_pool_distill_loss", True)),
            thinking_token=str(_cfg_get(cot_cfg, "thinking_token", "<|thinking|>")),
            start_thinking_token=str(_cfg_get(cot_cfg, "start_thinking_token", "<|start_of_thinking|>")),
            end_thinking_token=str(_cfg_get(cot_cfg, "end_thinking_token", "<|end_of_thinking|>")),
        )


class BridgeSplitImplicitCotDataset(LeRobotSingleDataset):
    def __init__(
        self,
        *,
        cot_cfg: Any,
        mode: str,
        dataset_path: Path | str,
        modality_configs: dict[str, ModalityConfig],
        embodiment_tag: str | EmbodimentTag,
        video_backend: str = "decord",
        video_backend_kwargs: dict | None = None,
        transforms=None,
        delete_pause_frame: bool = False,
        data_cfg=None,
        **kwargs,
    ):
        self._cot_dataset_cfg = CotDatasetConfig.from_cfg(cot_cfg)
        self._mode = mode
        self._cot_index_path = derive_cot_index_path(Path(dataset_path))
        self._sidecar_reader = BridgeCotSidecarReader(self._cot_index_path)
        self._manifest_by_episode = self._sidecar_reader.load_manifest()
        self._sidecar_reader.validate_against_dataset(Path(dataset_path))
        self._allowed_episode_ids = self._sidecar_reader.allowed_episode_ids_for_mode(self._mode)
        super().__init__(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend,
            video_backend_kwargs=video_backend_kwargs,
            transforms=transforms,
            delete_pause_frame=delete_pause_frame,
            data_cfg=data_cfg,
            **kwargs,
        )

    def _get_shadow_stats_path(self) -> Path:
        mode_suffix = self._mode or "all"
        return self.dataset_path / "meta" / f"stats_gr00t_cot_{mode_suffix}.json"

    def _get_shadow_steps_path(self) -> Path:
        return self.dataset_path / "meta" / f"steps_data_index_{self._get_steps_config_key()}.pkl"

    def _get_metadata(self, embodiment_tag: RegistryEmbodimentTag) -> DatasetMetadata:
        modality_meta_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
        assert modality_meta_path.exists(), f"Please provide a {LE_ROBOT_MODALITY_FILENAME} file in {self.dataset_path}"

        simplified_modality_meta: dict[str, dict] = {}
        with open(modality_meta_path, "r") as f:
            le_modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))

        for modality in ["state", "action"]:
            simplified_modality_meta[modality] = {}
            le_state_action_meta: dict[str, LeRobotStateActionMetadata] = getattr(le_modality_meta, modality)
            for subkey in le_state_action_meta:
                state_action_dtype = np.dtype(le_state_action_meta[subkey].dtype)
                simplified_modality_meta[modality][subkey] = {
                    "absolute": le_state_action_meta[subkey].absolute,
                    "rotation_type": le_state_action_meta[subkey].rotation_type,
                    "shape": [le_state_action_meta[subkey].end - le_state_action_meta[subkey].start],
                    "continuous": bool(np.issubdtype(state_action_dtype, np.floating)),
                }

        simplified_modality_meta["video"] = {}
        for new_key, le_video_meta in le_modality_meta.video.items():
            original_key = le_video_meta.original_key or new_key
            info_meta = self._get_lerobot_info_meta()
            original_video_info = info_meta["features"][original_key]
            height = original_video_info["shape"][original_video_info["names"].index("height")]
            width = original_video_info["shape"][original_video_info["names"].index("width")]
            try:
                channels = original_video_info["shape"][original_video_info["names"].index("channel")]
                fps = original_video_info["video_info"]["video.fps"]
            except (ValueError, KeyError):
                channels = original_video_info["info"]["video.channels"]
                fps = original_video_info["info"]["video.fps"]
            simplified_modality_meta["video"][new_key] = {
                "resolution": [width, height],
                "channels": channels,
                "fps": fps,
            }

        def is_main() -> bool:
            return (not dist.is_initialized()) or dist.get_rank() == 0

        action_mode = _normalize_action_mode(self.data_cfg.get("action_mode", "abs") if self.data_cfg else "abs")
        stats_path = self._get_shadow_stats_path()
        action_cfg = self.modality_configs.get("action")
        state_cfg = self.modality_configs.get("state")
        action_keys_full = list(action_cfg.modality_keys) if action_cfg else []
        state_keys_full = list(state_cfg.modality_keys) if state_cfg else []
        action_indices = list(action_cfg.delta_indices) if action_cfg else None
        state_indices = list(state_cfg.delta_indices) if state_cfg else None

        apply_keys = _normalize_action_mode_apply_keys(
            self.data_cfg.get("action_mode_apply_keys", None) if self.data_cfg else None,
            action_keys_full,
        )
        normalized_state_map = _normalize_action_mode_state_map(
            self.data_cfg.get("action_mode_state_map", {}) if self.data_cfg else {}
        )
        stats_cache_config = _build_stats_cache_config(action_mode=action_mode)
        parquet_files = list(self.dataset_path.glob("data/*/*.parquet"))
        parquet_files_filtered = [pf for pf in parquet_files if "episode_033675.parquet" not in pf.name]

        if is_main():
            le_statistics = _load_or_compute_statistics(
                stats_path,
                stats_cache_config=stats_cache_config,
                parquet_paths=parquet_files_filtered,
                dataset_name=self.dataset_name,
                action_mode=action_mode,
                lerobot_modality_meta=le_modality_meta,
                action_keys_full=action_keys_full,
                state_keys_full=state_keys_full,
                action_indices=action_indices,
                state_indices=state_indices,
                action_mode_apply_keys=apply_keys,
                action_mode_state_map=normalized_state_map,
            )
        else:
            le_statistics = None

        if dist.is_initialized():
            dist.barrier()

        if le_statistics is None:
            le_statistics = _load_stats_cache(
                stats_path,
                stats_cache_config,
                invalidate_legacy=False,
            )
            if le_statistics is None:
                raise RuntimeError(f"Dataset statistics cache is missing or invalid after sync: {stats_path}")

        for stat in le_statistics.values():
            DatasetStatisticalValues.model_validate(stat)

        dataset_statistics = {}
        for our_modality in ["state", "action"]:
            dataset_statistics[our_modality] = {}
            for subkey in simplified_modality_meta[our_modality]:
                dataset_statistics[our_modality][subkey] = {}
                state_action_meta = le_modality_meta.get_key_meta(f"{our_modality}.{subkey}")
                assert isinstance(state_action_meta, LeRobotStateActionMetadata)
                le_modality = state_action_meta.original_key
                for stat_name in le_statistics[le_modality]:
                    indices = np.arange(state_action_meta.start, state_action_meta.end)
                    stat = np.array(le_statistics[le_modality][stat_name])
                    dataset_statistics[our_modality][subkey][stat_name] = stat[indices].tolist()

        return DatasetMetadata(
            statistics=dataset_statistics,  # type: ignore[arg-type]
            modalities=simplified_modality_meta,  # type: ignore[arg-type]
            embodiment_tag=embodiment_tag,
        )

    def _get_trajectories(self) -> tuple[np.ndarray, np.ndarray]:
        if self._lerobot_version == "v2.0":
            file_path = self.dataset_path / LE_ROBOT_EPISODE_FILENAME
            with open(file_path, "r") as f:
                episode_metadata = [json.loads(line) for line in f]
            trajectory_ids = []
            trajectory_lengths = []
            for episode in episode_metadata:
                episode_index = int(episode["episode_index"])
                if episode_index not in self._allowed_episode_ids:
                    continue
                trajectory_ids.append(episode_index)
                trajectory_lengths.append(int(episode["length"]))
            return np.array(trajectory_ids), np.array(trajectory_lengths)

        if self._lerobot_version == "v3.0":
            file_paths = sorted(list(self.dataset_path.glob(LE_ROBOT3_EPISODE_FILENAME)))
            trajectory_ids = []
            trajectory_lengths = []
            self.trajectory_ids_to_metadata = {}
            for file_path in file_paths:
                episodes_data = pd.read_parquet(file_path)
                timestamp_cols = [
                    c
                    for c in episodes_data.columns
                    if str(c).startswith("videos/") and str(c).endswith("/from_timestamp")
                ]
                for index, episode in episodes_data.iterrows():
                    episode_index = int(episode["episode_index"])
                    if episode_index not in self._allowed_episode_ids:
                        continue
                    trajectory_ids.append(episode_index)
                    trajectory_lengths.append(int(episode["length"]))
                    from_timestamps = {}
                    for col in timestamp_cols:
                        value = episode[col]
                        if pd.isna(value):
                            continue
                        video_key = str(col)[len("videos/") : -len("/from_timestamp")]
                        from_timestamps[video_key] = float(value)
                    self.trajectory_ids_to_metadata[trajectory_ids[-1]] = {
                        "data/chunk_index": episode["data/chunk_index"],
                        "data/file_index": episode["data/file_index"],
                        "data/file_from_index": index,
                        "videos/from_timestamps": from_timestamps,
                    }
            return np.array(trajectory_ids), np.array(trajectory_lengths)

        raise ValueError(f"Unsupported LeRobot version: {self._lerobot_version}")

    def _get_steps_config_key(self) -> str:
        config_dict = {
            "delete_pause_frame": self.delete_pause_frame,
            "skip_empty_language": self.skip_empty_language,
            "dataset_name": self.dataset_name,
            "mode": self._mode,
            "cot_index_path": str(self._cot_index_path),
        }
        config_str = str(sorted(config_dict.items()))
        return hashlib.md5(config_str.encode()).hexdigest()[:12]

    def _get_all_steps(self) -> list[tuple[int, int]]:
        def is_main() -> bool:
            return (not dist.is_initialized()) or dist.get_rank() == 0

        config_key = self._get_steps_config_key()
        steps_path = self._get_shadow_steps_path()

        if steps_path.exists():
            try:
                with open(steps_path, "rb") as f:
                    cached_data = pickle.load(f)
                if cached_data.get("config_key") == config_key:
                    if is_main():
                        print(f"[RANK 0] Loaded cached steps from {steps_path}")
                    return cached_data["steps"]
            except Exception as exc:
                print(f"Failed to load cached split steps ({exc}), rebuilding.")

        if is_main():
            print(f"[RANK 0] Rebuilding steps cache at {steps_path}")
            all_steps = self._get_all_steps_single_process()
            cache_data = {
                "config_key": config_key,
                "steps": all_steps,
                "num_trajectories": len(self.trajectory_ids),
                "total_steps": len(all_steps),
                "computed_timestamp": pd.Timestamp.now().isoformat(),
                "delete_pause_frame": self.delete_pause_frame,
                "mode": self._mode,
            }
            steps_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = steps_path.with_suffix(".tmp")
            with open(tmp_path, "wb") as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, steps_path)
            print(f"[RANK 0] Cached steps saved to {steps_path}")

        if dist.is_initialized():
            dist.barrier()

        with open(steps_path, "rb") as f:
            cached_data = pickle.load(f)
        return cached_data["steps"]

    def _get_cot_record(self, episode_index: int, frame_index: int) -> dict[str, Any]:
        return self._sidecar_reader.lookup_episode_frame(episode_index, frame_index).record

    def build_step_sample(self, trajectory_id: int, base_index: int) -> dict[str, Any]:
        raw_data = self.get_step_data(trajectory_id, base_index)
        data = self.transforms(raw_data)
        sample = self._pack_sample(data)

        row = self.curr_traj_data.iloc[base_index]
        episode_index = int(row.get("episode_index", trajectory_id))
        frame_index = int(row.get("frame_index", base_index))
        task_index = int(row.get("task_index", -1))

        cot_record = self._get_cot_record(episode_index, frame_index)
        cot_fields_raw = {field_name: str(cot_record.get(field_name, "") or "") for field_name in self._cot_dataset_cfg.field_names}
        cot_slot_mask = np.asarray([bool(value.strip()) for value in cot_fields_raw.values()], dtype=np.bool_)

        sample["episode_index"] = episode_index
        sample["frame_index"] = frame_index
        sample["task_index"] = task_index
        sample["has_cot"] = bool(cot_slot_mask.any())
        sample["cot_slot_mask"] = cot_slot_mask
        sample["cot_fields_raw"] = cot_fields_raw
        return sample

    def __getitem__(self, index: int) -> dict[str, Any]:
        trajectory_id, base_index = self.all_steps[index]
        return self.build_step_sample(trajectory_id, base_index)


class ImplicitCotMixtureDataset(LeRobotMixtureDataset):
    def __getitem__(self, index: int) -> dict[str, Any]:
        self._getitem_count += 1
        max_retries = 10
        last_exception = None

        for attempt in range(max_retries):
            try:
                while True:
                    dataset, trajectory_id, step = self.sample_step(index)
                    key = dataset.modality_keys["video"][0].replace("video.", "")
                    video_path = dataset.get_video_path(trajectory_id, key)
                    if os.path.exists(video_path):
                        break
                    index = random.randint(0, len(self) - 1)

                if hasattr(dataset, "build_step_sample"):
                    sample = dataset.build_step_sample(trajectory_id, step)
                else:
                    raw_data = dataset.get_step_data(trajectory_id, step)
                    data = dataset.transforms(raw_data)
                    sample = dataset._pack_sample(data)
                sample["robot_tag"] = dataset.tag
                return sample
            except Exception as exc:
                last_exception = exc
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1}/{max_retries} failed for index {index}: {exc}")
                    index = random.randint(0, len(self) - 1)
                else:
                    raise last_exception


class ImplicitCotCollator:
    def __init__(self, cfg: Any):
        cot_cfg = _cfg_get(cfg, "cot", None)
        if not _cot_losses_enabled(cot_cfg):
            raise ValueError("ImplicitCotCollator requires at least one CoT loss to be enabled.")
        self.cot_cfg = CotDatasetConfig.from_cfg(cot_cfg)
        self.data_cfg = cfg.datasets.vla_data
        model_id = cfg.framework.qwenvl.base_vlm
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.processor.tokenizer.padding_side = "left"
        self.tokenizer = self.processor.tokenizer
        self.token_spec = parse_thinking_tokens(cot_cfg)
        self._thinking_token_id: int | None = None
        self._start_thinking_id: int | None = None
        self._end_thinking_id: int | None = None

    def set_tokenizer(self, tokenizer) -> None:
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer
        self.processor.tokenizer = tokenizer
        self._thinking_token_id = None
        self._start_thinking_id = None
        self._end_thinking_id = None

    def _register_special_tokens(self, token_texts: list[str]) -> None:
        additional = list(getattr(self.tokenizer, "additional_special_tokens", []) or [])
        updated = False
        for token_text in token_texts:
            if token_text not in additional:
                additional.append(token_text)
                updated = True
        if updated:
            self.tokenizer.add_special_tokens({"additional_special_tokens": additional})

    def _tokenize_step_label(self, text: str) -> torch.Tensor:
        if not text.strip():
            return torch.empty(0, dtype=torch.long)
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if self.tokenizer.eos_token_id is not None:
            token_ids = token_ids + [self.tokenizer.eos_token_id]
        token_ids = token_ids[: self.cot_cfg.max_cot_length]
        return torch.tensor(token_ids, dtype=torch.long)

    def _validate_prompt_lengths(
        self,
        attention_mask: torch.Tensor,
        limit: int,
        label: str,
    ) -> None:
        lengths = attention_mask.sum(dim=1)
        too_long = torch.nonzero(lengths > limit, as_tuple=False)
        if too_long.numel() == 0:
            return
        bad_index = int(too_long[0].item())
        bad_length = int(lengths[bad_index].item())
        raise ValueError(
            f"{label} prompt length {bad_length} exceeds {limit}. "
            "Increase the configured max prompt length; truncation is disabled."
        )

    def _apply_chat_template_checked(
        self,
        messages: list[list[dict[str, Any]]],
        *,
        add_generation_prompt: bool,
        limit: int,
        label: str,
    ) -> dict[str, torch.Tensor]:
        if limit <= 0:
            raise ValueError("Chat template max length must be positive.")
        batch_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            return_tensors="pt",
        )
        self._validate_prompt_lengths(
            batch_inputs["attention_mask"],
            limit,
            label,
        )
        return batch_inputs

    def _teacher_has_visible_cot(self, visible_cot_texts: list[str]) -> list[bool]:
        return [bool(text.strip()) for text in visible_cot_texts]

    def _get_special_token_id(self, token_text: str) -> int:
        token_id = self.tokenizer.convert_tokens_to_ids(token_text)
        if token_id is None or token_id == self.tokenizer.unk_token_id:
            self._register_special_tokens([token_text])
            token_id = self.tokenizer.convert_tokens_to_ids(token_text)
        if token_id is None or token_id == self.tokenizer.unk_token_id:
            raise ValueError(f"Failed to register CoT special token `{token_text}` in collator tokenizer.")
        return int(token_id)

    def _ensure_special_token_ids(self) -> None:
        if self._thinking_token_id is None:
            self._thinking_token_id = self._get_special_token_id(self.token_spec.thinking_token)
        if self._start_thinking_id is None:
            self._start_thinking_id = self._get_special_token_id(self.token_spec.start_token)
        if self._end_thinking_id is None:
            self._end_thinking_id = self._get_special_token_id(self.token_spec.end_token)

    def _field_end_offsets_from_visible_text(
        self,
        field_names: list[str],
        visible_text: str,
        field_char_spans: dict[str, tuple[int, int]],
    ) -> dict[str, int]:
        offsets: dict[str, int] = {field_name: -1 for field_name in field_names}
        if not field_char_spans:
            return offsets

        encoded_visible = self.tokenizer(
            visible_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        visible_token_ids = encoded_visible["input_ids"]
        if visible_token_ids and isinstance(visible_token_ids[0], list):
            visible_token_ids = visible_token_ids[0]
        offset_mapping = encoded_visible.get("offset_mapping", None)
        if offset_mapping and offset_mapping and isinstance(offset_mapping[0], list):
            offset_mapping = offset_mapping[0]
        if not visible_token_ids:
            raise ValueError("Visible CoT text produced no tokens.")

        for field_name, (char_start, char_end) in field_char_spans.items():
            last_char = char_end - 1
            local_token_index = -1
            if offset_mapping is not None:
                for token_idx, span in enumerate(offset_mapping):
                    token_start, token_end = int(span[0]), int(span[1])
                    if token_start <= last_char < token_end:
                        local_token_index = token_idx
                        break
            if local_token_index < 0:
                prefix_token_ids = self.tokenizer.encode(visible_text[:char_end], add_special_tokens=False)
                local_token_index = len(prefix_token_ids) - 1
            if local_token_index < 0 or local_token_index >= len(visible_token_ids):
                raise ValueError(
                    f"Failed to map `{field_name}` character span ({char_start}, {char_end}) "
                    f"to visible CoT token index."
                )
            offsets[field_name] = local_token_index
        return offsets

    def _find_subsequence_positions(self, sequence: list[int], subsequence: list[int]) -> int:
        if not subsequence:
            raise ValueError("subsequence must be non-empty.")
        max_start = len(sequence) - len(subsequence)
        for start in range(max_start + 1):
            if sequence[start : start + len(subsequence)] == subsequence:
                return start
        raise ValueError("Failed to locate tokenized visible CoT text inside teacher tokens.")

    def _build_teacher_supervision(
        self,
        *,
        teacher_batch: dict[str, torch.Tensor],
        teacher_has_visible_cot: list[bool],
        visible_cot_texts: list[str],
        visible_cot_field_spans: list[dict[str, tuple[int, int]]],
        field_names: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        teacher_labels = torch.full_like(teacher_batch["input_ids"], IGNORE_INDEX)
        teacher_loss_mask = torch.zeros_like(teacher_batch["attention_mask"], dtype=torch.bool)
        teacher_field_positions: list[torch.Tensor] = []

        teacher_input_ids = teacher_batch["input_ids"]
        teacher_attention_mask = teacher_batch["attention_mask"]

        for idx in range(len(visible_cot_texts)):
            has_visible_cot = teacher_has_visible_cot[idx]
            valid_positions = torch.nonzero(teacher_attention_mask[idx].bool(), as_tuple=False).squeeze(-1)
            teacher_valid_ids = teacher_input_ids[idx, valid_positions].tolist()

            field_positions = []
            if has_visible_cot:
                visible_cot_text = visible_cot_texts[idx]
                cot_token_ids = self.tokenizer.encode(visible_cot_text, add_special_tokens=False)
                cot_local_start = self._find_subsequence_positions(teacher_valid_ids, cot_token_ids)
                cot_local_end = cot_local_start + len(cot_token_ids)

                if self.cot_cfg.enable_teacher_cot_loss:
                    cot_start = int(valid_positions[cot_local_start].item())
                    cot_end_idx = min(cot_local_end, len(valid_positions) - 1)
                    cot_end = int(valid_positions[cot_end_idx].item()) + 1
                    teacher_labels[idx, cot_start:cot_end] = teacher_input_ids[idx, cot_start:cot_end]
                    teacher_loss_mask[idx, cot_start:cot_end] = True

                field_offsets = self._field_end_offsets_from_visible_text(
                    field_names,
                    visible_cot_text,
                    visible_cot_field_spans[idx],
                )
                for sample_field_name in field_names:
                    local_pos = field_offsets[sample_field_name]
                    if local_pos < 0:
                        field_positions.append(-1)
                        continue
                    if local_pos >= len(cot_token_ids):
                        raise ValueError(
                            f"Field `{sample_field_name}` local token index {local_pos} is out of range for "
                            f"visible CoT token length {len(cot_token_ids)}."
                        )
                    if cot_local_start + local_pos >= len(valid_positions):
                        raise ValueError(
                            f"Field `{sample_field_name}` absolute local token index {cot_local_start + local_pos} "
                            f"exceeds teacher valid token length {len(valid_positions)}."
                        )
                    absolute_token_pos = int(valid_positions[cot_local_start + local_pos].item())
                    if absolute_token_pos >= teacher_input_ids.shape[1]:
                        raise ValueError(
                            f"Computed teacher field position out of range for `{sample_field_name}`: "
                            f"{absolute_token_pos} >= {teacher_input_ids.shape[1]}."
                        )
                    field_positions.append(absolute_token_pos)
            else:
                field_positions = [-1] * len(field_names)

            teacher_field_positions.append(torch.tensor(field_positions, dtype=torch.long))

        return teacher_labels, teacher_loss_mask, torch.stack(teacher_field_positions, dim=0)

    def _compute_thinking_positions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        self._ensure_special_token_ids()
        num_latents = self.cot_cfg.num_latent_tokens
        valid_mask = attention_mask.bool()
        thinking_mask = (input_ids == self._thinking_token_id) & valid_mask
        counts = thinking_mask.sum(dim=1)
        if not bool(torch.all(counts == num_latents)):
            raise ValueError(
                f"Expected {num_latents} thinking tokens per sample, got counts={counts.tolist()}."
            )
        thinking_positions = torch.nonzero(thinking_mask, as_tuple=False)[:, 1].view(input_ids.shape[0], num_latents)
        return thinking_positions.to(dtype=torch.long)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        self._ensure_special_token_ids()
        field_names = self.cot_cfg.field_names
        visible_cot_items = [
            build_visible_cot_text_with_spans(field_names, sample["cot_fields_raw"])
            for sample in batch
        ]
        visible_cot_texts = [item[0] for item in visible_cot_items]
        visible_cot_field_spans = [item[1] for item in visible_cot_items]
        teacher_has_visible_cot = self._teacher_has_visible_cot(visible_cot_texts) if self.cot_cfg.needs_teacher_inputs else []

        ##student
        student_batch = None
        thinking_positions = None
        if self.cot_cfg.needs_student_inputs:
            student_messages = [
                build_student_message(
                    sample["image"],
                    sample["lang"],
                    self.data_cfg,
                    field_count=self.cot_cfg.num_latent_tokens,
                    token_spec=self.token_spec,
                )
                for sample in batch
            ]
            student_batch = self._apply_chat_template_checked(
                student_messages,
                add_generation_prompt=True,
                limit=self.cot_cfg.student_max_prompt_length,
                label="Student",
            )
            thinking_positions = self._compute_thinking_positions(
                student_batch["input_ids"],
                student_batch["attention_mask"],
            )

        ##teacher
        teacher_batch = None
        teacher_messages = []
        if self.cot_cfg.needs_teacher_inputs:
            teacher_messages = [
                build_teacher_message(sample["image"], sample["lang"], visible_cot_texts[idx], self.data_cfg)
                for idx, sample in enumerate(batch)
            ]
            teacher_batch = self._apply_chat_template_checked(
                teacher_messages,
                add_generation_prompt=False,
                limit=self.cot_cfg.teacher_max_prompt_length,
                label="Teacher",
            )

        teacher_labels = torch.empty(0, dtype=torch.long)
        teacher_loss_mask = torch.empty(0, dtype=torch.bool)
        teacher_field_positions = torch.empty(0, len(field_names), dtype=torch.long)
        if teacher_batch is not None:
            teacher_labels, teacher_loss_mask, teacher_field_positions = self._build_teacher_supervision(
                teacher_batch=teacher_batch,
                teacher_has_visible_cot=teacher_has_visible_cot,
                visible_cot_texts=visible_cot_texts,
                visible_cot_field_spans=visible_cot_field_spans,
                field_names=field_names,
            )

        action_labels = torch.tensor(np.array([sample["action"] for sample in batch]), dtype=torch.float32)
        state_tensor = None
        if "state" in batch[0]:
            state_tensor = torch.tensor(np.array([sample["state"] for sample in batch]), dtype=torch.float32)

        cot_labels = []
        cot_label_mask = []
        if self.cot_cfg.needs_decoder_targets:
            batch_max_cot_len = 0
            tokenized_steps: list[list[torch.Tensor]] = []
            for sample in batch:
                step_tokens = []
                for field_name in field_names:
                    tagged_text = (
                        f"{FIELD_TO_TAG[field_name]} {sample['cot_fields_raw'].get(field_name, '').strip()}".strip()
                        if sample["cot_fields_raw"].get(field_name, "").strip()
                        else ""
                    )
                    tokens = self._tokenize_step_label(tagged_text)
                    step_tokens.append(tokens)
                    batch_max_cot_len = max(batch_max_cot_len, int(tokens.shape[0]))
                tokenized_steps.append(step_tokens)

            batch_max_cot_len = max(batch_max_cot_len, 1)
            for step_tokens in tokenized_steps:
                padded_tokens = []
                padded_masks = []
                for tokens in step_tokens:
                    token_count = int(tokens.shape[0])
                    pad_len = batch_max_cot_len - token_count
                    if pad_len > 0:
                        pad = torch.zeros(pad_len, dtype=torch.long)
                        padded_tokens.append(torch.cat([tokens, pad], dim=0))
                    else:
                        padded_tokens.append(tokens)
                    padded_masks.append(
                        torch.tensor([True] * token_count + [False] * pad_len, dtype=torch.bool)
                    )
                cot_labels.append(torch.stack(padded_tokens, dim=0))
                cot_label_mask.append(torch.stack(padded_masks, dim=0))

        active_batch = teacher_batch if teacher_batch is not None else student_batch
        if active_batch is None:
            raise ValueError("ImplicitCotCollator requires at least one active teacher or student input branch.")
        batch_dict: dict[str, Any] = {
            "image": [sample["image"] for sample in batch],
            "lang": [sample["lang"] for sample in batch],
            "input_ids": active_batch["input_ids"],
            "attention_mask": active_batch["attention_mask"],
            "action_labels": action_labels,
            "episode_index": torch.tensor([sample["episode_index"] for sample in batch], dtype=torch.long),
            "frame_index": torch.tensor([sample["frame_index"] for sample in batch], dtype=torch.long),
            "task_index": torch.tensor([sample["task_index"] for sample in batch], dtype=torch.long),
            "has_cot": torch.tensor([sample["has_cot"] for sample in batch], dtype=torch.bool),
            "cot_slot_mask": torch.tensor(np.array([sample["cot_slot_mask"] for sample in batch]), dtype=torch.bool),
            "visible_cot_texts": visible_cot_texts,
        }

        if student_batch is not None:
            batch_dict["student_input_ids"] = student_batch["input_ids"]
            batch_dict["student_attention_mask"] = student_batch["attention_mask"]
            batch_dict["thinking_positions"] = thinking_positions

        if teacher_batch is not None:
            batch_dict["teacher_input_ids"] = teacher_batch["input_ids"]
            batch_dict["teacher_attention_mask"] = teacher_batch["attention_mask"]
            batch_dict["teacher_field_positions"] = teacher_field_positions
        if self.cot_cfg.enable_teacher_cot_loss:
            batch_dict["labels"] = teacher_labels
            batch_dict["loss_mask"] = teacher_loss_mask

        for key, value in active_batch.items():
            if key in {"input_ids", "attention_mask"}:
                continue
            batch_dict[key] = value

        if state_tensor is not None:
            batch_dict["state"] = state_tensor

        if self.cot_cfg.needs_decoder_targets:
            batch_dict["cot_labels"] = torch.stack(cot_labels, dim=0)
            batch_dict["cot_label_mask"] = torch.stack(cot_label_mask, dim=0)

        return batch_dict


def make_cot_lerobot_single_dataset(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,
    *,
    cot_cfg: Any,
    mode: str,
    delete_pause_frame: bool = False,
    data_cfg: dict | None = None,
) -> BridgeSplitImplicitCotDataset:
    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    modality_config = data_config.modality_config()
    transforms = data_config.transform()
    dataset_path = Path(data_root_dir) / data_name
    if robot_type not in ROBOT_TYPE_TO_EMBODIMENT_TAG:
        embodiment_tag = RegistryEmbodimentTag.NEW_EMBODIMENT
    else:
        embodiment_tag = ROBOT_TYPE_TO_EMBODIMENT_TAG[robot_type]
    video_backend = data_cfg.get("video_backend", "torchvision_av") if data_cfg else "torchvision_av"
    return BridgeSplitImplicitCotDataset(
        cot_cfg=cot_cfg,
        mode=mode,
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=video_backend,
        delete_pause_frame=delete_pause_frame,
        data_cfg=data_cfg,
    )


class BridgeSplitImplicitCotDatasetSmallFit(BridgeSplitImplicitCotDataset):
    def __init__(
        self,
        *,
        small_fit_cfg: Any,
        cot_cfg: Any,
        mode: str,
        dataset_path: Path | str,
        modality_configs: dict[str, ModalityConfig],
        embodiment_tag: str | EmbodimentTag,
        video_backend: str = "decord",
        video_backend_kwargs: dict | None = None,
        transforms=None,
        delete_pause_frame: bool = False,
        data_cfg=None,
        **kwargs,
    ):
        self._small_fit_cfg = small_fit_cfg
        manifest_path = _cfg_get(small_fit_cfg, "episode_manifest_path", None)
        if not manifest_path:
            raise ValueError("small_fit.episode_manifest_path must be provided for small-fit dataset.")
        manifest_data = load_json_or_literal(Path(manifest_path))
        if not isinstance(manifest_data, dict):
            raise ValueError(f"Small-fit manifest `{manifest_path}` must be a JSON object.")
        episode_indices_raw = manifest_data.get("episode_indices", [])
        if not isinstance(episode_indices_raw, list) or not episode_indices_raw:
            raise ValueError(f"Small-fit manifest `{manifest_path}` must contain a non-empty `episode_indices` list.")
        self._small_fit_manifest_path = str(manifest_path)
        self._small_fit_episode_ids = {int(ep_id) for ep_id in episode_indices_raw}
        self._cot_dataset_cfg = CotDatasetConfig.from_cfg(cot_cfg)
        self._mode = mode
        self._cot_index_path = derive_cot_index_path(Path(dataset_path))
        self._sidecar_reader = BridgeCotSidecarReader(self._cot_index_path)
        self._manifest_by_episode = self._sidecar_reader.load_manifest()
        self._sidecar_reader.validate_against_dataset(Path(dataset_path))
        self._allowed_episode_ids = self._sidecar_reader.allowed_episode_ids_for_mode(self._mode) & self._small_fit_episode_ids
        if not self._allowed_episode_ids:
            raise ValueError(
                f"No episodes remain after intersecting split `{mode}` with small-fit manifest `{manifest_path}`."
            )
        LeRobotSingleDataset.__init__(
            self,
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend,
            video_backend_kwargs=video_backend_kwargs,
            transforms=transforms,
            delete_pause_frame=delete_pause_frame,
            data_cfg=data_cfg,
            **kwargs,
        )

    def _get_shadow_steps_path(self) -> Path:
        return _repo_root() / ".small_fit_cache" / self.dataset_name / f"steps_data_index_{self._mode or 'all'}.pkl"

    def _get_steps_config_key(self) -> str:
        config_key = super()._get_steps_config_key()
        manifest_digest = hashlib.md5(
            ",".join(str(idx) for idx in sorted(self._small_fit_episode_ids)).encode("utf-8")
        ).hexdigest()[:12]
        return f"{config_key}_smallfit_{manifest_digest}"


def make_cot_lerobot_single_dataset_small_fit(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,
    *,
    cot_cfg: Any,
    small_fit_cfg: Any,
    mode: str,
    delete_pause_frame: bool = False,
    data_cfg: dict | None = None,
) -> BridgeSplitImplicitCotDatasetSmallFit:
    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    modality_config = data_config.modality_config()
    transforms = data_config.transform()
    dataset_path = Path(data_root_dir) / data_name
    if robot_type not in ROBOT_TYPE_TO_EMBODIMENT_TAG:
        embodiment_tag = RegistryEmbodimentTag.NEW_EMBODIMENT
    else:
        embodiment_tag = ROBOT_TYPE_TO_EMBODIMENT_TAG[robot_type]
    video_backend = data_cfg.get("video_backend", "torchvision_av") if data_cfg else "torchvision_av"
    return BridgeSplitImplicitCotDatasetSmallFit(
        small_fit_cfg=small_fit_cfg,
        cot_cfg=cot_cfg,
        mode=mode,
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=video_backend,
        delete_pause_frame=delete_pause_frame,
        data_cfg=data_cfg,
    )


def get_cot_vla_dataset(
    *,
    full_cfg: Any,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    **kwargs: dict,
) -> LeRobotMixtureDataset:
    data_cfg = full_cfg.datasets.vla_data
    cot_cfg = full_cfg.cot
    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix
    delete_pause_frame = data_cfg.get("delete_pause_frame", False)
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]

    included_datasets = set()
    filtered_mixture_spec = []
    for d_name, d_weight, robot_type in mixture_spec:
        dataset_key = (d_name, robot_type)
        if dataset_key in included_datasets:
            continue
        included_datasets.add(dataset_key)
        filtered_mixture_spec.append((d_name, d_weight, robot_type))

    dataset_mixture = []
    for d_name, d_weight, robot_type in filtered_mixture_spec:
        dataset_mixture.append(
            (
                make_cot_lerobot_single_dataset(
                    Path(data_root_dir),
                    d_name,
                    robot_type,
                    cot_cfg=cot_cfg,
                    mode=mode,
                    delete_pause_frame=delete_pause_frame,
                    data_cfg=data_cfg,
                ),
                d_weight,
            )
        )

    return ImplicitCotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        data_cfg=data_cfg,
        **kwargs,
    )


def get_cot_vla_dataset_small_fit(
    *,
    full_cfg: Any,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    **kwargs: dict,
) -> LeRobotMixtureDataset:
    data_cfg = full_cfg.datasets.vla_data
    cot_cfg = full_cfg.cot
    small_fit_cfg = full_cfg.small_fit
    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix
    delete_pause_frame = data_cfg.get("delete_pause_frame", False)
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]

    included_datasets = set()
    filtered_mixture_spec = []
    for d_name, d_weight, robot_type in mixture_spec:
        dataset_key = (d_name, robot_type)
        if dataset_key in included_datasets:
            continue
        included_datasets.add(dataset_key)
        filtered_mixture_spec.append((d_name, d_weight, robot_type))

    dataset_mixture = []
    for d_name, d_weight, robot_type in filtered_mixture_spec:
        dataset_mixture.append(
            (
                make_cot_lerobot_single_dataset_small_fit(
                    Path(data_root_dir),
                    d_name,
                    robot_type,
                    cot_cfg=cot_cfg,
                    small_fit_cfg=small_fit_cfg,
                    mode=mode,
                    delete_pause_frame=delete_pause_frame,
                    data_cfg=data_cfg,
                ),
                d_weight,
            )
        )

    return ImplicitCotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        data_cfg=data_cfg,
        **kwargs,
    )
