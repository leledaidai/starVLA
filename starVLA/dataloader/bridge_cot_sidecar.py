from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LE_ROBOT_INFO_FILENAME = "meta/info.json"
LE_ROBOT_EPISODE_FILENAME = "meta/episodes.jsonl"
LE_ROBOT_TASKS_FILENAME = "meta/tasks.jsonl"
_LRU_CACHE_SIZE = 32


def normalize_split_name(split: str | None) -> str:
    split_text = str(split or "").strip().lower()
    if split_text in {"val", "validation", "valid"}:
        return "valid"
    if split_text == "train":
        return "train"
    return split_text


def derive_cot_index_path(dataset_path: Path) -> Path:
    cot_index_path = dataset_path.parent / f"{dataset_path.name}_cot_index"
    if not cot_index_path.exists():
        raise FileNotFoundError(f"CoT index path `{cot_index_path}` does not exist.")
    return cot_index_path


def derive_source_db_relpath(source_file_path: str, source_episode_id: int | str) -> str:
    stable_key = f"{source_file_path}\t{source_episode_id}"
    digest = hashlib.sha1(stable_key.encode("utf-8")).hexdigest()
    return f"records/{digest[:2]}/{digest}.jsonl"


def load_json_or_literal(path: Path) -> Any:
    raw_text = path.read_text(encoding="utf-8")
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        import ast

        return ast.literal_eval(raw_text)


@dataclass(frozen=True)
class BridgeCotLookupResult:
    has_cot: bool
    reasoning_text: str
    record: dict[str, Any]


class BridgeCotSidecarReader:
    def __init__(self, index_dir: str | Path):
        self.index_dir = Path(index_dir)
        self.manifest_path = self.index_dir / "episode_manifest.jsonl"
        self.summary_path = self.index_dir / "summary.json"
        self.records_dir = self.index_dir / "records"
        self._manifest_cache: dict[int, dict[str, Any]] | None = None
        self._task_cache: dict[int, str] | None = None
        self._record_cache: OrderedDict[int, dict[int, dict[str, Any]]] = OrderedDict()

    def load_manifest(self) -> dict[int, dict[str, Any]]:
        if self._manifest_cache is None:
            manifest: dict[int, dict[str, Any]] = {}
            with self.manifest_path.open("r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    manifest[int(row["episode_index"])] = row
            self._manifest_cache = manifest
        return self._manifest_cache

    def get_summary(self) -> dict[str, Any]:
        if not self.summary_path.exists():
            return {}
        with self.summary_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def lookup_episode_frame(self, episode_index: int, frame_index: int) -> BridgeCotLookupResult:
        shard_records = self._load_episode_records(episode_index)
        record = shard_records.get(frame_index)
        if record is None:
            return BridgeCotLookupResult(has_cot=False, reasoning_text="", record={})
        return BridgeCotLookupResult(
            has_cot=bool(record.get("has_cot", False)),
            reasoning_text=str(record.get("reasoning_text", "") or ""),
            record=record,
        )

    def allowed_episode_ids_for_mode(self, mode: str) -> set[int]:
        normalized_mode = normalize_split_name(mode)
        manifest = self.load_manifest()
        if normalized_mode not in {"train", "valid"}:
            return set(manifest.keys())
        return {
            episode_index
            for episode_index, row in manifest.items()
            if normalize_split_name(row.get("split", "")) == normalized_mode
        }

    def validate_against_dataset(self, dataset_path: str | Path) -> None:
        dataset_dir = Path(dataset_path)
        info_path = dataset_dir / LE_ROBOT_INFO_FILENAME
        episodes_path = dataset_dir / LE_ROBOT_EPISODE_FILENAME
        tasks_path = dataset_dir / LE_ROBOT_TASKS_FILENAME

        if not info_path.exists() or not episodes_path.exists() or not tasks_path.exists():
            raise FileNotFoundError(
                f"Expected LeRobot metadata files under `{dataset_dir}`; "
                f"missing one of `{info_path}`, `{episodes_path}`, `{tasks_path}`."
            )

        with info_path.open("r", encoding="utf-8") as f:
            dataset_info = json.load(f)
        expected_total_episodes = int(dataset_info.get("total_episodes", -1))

        tasks_by_index: dict[int, str] = {}
        with tasks_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                tasks_by_index[int(row["task_index"])] = str(row.get("task", "") or "")

        manifest = self.load_manifest()
        if expected_total_episodes >= 0 and len(manifest) != expected_total_episodes:
            raise ValueError(
                f"Sidecar episode count mismatch for `{dataset_dir}`: "
                f"dataset has {expected_total_episodes}, sidecar has {len(manifest)}."
            )

        with episodes_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                episode_index = int(row["episode_index"])
                expected_length = int(row["length"])
                expected_tasks = row.get("tasks", []) or []
                expected_task = ""
                if expected_tasks:
                    expected_task = str(expected_tasks[0] or "")
                else:
                    task_index = row.get("task_index", None)
                    if task_index is not None:
                        expected_task = tasks_by_index.get(int(task_index), "")

                sidecar_row = manifest.get(episode_index)
                if sidecar_row is None:
                    raise ValueError(f"Sidecar is missing episode_index={episode_index}.")

                sidecar_length = int(sidecar_row.get("length", -1))
                if sidecar_length != expected_length:
                    raise ValueError(
                        f"Sidecar length mismatch for episode {episode_index}: "
                        f"dataset={expected_length}, sidecar={sidecar_length}."
                    )

                sidecar_task = str(sidecar_row.get("task", "") or "")
                if sidecar_task != expected_task:
                    raise ValueError(
                        f"Sidecar task mismatch for episode {episode_index}: "
                        f"dataset={expected_task!r}, sidecar={sidecar_task!r}."
                    )

    def _load_episode_records(self, episode_index: int) -> dict[int, dict[str, Any]]:
        if episode_index in self._record_cache:
            self._record_cache.move_to_end(episode_index)
            return self._record_cache[episode_index]

        shard_path = self.records_dir / f"episode_{episode_index:06d}.jsonl"
        records: dict[int, dict[str, Any]] = {}
        if shard_path.exists():
            with shard_path.open("r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    records[int(row["frame_index"])] = row

        self._record_cache[episode_index] = records
        if len(self._record_cache) > _LRU_CACHE_SIZE:
            self._record_cache.popitem(last=False)
        return records


class BridgeCotSourceDbReader:
    def __init__(self, source_db_dir: str | Path):
        self.source_db_dir = Path(source_db_dir)
        self.manifest_path = self.source_db_dir / "source_manifest.jsonl"
        self.summary_path = self.source_db_dir / "summary.json"
        self._manifest_cache: dict[tuple[str, int], dict[str, Any]] | None = None
        self._record_cache: OrderedDict[tuple[str, int], dict[int, dict[str, Any]]] = OrderedDict()

    def load_manifest(self) -> dict[tuple[str, int], dict[str, Any]]:
        if self._manifest_cache is None:
            manifest: dict[tuple[str, int], dict[str, Any]] = {}
            with self.manifest_path.open("r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    key = (str(row["source_file_path"]), int(row["source_episode_id"]))
                    manifest[key] = row
            self._manifest_cache = manifest
        return self._manifest_cache

    def get_summary(self) -> dict[str, Any]:
        if not self.summary_path.exists():
            return {}
        with self.summary_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def get_source_record(
        self,
        source_file_path: str,
        source_episode_id: int,
        source_frame_index: int,
    ) -> dict[str, Any]:
        records = self._load_source_records(source_file_path, source_episode_id)
        return records.get(int(source_frame_index), {})

    def _load_source_records(self, source_file_path: str, source_episode_id: int) -> dict[int, dict[str, Any]]:
        cache_key = (str(source_file_path), int(source_episode_id))
        if cache_key in self._record_cache:
            self._record_cache.move_to_end(cache_key)
            return self._record_cache[cache_key]

        manifest = self.load_manifest()
        manifest_row = manifest.get(cache_key)
        if manifest_row is None:
            records: dict[int, dict[str, Any]] = {}
        else:
            relpath = manifest_row.get("shard_relpath")
            if relpath is None:
                relpath = derive_source_db_relpath(source_file_path, source_episode_id)
            shard_path = self.source_db_dir / str(relpath)
            records = {}
            if shard_path.exists():
                with shard_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        row = json.loads(line)
                        records[int(row["source_frame_index"])] = row

        self._record_cache[cache_key] = records
        if len(self._record_cache) > _LRU_CACHE_SIZE:
            self._record_cache.popitem(last=False)
        return records
