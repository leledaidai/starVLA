from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@dataclass(frozen=True)
class CotLookupResult:
    has_cot: bool
    reasoning_text: str
    record: dict[str, Any]


class CotIndexReader:
    def __init__(self, index_dir: str | Path):
        self.index_dir = Path(index_dir)
        self.manifest_path = self.index_dir / "episode_manifest.jsonl"
        self.summary_path = self.index_dir / "summary.json"
        self.shards_dir = self.index_dir / "records"
        self._manifest_cache: dict[int, dict[str, Any]] | None = None
        self._record_cache: dict[int, dict[int, dict[str, Any]]] = {}

    def load_manifest(self) -> dict[int, dict[str, Any]]:
        if self._manifest_cache is None:
            manifest: dict[int, dict[str, Any]] = {}
            with self.manifest_path.open("r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    manifest[int(row["episode_index"])] = row
            self._manifest_cache = manifest
        return self._manifest_cache

    def get_episode_metadata(self, episode_index: int) -> dict[str, Any]:
        manifest = self.load_manifest()
        return manifest[episode_index]

    def lookup_episode_frame(self, episode_index: int, frame_index: int) -> CotLookupResult:
        shard_records = self._load_episode_records(episode_index)
        record = shard_records.get(frame_index)
        if record is None:
            return CotLookupResult(has_cot=False, reasoning_text="", record={})
        return CotLookupResult(
            has_cot=bool(record.get("has_cot", False)),
            reasoning_text=str(record.get("reasoning_text", "")),
            record=record,
        )

    def _load_episode_records(self, episode_index: int) -> dict[int, dict[str, Any]]:
        if episode_index in self._record_cache:
            return self._record_cache[episode_index]

        shard_path = self.shards_dir / f"episode_{episode_index:06d}.jsonl"
        records: dict[int, dict[str, Any]] = {}
        with shard_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                records[int(row["frame_index"])] = row
        self._record_cache[episode_index] = records
        return records

