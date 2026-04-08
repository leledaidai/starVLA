# Bridge RLDS Data Prep

This directory contains the offline conversion utilities for preparing Bridge data for StarVLA.

Outputs:

- a fresh LeRobot v2 dataset directory containing both `train` and `valid`
- a separate CoT sidecar index keyed by `(episode_index, frame_index)`

Main entrypoint:

```bash
conda activate starVLA
python -m starVLA.lerobot_data_process.convert_bridge_rlds_to_lerobot --overwrite
```

Important notes:

- The converter follows the same CoT keying logic as `openpi`:
  `source_file_path + source_episode_id + frame_idx`
- CoT is intentionally stored outside the LeRobot dataset because current StarVLA
  language loading expects `annotation -> task_index -> tasks.jsonl`, while CoT is
  frame-level text.
- The generated LeRobot dataset preserves all four Bridge cameras.

Sidecar files:

- `episode_manifest.jsonl`: episode-level mapping back to RLDS source ids
- `records/episode_XXXXXX.jsonl`: per-frame CoT records
- `summary.json`: match statistics
