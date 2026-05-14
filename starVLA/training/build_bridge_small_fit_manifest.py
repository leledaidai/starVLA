from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from starVLA.dataloader.bridge_cot_sidecar import BridgeCotSidecarReader, derive_cot_index_path, normalize_split_name
from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args


def build_bridge_small_fit_manifest_small_fit(cfg, output_path: Path, num_episodes: int) -> dict:
    data_cfg = cfg.datasets.vla_data
    mixture_spec = DATASET_NAMED_MIXTURES[data_cfg.data_mix]
    if len(mixture_spec) != 1:
        raise ValueError("Small-fit manifest builder currently supports a single dataset in the mixture.")

    data_name, _, _ = mixture_spec[0]
    dataset_path = Path(data_cfg.data_root_dir) / data_name
    sidecar_path = derive_cot_index_path(dataset_path)
    reader = BridgeCotSidecarReader(sidecar_path)
    manifest = reader.load_manifest()

    train_episode_ids = sorted(
        episode_index
        for episode_index, row in manifest.items()
        if normalize_split_name(row.get("split", "")) == "train"
    )
    if len(train_episode_ids) < num_episodes:
        raise ValueError(
            f"Requested {num_episodes} episodes, but only found {len(train_episode_ids)} train episodes in `{data_name}`."
        )

    selected_episode_ids = train_episode_ids[:num_episodes]
    payload = {
        "data_mix": str(data_cfg.data_mix),
        "data_name": str(data_name),
        "split": "train",
        "selection_policy": "first_n_sorted_episode_index",
        "num_episodes": int(num_episodes),
        "episode_indices": selected_episode_ids,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, default=100)
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    payload = build_bridge_small_fit_manifest_small_fit(
        cfg=cfg,
        output_path=Path(args.output_path),
        num_episodes=args.num_episodes,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
