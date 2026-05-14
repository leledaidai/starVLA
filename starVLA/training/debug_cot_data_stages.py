from __future__ import annotations

import argparse

from omegaconf import OmegaConf

from starVLA.dataloader.lerobot_datasets import build_vla_collate_fn, get_vla_dataset


def _load_cfg(config_yaml: str):
    cfg = OmegaConf.load(config_yaml)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, required=True)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "valid"])
    parser.add_argument("--sample_index", type=int, default=0)
    args = parser.parse_args()

    cfg = _load_cfg(args.config_yaml)

    dataset = get_vla_dataset(
        data_cfg=cfg.datasets.vla_data,
        full_cfg=cfg,
        mode=args.mode,
    )
    print("dataset_type:", type(dataset).__name__)
    print("dataset_len:", len(dataset))

    collator = build_vla_collate_fn(cfg)
    print("collator_type:", type(collator).__name__)

    sample = dataset[args.sample_index]
    print("sample_keys:", sorted(sample.keys()))

    batch = collator([sample])
    print("batch_keys:", sorted(batch.keys()))
    for key in sorted(batch.keys()):
        value = batch[key]
        shape = getattr(value, "shape", None)
        if shape is not None:
            print(f"[batch] {key}: shape={tuple(shape)}")
        elif isinstance(value, list):
            print(f"[batch] {key}: list len={len(value)}")
        else:
            print(f"[batch] {key}: type={type(value).__name__}")


if __name__ == "__main__":
    main()
