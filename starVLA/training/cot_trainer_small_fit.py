import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from starVLA.dataloader.lerobot_datasets import build_vla_collate_fn, get_vla_dataset_small_fit
from starVLA.model.framework.base_framework import build_framework
from starVLA.training.cot_trainer import (
    _build_accelerator,
    _dist_barrier,
    CotTrainer,
    setup_directories,
    setup_optimizer_and_scheduler,
)
from starVLA.training.trainer_utils.config_tracker import wrap_config
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args
from torch.utils.data import DataLoader


def build_dataloader_small_fit(cfg):
    dataset = get_vla_dataset_small_fit(data_cfg=cfg.datasets.vla_data, full_cfg=cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.datasets.vla_data.per_device_batch_size,
        collate_fn=build_vla_collate_fn(cfg),
        num_workers=4,
    )
    if (not dist.is_initialized()) or dist.get_rank() == 0:
        output_dir = Path(cfg.output_dir)
        dataset.save_dataset_statistics(output_dir / "dataset_statistics.small_fit.json")
    return dataloader


def main(cfg):
    cfg = wrap_config(cfg)
    accelerator = _build_accelerator()
    if not torch.cuda.is_available() and cfg.framework.qwenvl.get("attn_implementation", None) == "flash_attention_2":
        cfg.framework.qwenvl.attn_implementation = "eager"
    setup_directories(cfg)
    dataloader = build_dataloader_small_fit(cfg)
    model = build_framework(cfg)
    if hasattr(dataloader, "collate_fn") and hasattr(dataloader.collate_fn, "set_tokenizer"):
        dataloader.collate_fn.set_tokenizer(model.tokenizer)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model, cfg)
    trainer = CotTrainer(
        cfg=cfg,
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )
    trainer.prepare_training()
    trainer.train()
    _dist_barrier()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, required=True)
    args, clipargs = parser.parse_known_args()
    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    cfg.config_yaml = args.config_yaml
    main(cfg)
