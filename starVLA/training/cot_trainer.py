import argparse
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import get_scheduler

from starVLA.dataloader import build_dataloader
from starVLA.model.framework.base_framework import build_framework
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups, normalize_dotlist_args

logger = get_logger(__name__)


def _dist_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _dist_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _has_distributed_launcher_env() -> bool:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return world_size > 1 or "LOCAL_RANK" in os.environ or "RANK" in os.environ


def _build_accelerator() -> Accelerator:
    if _has_distributed_launcher_env():
        return Accelerator(deepspeed_plugin=DeepSpeedPlugin())
    return Accelerator()


def _print_trainable_parameters(model) -> tuple[int, int] | None:
    if _dist_rank() != 0:
        return None
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("model parameter statistics:")
    print(
        f"# Parameters (in millions): {num_params / 10**6:.3f} Total, "
        f"{num_trainable_params / 10**6:.3f} Trainable"
    )
    return num_params, num_trainable_params


def setup_directories(cfg) -> Path:
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)
    if _dist_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)
    return output_dir


def _save_initial_configs(cfg, output_dir: Path) -> None:
    if _dist_rank() != 0:
        return
    full_cfg = cfg.unwrap() if isinstance(cfg, AccessTrackedConfig) else cfg
    OmegaConf.save(full_cfg, output_dir / "config.full.yaml", resolve=True)
    if isinstance(cfg, AccessTrackedConfig):
        cfg.save_accessed_config(output_dir / "config.yaml", use_original_values=False)


def setup_optimizer_and_scheduler(model, cfg):
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,
    )
    return optimizer, lr_scheduler


class CotTrainer(TrainerUtils):
    def __init__(self, cfg, model, dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.completed_steps = 0
        self.total_batch_size = (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def prepare_training(self):
        rank = _dist_rank()
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(
                self.model,
                pretrained_checkpoint,
                reload_modules=reload_modules,
            )

        freeze_modules = self.config.trainer.freeze_modules if hasattr(self.config.trainer, "freeze_modules") else None
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        _print_trainable_parameters(self.model)
        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator, self.model, self.optimizer, self.vla_train_dataloader
        )
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="cot-train",
            )
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        _save_initial_configs(self.config, Path(self.config.output_dir))

    def _save_checkpoint(self):
        if not self.accelerator.is_main_process:
            return
        checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}_pytorch_model.pt")
        torch.save(self.accelerator.get_state_dict(self.model), checkpoint_path)
        with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
            f.write(json.dumps({"steps": self.completed_steps}) + "\n")
        if isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(Path(self.config.output_dir) / "config.yaml", use_original_values=False)

    def _log_metrics(self, metrics):
        if self.completed_steps % self.config.trainer.logging_frequency == 0 and _dist_rank() == 0:
            metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
            wandb.log(metrics, step=self.completed_steps)
            logger.info(f"Step {self.completed_steps}, metrics={metrics}")

    def _create_data_iterators(self):
        self.vla_iter = iter(self.vla_train_dataloader)

    def _get_next_batch(self):
        """就是这一句，触发了：
            dataset 的 __getitem__
            collator 的 __call__"""
        try:
            return next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            return next(self.vla_iter)

    def _gradients_are_finite(self) -> bool:
        bad_grads = []
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            if not torch.isfinite(grad).all():
                nan_count = int(torch.isnan(grad).sum().item())
                inf_count = int(torch.isinf(grad).sum().item())
                bad_grads.append((name, nan_count, inf_count))
                if len(bad_grads) >= 8:
                    break

        has_bad_grad = torch.tensor(
            [1 if bad_grads else 0],
            device=self.accelerator.device,
            dtype=torch.int,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(has_bad_grad, op=dist.ReduceOp.MAX)

        if int(has_bad_grad.item()) == 0:
            return True

        if bad_grads and _dist_rank() == 0:
            print("[NaN GRAD] Non-finite gradients before optimizer.step(); skipping update.", flush=True)
            for name, nan_count, inf_count in bad_grads:
                print(f"[NaN GRAD]   {name}: NaN={nan_count}, Inf={inf_count}", flush=True)
        return False

    def _train_step(self, batch):
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()
            autocast_ctx = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if torch.cuda.is_available()
                else nullcontext()
            )
            with autocast_ctx:
                output_dict = self.model.forward(batch)
                total_loss = output_dict["loss"]
            #$modify for nan debug
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                if dist.is_initialized() and dist.get_rank() == 0:
                    print(f"[SKIP] NaN/Inf loss at step {self.completed_steps}, skipping optimizer update",
                          flush=True)
                self.optimizer.zero_grad()
                skip_metrics = {}
                for key, value in output_dict.items():
                    if torch.is_tensor(value) and value.ndim == 0:
                        skip_metrics[key] = value.item()
                return skip_metrics
            #$modify for nan debug
            self.accelerator.backward(total_loss)
            if not self._gradients_are_finite():
                self.optimizer.zero_grad()
                skip_metrics = {}
                for key, value in output_dict.items():
                    if torch.is_tensor(value) and value.ndim == 0:
                        skip_metrics[key] = value.item()
                skip_metrics["skipped_nonfinite_grad"] = 1.0
                return skip_metrics
            if self.config.trainer.gradient_clipping is not None:
                grad_norm = self.accelerator.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.trainer.gradient_clipping,
                )
                if grad_norm is None:
                    grad_norm_is_finite = True
                elif torch.is_tensor(grad_norm):
                    grad_norm_is_finite = bool(torch.isfinite(grad_norm.detach()).all().item())
                else:
                    grad_norm_is_finite = bool(np.isfinite(float(grad_norm)))
                if not grad_norm_is_finite:
                    if _dist_rank() == 0:
                        print(
                            f"[NaN GRAD] Non-finite grad norm before optimizer.step(): {grad_norm}; skipping update.",
                            flush=True,
                        )
                    self.optimizer.zero_grad()
                    skip_metrics = {}
                    for key, value in output_dict.items():
                        if torch.is_tensor(value) and value.ndim == 0:
                            skip_metrics[key] = value.item()
                    skip_metrics["skipped_nonfinite_grad"] = 1.0
                    return skip_metrics
            if not self._gradients_are_finite():
                self.optimizer.zero_grad()
                skip_metrics = {}
                for key, value in output_dict.items():
                    if torch.is_tensor(value) and value.ndim == 0:
                        skip_metrics[key] = value.item()
                skip_metrics["skipped_nonfinite_grad"] = 1.0
                return skip_metrics
            self.optimizer.step()
            self.lr_scheduler.step()

        metrics = {}
        for key, value in output_dict.items():
            if torch.is_tensor(value) and value.ndim == 0:
                metrics[key] = value.item()
        return metrics

    def eval_action_model(self, step_metrics: dict):
        examples = self._get_next_batch()
        if not self.accelerator.is_main_process:
            _dist_barrier()
            return step_metrics
        model = self.accelerator.unwrap_model(self.model)
        if isinstance(examples, dict):
            batch_size = int(examples["action_labels"].shape[0])
            raw_examples = []
            for idx in range(batch_size):
                sample = {
                    "image": examples["image"][idx],
                    "lang": examples["lang"][idx],
                    "action": examples["action_labels"][idx].detach().to(dtype=torch.float32).cpu().numpy(),
                }
                if "state" in examples:
                    sample["state"] = examples["state"][idx].detach().to(dtype=torch.float32).cpu().numpy()
                raw_examples.append(sample)
        else:
            raw_examples = examples

        eval_mode = getattr(model, "eval_mode", "student")
        if eval_mode == "teacher":
            output_dict = model.predict_action_teacher_only(examples=raw_examples)
        else:
            output_dict = model.predict_action(examples=raw_examples)
        pred_actions = output_dict["normalized_actions"]
        gt_actions = np.array([example["action"] for example in raw_examples])
        step_metrics["mse_score"] = TrainerUtils.euclidean_distance(pred_actions, gt_actions) / np.prod(gt_actions.shape)
        _dist_barrier()
        return step_metrics

    def train(self):
        self._create_data_iterators()
        progress_bar = tqdm(range(self.config.trainer.max_train_steps), disable=not self.accelerator.is_local_main_process)
        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch = self._get_next_batch()
            t_end_data = time.perf_counter()
            t_start_model = time.perf_counter()
            metrics = self._train_step(batch)
            t_end_model = time.perf_counter()

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1
            if self.completed_steps % self.config.trainer.eval_interval == 0:
                metrics = self.eval_action_model(metrics)
            metrics["data_time"] = t_end_data - t_start_data
            metrics["model_time"] = t_end_model - t_start_model
            self._log_metrics(metrics)
            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()

        if self.accelerator.is_main_process:
            final_dir = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_dir, exist_ok=True)
            torch.save(self.accelerator.get_state_dict(self.model), os.path.join(final_dir, "pytorch_model.pt"))
            if isinstance(self.config, AccessTrackedConfig):
                self.config.save_accessed_config(Path(self.config.output_dir) / "config.yaml", use_original_values=False)
            wandb.finish()


def main(cfg):
    cfg = wrap_config(cfg)
    accelerator = _build_accelerator()
    if not torch.cuda.is_available() and cfg.framework.qwenvl.get("attn_implementation", None) == "flash_attention_2":
        cfg.framework.qwenvl.attn_implementation = "eager"
    output_dir = setup_directories(cfg)
    dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
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
