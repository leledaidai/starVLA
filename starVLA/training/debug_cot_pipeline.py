from __future__ import annotations

import argparse
from typing import Any

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from starVLA.dataloader.lerobot_datasets import build_vla_collate_fn, get_vla_dataset
from starVLA.model.framework.base_framework import build_framework


def _load_cfg(config_yaml: str, batch_size: int):
    cfg = OmegaConf.load(config_yaml)
    cfg.datasets.vla_data.per_device_batch_size = batch_size
    if not torch.cuda.is_available() and cfg.framework.qwenvl.get("attn_implementation", None) == "flash_attention_2":
        cfg.framework.qwenvl.attn_implementation = "eager"
    return cfg


def _build_loader(cfg, mode: str, batch_size: int, num_workers: int) -> DataLoader:
    dataset = get_vla_dataset(data_cfg=cfg.datasets.vla_data, full_cfg=cfg, mode=mode)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=build_vla_collate_fn(cfg),
        num_workers=num_workers,
    )


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _summarize_value(value: Any) -> str:
    if torch.is_tensor(value):
        return f"tensor shape={tuple(value.shape)} dtype={value.dtype} device={value.device}"
    if isinstance(value, list):
        return f"list len={len(value)}"
    if isinstance(value, tuple):
        return f"tuple len={len(value)}"
    if isinstance(value, dict):
        return f"dict keys={sorted(value.keys())}"
    return f"{type(value).__name__}: {value}"


def _print_batch_summary(batch: dict[str, Any]) -> None:
    print("batch keys:", sorted(batch.keys()))
    for key in sorted(batch.keys()):
        print(f"[batch] {key}: {_summarize_value(batch[key])}")


def _print_output_summary(outputs: dict[str, Any], prefix: str) -> None:
    print(f"{prefix} keys:", sorted(outputs.keys()))
    for key in sorted(outputs.keys()):
        value = outputs[key]
        if torch.is_tensor(value) and value.ndim == 0:
            print(f"[{prefix}] {key}: {float(value.detach().cpu())}")
        else:
            print(f"[{prefix}] {key}: {_summarize_value(value)}")


def _build_examples_from_batch(batch: dict[str, Any], num_examples: int) -> list[dict[str, Any]]:
    examples = []
    for idx in range(num_examples):
        example = {
            "image": batch["image"][idx],
            "lang": batch["lang"][idx],
            "action": batch["action_labels"][idx].detach().to(dtype=torch.float32).cpu().numpy(),
        }
        if "state" in batch:
            example["state"] = batch["state"][idx].detach().to(dtype=torch.float32).cpu().numpy()
        examples.append(example)
    return examples


def _run_forward_debug(model, batch: dict[str, Any], *, backward: bool) -> None:
    model.train(mode=backward)
    grad_ctx = torch.enable_grad() if backward else torch.no_grad()
    with grad_ctx:
        outputs = model.forward(batch)
    _print_output_summary(outputs, prefix="forward")

    if not backward:
        return

    loss = outputs["loss"]
    loss.backward()
    grad_param_count = 0
    grad_norm_sq = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        grad_param_count += 1
        grad_norm_sq += float(param.grad.detach().float().norm(2).cpu()) ** 2
    print("grad_param_count:", grad_param_count)
    print("grad_norm:", grad_norm_sq ** 0.5)


@torch.inference_mode()
def _run_predict_debug(model, batch: dict[str, Any], *, num_examples: int, decode_cot: bool, max_cot_tokens: int) -> None:
    model.eval()
    examples = _build_examples_from_batch(batch, num_examples)
    outputs = model.predict_action(
        examples,
        decode_cot=decode_cot,
        max_cot_tokens=max_cot_tokens,
    )
    _print_output_summary(outputs, prefix="predict")


@torch.inference_mode()
def _run_teacher_predict_debug(model, batch: dict[str, Any], *, num_examples: int, max_new_tokens: int) -> None:
    model.eval()
    examples = _build_examples_from_batch(batch, num_examples)
    outputs = model.predict_action_teacher_only(examples, max_new_tokens=max_new_tokens)
    _print_output_summary(outputs, prefix="teacher_predict")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, required=True)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "valid"])
    parser.add_argument("--phase", type=str, default="all", choices=["data", "forward", "predict", "teacher", "all"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--decode_cot", action="store_true")
    parser.add_argument("--max_cot_tokens", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    cfg = _load_cfg(args.config_yaml, args.batch_size)
    loader = _build_loader(cfg, args.mode, args.batch_size, args.num_workers)
    batch = next(iter(loader))
    _print_batch_summary(batch)

    if args.phase == "data":
        return

    model = build_framework(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    batch = _move_batch_to_device(batch, device)

    trainable_param_count = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print("trainable_param_count:", trainable_param_count)

    if args.phase in {"forward", "all"}:
        _run_forward_debug(model, batch, backward=args.backward)

    if args.phase in {"predict", "all"}:
        _run_predict_debug(
            model,
            batch,
            num_examples=args.batch_size,
            decode_cot=args.decode_cot,
            max_cot_tokens=args.max_cot_tokens,
        )

    if args.phase in {"teacher", "all"}:
        _run_teacher_predict_debug(
            model,
            batch,
            num_examples=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )


if __name__ == "__main__":
    main()
