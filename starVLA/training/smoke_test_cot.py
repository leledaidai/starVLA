import argparse

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from starVLA.dataloader.lerobot_datasets import build_vla_collate_fn, get_vla_dataset
from starVLA.model.framework.base_framework import build_framework


def _move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, required=True)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "valid"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--optimizer_step", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-6)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_yaml)
    cfg.datasets.vla_data.per_device_batch_size = args.batch_size
    if not torch.cuda.is_available() and cfg.framework.qwenvl.get("attn_implementation", None) == "flash_attention_2":
        cfg.framework.qwenvl.attn_implementation = "eager"

    dataset = get_vla_dataset(data_cfg=cfg.datasets.vla_data, full_cfg=cfg, mode=args.mode)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=build_vla_collate_fn(cfg),
        num_workers=0,
    )
    batch = next(iter(loader))

    model = build_framework(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    batch = _move_batch_to_device(batch, device)
    model.train(mode=args.backward)

    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("trainable_param_count", trainable_param_count)
    if hasattr(model, "_decoder_uses_lora"):
        print("decoder_uses_lora", bool(model._decoder_uses_lora()))

    grad_ctx = torch.enable_grad() if args.backward else torch.no_grad()
    with grad_ctx:
        outputs = model.forward(batch)

    print("forward_keys", sorted(outputs.keys()))
    for key, value in outputs.items():
        if torch.is_tensor(value):
            if value.ndim == 0:
                print(f"{key}: {float(value.detach().cpu())}")
            else:
                print(f"{key}: shape={tuple(value.shape)} dtype={value.dtype}")

    if args.backward:
        loss = outputs["loss"]
        loss.backward()
        total_norm_sq = 0.0
        grad_param_count = 0
        for param in model.parameters():
            if param.grad is None:
                continue
            grad_param_count += 1
            total_norm_sq += float(param.grad.detach().float().norm(2).cpu()) ** 2
        print("grad_param_count", grad_param_count)
        print("grad_norm", total_norm_sq ** 0.5)
        if args.optimizer_step:
            optimizer = torch.optim.AdamW(
                (p for p in model.parameters() if p.requires_grad),
                lr=args.lr,
                foreach=False,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            print("optimizer_step ok")

    if args.predict:
        model.eval()
        raw_examples = []
        batch_size = int(batch["action_labels"].shape[0])
        for idx in range(batch_size):
            example = {
                "image": batch["image"][idx],
                "lang": batch["lang"][idx],
            }
            if "state" in batch:
                example["state"] = batch["state"][idx].detach().to(dtype=torch.float32).cpu().numpy()
            raw_examples.append(example)

        pred = model.predict_action(raw_examples)
        print("predict_action keys", sorted(pred.keys()))
        print("normalized_actions", pred["normalized_actions"].shape)

        teacher = model.predict_action_teacher_only(raw_examples)
        print("teacher_only keys", sorted(teacher.keys()))
        print("teacher normalized_actions", teacher["normalized_actions"].shape)


if __name__ == "__main__":
    main()
