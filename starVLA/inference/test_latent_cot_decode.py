from __future__ import annotations

import argparse
import json
import random
import textwrap
from types import MethodType
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from starVLA.dataloader.lerobot_datasets import build_vla_collate_fn, get_vla_dataset
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import dict_to_namespace, read_mode_config


DEFAULT_CHECKPOINT = (
    "results/Checkpoints/"
    "latent_cot_distill_off_test_bridge_train_cot_qwen3vl4b_test_bs16_fields_5/"
    "checkpoints/steps_55000_pytorch_model.pt"
)
DEFAULT_SAMPLE_SEED = 42
DEFAULT_SAMPLE_IMAGE_DIR = Path("results/debug/latent_cot_test_samples")


def _resolve_checkpoint(path: str | Path) -> Path:
    ckpt_path = Path(path)
    if ckpt_path.is_dir():
        candidates = sorted(ckpt_path.glob("checkpoints/steps_*_pytorch_model.pt"))
        if not candidates:
            raise FileNotFoundError(f"No checkpoint files found under {ckpt_path}/checkpoints")
        return candidates[-1]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")
    return ckpt_path


def _set_if_present(cfg: Any, dotted_key: str, value: Any) -> None:
    node = cfg
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        node = getattr(node, part)
    setattr(node, parts[-1], value)


def _load_state_dict(checkpoint_path: Path, *, mmap: bool) -> dict[str, torch.Tensor]:
    kwargs: dict[str, Any] = {"map_location": "cpu"}
    if mmap:
        kwargs["mmap"] = True
    try:
        return torch.load(str(checkpoint_path), **kwargs)
    except TypeError:
        kwargs.pop("mmap", None)
        return torch.load(str(checkpoint_path), **kwargs)


def set_debug_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_latent_cot_debug_checkpoint(
    checkpoint_path: str | Path,
    *,
    attn_implementation: str | None = "sdpa",
    mmap: bool = True,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint_path = _resolve_checkpoint(checkpoint_path)
    model_config, norm_stats = read_mode_config(str(checkpoint_path))
    cfg = dict_to_namespace(model_config)

    cfg.trainer.pretrained_checkpoint = None
    cfg.cot.enable_teacher_cot_loss = False
    cfg.cot.enable_teacher_action_loss = False
    cfg.cot.enable_student_action_loss = True
    cfg.cot.enable_decoder_loss = True
    cfg.cot.enable_slot_distill_loss = False
    cfg.cot.enable_pool_distill_loss = False
    cfg.cot.decode_cot_in_inference = True

    if attn_implementation:
        _set_if_present(cfg, "framework.qwenvl.attn_implementation", attn_implementation)
    if not torch.cuda.is_available() and cfg.framework.qwenvl.get("attn_implementation", None) == "flash_attention_2":
        cfg.framework.qwenvl.attn_implementation = "eager"

    model = build_framework(cfg)
    state_dict = _load_state_dict(checkpoint_path, mmap=mmap)
    incompatible = model.load_state_dict(state_dict, strict=False)
    del state_dict
    model.norm_stats = norm_stats

    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    decoder_missing = [
        key
        for key in missing
        if key.startswith(("decoder_language_model.", "decoder_lm_head.", "decoder_projection."))
    ]
    if decoder_missing:
        preview = ", ".join(decoder_missing[:8])
        raise RuntimeError(
            "Decoder weights were not fully loaded. "
            f"Missing decoder keys ({len(decoder_missing)}): {preview}"
        )

    print(f"[*] Loaded checkpoint: {checkpoint_path}")
    print(f"[*] Missing keys: {len(missing)}; unexpected keys: {len(unexpected)}")
    if unexpected:
        print(f"[*] Unexpected key preview: {unexpected[:8]}")
    install_debug_latent_decoder(model)
    return model, norm_stats


def _generate_from_latent_debug(self, latent_hidden: torch.Tensor, *, max_new_tokens: int) -> list[str]:
    """Debug-only latent decoder with explicit dtype alignment.

    The training/inference model method is left untouched.  This script installs
    this helper on the loaded model instance so fp32 decoder checkpoints can
    decode bf16 VLA hidden states without changing training behavior.
    """
    if max_new_tokens <= 0:
        return ["" for _ in range(latent_hidden.shape[0])]

    lm, lm_head = self._require_text_decoder()
    dtype_info = get_debug_decoder_dtype_info(self)
    projection_dtype = dtype_info["projection_dtype"]
    lm_input_dtype = dtype_info["lm_input_dtype"]
    lm_head_dtype = dtype_info["lm_head_dtype"]
    latent_hidden = latent_hidden.to(device=self.decoder_projection.weight.device, dtype=projection_dtype)
    eos_token_id = self.tokenizer.eos_token_id
    decoder_latent = self.decoder_projection(latent_hidden).to(dtype=lm_input_dtype)
    generated: list[list[int]] = [[] for _ in range(latent_hidden.shape[0])]
    generated_ids = torch.empty((latent_hidden.shape[0], 0), dtype=torch.long, device=decoder_latent.device)
    finished = torch.zeros(latent_hidden.shape[0], dtype=torch.bool, device=decoder_latent.device)

    for _ in range(max_new_tokens):
        if generated_ids.shape[1] > 0:
            text_embeds = self._embed_decoder_tokens(generated_ids).to(dtype=lm_input_dtype)
            current_embeds = torch.cat([decoder_latent.unsqueeze(1), text_embeds], dim=1)
        else:
            current_embeds = decoder_latent.unsqueeze(1)
        current_embeds = current_embeds.to(dtype=lm_input_dtype)
        step_outputs = lm(
            inputs_embeds=current_embeds,
            attention_mask=torch.ones(current_embeds.shape[:2], dtype=torch.long, device=current_embeds.device),
            use_cache=False,
            return_dict=True,
        )
        next_logits = lm_head(step_outputs.last_hidden_state[:, -1, :].to(dtype=lm_head_dtype))
        next_token = torch.argmax(next_logits, dim=-1)
        if eos_token_id is not None:
            next_token = torch.where(finished, torch.full_like(next_token, eos_token_id), next_token)
        for batch_idx, token_id in enumerate(next_token.tolist()):
            if finished[batch_idx]:
                continue
            if eos_token_id is not None and token_id == eos_token_id:
                finished[batch_idx] = True
                continue
            generated[batch_idx].append(token_id)
        if finished.all():
            break
        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(1)], dim=1)

    return [self.tokenizer.decode(token_ids, skip_special_tokens=True).strip() for token_ids in generated]


def get_debug_decoder_dtype_info(model: torch.nn.Module) -> dict[str, torch.dtype]:
    lm, lm_head = model._require_text_decoder()
    lm_input_dtype = next(lm.embed_tokens.parameters()).dtype
    if hasattr(lm, "layers") and len(lm.layers) > 0 and hasattr(lm.layers[0], "self_attn"):
        lm_input_dtype = lm.layers[0].self_attn.q_proj.weight.dtype
    return {
        "projection_dtype": model.decoder_projection.weight.dtype,
        "embed_dtype": lm.embed_tokens.weight.dtype,
        "lm_input_dtype": lm_input_dtype,
        "lm_head_dtype": lm_head.weight.dtype,
    }


def install_debug_latent_decoder(model: torch.nn.Module) -> None:
    info = get_debug_decoder_dtype_info(model)
    print(
        "[*] decoder dtypes: "
        f"projection={info['projection_dtype']}, "
        f"embed={info['embed_dtype']}, "
        f"lm_input={info['lm_input_dtype']}, "
        f"lm_head={info['lm_head_dtype']}"
    )
    model._generate_from_latent = MethodType(_generate_from_latent_debug, model)


def _build_examples_from_batch(batch: dict[str, Any], num_examples: int) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    actual_examples = min(num_examples, len(batch["image"]))
    for idx in range(num_examples):
        if idx >= actual_examples:
            break
        example: dict[str, Any] = {
            "image": batch["image"][idx],
            "lang": batch["lang"][idx],
            "action": batch["action_labels"][idx].detach().to(dtype=torch.float32).cpu().numpy(),
            "visible_cot_text": batch.get("visible_cot_texts", [""] * actual_examples)[idx],
        }
        for meta_key in ("episode_index", "frame_index", "task_index"):
            if meta_key in batch:
                example[meta_key] = int(batch[meta_key][idx].detach().cpu().item())
        if "state" in batch:
            example["state"] = batch["state"][idx].detach().to(dtype=torch.float32).cpu().numpy()
        examples.append(example)
    return examples


def load_dataset_examples(
    checkpoint_path: Path,
    *,
    mode: str,
    num_samples: int,
    num_workers: int,
    attn_implementation: str | None,
    seed: int,
) -> list[dict[str, Any]]:
    model_config, _ = read_mode_config(str(checkpoint_path))
    cfg = OmegaConf.create(model_config)
    cfg.datasets.vla_data.per_device_batch_size = num_samples
    cfg.cot.enable_teacher_cot_loss = False
    cfg.cot.enable_teacher_action_loss = False
    cfg.cot.enable_student_action_loss = True
    cfg.cot.enable_decoder_loss = True
    cfg.cot.enable_slot_distill_loss = False
    cfg.cot.enable_pool_distill_loss = False
    if attn_implementation:
        cfg.framework.qwenvl.attn_implementation = attn_implementation
    if not torch.cuda.is_available() and cfg.framework.qwenvl.get("attn_implementation", None) == "flash_attention_2":
        cfg.framework.qwenvl.attn_implementation = "eager"

    set_debug_seed(seed)
    dataset = get_vla_dataset(data_cfg=cfg.datasets.vla_data, full_cfg=cfg, mode=mode, seed=seed)
    loader = DataLoader(
        dataset,
        batch_size=num_samples,
        collate_fn=build_vla_collate_fn(cfg),
        num_workers=num_workers,
    )
    batch = next(iter(loader))
    return _build_examples_from_batch(batch, num_samples)


def load_custom_example(image_path: str | Path, prompt: str) -> list[dict[str, Any]]:
    image = Image.open(image_path).convert("RGB")
    return [{"image": [image], "lang": prompt}]


def _check_unnorm_key(norm_stats: dict[str, Any], unnorm_key: str | None) -> str:
    if unnorm_key is None:
        if len(norm_stats) != 1:
            raise ValueError(f"Multiple norm stat keys available; pass --unnorm-key from {list(norm_stats)}")
        return next(iter(norm_stats))
    if unnorm_key not in norm_stats:
        raise ValueError(f"Unknown --unnorm-key {unnorm_key!r}; available keys: {list(norm_stats)}")
    return unnorm_key


def unnormalize_actions(normalized_actions: np.ndarray, action_stats: dict[str, Any]) -> np.ndarray:
    high_key = "q99" if "q99" in action_stats else "max"
    low_key = "q01" if "q01" in action_stats else "min"
    action_high = np.asarray(action_stats[high_key], dtype=np.float32)
    action_low = np.asarray(action_stats[low_key], dtype=np.float32)
    mask = np.asarray(action_stats.get("mask", np.ones_like(action_low, dtype=bool)), dtype=bool)
    clipped = np.clip(normalized_actions, -1.0, 1.0)
    if clipped.shape[-1] >= 7:
        clipped[..., 6] = np.where(clipped[..., 6] < 0.5, 0.0, 1.0)
    return np.where(mask, 0.5 * (clipped + 1.0) * (action_high - action_low) + action_low, clipped)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _ensure_pil_image(image_obj: Any) -> Image.Image:
    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")
    arr = np.asarray(image_obj)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def _build_annotated_image(image: Image.Image, prompt: str) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    font = ImageFont.load_default()
    wrapped = textwrap.wrap(f"task: {prompt}", width=max(20, width // 10))
    line_height = font.getbbox("Ag")[3] + 6
    top_pad = 12
    bottom_pad = 12
    banner_height = top_pad + bottom_pad + max(1, len(wrapped)) * line_height

    canvas = Image.new("RGB", (width, height + banner_height), color=(0, 0, 0))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([(0, height), (width, height + banner_height)], fill=(15, 15, 15))

    y = height + top_pad
    for line in wrapped:
        draw.text((12, y), line, fill=(255, 255, 255), font=font)
        y += line_height
    return canvas


def save_extracted_samples(examples: list[dict[str, Any]], output_dir: str | Path) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / "samples.jsonl"
    with metadata_path.open("w", encoding="utf-8") as f:
        for idx, example in enumerate(examples):
            image_obj = example["image"][0] if isinstance(example["image"], list) else example["image"]
            base_image = _ensure_pil_image(image_obj)
            raw_image_path = out_dir / f"sample_{idx:02d}_raw.png"
            annotated_image_path = out_dir / f"sample_{idx:02d}_prompt.png"
            base_image.save(raw_image_path)
            _build_annotated_image(base_image, example["lang"]).save(annotated_image_path)
            record = {
                "sample_index": idx,
                "raw_image_path": str(raw_image_path),
                "annotated_image_path": str(annotated_image_path),
                "lang": example["lang"],
                "visible_cot_text": example.get("visible_cot_text", ""),
                "episode_index": example.get("episode_index"),
                "frame_index": example.get("frame_index"),
                "task_index": example.get("task_index"),
            }
            f.write(json.dumps(_to_jsonable(record), ensure_ascii=False) + "\n")
    print(f"[*] Saved extracted sample images/metadata to {out_dir}")


def print_result(
    *,
    sample_idx: int,
    example: dict[str, Any],
    outputs: dict[str, Any],
    norm_stats: dict[str, Any],
    print_gt: bool,
    print_unnormalized: bool,
    unnorm_key: str | None,
) -> None:
    action = np.asarray(outputs["normalized_actions"][0], dtype=np.float32)
    print("=" * 100)
    print(f"[sample {sample_idx}]")
    for meta_key in ("episode_index", "frame_index", "task_index"):
        if meta_key in example:
            print(f"{meta_key}: {example[meta_key]}")
    print(f"instruction: {example['lang']}")
    if print_gt and example.get("visible_cot_text"):
        print("\n[ground_truth_visible_cot]")
        print(example["visible_cot_text"])
    print("\n[decoded_latent_cot_by_field]")
    decoded = outputs.get("decoded_cot_by_field", [{}])[0]
    for field_name in outputs.get("cot_field_names", decoded.keys()):
        print(f"{field_name}: {decoded.get(field_name, '')}")
    print(f"\nnum_reasoning_passes: {outputs.get('num_reasoning_passes')}")
    print(f"normalized_action_shape: {tuple(action.shape)}")
    print("[normalized_actions]")
    print(np.array2string(action, precision=5, suppress_small=False))

    if print_unnormalized:
        key = _check_unnorm_key(norm_stats, unnorm_key)
        raw_action = unnormalize_actions(action, norm_stats[key]["action"])
        print(f"\n[unnormalized_actions] key={key}")
        print(np.array2string(raw_action, precision=5, suppress_small=False))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decode latent CoT slots from a trained QwenGR00TImplicitCoT checkpoint.")
    parser.add_argument("--checkpoint-path", type=Path, default=Path(DEFAULT_CHECKPOINT))
    parser.add_argument("--mode", choices=["train", "valid"], default="train")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--image", type=Path, default=None, help="Optional custom image path.")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for --image.")
    parser.add_argument("--max-cot-tokens", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--attn-implementation", type=str, default="sdpa", help="Use empty string to keep checkpoint config.")
    parser.add_argument("--no-mmap", action="store_true", help="Disable torch.load(..., mmap=True).")
    parser.add_argument("--save-samples-dir", type=Path, default=None)
    parser.add_argument("--print-ground-truth-cot", action="store_true")
    parser.add_argument("--print-unnormalized-actions", action="store_true")
    parser.add_argument("--unnorm-key", type=str, default=None)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    checkpoint_path = _resolve_checkpoint(args.checkpoint_path)
    attn_impl = args.attn_implementation or None
    set_debug_seed(args.seed)

    if args.image is not None:
        if not args.prompt:
            raise ValueError("--prompt is required when --image is provided.")
        examples = load_custom_example(args.image, args.prompt)
    else:
        examples = load_dataset_examples(
            checkpoint_path,
            mode=args.mode,
            num_samples=args.num_samples,
            num_workers=args.num_workers,
            attn_implementation=attn_impl,
            seed=args.seed,
        )

    sample_image_dir = args.save_samples_dir
    if sample_image_dir is None:
        checkpoint_tag = checkpoint_path.parent.parent.name
        sample_image_dir = DEFAULT_SAMPLE_IMAGE_DIR / checkpoint_tag / f"{args.mode}_seed_{args.seed}"
    save_extracted_samples(examples, sample_image_dir)

    model, norm_stats = load_latent_cot_debug_checkpoint(
        checkpoint_path,
        attn_implementation=attn_impl,
        mmap=not args.no_mmap,
    )
    model = model.to(torch.device(args.device)).eval()

    print(f"[*] field_names: {getattr(model, 'field_names', [])}")
    print(f"[*] max_cot_tokens: {args.max_cot_tokens}")
    print(f"[*] sample_seed: {args.seed}")
    print(f"[*] saved_sample_images: {sample_image_dir}")
    print(f"[*] running {len(examples)} sample(s) on {args.device}")

    for idx, example in enumerate(examples):
        with torch.inference_mode():
            outputs = model.predict_action(
                [example],
                decode_cot=True,
                max_cot_tokens=args.max_cot_tokens,
            )
        print_result(
            sample_idx=idx,
            example=example,
            outputs=outputs,
            norm_stats=norm_stats,
            print_gt=args.print_ground_truth_cot,
            print_unnormalized=args.print_unnormalized_actions,
            unnorm_key=args.unnorm_key,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
