from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from starVLA.dataloader.cot_dataset import make_cot_lerobot_single_dataset_small_fit
from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES
from starVLA.model.framework.base_framework import build_framework
from starVLA.training.trainer_utils.config_tracker import wrap_config
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args


def build_eval_dataset_small_fit(cfg):
    mixture_spec = DATASET_NAMED_MIXTURES[cfg.datasets.vla_data.data_mix]
    if len(mixture_spec) != 1:
        raise ValueError("Small-fit eval currently supports a single dataset in the mixture.")
    data_name, _, robot_type = mixture_spec[0]
    return make_cot_lerobot_single_dataset_small_fit(
        data_root_dir=cfg.datasets.vla_data.data_root_dir,
        data_name=data_name,
        robot_type=robot_type,
        cot_cfg=cfg.cot,
        small_fit_cfg=cfg.small_fit,
        mode="train",
        delete_pause_frame=cfg.datasets.vla_data.get("delete_pause_frame", False),
        data_cfg=cfg.datasets.vla_data,
    )


def _to_raw_example_small_fit(sample: dict) -> dict:
    raw_example = {
        "image": sample["image"],
        "lang": sample["lang"],
        "action": np.asarray(sample["action"], dtype=np.float32),
    }
    if "state" in sample:
        raw_example["state"] = np.asarray(sample["state"], dtype=np.float32)
    if "cot_fields_raw" in sample:
        raw_example["cot_fields_raw"] = sample["cot_fields_raw"]
    return raw_example


@torch.inference_mode()
def evaluate_small_fit(cfg, checkpoint_path: Path) -> dict:
    if not torch.cuda.is_available() and cfg.framework.qwenvl.get("attn_implementation", None) == "flash_attention_2":
        cfg.framework.qwenvl.attn_implementation = "eager"
    model = build_framework(cfg)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    # Filter out corrupted NaN weights (e.g. embed_tokens corrupted by
    # training NaN) so the pretrained weights are retained instead.
    clean_state_dict = {}
    for k, v in state_dict.items():
        if torch.isnan(v).any():
            print(f"[WARNING] Skipping NaN weight: {k}")
            continue
        clean_state_dict[k] = v
    if len(clean_state_dict) < len(state_dict):
        print(f"[INFO] Filtered {len(state_dict) - len(clean_state_dict)} NaN key(s) from checkpoint, "
              f"loading {len(clean_state_dict)} clean key(s)")
    model.load_state_dict(clean_state_dict, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    dataset = build_eval_dataset_small_fit(cfg)
    mse_sum = 0.0
    l2_sum = 0.0
    sample_count = 0

    eval_batch_size = int(cfg.small_fit.get("eval_batch_size", 8))
    predict_mode = str(cfg.small_fit.get("predict_mode", "student"))
    batch_examples = []
    batch_targets = []

    def flush_batch() -> tuple[float, float, int]:
        if not batch_examples:
            return 0.0, 0.0, 0
        if predict_mode == "teacher":
            output_dict = model.predict_action_teacher_only(examples=batch_examples)
        elif predict_mode == "teacher_debug":
            output_dict = model.predict_action_teacher_only_small_fit_debug(examples=batch_examples)
        elif predict_mode == "student":
            output_dict = model.predict_action(examples=batch_examples)
        else:
            raise ValueError(
                f"Unsupported small_fit.predict_mode `{predict_mode}`. "
                "Expected `student`, `teacher`, or `teacher_debug`."
            )
        pred_actions = np.asarray(output_dict["normalized_actions"], dtype=np.float32)
        gt_actions = np.asarray(batch_targets, dtype=np.float32)
        batch_mse = float(np.square(pred_actions - gt_actions).mean())
        batch_l2 = float(np.linalg.norm((pred_actions - gt_actions).reshape(pred_actions.shape[0], -1), axis=1).mean())
        return batch_mse * pred_actions.shape[0], batch_l2 * pred_actions.shape[0], pred_actions.shape[0]

    for index in tqdm(range(len(dataset)), desc="Evaluating small-fit train subset"):
        sample = dataset[index]
        batch_examples.append(_to_raw_example_small_fit(sample))
        batch_targets.append(np.asarray(sample["action"], dtype=np.float32))
        if len(batch_examples) >= eval_batch_size:
            batch_mse_sum, batch_l2_sum, batch_size = flush_batch()
            mse_sum += batch_mse_sum
            l2_sum += batch_l2_sum
            sample_count += batch_size
            batch_examples = []
            batch_targets = []

    batch_mse_sum, batch_l2_sum, batch_size = flush_batch()
    mse_sum += batch_mse_sum
    l2_sum += batch_l2_sum
    sample_count += batch_size

    if sample_count == 0:
        raise ValueError("Small-fit evaluation dataset is empty.")

    return {
        "checkpoint_path": str(checkpoint_path),
        "manifest_path": str(cfg.small_fit.episode_manifest_path),
        "num_episodes": int(cfg.small_fit.get("num_episodes", 100)),
        "num_samples": int(sample_count),
        "predict_mode": predict_mode,
        "mean_mse": float(mse_sum / sample_count),
        "mean_l2": float(l2_sum / sample_count),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--summary_jsonl", type=str, default=None)
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    cfg = wrap_config(cfg)

    result = evaluate_small_fit(cfg, Path(args.checkpoint_path))
    output_path = Path(args.output_path) if args.output_path else Path(cfg.run_root_dir) / cfg.run_id / "small_fit_eval.json"
    summary_jsonl = (
        Path(args.summary_jsonl)
        if args.summary_jsonl
        else Path(cfg.run_root_dir) / cfg.run_id / "small_fit_eval_results.jsonl"
    )
    result["run_id"] = str(cfg.run_id)
    result["output_path"] = str(output_path)
    result["evaluated_at"] = datetime.now(timezone.utc).isoformat()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with summary_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Saved small-fit eval JSON to {output_path}")
    print(f"Appended small-fit eval summary to {summary_jsonl}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
