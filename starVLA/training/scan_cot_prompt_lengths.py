import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoProcessor

from starVLA.dataloader.cot_dataset import get_cot_vla_dataset
from starVLA.dataloader.cot_formatter import (
    build_prefix_message,
    build_student_message,
    build_teacher_message,
    build_visible_cot_text,
    parse_cot_fields,
    parse_thinking_tokens,
)


def _tokenized_length(
    processor: AutoProcessor,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
) -> int:
    batch = processor.apply_chat_template(
        [messages],
        tokenize=True,
        padding=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=True,
        return_tensors="pt",
    )
    return int(batch["attention_mask"][0].sum().item())


def _summary(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "p99_9": 0.0,
        }

    array = np.asarray(values, dtype=np.int32)
    return {
        "count": int(array.shape[0]),
        "min": int(array.min()),
        "max": int(array.max()),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "p99_9": float(np.percentile(array, 99.9)),
    }


def _top_examples(records: list[dict[str, Any]], key: str, top_k: int) -> list[dict[str, Any]]:
    ranked = sorted(records, key=lambda item: item[key], reverse=True)
    trimmed = []
    for item in ranked[:top_k]:
        trimmed.append(
            {
                "dataset_name": item["dataset_name"],
                "episode_index": int(item["episode_index"]),
                "frame_index": int(item["frame_index"]),
                "task_index": int(item["task_index"]),
                "has_cot": bool(item["has_cot"]),
                "prefix_with_generation_length": int(item["prefix_with_generation_length"]),
                "prefix_without_generation_length": int(item["prefix_without_generation_length"]),
                "student_length": int(item["student_length"]),
                "teacher_length": int(item["teacher_length"]),
                "lang": item["lang"],
                "visible_cot_text": item["visible_cot_text"],
            }
        )
    return trimmed


def _print_summary(name: str, stats: dict[str, float | int]) -> None:
    print(
        f"{name}: "
        f"count={stats['count']} min={stats['min']} max={stats['max']} "
        f"mean={stats['mean']:.2f} p50={stats['p50']:.2f} p90={stats['p90']:.2f} "
        f"p95={stats['p95']:.2f} p99={stats['p99']:.2f} p99.9={stats['p99_9']:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, required=True)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "valid"])
    parser.add_argument("--max_samples", type=int, default=0, help="0 means scan all samples.")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--output_json", type=str, default="")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_yaml)
    dataset = get_cot_vla_dataset(full_cfg=cfg, mode=args.mode)
    processor = AutoProcessor.from_pretrained(cfg.framework.qwenvl.base_vlm)
    processor.tokenizer.padding_side = "left"

    field_names = [spec.name for spec in parse_cot_fields(cfg.cot)]
    token_spec = parse_thinking_tokens(cfg.cot)

    records: list[dict[str, Any]] = []
    prefix_with_generation_lengths: list[int] = []
    prefix_without_generation_lengths: list[int] = []
    student_lengths: list[int] = []
    teacher_lengths: list[int] = []
    teacher_lengths_with_cot: list[int] = []
    teacher_lengths_without_cot: list[int] = []

    total_available = sum(len(single_dataset.all_steps) for single_dataset in dataset.datasets)
    target_total = total_available if args.max_samples <= 0 else min(total_available, args.max_samples)
    processed = 0

    print(f"[CoT][Scan] mode={args.mode} total_available={total_available} target_total={target_total}")
    print(f"[CoT][Scan] field_order={field_names}")

    progress = tqdm(total=target_total, desc="Scanning CoT prompt lengths")
    for single_dataset in dataset.datasets:
        dataset_name = getattr(single_dataset, "dataset_name", single_dataset.__class__.__name__)
        for trajectory_id, base_index in single_dataset.all_steps:
            if args.max_samples > 0 and processed >= args.max_samples:
                break

            sample = single_dataset.build_step_sample(trajectory_id, base_index)
            visible_cot_text = build_visible_cot_text(field_names, sample["cot_fields_raw"])
            prefix_message = build_prefix_message(sample["image"], sample["lang"], cfg.datasets.vla_data)
            student_message = build_student_message(
                sample["image"],
                sample["lang"],
                cfg.datasets.vla_data,
                field_count=len(field_names),
                token_spec=token_spec,
            )
            teacher_message = build_teacher_message(
                sample["image"],
                sample["lang"],
                visible_cot_text,
                cfg.datasets.vla_data,
            )

            prefix_with_generation_length = _tokenized_length(
                processor,
                prefix_message,
                add_generation_prompt=True,
            )
            prefix_without_generation_length = _tokenized_length(
                processor,
                prefix_message,
                add_generation_prompt=False,
            )
            student_length = _tokenized_length(
                processor,
                student_message,
                add_generation_prompt=True,
            )
            teacher_length = _tokenized_length(
                processor,
                teacher_message,
                add_generation_prompt=not bool(visible_cot_text.strip()),
            )

            record = {
                "dataset_name": dataset_name,
                "episode_index": sample.get("episode_index", -1),
                "frame_index": sample.get("frame_index", -1),
                "task_index": sample.get("task_index", -1),
                "has_cot": sample.get("has_cot", False),
                "prefix_with_generation_length": prefix_with_generation_length,
                "prefix_without_generation_length": prefix_without_generation_length,
                "student_length": student_length,
                "teacher_length": teacher_length,
                "lang": str(sample.get("lang", "")).strip().replace("\n", " "),
                "visible_cot_text": visible_cot_text,
            }
            records.append(record)

            prefix_with_generation_lengths.append(prefix_with_generation_length)
            prefix_without_generation_lengths.append(prefix_without_generation_length)
            student_lengths.append(student_length)
            teacher_lengths.append(teacher_length)
            if record["has_cot"]:
                teacher_lengths_with_cot.append(teacher_length)
            else:
                teacher_lengths_without_cot.append(teacher_length)

            processed += 1
            progress.update(1)

        if args.max_samples > 0 and processed >= args.max_samples:
            break

    progress.close()

    result = {
        "config_yaml": args.config_yaml,
        "mode": args.mode,
        "field_order": field_names,
        "thinking_tokens": {
            "thinking_token": token_spec.thinking_token,
            "start_token": token_spec.start_token,
            "end_token": token_spec.end_token,
        },
        "scanned_samples": processed,
        "total_available_samples": total_available,
        "length_stats": {
            "prefix_with_generation": _summary(prefix_with_generation_lengths),
            "prefix_without_generation": _summary(prefix_without_generation_lengths),
            "student": _summary(student_lengths),
            "teacher": _summary(teacher_lengths),
            "teacher_with_cot": _summary(teacher_lengths_with_cot),
            "teacher_without_cot": _summary(teacher_lengths_without_cot),
        },
        "top_examples": {
            "student": _top_examples(records, "student_length", args.top_k),
            "teacher": _top_examples(records, "teacher_length", args.top_k),
            "prefix_with_generation": _top_examples(records, "prefix_with_generation_length", args.top_k),
            "prefix_without_generation": _top_examples(records, "prefix_without_generation_length", args.top_k),
        },
    }

    print("\n[CoT][Scan] Length Summary")
    _print_summary("prefix_with_generation", result["length_stats"]["prefix_with_generation"])
    _print_summary("prefix_without_generation", result["length_stats"]["prefix_without_generation"])
    _print_summary("student", result["length_stats"]["student"])
    _print_summary("teacher", result["length_stats"]["teacher"])
    _print_summary("teacher_with_cot", result["length_stats"]["teacher_with_cot"])
    _print_summary("teacher_without_cot", result["length_stats"]["teacher_without_cot"])

    print("\n[CoT][Scan] Recommended YAML floors")
    print(
        "cot.student_max_prompt_length >= "
        f"{int(result['length_stats']['student']['max'])}"
    )
    print(
        "cot.teacher_max_prompt_length >= "
        f"{int(result['length_stats']['teacher']['max'])}"
    )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n[CoT][Scan] Saved report to {output_path}")


if __name__ == "__main__":
    main()
