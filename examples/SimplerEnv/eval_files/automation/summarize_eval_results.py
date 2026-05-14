#!/usr/bin/env python3

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "EvalRuns"


def parse_step_from_name(name: str) -> Optional[int]:
    match = re.search(r"steps_(\d+)_pytorch_model\.pt$", name)
    if not match:
        return None
    return int(match.group(1))


def derive_exp_name(ckpt_path: Path) -> str:
    parts = ckpt_path.resolve().parts
    if len(parts) >= 3 and parts[-2] == "checkpoints":
        return parts[-3]
    return ckpt_path.parent.name


def derive_ckpt_name(ckpt_path: Path) -> str:
    return ckpt_path.stem


TASK_COLUMN_ALIASES = {
    "StackGreenCubeOnYellowCubeBakedTexInScene-v0": "stack_success",
    "PutCarrotOnPlateInScene-v0": "carrot_success",
    "PutSpoonOnTableClothInScene-v0": "spoon_success",
    "PutEggplantInBasketScene-v0": "eggplant_success",
}

FOUR_TASK_ORDER = [
    "StackGreenCubeOnYellowCubeBakedTexInScene-v0",
    "PutCarrotOnPlateInScene-v0",
    "PutSpoonOnTableClothInScene-v0",
    "PutEggplantInBasketScene-v0",
]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate SimplerEnv batch evaluation summaries.")
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory that contains batch directories under results/EvalRuns",
    )
    parser.add_argument(
        "--batch-dir",
        nargs="*",
        default=[],
        type=Path,
        help="One or more explicit batch directories to aggregate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output TSV path (sorted by success rate). Default: print to stdout",
    )
    parser.add_argument(
        "--output-step",
        type=Path,
        default=None,
        help="Optional output TSV path sorted by training step ascending.",
    )
    parser.add_argument(
        "--match",
        type=str,
        default=None,
        help="Keep only rows whose summary.json path contains this substring",
    )
    parser.add_argument(
        "--exclude-match",
        type=str,
        default=None,
        help="Drop rows whose summary.json path contains this substring",
    )
    return parser


def resolve_batch_dirs(eval_root: Path, batch_dirs: Iterable[Path]) -> List[Path]:
    if batch_dirs:
        resolved = []
        for batch_dir in batch_dirs:
            path = batch_dir.expanduser()
            if not path.is_absolute():
                path = (REPO_ROOT / path).resolve()
            else:
                path = path.resolve()
            if not path.is_dir():
                raise FileNotFoundError(f"Batch directory not found: {path}")
            resolved.append(path)
        return resolved

    root = eval_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Eval root not found: {root}")
    return sorted(path for path in root.iterdir() if path.is_dir())


def discover_summary_files(batch_dirs: Iterable[Path], match_text: Optional[str], exclude_match: Optional[str]) -> List[Path]:
    summary_files: List[Path] = []
    for batch_dir in batch_dirs:
        for summary_path in sorted(batch_dir.glob("*/*/summary.json")):
            summary_text = str(summary_path)
            if match_text and match_text not in summary_text:
                continue
            if exclude_match and exclude_match in summary_text:
                continue
            summary_files.append(summary_path.resolve())
    return summary_files


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_task_results(task_entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for entry in task_entries:
        grouped[str(entry.get("task", "unknown"))].append(entry)

    summary: Dict[str, Dict[str, Any]] = {}
    for task_name, entries in grouped.items():
        numeric_scores = [
            float(entry["average_success"])
            for entry in entries
            if entry.get("average_success") is not None
        ]
        states = [str(entry.get("state", "")) for entry in entries]
        summary[task_name] = {
            "mean_average_success": round(sum(numeric_scores) / len(numeric_scores), 6) if numeric_scores else "",
            "num_runs": len(entries),
            "num_success_state": sum(1 for state in states if state == "success"),
            "num_failed_state": sum(1 for state in states if state != "success"),
        }
    return summary


def task_alias(task_name: str) -> str:
    return TASK_COLUMN_ALIASES.get(task_name, task_name.replace("-", "_"))


def format_scalar(value: Any) -> str:
    if value == "":
        return ""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".") if value != int(value) else str(int(value))
    return str(value)


def derive_batch_name(summary_path: Path, eval_root: Path) -> str:
    try:
        return summary_path.relative_to(eval_root.resolve()).parts[0]
    except ValueError:
        return summary_path.parents[2].name


def sort_score(value: Any) -> tuple[int, float]:
    if value == "" or value is None:
        return (1, 0.0)
    return (0, -float(value))


def sort_step(value: Any) -> float:
    if value == "" or value is None:
        return float("inf")
    return float(value)


def build_rows(summary_files: Iterable[Path], eval_root: Path) -> tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    task_names: set[str] = set()

    for summary_path in summary_files:
        payload = load_json(summary_path)
        ckpt_path = Path(payload["ckpt_path"]).resolve()
        task_summary = summarize_task_results(payload.get("tasks", []))
        task_names.update(task_summary.keys())

        mean_scores = [
            float(task_info["mean_average_success"])
            for task_info in task_summary.values()
            if task_info["mean_average_success"] != ""
        ]
        four_task_scores = [
            float(task_summary[task_name]["mean_average_success"])
            for task_name in FOUR_TASK_ORDER
            if task_name in task_summary and task_summary[task_name]["mean_average_success"] != ""
        ]
        four_task_mean_success = round(sum(four_task_scores) / len(four_task_scores), 6) if four_task_scores else ""
        row: Dict[str, Any] = {
            "batch_name": derive_batch_name(summary_path, eval_root),
            "exp_name": payload.get("exp_name", derive_exp_name(ckpt_path)),
            "ckpt_name": payload.get("ckpt_name", derive_ckpt_name(ckpt_path)),
            "step": parse_step_from_name(ckpt_path.name) or "",
            "state": payload.get("state", ""),
            "duration_sec": payload.get("duration_sec", ""),
            "gpu_id": payload.get("gpu_id", ""),
            "port": payload.get("port", ""),
            "num_task_runs": len(payload.get("tasks", [])),
            "num_tasks": len(task_summary),
            "mean_average_success": round(sum(mean_scores) / len(mean_scores), 6) if mean_scores else "",
            "four_task_mean_success": four_task_mean_success,
            "summary_json": str(summary_path),
            "ckpt_path": str(ckpt_path),
        }
        for task_name, task_info in task_summary.items():
            alias = task_alias(task_name)
            row[f"{alias}"] = task_info["mean_average_success"]
            row[f"{alias}_runs"] = task_info["num_runs"]
            row[f"{alias}_ok"] = task_info["num_success_state"]
            row[f"{alias}_fail"] = task_info["num_failed_state"]
        rows.append(row)

    return rows, sorted(task_names)


def build_table(rows: List[Dict[str, Any]], task_names: List[str]) -> str:
    columns = [
        "batch_name",
        "exp_name",
        "ckpt_name",
        "step",
        "state",
        "duration_sec",
        "gpu_id",
        "port",
    ]
    for task_name in task_names:
        alias = task_alias(task_name)
        columns.extend(
            [
                alias,
            ]
        )
    columns.append("four_task_mean_success")

    numeric_columns = {
        "step",
        "duration_sec",
        "gpu_id",
        "port",
        "four_task_mean_success",
    }
    numeric_columns.update(task_alias(task_name) for task_name in task_names)

    formatted_rows: List[Dict[str, str]] = []
    for row in rows:
        formatted_rows.append({column: format_scalar(row.get(column, "")) for column in columns})

    widths = {column: len(column) for column in columns}
    for row in formatted_rows:
        for column in columns:
            widths[column] = max(widths[column], len(row[column]))

    header = "  ".join(
        column.rjust(widths[column]) if column in numeric_columns else column.ljust(widths[column])
        for column in columns
    )
    separator = "  ".join("-" * widths[column] for column in columns)
    lines = [header, separator]
    for row in formatted_rows:
        line = "  ".join(
            row[column].rjust(widths[column]) if column in numeric_columns else row[column].ljust(widths[column])
            for column in columns
        )
        lines.append(line)
    return "\n".join(lines) + "\n"


def main() -> int:
    args = build_argparser().parse_args()
    eval_root = args.eval_root.expanduser().resolve()
    batch_dirs = resolve_batch_dirs(eval_root, args.batch_dir)
    summary_files = discover_summary_files(batch_dirs, args.match, args.exclude_match)
    if not summary_files:
        raise FileNotFoundError("No summary.json files matched the provided inputs")

    rows, task_names = build_rows(summary_files, eval_root)

    success_sorted = sorted(
        rows,
        key=lambda row: (
            sort_score(row["four_task_mean_success"]),
            sort_step(row["step"]),
            str(row["exp_name"]),
            str(row["batch_name"]),
            str(row["ckpt_name"]),
        ),
    )
    table_text = build_table(success_sorted, task_names)

    if args.output is not None:
        output_path = args.output.expanduser()
        if not output_path.is_absolute():
            output_path = (REPO_ROOT / output_path).resolve()
        else:
            output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(table_text, encoding="utf-8")
        print(output_path, flush=True)
    else:
        print(table_text, end="")

    if args.output_step is not None:
        step_sorted = sorted(rows, key=lambda row: sort_step(row["step"]))
        step_table = build_table(step_sorted, task_names)
        step_path = args.output_step.expanduser()
        if not step_path.is_absolute():
            step_path = (REPO_ROOT / step_path).resolve()
        else:
            step_path = step_path.resolve()
        step_path.parent.mkdir(parents=True, exist_ok=True)
        step_path.write_text(step_table, encoding="utf-8")
        print(step_path, flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
