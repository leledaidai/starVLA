#!/usr/bin/env python3

import argparse
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
DEFAULT_PROFILE = REPO_ROOT / "examples" / "SimplerEnv" / "eval_yaml" / "bridge_default.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "EvalRuns"
DEFAULT_RUNNER_PYTHON = Path("/root/miniconda3/envs/simpler/bin/python")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SimplerEnv batch evaluation and then aggregate results into a sorted table."
    )
    parser.add_argument(
        "--ckpt-path",
        nargs="+",
        required=True,
        type=Path,
        help="Checkpoint .pt file(s) or experiment directory/directories containing checkpoints.",
    )
    parser.add_argument(
        "--gpus",
        required=True,
        type=str,
        help="GPU count or comma-separated GPU ids, e.g. 8 or 0,1,2,3.",
    )
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument(
        "--runner-python",
        type=Path,
        default=DEFAULT_RUNNER_PYTHON,
        help="Python executable used to run run_eval_batch.py and summarize_eval_results.py.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--batch-name",
        type=str,
        default=None,
        help="Batch output directory name under output-root. Default: derived from ckpt-path and timestamp.",
    )
    parser.add_argument(
        "--aggregate-name",
        type=str,
        default="aggregate.tsv",
        help="Aggregate table filename written under the batch directory.",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=None,
        help="If set, rollout videos are saved under this root (e.g. a large disk) while "
        "result files and logs stay under --output-root. Mirrors the same batch/exp/ckpt "
        "directory structure.",
    )
    parser.add_argument(
        "--min-step",
        type=int,
        default=20000,
        help="Minimum checkpoint step to evaluate. Default: 20000. Use --min-step 0 to include all steps.",
    )
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--step-list", type=str, default=None, help="Comma-separated step list, e.g. 35000,40000.")
    parser.add_argument("--match", type=str, default=None)
    parser.add_argument("--exclude-match", type=str, default=None)
    parser.add_argument("--only-task", nargs="*", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Show selected eval jobs without running evaluation.")
    return parser


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        return (REPO_ROOT / path).resolve()
    return path.resolve()


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def format_gpus_for_name(raw_gpus: str) -> str:
    stripped = raw_gpus.strip()
    if "," not in stripped:
        return f"{sanitize_name(stripped)}gpu"
    return f"gpus{sanitize_name(stripped.replace(',', '-'))}"


def format_step_filter_for_name(args: argparse.Namespace) -> str:
    parts = []
    if args.step_list:
        parts.append(f"steps{sanitize_name(args.step_list.replace(',', '-'))}")
    else:
        if args.min_step is not None:
            parts.append(f"ge{args.min_step}")
        if args.max_step is not None:
            parts.append(f"le{args.max_step}")
    if args.match:
        parts.append(f"match_{sanitize_name(args.match)}")
    if args.exclude_match:
        parts.append(f"exclude_{sanitize_name(args.exclude_match)}")
    return "_".join(parts)


def make_default_batch_name(paths: List[Path], args: argparse.Namespace) -> str:
    if len(paths) == 1:
        stem = paths[0].stem if paths[0].is_file() else paths[0].name
    else:
        stem = "selected_ckpts"
    name_parts = [
        sanitize_name(stem),
        "simpler",
        format_gpus_for_name(args.gpus),
    ]
    step_filter = format_step_filter_for_name(args)
    if step_filter:
        name_parts.append(step_filter)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name_parts.append(timestamp)
    return "_".join(part for part in name_parts if part)


def append_optional(command: List[str], flag: str, value: object | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def build_eval_command(args: argparse.Namespace, ckpt_paths: List[Path], batch_name: str) -> List[str]:
    runner_python = resolve_runner_python(args.runner_python)
    command = [
        str(runner_python),
        str(SCRIPT_DIR / "run_eval_batch.py"),
        "--profile",
        str(resolve_path(args.profile)),
        "--gpus",
        args.gpus,
        "--batch-name",
        batch_name,
        "--output-root",
        str(resolve_path(args.output_root)),
    ]

    for ckpt_path in ckpt_paths:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt_path}")
        if ckpt_path.is_file():
            command.extend(["--ckpt", str(ckpt_path)])
        elif ckpt_path.is_dir():
            command.extend(["--exp-dir", str(ckpt_path)])
        else:
            raise ValueError(f"Unsupported checkpoint path: {ckpt_path}")

    append_optional(command, "--video-root", args.video_root)
    append_optional(command, "--min-step", args.min_step)
    append_optional(command, "--max-step", args.max_step)
    append_optional(command, "--step-list", args.step_list)
    append_optional(command, "--match", args.match)
    append_optional(command, "--exclude-match", args.exclude_match)
    if args.only_task:
        command.extend(["--only-task", *args.only_task])
    if args.resume:
        command.append("--resume")
    if args.retry_failed:
        command.append("--retry-failed")
    if args.dry_run:
        command.append("--dry-run")
    return command


def build_summary_command(args: argparse.Namespace, batch_dir: Path, aggregate_path: Path, aggregate_step_path: Path) -> List[str]:
    runner_python = resolve_runner_python(args.runner_python)
    return [
        str(runner_python),
        str(SCRIPT_DIR / "summarize_eval_results.py"),
        "--batch-dir",
        str(batch_dir),
        "--output",
        str(aggregate_path),
        "--output-step",
        str(aggregate_step_path),
    ]


def resolve_runner_python(path: Path) -> Path:
    resolved = resolve_path(path)
    if resolved.exists():
        return resolved
    if path == DEFAULT_RUNNER_PYTHON:
        return Path(sys.executable).resolve()
    raise FileNotFoundError(f"Runner python does not exist: {resolved}")


def main() -> int:
    args = build_argparser().parse_args()
    ckpt_paths = [resolve_path(path) for path in args.ckpt_path]
    batch_name = args.batch_name or make_default_batch_name(ckpt_paths, args)
    output_root = resolve_path(args.output_root)
    batch_dir = output_root / batch_name
    aggregate_path = batch_dir / args.aggregate_name
    aggregate_name_stem = Path(args.aggregate_name).stem
    aggregate_step_path = batch_dir / f"{aggregate_name_stem}_step.tsv"

    eval_command = build_eval_command(args, ckpt_paths, batch_name)
    print("[eval_and_summarize] running evaluation:", flush=True)
    print(" ".join(eval_command), flush=True)
    eval_result = subprocess.run(eval_command, cwd=str(REPO_ROOT))
    if args.dry_run:
        return eval_result.returncode

    print("[eval_and_summarize] aggregating results:", flush=True)
    summary_command = build_summary_command(args, batch_dir, aggregate_path, aggregate_step_path)
    print(" ".join(summary_command), flush=True)
    summary_result = subprocess.run(summary_command, cwd=str(REPO_ROOT))
    if summary_result.returncode == 0:
        print(f"[eval_and_summarize] aggregate table (by success): {aggregate_path}", flush=True)
        print(f"[eval_and_summarize] aggregate table (by step):    {aggregate_step_path}", flush=True)

    return eval_result.returncode or summary_result.returncode


if __name__ == "__main__":
    sys.exit(main())
