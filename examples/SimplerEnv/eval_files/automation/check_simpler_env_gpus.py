#!/usr/bin/env python3

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_VK_ICD_PATH = REPO_ROOT / "local_nvidia_icd.json"
DEFAULT_SIM_PYTHON = Path("/root/miniconda3/envs/simpler_env/bin/python")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "SimplerEnvGpuChecks"


DEVICE_LOST_PATTERNS = [
    "ErrorDeviceLost",
    "DeviceLostError",
    "vk::Device::waitForFences",
    "vk::Device::waitIdle",
]

VULKAN_EXTENSION_PATTERNS = [
    "ErrorExtensionNotPresent",
]

PATH_ERROR_PATTERNS = [
    "FileNotFoundError",
    "rgb_overlay_path",
]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check whether each GPU can build the SimplerEnv test environment.")
    parser.add_argument("--gpus", type=str, default="8", help="GPU count or comma-separated GPU ids, e.g. 8 or 0,1,2,3")
    parser.add_argument("--timeout-sec", type=int, default=60, help="Per-GPU timeout in seconds")
    parser.add_argument("--sim-python", type=Path, default=DEFAULT_SIM_PYTHON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def parse_gpu_list(raw: str) -> List[int]:
    stripped = raw.strip()
    if not stripped:
        raise ValueError("--gpus must not be empty")
    if "," not in stripped:
        count = int(stripped)
        if count <= 0:
            raise ValueError("--gpus must be a positive integer or a comma-separated list")
        return list(range(count))
    gpu_ids = []
    for part in stripped.split(","):
        text = part.strip()
        if text:
            gpu_ids.append(int(text))
    if not gpu_ids:
        raise ValueError("--gpus must resolve to at least one GPU id")
    return gpu_ids


def classify_result(log_text: str, return_code: int, timed_out: bool) -> str:
    if timed_out:
        return "timeout"
    if "✅ Env built successfully" in log_text:
        return "success"
    if any(pattern in log_text for pattern in DEVICE_LOST_PATTERNS):
        return "device_lost"
    if any(pattern in log_text for pattern in VULKAN_EXTENSION_PATTERNS):
        return "vulkan_extension"
    if any(pattern in log_text for pattern in PATH_ERROR_PATTERNS):
        return "path_error"
    if return_code != 0:
        return "failed"
    return "unknown"


def summarize_error(log_text: str) -> str:
    for line in reversed(log_text.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        if "Traceback" in stripped:
            continue
        return stripped[:300]
    return ""


def build_tsv(rows: List[Dict[str, object]]) -> str:
    columns = ["gpu_id", "state", "return_code", "timed_out", "duration_sec", "log_path", "error_summary"]
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(str(row.get(column, "")) for column in columns))
    return "\n".join(lines) + "\n"


def main() -> int:
    args = build_argparser().parse_args()
    gpu_ids = parse_gpu_list(args.gpus)
    sim_python = args.sim_python.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    script_path = REPO_ROOT / "examples" / "SimplerEnv" / "eval_files" / "test_your_simplerEnv.py"

    for gpu_id in gpu_ids:
        log_path = output_dir / f"gpu_{gpu_id}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONUNBUFFERED"] = "1"
        env["VK_ICD_FILENAMES"] = str(DEFAULT_VK_ICD_PATH)
        env["DISPLAY"] = ""
        started = time.monotonic()
        timed_out = False
        return_code = -1

        print(f"[GPU {gpu_id}] running {script_path}", flush=True)
        with log_path.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(
                [str(sim_python), str(script_path)],
                cwd=str(REPO_ROOT),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            try:
                return_code = process.wait(timeout=args.timeout_sec)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=10)
                return_code = process.returncode if process.returncode is not None else -1

        duration_sec = round(time.monotonic() - started, 3)
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        state = classify_result(log_text, return_code=return_code, timed_out=timed_out)
        row = {
            "gpu_id": gpu_id,
            "state": state,
            "return_code": return_code,
            "timed_out": timed_out,
            "duration_sec": duration_sec,
            "log_path": str(log_path),
            "error_summary": summarize_error(log_text),
        }
        rows.append(row)
        print(f"[GPU {gpu_id}] {state} ({duration_sec}s)", flush=True)

    json_path = output_dir / "summary.json"
    tsv_path = output_dir / "summary.tsv"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tsv_path.write_text(build_tsv(rows), encoding="utf-8")

    print(json_path, flush=True)
    print(tsv_path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
