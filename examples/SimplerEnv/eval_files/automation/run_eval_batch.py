#!/usr/bin/env python3

import argparse
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_automation import (
    DEFAULT_OUTPUT_ROOT,
    PortAllocator,
    build_job_output_dir,
    build_summary_tsv,
    derive_ckpt_name,
    derive_exp_name,
    ensure_dir,
    evaluate_resume_decision,
    filter_jobs,
    format_duration,
    get_port_range,
    load_yaml_profile,
    parse_step_from_name,
    resolve_checkpoint_jobs,
    utc_now_iso,
    write_json,
)


@dataclass
class JobSpec:
    ckpt_path: Path
    output_dir: Path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch SimplerEnv evaluation scheduler.")
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument(
        "--gpus",
        required=True,
        type=str,
        help="GPU count or comma-separated GPU ids, e.g. 8 or 0,1,2,3",
    )
    parser.add_argument("--batch-name", required=True, type=str)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--ckpt", nargs="*", default=[])
    parser.add_argument("--exp-dir", nargs="*", default=[])
    parser.add_argument("--ckpt-glob", nargs="*", default=[])
    parser.add_argument("--manifest", nargs="*", default=[])
    parser.add_argument("--min-step", type=int, default=None)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--step-list", type=str, default=None, help="Comma-separated step list")
    parser.add_argument("--match", type=str, default=None)
    parser.add_argument("--exclude-match", type=str, default=None)
    parser.add_argument("--video-root", type=Path, default=None)
    parser.add_argument("--only-task", nargs="*", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_gpu_list(raw: str) -> List[int]:
    stripped = raw.strip()
    if not stripped:
        raise ValueError("--gpus must specify at least one GPU id")

    # Convenience mode: `--gpus 8` means use GPUs [0..7].
    # Explicit lists like `--gpus 0,2,4,6` still work as before.
    if "," not in stripped:
        gpu_count = int(stripped)
        if gpu_count <= 0:
            raise ValueError("--gpus must be a positive integer or a comma-separated GPU id list")
        return list(range(gpu_count))

    gpu_ids = []
    for part in stripped.split(","):
        gpu_id_text = part.strip()
        if not gpu_id_text:
            continue
        gpu_ids.append(int(gpu_id_text))
    if not gpu_ids:
        raise ValueError("--gpus must specify at least one GPU id")
    return gpu_ids


def parse_step_list(raw: Optional[str]) -> Optional[List[int]]:
    if not raw:
        return None
    steps = []
    for part in raw.split(","):
        stripped = part.strip()
        if stripped:
            steps.append(int(stripped))
    return steps or None


def render_job_line(index: int, total: int, job: JobSpec) -> str:
    return (
        f"[{index:03d}/{total:03d}] "
        f"{derive_exp_name(job.ckpt_path)}/{derive_ckpt_name(job.ckpt_path)} "
        f"-> {job.output_dir}"
    )


def build_jobs(args: argparse.Namespace) -> List[JobSpec]:
    resolved = resolve_checkpoint_jobs(
        ckpt_paths=args.ckpt,
        exp_dirs=args.exp_dir,
        glob_patterns=args.ckpt_glob,
        manifests=args.manifest,
    )
    if not resolved:
        raise ValueError("No checkpoints matched the provided inputs")

    filtered = filter_jobs(
        ckpt_paths=resolved,
        min_step=args.min_step,
        max_step=args.max_step,
        step_list=parse_step_list(args.step_list),
        match_text=args.match,
        exclude_match=args.exclude_match,
    )
    if not filtered:
        raise ValueError("No checkpoints remained after filtering")

    output_root = args.output_root.expanduser().resolve()
    return [JobSpec(ckpt_path=path, output_dir=build_job_output_dir(output_root, args.batch_name, path)) for path in filtered]


def write_batch_files(batch_root: Path, jobs: List[JobSpec], rows: List[Dict[str, Any]], live_status: Dict[str, Any]) -> None:
    ensure_dir(batch_root)
    resolved_jobs_path = batch_root / "resolved_jobs.txt"
    resolved_jobs_path.write_text(
        "".join(f"{job.ckpt_path}\t{job.output_dir}\n" for job in jobs),
        encoding="utf-8",
    )
    (batch_root / "summary.tsv").write_text(build_summary_tsv(rows), encoding="utf-8")
    write_json(batch_root / "live_status.json", live_status)


def main() -> int:
    args = build_argparser().parse_args()
    if not any([args.ckpt, args.exp_dir, args.ckpt_glob, args.manifest]):
        raise ValueError("At least one of --ckpt/--exp-dir/--ckpt-glob/--manifest must be provided")
    profile = load_yaml_profile(args.profile.expanduser().resolve())
    jobs = build_jobs(args)
    batch_root = args.output_root.expanduser().resolve() / args.batch_name
    gpu_ids = parse_gpu_list(args.gpus)
    start_port, end_port = get_port_range(profile)
    port_allocator = PortAllocator(start_port=start_port, end_port=end_port)

    if args.dry_run:
        print(f"Profile: {args.profile.expanduser().resolve()}")
        print(f"Batch: {args.batch_name}")
        print(f"Output root: {args.output_root.expanduser().resolve()}")
        print(f"GPUs: {gpu_ids}")
        print(f"Job count: {len(jobs)}")
        for idx, job in enumerate(jobs, start=1):
            print(render_job_line(idx, len(jobs), job))
        return 0

    ensure_dir(batch_root)
    queue_items: "queue.Queue[JobSpec]" = queue.Queue()
    for job in jobs:
        queue_items.put(job)

    rows_lock = threading.Lock()
    rows: List[Dict[str, Any]] = []
    live_status: Dict[str, Any] = {
        "batch_name": args.batch_name,
        "profile": str(args.profile.expanduser().resolve()),
        "started_at": utc_now_iso(),
        "workers": {},
        "totals": {"queued": len(jobs), "running": 0, "success": 0, "failed": 0, "skipped": 0},
    }
    shutdown_event = threading.Event()

    def update_files() -> None:
        write_batch_files(batch_root, jobs, rows, live_status)

    def record_row(row: Dict[str, Any]) -> None:
        with rows_lock:
            rows.append(row)
            update_files()

    def set_worker_state(worker_name: str, payload: Dict[str, Any]) -> None:
        with rows_lock:
            live_status["workers"][worker_name] = payload
            update_files()

    def change_total(key: str, delta: int) -> None:
        with rows_lock:
            live_status["totals"][key] = max(0, live_status["totals"][key] + delta)
            update_files()

    def handle_signal(signum, frame) -> None:  # type: ignore[no-untyped-def]
        shutdown_event.set()
        print(f"[batch] received signal {signum}, stopping after current child cleanup", flush=True)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    update_files()

    def worker_loop(gpu_id: int) -> None:
        worker_name = f"gpu{gpu_id}"
        active_process: Optional[subprocess.Popen[Any]] = None
        active_port: Optional[int] = None
        while not shutdown_event.is_set():
            try:
                job = queue_items.get_nowait()
            except queue.Empty:
                set_worker_state(worker_name, {"state": "idle", "gpu_id": gpu_id})
                return

            output_dir = ensure_dir(job.output_dir)
            status_path = output_dir / "status.json"
            if args.resume:
                decision, existing = evaluate_resume_decision(status_path, retry_failed=args.retry_failed)
                if decision != "run":
                    state = decision.replace("skip_", "")
                    row = {
                        "exp_name": derive_exp_name(job.ckpt_path),
                        "ckpt_name": derive_ckpt_name(job.ckpt_path),
                        "step": parse_step_from_name(job.ckpt_path.name) or "",
                        "state": state,
                        "gpu_id": gpu_id,
                        "port": existing.get("port", "") if existing else "",
                        "duration_sec": existing.get("duration_sec", "") if existing else "",
                        "output_dir": str(output_dir),
                    }
                    record_row(row)
                    change_total("skipped", 1)
                    queue_items.task_done()
                    continue

            try:
                active_port = port_allocator.acquire()
                set_worker_state(
                    worker_name,
                    {
                        "state": "running",
                        "gpu_id": gpu_id,
                        "port": active_port,
                        "ckpt_path": str(job.ckpt_path),
                        "output_dir": str(output_dir),
                        "started_at": utc_now_iso(),
                    },
                )
                change_total("running", 1)
                started = time.monotonic()
                command = [
                    sys.executable,
                    str((SCRIPT_DIR / "run_eval_job.py").resolve()),
                    "--profile",
                    str(args.profile.expanduser().resolve()),
                    "--ckpt",
                    str(job.ckpt_path),
                    "--gpu-id",
                    str(gpu_id),
                    "--port",
                    str(active_port),
                    "--output-dir",
                    str(output_dir),
                ]
                if args.video_root:
                    command.extend(["--video-root", str(args.video_root)])
                if args.only_task:
                    command.extend(["--only-task", *args.only_task])
                active_process = subprocess.Popen(
                    command,
                    cwd=str(REPO_ROOT),
                    env=os.environ.copy(),
                    start_new_session=True,
                )
                while True:
                    return_code = active_process.poll()
                    if return_code is not None:
                        break
                    if shutdown_event.is_set():
                        try:
                            os.killpg(os.getpgid(active_process.pid), signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        return_code = active_process.wait(timeout=10)
                        break
                    time.sleep(1.0)
                duration_sec = round(time.monotonic() - started, 3)
                state = "cancelled" if shutdown_event.is_set() else ("success" if return_code == 0 else "failed")
                row = {
                    "exp_name": derive_exp_name(job.ckpt_path),
                    "ckpt_name": derive_ckpt_name(job.ckpt_path),
                    "step": parse_step_from_name(job.ckpt_path.name) or "",
                    "state": state,
                    "gpu_id": gpu_id,
                    "port": active_port,
                    "duration_sec": duration_sec,
                    "output_dir": str(output_dir),
                }
                record_row(row)
                change_total("running", -1)
                if state == "success":
                    change_total("success", 1)
                elif state == "failed":
                    change_total("failed", 1)
                print(
                    f"[{worker_name}] {state} {derive_exp_name(job.ckpt_path)}/{derive_ckpt_name(job.ckpt_path)} "
                    f"in {format_duration(duration_sec)}",
                    flush=True,
                )
            except Exception as exc:
                duration_sec = 0.0
                row = {
                    "exp_name": derive_exp_name(job.ckpt_path),
                    "ckpt_name": derive_ckpt_name(job.ckpt_path),
                    "step": parse_step_from_name(job.ckpt_path.name) or "",
                    "state": "failed",
                    "gpu_id": gpu_id,
                    "port": active_port or "",
                    "duration_sec": duration_sec,
                    "output_dir": str(output_dir),
                }
                record_row(row)
                if live_status["totals"]["running"] > 0:
                    change_total("running", -1)
                change_total("failed", 1)
                print(f"[{worker_name}] failed before job completion: {exc}", flush=True)
            finally:
                if active_process is not None and active_process.poll() is None:
                    try:
                        os.killpg(os.getpgid(active_process.pid), signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                if active_port is not None:
                    port_allocator.release(active_port)
                set_worker_state(worker_name, {"state": "idle", "gpu_id": gpu_id})
                active_process = None
                active_port = None
                queue_items.task_done()

    threads = [threading.Thread(target=worker_loop, args=(gpu_id,), daemon=True) for gpu_id in gpu_ids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    with rows_lock:
        live_status["ended_at"] = utc_now_iso()
        update_files()

    failures = [row for row in rows if row["state"] == "failed"]
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
