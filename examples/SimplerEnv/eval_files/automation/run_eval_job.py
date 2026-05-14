#!/usr/bin/env python3

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_automation import (
    CancellationRequested,
    append_log,
    append_runner_line,
    derive_ckpt_name,
    derive_exp_name,
    ensure_dir,
    format_duration,
    load_yaml_profile,
    make_runtime_env,
    parse_average_success,
    read_structured_eval_result,
    select_tasks,
    terminate_process_group,
    utc_now_iso,
    wait_for_process,
    wait_for_websocket_server,
    write_json,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SimplerEnv evaluation for a single checkpoint.")
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--gpu-id", required=True, type=int)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--video-root", type=Path, default=None)
    parser.add_argument("--only-task", nargs="*", default=None)
    return parser


def build_server_command(profile: Dict[str, Any], ckpt_path: Path, port: int) -> List[str]:
    command = [
        str(Path(profile["runtime"]["server_python"]).expanduser().resolve()),
        "-u",
        "deployment/model_server/server_policy.py",
        "--ckpt_path",
        str(ckpt_path),
        "--port",
        str(port),
    ]
    if profile["server"].get("use_bf16", False):
        command.append("--use_bf16")
    command.extend(["--idle_timeout", str(profile["server"].get("idle_timeout", -1))])
    return command


def build_eval_command(
    profile: Dict[str, Any],
    task: Dict[str, Any],
    ckpt_path: Path,
    port: int,
    logging_dir: Path,
    additional_env_save_tags: str,
    result_json_path: Path,
) -> List[str]:
    command = [
        str(Path(profile["runtime"]["sim_python"]).expanduser().resolve()),
        "-u",
        "examples/SimplerEnv/eval_files/start_simpler_env.py",
        "--ckpt-path",
        str(ckpt_path),
        "--port",
        str(port),
        "--host",
        str(profile["server"].get("host", "127.0.0.1")),
        "--policy-setup",
        str(task["policy_setup"]),
        "--action-scale",
        str(task.get("action_scale", 1.0)),
        "--control-freq",
        str(task["control_freq"]),
        "--sim-freq",
        str(task["sim_freq"]),
        "--max-episode-steps",
        str(task["max_episode_steps"]),
        "--env-name",
        str(task["env_name"]),
        "--scene-name",
        str(task["scene_name"]),
        "--robot",
        str(task["robot"]),
        "--logging-dir",
        str(logging_dir),
        "--result-json",
        str(result_json_path),
        "--additional-env-save-tags",
        additional_env_save_tags,
        "--obj-variation-mode",
        str(task["obj_variation_mode"]),
        "--robot-init-x-range",
        *[str(value) for value in task["robot_init_x_range"]],
        "--robot-init-y-range",
        *[str(value) for value in task["robot_init_y_range"]],
        "--robot-init-rot-quat-center",
        *[str(value) for value in task["robot_init_rot_quat_center"]],
        "--robot-init-rot-rpy-range",
        *[str(value) for value in task["robot_init_rot_rpy_range"]],
    ]

    if task.get("rgb_overlay_path"):
        command.extend(["--rgb-overlay-path", str(task["rgb_overlay_path"])])
    if task["obj_variation_mode"] == "episode":
        command.extend(["--obj-episode-range", *[str(value) for value in task["obj_episode_range"]]])
    if task["obj_variation_mode"] == "xy":
        command.extend(["--obj-init-x-range", *[str(value) for value in task["obj_init_x_range"]]])
        command.extend(["--obj-init-y-range", *[str(value) for value in task["obj_init_y_range"]]])
    if task.get("obs_camera_name"):
        command.extend(["--obs-camera-name", str(task["obs_camera_name"])])
    if task.get("enable_raytracing"):
        command.append("--enable-raytracing")
    if task.get("additional_env_build_kwargs"):
        command.append("--additional-env-build-kwargs")
        for key, value in task["additional_env_build_kwargs"].items():
            command.append(f"{key}={value}")
    return command


def main() -> int:
    args = build_argparser().parse_args()
    profile = load_yaml_profile(args.profile.resolve())
    ckpt_path = args.ckpt.expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    output_dir = ensure_dir(args.output_dir.expanduser().resolve())
    env_logs_dir = ensure_dir(output_dir / "env_logs")
    if args.video_root:
        # Mirror the batch/exp/ckpt structure under video_root so videos land on the large disk.
        # output_dir layout: output_root / batch_name / exp_name / ckpt_name
        batch_name = output_dir.parent.parent.name
        exp_name = output_dir.parent.name
        ckpt_name = output_dir.name
        video_root = args.video_root.expanduser().resolve()
        logging_dir = ensure_dir(video_root / batch_name / exp_name / ckpt_name / profile["eval_defaults"].get("logging_subdir", "simpler_env_results"))
    else:
        logging_dir = ensure_dir(output_dir / profile["eval_defaults"].get("logging_subdir", "simpler_env_results"))
    runner_log = output_dir / "runner.log"
    server_log = output_dir / "server.log"
    status_path = output_dir / "status.json"
    summary_path = output_dir / "summary.json"
    meta_path = output_dir / "meta.json"

    runtime_env = make_runtime_env(profile, args.gpu_id)
    tasks = select_tasks(profile, args.only_task)
    job_pid = os.getpid()
    shutdown_event = threading.Event()
    server_process: subprocess.Popen[Any] | None = None

    def request_shutdown(signum, frame) -> None:  # type: ignore[no-untyped-def]
        shutdown_event.set()
        append_log(runner_log, f"Received signal {signum}, shutting down")
        terminate_process_group(server_process)
        write_json(
            status_path,
            {
                "state": "cancelled",
                "ended_at": utc_now_iso(),
                "job_pid": job_pid,
                "signal": signum,
            },
        )
        raise SystemExit(130)

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)

    meta = {
        "ckpt_path": str(ckpt_path),
        "exp_name": derive_exp_name(ckpt_path),
        "ckpt_name": derive_ckpt_name(ckpt_path),
        "gpu_id": args.gpu_id,
        "port": args.port,
        "profile": str(args.profile.resolve()),
        "output_dir": str(output_dir),
        "tasks": [task["name"] for task in tasks],
        "started_at": utc_now_iso(),
    }
    write_json(meta_path, meta)
    write_json(
        status_path,
        {
            "state": "running",
            "started_at": meta["started_at"],
            "job_pid": job_pid,
            "gpu_id": args.gpu_id,
            "port": args.port,
            "ckpt_path": str(ckpt_path),
            "output_dir": str(output_dir),
        },
    )

    start_time = time.monotonic()
    task_results: List[Dict[str, Any]] = []
    summary_state = "success"

    try:
        append_runner_line(runner_log, f"Starting server for {ckpt_path}")
        with server_log.open("a", encoding="utf-8") as server_handle:
            server_process = subprocess.Popen(
                build_server_command(profile, ckpt_path, args.port),
                cwd=str(REPO_ROOT),
                env=runtime_env,
                stdout=server_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        wait_for_websocket_server(
            host=str(profile["server"].get("host", "127.0.0.1")),
            port=args.port,
            timeout_sec=int(profile["server"].get("start_timeout_sec", 180)),
            process=server_process,
        )
        append_runner_line(runner_log, f"Server is ready on port {args.port}")

        for task in tasks:
            repeats = int(task.get("repeats", 1))
            timeout_sec = int(float(task["per_task_timeout_min"]) * 60)
            for repeat_idx in range(1, repeats + 1):
                task_name = task["name"]
                env_log_path = env_logs_dir / f"{task_name}_run{repeat_idx}.log"
                result_json_path = env_logs_dir / f"{task_name}_run{repeat_idx}_result.json"
                task_started_at = utc_now_iso()
                additional_tags = f"{task_name}_run{repeat_idx}"
                append_runner_line(runner_log, f"Running task={task_name} repeat={repeat_idx}")
                with env_log_path.open("a", encoding="utf-8") as env_log_handle:
                    eval_process = subprocess.Popen(
                        build_eval_command(
                            profile=profile,
                            task=task,
                            ckpt_path=ckpt_path,
                            port=args.port,
                            logging_dir=logging_dir,
                            additional_env_save_tags=additional_tags,
                            result_json_path=result_json_path,
                        ),
                        cwd=str(REPO_ROOT),
                        env=runtime_env,
                        stdout=env_log_handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                        start_new_session=True,
                    )

                result: Dict[str, Any] = {
                    "task": task_name,
                    "repeat": repeat_idx,
                    "started_at": task_started_at,
                    "log_path": str(env_log_path),
                    "result_json_path": str(result_json_path),
                }
                try:
                    return_code = wait_for_process(
                        eval_process,
                        timeout_sec=timeout_sec,
                        shutdown_event=shutdown_event,
                    )
                    result["return_code"] = return_code
                    structured_result = read_structured_eval_result(result_json_path)
                    result["average_success"] = (
                        structured_result.get("average_success")
                        if structured_result is not None
                        else parse_average_success(env_log_path)
                    )
                    if structured_result is not None:
                        result["num_episodes"] = structured_result.get("num_episodes")
                        result["num_success"] = structured_result.get("num_success")
                    result["ended_at"] = utc_now_iso()
                    result["state"] = "success" if return_code == 0 else "failed"
                    task_results.append(result)
                    if return_code != 0:
                        raise RuntimeError(f"Task {task_name} repeat {repeat_idx} failed with code {return_code}")
                except (TimeoutError, CancellationRequested, Exception) as exc:
                    terminate_process_group(eval_process)
                    result["ended_at"] = utc_now_iso()
                    result["state"] = "timeout" if isinstance(exc, TimeoutError) else "failed"
                    result["error"] = str(exc)
                    structured_result = read_structured_eval_result(result_json_path)
                    result["average_success"] = (
                        structured_result.get("average_success")
                        if structured_result is not None
                        else parse_average_success(env_log_path)
                    )
                    if structured_result is not None:
                        result["num_episodes"] = structured_result.get("num_episodes")
                        result["num_success"] = structured_result.get("num_success")
                    task_results.append(result)
                    raise

        append_runner_line(runner_log, "All tasks completed successfully")
    except SystemExit:
        raise
    except Exception as exc:
        summary_state = "failed"
        append_runner_line(runner_log, f"Job failed: {exc}")
    finally:
        terminate_process_group(server_process)
        duration_sec = round(time.monotonic() - start_time, 3)
        summary = {
            "state": summary_state,
            "ckpt_path": str(ckpt_path),
            "exp_name": derive_exp_name(ckpt_path),
            "ckpt_name": derive_ckpt_name(ckpt_path),
            "gpu_id": args.gpu_id,
            "port": args.port,
            "duration_sec": duration_sec,
            "duration_human": format_duration(duration_sec),
            "tasks": task_results,
            "started_at": meta["started_at"],
            "ended_at": utc_now_iso(),
        }
        write_json(summary_path, summary)
        write_json(
            status_path,
            {
                "state": summary_state,
                "started_at": meta["started_at"],
                "ended_at": summary["ended_at"],
                "job_pid": job_pid,
                "gpu_id": args.gpu_id,
                "port": args.port,
                "ckpt_path": str(ckpt_path),
                "output_dir": str(output_dir),
            },
        )

    return 0 if summary_state == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
