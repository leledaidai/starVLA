import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import yaml
import websockets.sync.client


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "EvalRuns"
CHECKPOINT_PATTERN = re.compile(r"(steps_(\d+)_pytorch_model\.pt|pytorch_model\.pt)$")
AVERAGE_SUCCESS_PATTERN = re.compile(r"Average success\s+([0-9]*\.?[0-9]+)")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_log(log_path: Path, message: str, also_stdout: bool = False) -> None:
    ensure_dir(log_path.parent)
    line = f"[{utc_now_iso()}] {message}"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    if also_stdout:
        print(line, flush=True)


def load_yaml_profile(profile_path: Path) -> Dict[str, Any]:
    with profile_path.open("r", encoding="utf-8") as handle:
        profile = yaml.safe_load(handle)
    if not isinstance(profile, dict):
        raise ValueError(f"Invalid profile at {profile_path}: root must be a mapping")
    for key in ("runtime", "server", "eval_defaults", "tasks"):
        if key not in profile:
            raise ValueError(f"Profile {profile_path} is missing '{key}'")
    if not isinstance(profile["tasks"], list) or not profile["tasks"]:
        raise ValueError(f"Profile {profile_path} must define a non-empty tasks list")
    return profile


def normalize_path(value: str) -> str:
    return str(Path(value).expanduser().resolve())


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


def matches_checkpoint_name(path: Path) -> bool:
    return bool(CHECKPOINT_PATTERN.search(path.name))


def collect_checkpoints_from_exp_dir(exp_dir: Path) -> List[Path]:
    exp_dir = exp_dir.expanduser().resolve()
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
    checkpoints_dir = exp_dir / "checkpoints"
    search_root = checkpoints_dir if checkpoints_dir.is_dir() else exp_dir
    paths = [path.resolve() for path in search_root.rglob("*") if path.is_file() and matches_checkpoint_name(path)]
    return sorted(paths)


def collect_checkpoints_from_glob(pattern: str) -> List[Path]:
    matches = [path.resolve() for path in REPO_ROOT.glob(pattern) if path.is_file() and matches_checkpoint_name(path)]
    if matches:
        return sorted(matches)
    absolute_pattern = Path(pattern).expanduser()
    if absolute_pattern.is_absolute():
        return sorted(
            path.resolve()
            for path in absolute_pattern.parent.glob(absolute_pattern.name)
            if path.is_file() and matches_checkpoint_name(path)
        )
    return []


def read_manifest(manifest_path: Path) -> List[Path]:
    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    paths: List[Path] = []
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        candidate = Path(line).expanduser()
        if not candidate.is_absolute():
            candidate = (manifest_path.parent / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Manifest entry not found: {candidate}")
        paths.append(candidate)
    return paths


def resolve_checkpoint_jobs(
    ckpt_paths: Iterable[str],
    exp_dirs: Iterable[str],
    glob_patterns: Iterable[str],
    manifests: Iterable[str],
) -> List[Path]:
    resolved: Dict[str, Path] = {}
    for raw_path in ckpt_paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        if not matches_checkpoint_name(path):
            raise ValueError(f"Unsupported checkpoint filename: {path}")
        resolved[str(path)] = path
    for exp_dir in exp_dirs:
        for path in collect_checkpoints_from_exp_dir(Path(exp_dir)):
            resolved[str(path)] = path
    for pattern in glob_patterns:
        for path in collect_checkpoints_from_glob(pattern):
            resolved[str(path)] = path
    for manifest in manifests:
        for path in read_manifest(Path(manifest)):
            if not matches_checkpoint_name(path):
                raise ValueError(f"Unsupported checkpoint filename: {path}")
            resolved[str(path)] = path
    return list(resolved.values())


def filter_jobs(
    ckpt_paths: Iterable[Path],
    min_step: Optional[int],
    max_step: Optional[int],
    step_list: Optional[List[int]],
    match_text: Optional[str],
    exclude_match: Optional[str],
) -> List[Path]:
    selected: List[Path] = []
    step_filter = set(step_list or [])
    for path in ckpt_paths:
        path_text = str(path)
        step = parse_step_from_name(path.name)
        if min_step is not None and (step is None or step < min_step):
            continue
        if max_step is not None and (step is None or step > max_step):
            continue
        if step_filter and step not in step_filter:
            continue
        if match_text and match_text not in path_text:
            continue
        if exclude_match and exclude_match in path_text:
            continue
        selected.append(path)
    return sorted(
        selected,
        key=lambda path: (
            derive_exp_name(path),
            parse_step_from_name(path.name) if parse_step_from_name(path.name) is not None else float("inf"),
            path.name,
        ),
    )


def build_job_output_dir(output_root: Path, batch_name: str, ckpt_path: Path) -> Path:
    return output_root / batch_name / derive_exp_name(ckpt_path) / derive_ckpt_name(ckpt_path)


def merge_task_config(task: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(defaults)
    merged.update(task)
    return merged


def select_tasks(profile: Dict[str, Any], only_tasks: Optional[List[str]]) -> List[Dict[str, Any]]:
    selected_names = set(only_tasks or [])
    tasks = []
    for task in profile["tasks"]:
        if "name" not in task:
            raise ValueError("Each task in the profile must have a 'name'")
        if only_tasks and task["name"] not in selected_names:
            continue
        tasks.append(merge_task_config(task, profile["eval_defaults"]))
    if not tasks:
        if only_tasks:
            raise ValueError(f"No tasks matched --only-task {only_tasks}")
        raise ValueError("Profile selected zero tasks")
    return tasks


def make_runtime_env(profile: Dict[str, Any], gpu_id: Optional[int]) -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("PYTHONUNBUFFERED", "1")
    for key, value in profile["runtime"].get("extra_env", {}).items():
        env[str(key)] = str(value)
    env["SimplerEnv_PATH"] = str(Path(profile["runtime"]["simpler_env_path"]).expanduser().resolve())
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return env


def check_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((host, port)) == 0


def wait_for_port(host: str, port: int, timeout_sec: int, process: subprocess.Popen[Any]) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited early with code {process.returncode}")
        if check_port_open(host, port):
            return
        time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for {host}:{port}")


def wait_for_websocket_server(host: str, port: int, timeout_sec: int, process: subprocess.Popen[Any]) -> None:
    """
    Wait until the websocket server accepts a valid websocket handshake and
    sends its metadata frame. This avoids generating fake handshake errors in
    the server log that come from raw TCP probing.
    """
    deadline = time.monotonic() + timeout_sec
    uri = f"ws://{host}:{port}"

    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited early with code {process.returncode}")
        try:
            with websockets.sync.client.connect(
                uri,
                compression=None,
                max_size=None,
                open_timeout=5,
                ping_interval=None,
            ) as conn:
                _ = msgpack_safe_recv_metadata(conn)
                return
        except Exception:
            time.sleep(1.0)

    raise TimeoutError(f"Timed out waiting for websocket server {uri}")


def msgpack_safe_recv_metadata(conn: Any) -> Any:
    """
    Best-effort metadata receive for server readiness checks. We intentionally
    do not fully decode the payload here because readiness only requires a
    successful websocket handshake plus one first frame from the server.
    """
    return conn.recv()


def terminate_process_group(process: Optional[subprocess.Popen[Any]], grace_sec: int = 10) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_sec
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.5)
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except ProcessLookupError:
        return


def parse_average_success(log_path: Path) -> Optional[float]:
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = AVERAGE_SUCCESS_PATTERN.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def read_structured_eval_result(result_json_path: Path) -> Optional[Dict[str, Any]]:
    if not result_json_path.exists():
        return None
    try:
        return json.loads(result_json_path.read_text(encoding="utf-8"))
    except Exception:
        return None


class CancellationRequested(Exception):
    pass


def wait_for_process(
    process: subprocess.Popen[Any],
    timeout_sec: int,
    shutdown_event: Optional[threading.Event] = None,
    poll_interval_sec: float = 1.0,
) -> int:
    deadline = time.monotonic() + timeout_sec
    while True:
        if shutdown_event is not None and shutdown_event.is_set():
            raise CancellationRequested("Shutdown requested")
        return_code = process.poll()
        if return_code is not None:
            return return_code
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Process timed out after {timeout_sec} seconds")
        time.sleep(poll_interval_sec)


def process_is_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


@dataclass
class PortAllocator:
    start_port: int
    end_port: int

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._in_use: set[int] = set()

    def acquire(self) -> int:
        with self._lock:
            for port in range(self.start_port, self.end_port + 1):
                if port in self._in_use:
                    continue
                if check_port_open("127.0.0.1", port):
                    continue
                self._in_use.add(port)
                return port
        raise RuntimeError(f"No free port available in range {self.start_port}-{self.end_port}")

    def release(self, port: int) -> None:
        with self._lock:
            self._in_use.discard(port)


def get_port_range(profile: Dict[str, Any]) -> tuple[int, int]:
    server_cfg = profile["server"]
    if "port_range" in server_cfg:
        start_port, end_port = server_cfg["port_range"]
        return int(start_port), int(end_port)
    base_port = int(server_cfg.get("base_port", 20000))
    span = int(server_cfg.get("port_span", 200))
    return base_port, base_port + span - 1


def load_existing_status(status_path: Path) -> Optional[Dict[str, Any]]:
    if not status_path.exists():
        return None
    return json.loads(status_path.read_text(encoding="utf-8"))


def evaluate_resume_decision(
    status_path: Path,
    retry_failed: bool,
) -> tuple[str, Optional[Dict[str, Any]]]:
    existing = load_existing_status(status_path)
    if existing is None:
        return "run", None
    state = existing.get("state")
    if state == "success":
        return "skip_success", existing
    if state == "failed":
        return ("run", existing) if retry_failed else ("skip_failed", existing)
    if state == "cancelled":
        return ("run", existing) if retry_failed else ("skip_cancelled", existing)
    if state == "running":
        pid = existing.get("job_pid")
        if process_is_alive(pid):
            return "skip_running", existing
        stale = dict(existing)
        stale["state"] = "stale_failed"
        stale["ended_at"] = utc_now_iso()
        write_json(status_path, stale)
        return ("run", stale) if retry_failed else ("skip_stale_failed", stale)
    return ("run", existing) if retry_failed else ("skip_unknown", existing)


def format_duration(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def build_summary_tsv(rows: List[Dict[str, Any]]) -> str:
    header = [
        "exp_name",
        "ckpt_name",
        "step",
        "state",
        "gpu_id",
        "port",
        "duration_sec",
        "output_dir",
    ]
    lines = ["\t".join(header)]
    for row in rows:
        values = [str(row.get(column, "")) for column in header]
        lines.append("\t".join(values))
    return "\n".join(lines) + "\n"


def append_runner_line(log_path: Path, message: str) -> None:
    append_log(log_path, message, also_stdout=False)
