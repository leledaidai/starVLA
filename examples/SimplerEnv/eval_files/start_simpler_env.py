import os
import json
from pathlib import Path

import numpy as np
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator

# from IPython import embed; embed()
from examples.SimplerEnv.eval_files.custom_argparse import get_args
from examples.SimplerEnv.eval_files.model2simpler_interface import ModelClient


def start_debugpy_once():
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    args = get_args()

    os.environ["DISPLAY"] = ""
    # prevent a single jax process from taking up all the GPU memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    if os.getenv("DEBUG", False):
        start_debugpy_once()
    model = ModelClient(
        policy_ckpt_path=args.ckpt_path,  # to get unnormalization stats
        policy_setup=args.policy_setup,
        port=args.port,
        action_scale=args.action_scale,
        cfg_scale=1.5,  # cfg from 1.5 to 7 also performs well
    )

    # policy model creation; update this if you are using a new policy model
    # run real-to-sim evaluation
    success_arr = maniskill2_evaluator(model, args)
    average_success = float(np.mean(success_arr)) if len(success_arr) > 0 else None
    if args.result_json:
        result_path = Path(args.result_json).expanduser().resolve()
        result_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "average_success": average_success,
            "num_episodes": int(len(success_arr)),
            "num_success": int(np.sum(success_arr)),
            "success_arr": [bool(x) for x in success_arr],
            "env_name": args.env_name,
            "ckpt_path": args.ckpt_path,
            "port": args.port,
            "host": args.host,
        }
        result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args)
    print(" " * 10, "Average success", average_success)
