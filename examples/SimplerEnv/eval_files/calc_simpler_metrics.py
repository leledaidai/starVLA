#!/usr/bin/env python3
"""Calculate SimplerEnv evaluation metrics from video results."""
import argparse
import glob
from pathlib import Path
import numpy as np
import pandas as pd


def get_dir_stats(result_dir, task_pattern, pattern_type="success"):
    """
    Count success/failure from video filenames.

    Args:
        result_dir: Directory containing result videos
        task_pattern: Task name pattern to filter (e.g., "PutCarrotOnPlateInScene-v0")
        pattern_type: "success" for entire success, "partial" for partial success (grasp)

    Returns:
        List of success (1) or failure (0) for each episode
    """
    if pattern_type == "success":
        succ_pattern = "success"
        fail_pattern = "failure"
    else:  # partial success - check if object was grasped
        succ_pattern = "is_src_obj_grasped_True"
        fail_pattern = "is_src_obj_grasped_False"

    results = []
    all_videos = glob.glob(f"{result_dir}/**/*.mp4", recursive=True)

    for fname in all_videos:
        # Filter by task pattern
        if task_pattern not in fname:
            continue

        fname_stem = Path(fname).stem
        if succ_pattern in fname_stem:
            results.append(1)
        elif fail_pattern in fname_stem:
            results.append(0)

    return results


def calc_bridge_stats(result_dir):
    """Calculate Bridge dataset task statistics."""
    tasks = {
        "PutSpoonOnTableClothInScene-v0": "put_spoon_on_tablecloth",
        "PutCarrotOnPlateInScene-v0": "put_carrot_on_plate",
        "StackGreenCubeOnYellowCubeBakedTexInScene-v0": "stack_green_block_on_yellow_block",
        "PutEggplantInBasketScene-v0": "put_eggplant_in_basket",
    }

    results = {}
    print("Task Success Rates:")
    print("-" * 60)

    for task_env, task_name in tasks.items():
        # Entire success
        stats_entire = get_dir_stats(result_dir, task_env, "success")
        # Partial success (grasp)
        stats_partial = get_dir_stats(result_dir, task_env, "partial")

        if stats_entire:
            entire_rate = np.mean(stats_entire)
            results[f"{task_name}_entire"] = entire_rate
            print(f"{task_name} (entire): {entire_rate:.3f} ({sum(stats_entire)}/{len(stats_entire)})")

        if stats_partial:
            partial_rate = np.mean(stats_partial)
            results[f"{task_name}_partial"] = partial_rate
            print(f"{task_name} (partial): {partial_rate:.3f} ({sum(stats_partial)}/{len(stats_partial)})")

        if not stats_entire and not stats_partial:
            print(f"{task_name}: No results found")

    # Calculate averages
    entire_rates = [v for k, v in results.items() if k.endswith("_entire")]
    partial_rates = [v for k, v in results.items() if k.endswith("_partial")]

    if entire_rates:
        results["average_entire"] = np.mean(entire_rates)
        print(f"\nAverage Entire Success: {results['average_entire']:.3f}")

    if partial_rates:
        results["average_partial"] = np.mean(partial_rates)
        print(f"Average Partial Success: {results['average_partial']:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=str, required=True, help="Result directory")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        print(f"Error: {result_dir} does not exist")
        return

    print("=" * 60)
    print("SimplerEnv Evaluation Results")
    print("=" * 60)
    print(f"Result directory: {result_dir}\n")

    bridge_results = calc_bridge_stats(result_dir)

    # Save to CSV
    if bridge_results:
        df = pd.DataFrame([bridge_results])
        csv_path = result_dir / "metrics.csv"
        df.to_csv(csv_path, index=False, float_format="%.3f")
        print(f"\n✅ Results saved to {csv_path}")


if __name__ == "__main__":
    main()
