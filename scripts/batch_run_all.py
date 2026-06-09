#!/usr/bin/env python3
"""
Chemia Batch Runner — auto-discovers YAML configs, runs 3 seeds each, collects error bars.
Supports checkpoint/resume and multi-target configs (target_cols: [...]).
Usage:
    python scripts/batch_run_all.py                        # run everything
    python scripts/batch_run_all.py --datasets BBBP SIDER  # specific datasets
    python scripts/batch_run_all.py --trials 30            # fewer Optuna trials
    python scripts/batch_run_all.py --resume               # skip completed tasks
    python scripts/batch_run_all.py --resume --retry-failed
"""

import os, sys, yaml, argparse, glob, subprocess, json
import numpy as np
from datetime import datetime
from collections import OrderedDict

SEEDS = [42, 123, 456]
CONFIG_DIR = "examples/configs"

METRIC_MAP = {
    "regression":              {"primary": "test_r2",       "higher_is_better": True},
    "binary_classification":   {"primary": "test_auc",      "higher_is_better": True},
    "multiclass_classification":{"primary": "test_f1_macro", "higher_is_better": True},
}

DATASET_ORDER = ["FreeSolv", "BACE", "BBBP", "ClinTox", "Tox21", "SIDER"]


def discover_configs(datasets=None):
    """Discover all YAML configs, optionally filtered by dataset name.
    Multi-target configs (with target_cols) are expanded into one entry per target."""
    configs = []
    for path in sorted(glob.glob(os.path.join(CONFIG_DIR, "**/*.yaml"), recursive=True)):
        rel = os.path.relpath(path, CONFIG_DIR)
        parts = rel.replace(".yaml", "").split(os.sep)
        ds_name = parts[0].split("_")[0].upper() if len(parts) == 1 else parts[0].upper()

        if datasets and ds_name not in datasets:
            continue

        with open(path) as f:
            cfg = yaml.safe_load(f)
        target_cols = cfg.get('data', {}).get('single_file_config', {}).get('target_cols')

        if target_cols and isinstance(target_cols, list):
            # Multi-target: expand into one entry per target
            for tgt in target_cols:
                configs.append({
                    "path": path,
                    "dataset": ds_name,
                    "target": tgt,
                    "is_multi_target": True,
                })
        else:
            # Single target
            target_name = parts[-1] if len(parts) > 1 else parts[0].replace("_classification", "").replace("_regression", "")
            configs.append({
                "path": path,
                "dataset": ds_name,
                "target": target_name,
                "is_multi_target": False,
            })
    return configs


def load_progress(progress_path):
    """Load checkpoint file. Returns (completed_keys, failed_keys) sets."""
    if not os.path.exists(progress_path):
        return set(), set()
    try:
        with open(progress_path) as f:
            data = json.load(f)
        completed = {(e["dataset"], e["target"], e["seed"]) for e in data.get("completed", []) if e.get("success")}
        failed = {(e["dataset"], e["target"], e["seed"]) for e in data.get("failed", [])}
        return completed, failed
    except Exception:
        return set(), set()


def save_progress(progress_path, completed_tasks, failed_tasks, total):
    """Atomically save progress to JSON."""
    data = {
        "last_updated": datetime.now().isoformat(),
        "total_tasks": total,
        "completed": [
            {"dataset": ds, "target": tgt, "seed": sd, "success": True, "completed_at": ts}
            for ds, tgt, sd, ts in completed_tasks
        ],
        "failed": [
            {"dataset": ds, "target": tgt, "seed": sd, "error": err}
            for ds, tgt, sd, err in failed_tasks
        ],
    }
    tmp_path = progress_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, progress_path)


def run_training(config_path):
    cmd = f"PYTHONPATH=. python scripts/run_training_only.py --config {config_path}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=os.getcwd())
    if result.returncode != 0:
        lines = result.stderr.strip().split("\n")
        return False, (lines[-1] if lines else "Unknown error")
    return True, None


def find_latest_output():
    output_dirs = sorted(
        [d for d in os.listdir("output") if d.startswith("CHEMIA_run")],
        key=lambda d: os.path.getmtime(os.path.join("output", d)),
        reverse=True,
    )
    return os.path.join("output", output_dirs[0]) if output_dirs else None


def parse_results(run_dir, task_type):
    import pandas as pd
    csv_path = os.path.join(run_dir, "model_comparison.csv")
    # Multi-target runs write combined_results.csv at the parent level
    if not os.path.exists(csv_path):
        csv_path = os.path.join(run_dir, "combined_results.csv")
    if not os.path.exists(csv_path):
        # Try one level up (for multi-target per-target subdirs)
        parent = os.path.dirname(run_dir.rstrip('/'))
        csv_path = os.path.join(parent, "combined_results.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    primary = METRIC_MAP[task_type]["primary"]
    results = {}
    for _, row in df.iterrows():
        model = row.get("model_name", row.get("model", "unknown"))
        results[model] = {col: row[col] for col in df.columns if col.startswith("test_")}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--output", default="batch_results")
    parser.add_argument("--resume", action="store_true", help="Skip completed tasks")
    parser.add_argument("--retry-failed", action="store_true", help="Re-run previously failed tasks")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    progress_path = os.path.join(args.output, "batch_progress.json")

    # Load checkpoint
    completed_set, failed_set = set(), set()
    if args.resume:
        completed_set, failed_set = load_progress(progress_path)
        print(f"Resume mode: {len(completed_set)} completed, {len(failed_set)} failed")

    # Discover configs
    dataset_filter = {d.upper() for d in args.datasets} if args.datasets else None
    configs = discover_configs(dataset_filter)

    if not configs:
        print("No configs found.")
        return

    # Count tasks (configs × seeds)
    total_tasks = len(configs) * len(args.seeds)
    print(f"Found {len(configs)} configs × {len(args.seeds)} seeds = {total_tasks} tasks")

    all_rows = []
    completed_tasks = []
    failed_tasks = []
    task_idx = 0

    for ds_name in DATASET_ORDER:
        ds_cfgs = [c for c in configs if c["dataset"] == ds_name]
        if not ds_cfgs:
            continue

        multi = ds_cfgs[0].get("is_multi_target", False)
        n_cfgs = len({c["path"] for c in ds_cfgs})  # Unique YAML files
        n_targets = len({c["target"] for c in ds_cfgs})
        print(f"\n{'#'*60}")
        if multi:
            print(f"  {ds_name} ({n_cfgs} configs × {n_targets} targets × {len(args.seeds)} seeds)")
        else:
            print(f"  {ds_name} ({n_cfgs} configs × {len(args.seeds)} seeds)")
        print(f"{'#'*60}")

        # For multi-target: group by YAML path, run once with target_cols
        if multi:
            seen_paths = set()
            for cfg in ds_cfgs:
                config_path = cfg["path"]
                if config_path in seen_paths:
                    continue
                seen_paths.add(config_path)

                # All targets from this YAML
                yaml_targets = [c["target"] for c in ds_cfgs if c["path"] == config_path]
                label = f"{ds_name} [{len(yaml_targets)} targets]"

                with open(config_path) as f:
                    base_config = yaml.safe_load(f)

                task_type = base_config["task_type"]
                primary_metric = METRIC_MAP[task_type]["primary"]

                for seed in args.seeds:
                    task_idx += 1
                    task_key = (ds_name, "__multi__", seed)
                    print(f"\n[{task_idx}/{total_tasks}] {label} seed={seed}")

                    if args.resume and task_key in completed_set:
                        print(f"  [SKIP] already completed")
                        continue
                    if args.resume and not args.retry_failed and task_key in failed_set:
                        print(f"  [SKIP] previously failed (use --retry-failed)")
                        continue

                    config = yaml.safe_load(yaml.dump(base_config))
                    config["split_config"]["hold_out"]["random_state"] = seed
                    config["training"]["random_state"] = seed
                    config["training"]["n_trials"] = args.trials
                    config["data"]["single_file_config"]["target_cols"] = yaml_targets

                    tmp_config = f"/tmp/chemia_{ds_name}_multi_seed{seed}.yaml"
                    with open(tmp_config, "w") as f:
                        yaml.dump(config, f)

                    ok, err = run_training(tmp_config)
                    if not ok:
                        print(f"  seed={seed} FAILED: {err}")
                        failed_tasks.append((ds_name, "__multi__", seed, err))
                        continue

                    run_dir = find_latest_output()
                    if run_dir:
                        results = parse_results(run_dir, task_type)
                        if results:
                            for model, metrics in results.items():
                                tgt_col = metrics.get("target_col", "unknown")
                                val = metrics.get(primary_metric)
                                if val is not None:
                                    all_rows.append({
                                        "dataset": ds_name, "target": tgt_col, "model": model,
                                        f"{primary_metric}_mean": round(float(val), 4),
                                        f"{primary_metric}_std": 0.0, "n_runs": 1,
                                        "raw_values": str([round(float(val), 4)]),
                                        "metric": primary_metric,
                                    })

                    completed_tasks.append((ds_name, "__multi__", seed, datetime.now().isoformat()))
                    save_progress(progress_path, completed_tasks, failed_tasks, total_tasks)
        else:
            # Single-target: one YAML per target
            for cfg in ds_cfgs:
                config_path = cfg["path"]
                label = f"{cfg['dataset']}/{cfg['target']}"

                with open(config_path) as f:
                    base_config = yaml.safe_load(f)

                task_type = base_config["task_type"]
                primary_metric = METRIC_MAP[task_type]["primary"]

                run_dirs = []
                for seed in args.seeds:
                    task_idx += 1
                    task_key = (cfg["dataset"], cfg["target"], seed)
                    print(f"\n[{task_idx}/{total_tasks}] {label} seed={seed}")

                    if args.resume and task_key in completed_set:
                        print(f"  [SKIP] already completed")
                        continue
                    if args.resume and not args.retry_failed and task_key in failed_set:
                        print(f"  [SKIP] previously failed (use --retry-failed)")
                        continue

                    config = yaml.safe_load(yaml.dump(base_config))
                    config["split_config"]["hold_out"]["random_state"] = seed
                    config["training"]["random_state"] = seed
                    config["training"]["n_trials"] = args.trials

                    tmp_config = f"/tmp/chemia_{cfg['dataset']}_{cfg['target']}_seed{seed}.yaml"
                    with open(tmp_config, "w") as f:
                        yaml.dump(config, f)

                    ok, err = run_training(tmp_config)
                    if not ok:
                        print(f"  seed={seed} FAILED: {err}")
                        failed_tasks.append((cfg["dataset"], cfg["target"], seed, err))
                        continue

                    run_dir = find_latest_output()
                    if run_dir:
                        run_dirs.append(run_dir)
                        completed_tasks.append((cfg["dataset"], cfg["target"], seed, datetime.now().isoformat()))

                    save_progress(progress_path, completed_tasks, failed_tasks, total_tasks)

                if len(run_dirs) < 2:
                    continue

                # Aggregate across seeds
                all_model_results = {}
                for rd in run_dirs:
                    results = parse_results(rd, task_type)
                    if results:
                        for model, metrics in results.items():
                            all_model_results.setdefault(model, []).append(metrics[primary_metric])

                for model, vals in sorted(all_model_results.items(), key=lambda x: -np.mean(x[1])):
                    mean_v = np.mean(vals)
                    std_v = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                    all_rows.append({
                        "dataset": cfg["dataset"], "target": cfg["target"], "model": model,
                        "n_runs": len(vals),
                        f"{primary_metric}_mean": round(mean_v, 4),
                        f"{primary_metric}_std": round(std_v, 4),
                        "raw_values": str([round(v, 4) for v in vals]),
                        "metric": primary_metric,
                    })

    # Final: clear embedding cache, write summary
    from utils.embedding_cache import clear_cache
    clear_cache()

    if all_rows:
        import pandas as pd
        df = pd.DataFrame(all_rows)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_out = os.path.join(args.output, f"batch_summary_{timestamp}.csv")
        df.to_csv(csv_out, index=False)
        print(f"\n{'='*60}")
        print(f"Summary saved to {csv_out}")
        print(f"Completed: {len(completed_tasks)}, Failed: {len(failed_tasks)}")
    else:
        print("\nNo results collected.")


if __name__ == "__main__":
    main()
