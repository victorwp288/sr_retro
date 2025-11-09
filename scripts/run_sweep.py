import argparse
import itertools
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def build_search_space():
    trunk_lrs = [5e-5, 7.5e-5, 1e-4]
    warmup_steps = [5000, 10000]
    warmup_start_factors = [5e-4, 1e-3]
    sobel_weights = [0.0, 0.01]
    jpeg_probs = [0.1, 0.2]

    combos = []
    for lr_trunk, warm_steps, warm_factor, sobel, jpeg_prob in itertools.product(
        trunk_lrs, warmup_steps, warmup_start_factors, sobel_weights, jpeg_probs
    ):
        lr_tail = lr_trunk * 2.0
        combos.append(
            {
                "optimizer": {"lr_trunk": lr_trunk, "lr_tail": lr_tail},
                "training": {
                    "warmup_steps": warm_steps,
                    "warmup_start_factor": warm_factor,
                },
                "loss": {"sobel_weight": sobel},
                "degradations": {"jpeg_prob": jpeg_prob},
            }
        )
    return combos


def apply_overrides(config, overrides):
    for section, values in overrides.items():
        target = config.setdefault(section, {})
        for key, value in values.items():
            target[key] = value


def run_trial(base_config_path, overrides, steps, extra_args):
    config = yaml.safe_load(Path(base_config_path).read_text())

    # ensure deterministic compile behaviour for sweeps
    config.setdefault("training", {})["compile"] = False
    config["training"]["pilot_steps"] = steps

    apply_overrides(config, overrides)

    # ensure resample probabilities (nearest/box/bilinear) remain valid
    if "degradations" in overrides:
        total = (
            config["degradations"]["nearest_prob"]
            + config["degradations"]["box_prob"]
            + config["degradations"].get("bilinear_prob", 0.0)
        )
        if total > 1.0 + 1e-6:
            raise ValueError("resample probabilities exceed 1.0")

    # write to temp file
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(yaml.safe_dump(config).encode("utf-8"))

    try:
        env = os.environ.copy()
        env.setdefault("PYTHONPATH", ".")
        cmd = [sys.executable, "-m", "src.train", "--config", str(tmp_path), "--pilot"]
        if extra_args:
            cmd.extend(extra_args)
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        stdout = proc.stdout
        stderr = proc.stderr
        if proc.returncode != 0:
            metric = float("inf")
            reason = f"returncode {proc.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}"
            return metric, stdout, reason

        val_lines = [
            line for line in stdout.splitlines() if "validation mean L1:" in line
        ]
        if not val_lines:
            metric = float("inf")
            reason = f"no validation outputs\nstdout:\n{stdout}\nstderr:\n{stderr}"
            return metric, stdout, reason
        metrics = []
        for line in val_lines:
            try:
                value = float(line.split("validation mean L1:")[-1].strip())
                metrics.append(value)
            except ValueError:
                continue
        if not metrics:
            metric = float("inf")
            reason = f"no parsable validation outputs\nstdout:\n{stdout}\nstderr:\n{stderr}"
            return metric, stdout, reason
        metric = min(metrics)
        return metric, stdout, None
    finally:
        tmp_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep helper")
    parser.add_argument("--config", required=True, help="Base YAML config path")
    parser.add_argument(
        "--steps",
        type=int,
        default=10000,
        help="Pilot steps per trial",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="How many best trials to show",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Additional arg(s) passed to training command",
    )
    args = parser.parse_args()

    combos = build_search_space()
    results = []

    for idx, overrides in enumerate(combos, start=1):
        print(f"Running trial {idx}/{len(combos)} with overrides {overrides}")
        metric, stdout, reason = run_trial(
            args.config, overrides, args.steps, args.extra_arg
        )
        if reason:
            print(f"  trial failed: {reason}")
        else:
            print(f"  best validation L1: {metric:.6f}")
        results.append((metric, overrides, stdout, reason))

    results.sort(key=lambda x: x[0])
    print("\n=== Top trials ===")
    for metric, overrides, _, reason in results[: args.top]:
        status = "OK" if reason is None else f"FAIL ({reason.splitlines()[0]})"
        print(f"metric={metric:.6f} status={status} overrides={overrides}")

    output_path = Path("sweep_results.yaml")
    serializable = []
    for metric, overrides, stdout, reason in results:
        entry = {
            "metric": None if metric == float("inf") else metric,
            "overrides": overrides,
            "status": "ok" if reason is None else "fail",
        }
        serializable.append(entry)
    with output_path.open("w") as f:
        yaml.safe_dump(serializable, f)
    print(f"saved summary to {output_path}")


if __name__ == "__main__":
    main()
