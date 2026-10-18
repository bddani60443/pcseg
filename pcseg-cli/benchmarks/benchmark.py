# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Baranyai Dániel
"""
Runtime benchmark of the three pcseg commands on one structured scan.

Each configuration runs `cli.main` in its own fresh Python process (so
no model or CUDA state carries over between runs), and its log timestamps are
split into stages:

    load     reading + decimating the E57 file
    models   loading GroundingDINO / SAM2 weights
    segment  the view sweep (or single-point pick) incl. outlier filtering / merging
    export   writing the requested .las / .png outputs

plus total wall time, peak PyTorch GPU memory and the number of instances.
A warm-up load runs first, so every configuration reads the file from a warm
OS page cache.

Usage:
    python benchmarks/benchmark.py --scan outdoor.e57 --work-dir /tmp/bench \\
        --results benchmarks/results/gb10.json
    python benchmarks/benchmark.py ... --only run-d4 auto-d4     # a subset
    python benchmarks/benchmark.py ... --cpu                     # add the CPU run
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PAPER_PROMPT = ["--prompt", "window, building, road, curb", "--auto-coverage",
                "--dino-box-threshold", "0.35", "--dino-text-threshold", "0.35", "--sam-score-threshold", "0.90"]

CONFIGS = {
    "run-d2": ["run", *PAPER_PROMPT, "--decimation", "2"],
    "run-d4": ["run", *PAPER_PROMPT, "--decimation", "4"],
    "run-d8": ["run", *PAPER_PROMPT, "--decimation", "8"],
    "auto-d4": ["auto-segment", "--auto-coverage", "--decimation", "4"],
    "auto-d8": ["auto-segment", "--auto-coverage", "--decimation", "8"],
    "pick-d4": ["pick-segment", "--point-xyz", "5", "0", "0", "--keep-all-masks", "--decimation", "4"],
}
CPU_CONFIG = {"run-d8-cpu": ["run", *PAPER_PROMPT, "--decimation", "8", "--device", "cpu"]}

# Runs inside the child process: the real CLI, then report peak GPU memory.
_CHILD = """
import json, sys
from cli import main
code = main(sys.argv[1:])
peak = None
try:
    import torch
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated()
except ImportError:
    pass
print("BENCH_RESULT " + json.dumps({"exit_code": code, "peak_gpu_bytes": peak}), flush=True)
"""

_TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}) \[\w+\] (.*)$")


def _parse_log(text: str) -> list[tuple[float, str]]:
    events = []
    for line in text.splitlines():
        m = _TS.match(line)
        if m:
            t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f").timestamp()
            events.append((t, m.group(2)))
    return events


def _first(events, pattern, after=None):
    for t, msg in events:
        if (after is None or t >= after) and re.search(pattern, msg):
            return t
    return None


def _last(events, pattern):
    found = [t for t, msg in events if re.search(pattern, msg)]
    return found[-1] if found else None


def stage_times(events: list[tuple[float, str]]) -> dict:
    load_start = _first(events, r"^Loading .* \(decimation=")
    load_end = _first(events, r"^ScanNode ")

    model_spans = []
    for start_pat, end_pat in ((r"^Loading Grounding DINO", r"^Grounding DINO loaded"),
                               (r"^Loading SAM2 model", r"^SAM2 loaded")):
        s = _first(events, start_pat)
        e = _first(events, end_pat, after=s) if s is not None else None
        if s is not None and e is not None:
            model_spans.append((s, e))
    models = sum(e - s for s, e in model_spans)

    # Everything between the loaded scan and the first export, minus model loading
    # (which happens lazily somewhere inside that span, depending on the command).
    seg_end = _first(events, r"Sweep complete: \d+ instances detected|Auto-segmentation complete|candidate mask\(s\) kept")
    export_end = _last(events, r"^Saved ")

    return {
        "load_s": round(load_end - load_start, 1),
        "models_s": round(models, 1),
        "segment_s": round(seg_end - load_end - models, 1),
        "export_s": round(export_end - seg_end, 1),
    }


def count_instances(events) -> int | None:
    for _, msg in events:
        m = re.search(r"Sweep complete: (\d+) instances detected|merged into (\d+) objects|: (\d+) candidate mask", msg)
        if m:
            return int(next(g for g in m.groups() if g is not None))
    return None


def count_points(events) -> int | None:
    for _, msg in events:
        m = re.search(r"Saved LAS \(([\d,]+) points\)", msg)
        if m:
            return int(m.group(1).replace(",", ""))
    return None


def run_config(name: str, argv: list[str], scan: str, work_dir: Path, timeout: int) -> dict:
    out_las, out_png = work_dir / f"{name}.las", work_dir / f"{name}.png"
    full_argv = [argv[0], "--input", scan, *argv[1:], "--out", str(out_las), "--out", str(out_png)]
    print(f"[{name}] pcseg {' '.join(full_argv)}", flush=True)

    t0 = time.perf_counter()
    try:
        proc = subprocess.run([sys.executable, "-c", _CHILD, *full_argv], capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[{name}] TIMEOUT after {timeout}s", flush=True)
        return {"name": name, "argv": full_argv, "status": f"timeout>{timeout}s"}
    wall = time.perf_counter() - t0

    log = proc.stdout + proc.stderr
    (work_dir / f"{name}.log").write_text(log)
    child = next((json.loads(line.split(" ", 1)[1]) for line in proc.stdout.splitlines()
                  if line.startswith("BENCH_RESULT ")), {})
    events = _parse_log(log)

    result = {"name": name, "argv": full_argv, "status": "ok" if child.get("exit_code") == 0 else "failed",
              "wall_s": round(wall, 1), "instances": count_instances(events), "exported_points": count_points(events)}
    if result["status"] == "ok":
        result.update(stage_times(events))
    if child.get("peak_gpu_bytes"):
        result["peak_gpu_gib"] = round(child["peak_gpu_bytes"] / 2**30, 2)
    print(f"[{name}] {result}", flush=True)
    return result


def hardware() -> dict:
    info = {"python": platform.python_version(), "machine": platform.machine(), "platform": platform.platform()}
    try:
        info["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        info["gpu"] = None
    try:
        lscpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        info["cpu"] = "; ".join(line.split(":", 1)[1].strip() for line in lscpu.splitlines()
                                if line.startswith(("Model name", "CPU(s)")))
    except FileNotFoundError:
        info["cpu"] = platform.processor()
    try:
        import torch
        info["torch"] = torch.__version__
    except ImportError:
        pass
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scan", required=True)
    parser.add_argument("--work-dir", required=True, help="Where outputs and per-run logs go (large files).")
    parser.add_argument("--results", required=True, help="JSON file to write the summary to.")
    parser.add_argument("--only", nargs="+", help="Run only these configurations.")
    parser.add_argument("--cpu", action="store_true", help="Also run the (slow) CPU-only configuration.")
    parser.add_argument("--timeout", type=int, default=3600, help="Per-configuration timeout, seconds.")
    args = parser.parse_args()

    configs = dict(CONFIGS, **(CPU_CONFIG if args.cpu else {}))
    if args.only:
        configs = {k: v for k, v in configs.items() if k in args.only}
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print("Warm-up load (not timed)...", flush=True)
    subprocess.run([sys.executable, "-c",
                    f"from fileio.e57_loader import ScanData; ScanData({args.scan!r}, decimation_step=8).load()"],
                   check=True, capture_output=True)

    results = [run_config(name, argv, args.scan, work_dir, args.timeout) for name, argv in configs.items()]
    summary = {"scan": args.scan, "date": datetime.now().isoformat(timespec="seconds"),
               "hardware": hardware(), "results": results}
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {args.results}")


if __name__ == "__main__":
    main()
