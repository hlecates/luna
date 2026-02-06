#!/usr/bin/env python3
"""
Run Luna's alpha-CROWN on all TLLVerifyBench 2023 benchmark instances, collect output bounds, and compute average bound width.

Runs on all instances (no subsetting).

Usage:
  python run_tllverifybench_alpha_crown.py --luna ./luna --benchmark-dir resources/regular_benchmarks/benchmarks/tllverifybench_2023 [options]

  Or with explicit paths:
  python run_tllverifybench_alpha_crown.py --luna ./luna --onnx-dir path/to/onnx --vnnlib-dir path/to/vnnlib [options]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def decompress_if_needed(filepath: Path, temp_dir: Path) -> Path:
    """
    Decompress a .gz file if needed and return path to uncompressed file.
    If filepath is not .gz, return it as-is.
    """
    if filepath.suffix == ".gz":
        uncompressed_name = filepath.stem
        temp_file = temp_dir / f"{filepath.parent.name}_{uncompressed_name}"
        if not temp_file.exists():
            with gzip.open(filepath, "rb") as f_in:
                with open(temp_file, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
        return temp_file
    return filepath


def parse_output_bounds(stdout: str) -> list[tuple[float, float]] | None:
    """
    Parse luna stdout for 'Output Bounds:' and the following line of [lb, ub] intervals.
    Returns list of (lower, upper) pairs, or None if not found.
    """
    lines = stdout.splitlines()
    # Match [number, number] with optional sign and exponent
    interval_re = re.compile(r"\[\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\]")

    for i, line in enumerate(lines):
        if "Output Bounds:" not in line:
            continue
        bounds: list[tuple[float, float]] = []
        # Bounds are on the same line (after colon) or on the next line
        rest = line.split("Output Bounds:", 1)[-1].strip()
        if rest:
            for m in interval_re.finditer(rest):
                bounds.append((float(m.group(1)), float(m.group(2))))
        if i + 1 < len(lines):
            for m in interval_re.finditer(lines[i + 1]):
                bounds.append((float(m.group(1)), float(m.group(2))))
        return bounds if bounds else None
    return None


def bound_widths(bounds: list[tuple[float, float]]) -> list[float]:
    """Return list of widths (upper - lower) for each interval."""
    return [u - l for l, u in bounds]


def run_luna(
    luna_binary: Path,
    onnx_path: Path,
    vnnlib_path: Path,
    *,
    iterations: int = 20,
    timeout_s: int | None = 300,
) -> tuple[str, str, int]:
    """Run luna alpha-crown. Returns (stdout, stderr, returncode)."""
    cmd = [
        str(luna_binary),
        "--input", str(onnx_path),
        "--property", str(vnnlib_path),
        "--method", "alpha-crown",
        "--iterations", str(iterations),
        "--optimize-lower",
        "--optimize-upper",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=luna_binary.parent if luna_binary.is_absolute() else None,
        )
        return (result.stdout, result.stderr, result.returncode)
    except subprocess.TimeoutExpired:
        return ("", "timeout", -1)
    except FileNotFoundError:
        return ("", f"luna binary not found: {luna_binary}", -1)


def discover_instances(
    benchmark_dir: Path | None = None,
    onnx_dir: Path | None = None,
    vnnlib_dir: Path | None = None,
    instances_csv: Path | None = None,
) -> list[tuple[Path, Path]]:
    """
    Discover (onnx, vnnlib) instance pairs.
    If instances_csv is set or benchmark_dir/instances.csv exists, use CSV.
    Otherwise use onnx_dir and vnnlib_dir (or benchmark_dir/onnx and benchmark_dir/vnnlib)
    and form all pairs.
    """
    if benchmark_dir is not None:
        csv_path = benchmark_dir / "instances.csv"
        onnx_base = benchmark_dir / "onnx"
        vnn_base = benchmark_dir / "vnnlib"
    else:
        csv_path = None
        onnx_base = onnx_dir
        vnn_base = vnnlib_dir

    if instances_csv is not None:
        csv_path = instances_csv

    if csv_path is not None and csv_path.exists():
        instances = []
        base = csv_path.parent if benchmark_dir is None else (benchmark_dir or csv_path.parent)
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                onnx_rel, vnn_rel = row[0].strip(), row[1].strip()
                onnx_path = (base / onnx_rel).resolve()
                vnn_path = (base / vnn_rel).resolve()
                # Accept .gz if uncompressed file does not exist
                if not onnx_path.exists() and (base / (onnx_rel + ".gz")).exists():
                    onnx_path = (base / (onnx_rel + ".gz")).resolve()
                if not vnn_path.exists() and (base / (vnn_rel + ".gz")).exists():
                    vnn_path = (base / (vnn_rel + ".gz")).resolve()
                if onnx_path.exists() and vnn_path.exists():
                    instances.append((onnx_path, vnn_path))
        return instances

    if onnx_base is None or vnn_base is None or not onnx_base.exists() or not vnn_base.exists():
        return []

    onnx_files = sorted(onnx_base.glob("*.onnx"))
    vnn_files = sorted(vnn_base.glob("*.vnnlib"))
    return [(o, v) for o in onnx_files for v in vnn_files]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Luna alpha-CROWN on all TLLVerifyBench 2023 benchmark instances and compute average output bound width.",
    )
    parser.add_argument(
        "--luna",
        type=Path,
        default=Path("./luna"),
        help="Path to luna executable (default: ./luna)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--benchmark-dir",
        type=Path,
        default=None,
        help="Benchmark root (expects onnx/ and vnnlib/ subdirs, or instances.csv)",
    )
    group.add_argument(
        "--instances-csv",
        type=Path,
        default=None,
        help="Path to instances.csv (each row: onnx_path, vnnlib_path [, timeout])",
    )
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        default=None,
        help="Directory of ONNX files (use with --vnnlib-dir if no instances.csv)",
    )
    parser.add_argument(
        "--vnnlib-dir",
        type=Path,
        default=None,
        help="Directory of VNN-LIB files (use with --onnx-dir if no instances.csv)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Alpha-CROWN iterations (default: 20)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per instance in seconds (default: 300)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print summary and errors",
    )
    args = parser.parse_args()

    if args.benchmark_dir is None and args.instances_csv is None and (args.onnx_dir is None or args.vnnlib_dir is None):
        parser.error("Provide --benchmark-dir, or --instances-csv, or both --onnx-dir and --vnnlib-dir")

    instances = discover_instances(
        benchmark_dir=args.benchmark_dir,
        onnx_dir=args.onnx_dir,
        vnnlib_dir=args.vnnlib_dir,
        instances_csv=args.instances_csv,
    )

    if not instances:
        print("No instances found.", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"Found {len(instances)} instance(s). Running alpha-CROWN (iterations={args.iterations})...")
        print()

    all_widths: list[float] = []
    results: list[dict] = []
    failed = 0
    temp_dir = Path(tempfile.mkdtemp(prefix="luna_tllverifybench_"))

    try:
        for idx, (onnx_path, vnnlib_path) in enumerate(instances):
            if not args.quiet:
                print(f"[{idx + 1}/{len(instances)}] {onnx_path.name} + {vnnlib_path.name} ... ", end="", flush=True)

            # Decompress .gz to temp dir if needed
            onnx_resolved = decompress_if_needed(onnx_path, temp_dir)
            vnnlib_resolved = decompress_if_needed(vnnlib_path, temp_dir)

            stdout, stderr, ret = run_luna(
                args.luna,
                onnx_resolved,
                vnnlib_resolved,
                iterations=args.iterations,
                timeout_s=args.timeout,
            )

            if ret != 0:
                failed += 1
                if not args.quiet:
                    print("FAILED")
                    if stderr:
                        print(stderr[:500])
                results.append({
                    "onnx": onnx_path.name,
                    "vnnlib": vnnlib_path.name,
                    "widths": [],
                    "error": stderr[:200] if stderr else "non-zero exit",
                })
                continue

            bounds = parse_output_bounds(stdout)
            if bounds is None:
                failed += 1
                if not args.quiet:
                    print("FAILED (no Output Bounds in stdout)")
                results.append({
                    "onnx": onnx_path.name,
                    "vnnlib": vnnlib_path.name,
                    "widths": [],
                    "error": "could not parse Output Bounds",
                })
                continue

            widths = bound_widths(bounds)
            all_widths.extend(widths)
            results.append({
                "onnx": onnx_path.name,
                "vnnlib": vnnlib_path.name,
                "widths": widths,
                "error": None,
            })

            if not args.quiet:
                avg_w = sum(widths) / len(widths) if widths else 0
                print(f"OK  (bounds={len(widths)}, avg_width={avg_w:.6f})")

        # Summary
        print()
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Instances: {len(instances)} total, {len(instances) - failed} succeeded, {failed} failed")
        if all_widths:
            avg_bound_width = sum(all_widths) / len(all_widths)
            print(f"Total output bounds: {len(all_widths)}")
            print(f"Average bound width: {avg_bound_width:.6f}")
        else:
            print("No bounds collected (all runs failed or no parseable output).")
            return 1
    finally:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
