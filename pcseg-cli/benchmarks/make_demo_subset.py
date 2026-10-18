# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Baranyai Dániel
"""
Writes a smaller copy of a structured E57 scan that keeps every `step`-th grid
row and column, for a quick-start/demo download.

Unlike loading with `pcseg --decimation N` and re-exporting (which quantizes
coordinates to 16 bits and keeps the ORIGINAL grid indices, so the result only
reads back densely at the same --decimation), this copies the raw float
coordinates, colors and intensities, carries over the scan pose, and renumbers
the grid (row // step, column // step) - so the demo file loads as a dense
panorama with `--decimation 1`.

Usage:
    python benchmarks/make_demo_subset.py outdoor.e57 outdoor_demo.e57 --step 4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pye57


def make_demo_subset(src: str, dst: str, step: int) -> int:
    e57 = pye57.E57(src)
    try:
        header = e57.get_header(0)
        rotation, translation = header.rotation, header.translation
        try:
            name = header["name"].value()
        except pye57.libe57.E57Exception:  # "name" is optional in the E57 standard
            name = Path(src).stem
        raw = e57.read_scan_raw(0, ignore_unsupported_fields=True)
    finally:
        e57.close()

    keep = (raw["rowIndex"] % step == 0) & (raw["columnIndex"] % step == 0)

    if "cartesianX" in raw:
        x, y, z = (raw[k][keep] for k in ("cartesianX", "cartesianY", "cartesianZ"))
    else:
        r = raw["sphericalRange"][keep]
        az = raw["sphericalAzimuth"][keep]
        el = raw["sphericalElevation"][keep]
        x = r * np.cos(el) * np.cos(az)
        y = r * np.cos(el) * np.sin(az)
        z = r * np.sin(el)

    data = {
        "cartesianX": x.astype(np.float64),
        "cartesianY": y.astype(np.float64),
        "cartesianZ": z.astype(np.float64),
        "rowIndex": (raw["rowIndex"][keep] // step).astype(np.int32),
        "columnIndex": (raw["columnIndex"][keep] // step).astype(np.int32),
    }
    for field in ("colorRed", "colorGreen", "colorBlue", "intensity"):
        if field in raw:
            data[field] = raw[field][keep]

    out = pye57.E57(dst, mode="w")
    try:
        out.write_scan_raw(data, name=f"{name} (every {step}th row/column)",
                           rotation=np.asarray(rotation), translation=np.asarray(translation))
    finally:
        out.close()
    return int(keep.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", help="Structured source .e57")
    parser.add_argument("dst", help="Output .e57")
    parser.add_argument("--step", type=int, default=4, help="Keep every STEP-th row and column (default: 4).")
    args = parser.parse_args()
    n = make_demo_subset(args.src, args.dst, args.step)
    print(f"Wrote {n:,} points to {args.dst}")


if __name__ == "__main__":
    main()
