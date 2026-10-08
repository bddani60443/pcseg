"""
pcseg - command-line point cloud segmentation pipeline.

Loads a structured E57 scan and sweeps a full 360 view around it, either:

  run           prompt-driven: GroundingDINO finds objects matching a text
                prompt, SAM2 segments them.
  auto-segment  prompt-free: an automatic mask backend (SAM2 by default)
                proposes every object it sees; overlapping detections from
                different views are merged in 3D.

Either way the output format is picked from each --out path's extension:

    .las / .e57         -> segmented point cloud (with per-label classification)
    .png / .jpg / .jpeg -> panorama preview with detected instances highlighted

Examples:
    pcseg run --input scan.e57 --prompt "door, window" \\
        --dino-channel intensity --sam-channel normals \\
        --out result.las --out preview.png

    pcseg auto-segment --input scan.e57 --backend sam2 \\
        --out result.las --out preview.png
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from pcseg.core.mask_filters import FILTERS
from pcseg.core.projection import default_channel

CHANNEL_CHOICES = ["auto", "rgb", "intensity", "normals"]
AUTO_BACKEND_CHOICES = ["sam2"]

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
POINTCLOUD_SUFFIXES = {".las", ".e57"}


def _add_common_flags(sub: argparse.ArgumentParser) -> None:
    """Flags shared by every pipeline subcommand (loading, sweep geometry, export, device)."""
    sub.add_argument("--input", "-i", required=True, help="Path to a structured (grid) .e57 scan.")
    sub.add_argument("--out", "-o", action="append", required=True, dest="outputs",
                      help="Output path; format is inferred from the extension (.las/.e57/.png/.jpg). Repeatable.")
    sub.add_argument("--decimation", type=int, default=2,
                      help="E57 grid decimation step (default: 2). Controls how many real points get loaded, "
                           "which sets the *actual optical detail* available (a coarser decimation gives DINO/SAM2 "
                           "a blurrier source panorama to detect from, however large --view-size renders it) - see "
                           "--view-size, which does NOT compensate for this.")
    sub.add_argument("--export-decimation", type=int, default=None,
                      help="If set finer (lower) than --decimation, re-loads the scan at this resolution after "
                           "segmentation and propagates the classification onto it via nearest-grid-cell lookup, "
                           "so .las/.e57 exports get full detail while segmentation itself stays fast on the "
                           "coarser --decimation. PNG/JPG previews always use --decimation. "
                           "Default: same as --decimation (no upsampling).")
    sub.add_argument("--step-deg", type=int, default=30, help="Yaw step between sweep views, in degrees (default: 30). "
                                                                "Ignored with --auto-coverage.")
    sub.add_argument("--pitch", type=float, nargs="+", default=[0.0], help="Pitch angle(s) to sweep at, in degrees "
                                                                            "(default: 0). Ignored with --auto-coverage.")
    sub.add_argument("--fov", type=float, default=90.0, help="Field of view per view, in degrees (default: 90).")
    sub.add_argument("--auto-coverage", action="store_true",
                      help="Replace --step-deg/--pitch with an automatically computed view grid: guarantees "
                           "--overlap between neighboring views (so nothing near a view's edge gets missed) and "
                           "skips any view whose footprint has no real scan data at all, instead of a manually "
                           "guessed step/pitch combination. See --phi-range to control how far up/down it reaches.")
    sub.add_argument("--phi-range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                      help="Vertical range --auto-coverage may sweep, in degrees. Default: derived from the scan "
                           "itself (exactly as far up/down as it actually has real data, no further) - pass this "
                           "only to deliberately sweep a narrower range than the scan's real extent (e.g. a "
                           "detector known to misbehave on a specific viewing angle for a specific prompt). "
                           "Negative looks up, positive looks down (same convention as --pitch). Ignored without "
                           "--auto-coverage.")
    sub.add_argument("--overlap", type=float, default=0.2,
                      help="Fractional angular overlap between neighboring --auto-coverage views (default: 0.2). "
                           "Ignored without --auto-coverage.")
    sub.add_argument("--view-size", type=int, default=1024,
                      help="Rendered view resolution in pixels (default: 1024). Rarely worth changing: "
                           "GroundingDINO always internally resizes to ~800px and SAM2 to exactly 1024x1024 "
                           "regardless of this value, so it neither compensates for a coarse --decimation nor "
                           "gains real detail from a value above ~1024-1333 - it just costs extra render time.")
    sub.add_argument("--filter", choices=list(FILTERS), default="none",
                      help="Per-instance 3D noise filter applied before export (default: none).")
    sub.add_argument("--preview-channel", choices=CHANNEL_CHOICES, default="auto",
                      help="Channel used as the background for .png/.jpg outputs (default: auto).")
    sub.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                      help="Device to run the model(s) on (default: auto = cuda if available, else cpu).")
    sub.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pcseg", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Prompt-driven: GroundingDINO + SAM2, load -> segment -> export.")
    run.add_argument("--prompt", "-p", required=True, help="Text prompt for GroundingDINO, e.g. 'door, window'.")
    _add_common_flags(run)
    run.add_argument("--dino-channel", choices=CHANNEL_CHOICES, default="auto",
                      help="Rendered channel GroundingDINO runs detection on (default: auto = best available).")
    run.add_argument("--sam-channel", choices=CHANNEL_CHOICES, default="auto",
                      help="Rendered channel SAM2 runs segmentation on (default: auto = best available).")
    run.add_argument("--dino-box-threshold", type=float, default=0.3)
    run.add_argument("--dino-text-threshold", type=float, default=0.25)
    run.add_argument("--sam-score-threshold", type=float, default=0.0,
                      help="Minimum SAM2 mask-quality score to keep a detection (default: 0.0 = keep everything "
                           "DINO's box passed). This is SAM2's OWN confidence in the mask boundary it drew, not "
                           "whether it segmented the right object - a wrong-but-clean-edged mask (e.g. pavement "
                           "mistaken for a building) can score just as high as, or higher than, a correct one, so "
                           "this filters low-quality/fragmented mask boundaries, not semantically wrong detections.")
    run.add_argument("--min-score", type=float, default=0.0,
                      help="Minimum FINAL instance score to keep (default: 0.0 = keep everything). This is "
                           "GroundingDINO's own text-match confidence (averaged across fragments if "
                           "--merge-duplicates merged more than one into an instance), applied last, after "
                           "outlier-filtering and merging - the semantic-confidence counterpart to "
                           "--sam-score-threshold's mask-quality one, not a replacement for it.")
    run.add_argument("--merge-duplicates", action="store_true",
                      help="Deduplicate instances of the same label that are really one physical object seen "
                           "from multiple overlapping sweep views (default: off - each view's detections stay "
                           "separate). Off by default because merging can hide exactly the kind of segmentation "
                           "mistake worth seeing raw, e.g. two classes fighting over the same points; turn it on "
                           "once you trust the detections and just want one instance per real object.")
    run.add_argument("--merge-iou", type=float, default=0.5,
                      help="3D point-index IoU threshold for --merge-duplicates (default: 0.5). Ignored without "
                           "--merge-duplicates.")
    run.add_argument("--merge-overlap", type=float, default=0.3,
                      help="Looser merge rule added ON TOP of --merge-iou for --merge-duplicates (either being "
                           "satisfied is enough): merges two fragments once the SMALLER one is this fraction "
                           "covered by the other, regardless of size mismatch - catches the same real object seen "
                           "near-complete in one view and only a sliver of it in another, which plain IoU misses "
                           "since the union balloons relative to the intersection. Set to a value > 1 (e.g. 999) to "
                           "disable this rule and keep only --merge-iou. NOT YET VALIDATED against a real scan "
                           "(see TODO.md) - default is a first guess, not tuned. Ignored without --merge-duplicates.")
    run.add_argument("--dino-config", default=None, help="Override the GroundingDINO config path.")
    run.add_argument("--dino-checkpoint", default=None, help="Override the GroundingDINO checkpoint path.")
    run.add_argument("--sam-config", default=None, help="Override the SAM2 config path.")
    run.add_argument("--sam-checkpoint", default=None, help="Override the SAM2 checkpoint path.")

    auto = subparsers.add_parser("auto-segment", help="Prompt-free: propose+merge every object an auto-mask backend finds.")
    _add_common_flags(auto)
    auto.add_argument("--backend", choices=AUTO_BACKEND_CHOICES, default="sam2",
                       help="Automatic mask proposal backend (default: sam2). See AutoMaskBackend for adding more.")
    auto.add_argument("--channel", choices=CHANNEL_CHOICES, default="auto",
                       help="Rendered channel the backend runs on (default: auto = best available).")
    auto.add_argument("--merge-iou", type=float, default=0.5,
                       help="3D index-overlap (IoU) threshold above which two views' masks are treated as the same object (default: 0.5).")
    auto.add_argument("--border-merge-overlap", type=float, default=0.6,
                       help="Looser containment-overlap threshold used only between two masks that both touched their "
                            "view's crop edge - stitches a large surface (e.g. pavement) cut apart by the sweep's view "
                            "boundaries back into one object, without affecting compact fully-visible objects (default: 0.6).")
    auto.add_argument("--surface-merge-distance", type=float, default=0.3,
                       help="Fallback for two border-touching masks with zero point-index overlap (common for a ground "
                            "plane, since different views resample it into different point subsets): merge if they come "
                            "within this many meters of each other in 3D. Set to 0 to disable (default: 0.3).")
    auto.add_argument("--merge-rounds", type=int, default=3,
                       help="Re-run the merge on its own output this many times, so growing clusters can keep "
                            "stitching to newly-adjacent neighbors; stops early once a round finds nothing more to "
                            "merge (default: 3).")
    auto.add_argument("--min-points", type=int, default=50,
                       help="Discard a single-view mask if it back-projects to fewer than this many 3D points (default: 50).")
    auto.add_argument("--points-per-side", type=int, default=32,
                       help="SAM2 automatic-mask-generator prompt grid density per view (default: 32). Higher = slower, finer-grained.")
    auto.add_argument("--pred-iou-thresh", type=float, default=0.86)
    auto.add_argument("--stability-score-thresh", type=float, default=0.92)
    auto.add_argument("--sam-config", default=None, help="Override the SAM2 config path.")
    auto.add_argument("--sam-checkpoint", default=None, help="Override the SAM2 checkpoint path.")

    pick = subparsers.add_parser(
        "pick-segment",
        help="Prompt-free, single-point: segment whatever SAM2 finds at one given 3D location.",
    )
    pick.add_argument("--input", "-i", required=True, help="Path to a structured (grid) .e57 scan.")
    pick.add_argument("--out", "-o", action="append", required=True, dest="outputs",
                       help="Output path; format is inferred from the extension (.las/.e57/.png/.jpg). Repeatable.")
    pick.add_argument("--decimation", type=int, default=2, help="E57 grid decimation step (default: 2).")
    pick.add_argument("--export-decimation", type=int, default=None,
                       help="If set finer (lower) than --decimation, re-loads the scan at this resolution after "
                            "segmentation and propagates the classification onto it. See `run --help` for the "
                            "full explanation. Default: same as --decimation (no upsampling).")

    point_group = pick.add_mutually_exclusive_group(required=True)
    point_group.add_argument("--point-xyz", type=float, nargs=3, metavar=("X", "Y", "Z"),
                              help="Real-world 3D coordinate; the nearest actual scan point is used - "
                                   "the CLI's substitute for clicking a point in a viewer.")
    point_group.add_argument("--point-index", type=int, help="A raw point-cloud array index, if already known.")

    pick.add_argument("--channel", choices=CHANNEL_CHOICES, default="auto",
                       help="Rendered channel SAM2 runs on (default: auto = best available).")
    pick.add_argument("--fov", type=float, default=90.0, help="Field of view of the view rendered around the point (default: 90).")
    pick.add_argument("--view-size", type=int, default=None,
                       help="Rendered view resolution in pixels (default: auto-computed from --fov and the "
                           "panorama's native resolution). Rarely worth raising further: SAM2 always internally "
                           "resizes to exactly 1024x1024 regardless, so it won't reveal detail --decimation "
                           "didn't capture in the first place.")
    pick.add_argument("--keep-all-masks", action="store_true",
                       help="Export all 3 SAM2 candidate masks (ranked by score) instead of only the best-scoring "
                            "one - useful when the point alone is ambiguous (e.g. door vs. door panel vs. wall) "
                            "and you want to inspect and choose offline.")
    pick.add_argument("--filter", choices=list(FILTERS), default="none",
                       help="Per-instance 3D noise filter applied before export (default: none).")
    pick.add_argument("--preview-channel", choices=CHANNEL_CHOICES, default="auto",
                       help="Channel used as the background for .png/.jpg outputs (default: auto).")
    pick.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                       help="Device to run SAM2 on (default: auto = cuda if available, else cpu).")
    pick.add_argument("--sam-config", default=None, help="Override the SAM2 config path.")
    pick.add_argument("--sam-checkpoint", default=None, help="Override the SAM2 checkpoint path.")
    pick.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")

    return parser


def _resolve_channel(node, choice: str):
    return default_channel(node) if choice == "auto" else choice


def _log_sweep_progress(percent: int, message: str) -> None:
    """Default progress_callback for run_sweep(): surfaces which view a sweep is on at INFO
    level, since -v's DEBUG output is dominated by noisy, uninformative third-party library
    lines (torch/transformers/groundingdino) that don't say anything about actual progress."""
    logging.info(f"[{percent:3d}%] {message}")


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                         format="%(asctime)s [%(levelname)s] %(message)s")
    if not verbose:
        # Deprecation noise from torch/transformers/groundingdino internals, not from pcseg
        # itself - real errors still surface since only these two warning categories are hit.
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)


def _validate_output_extensions(outputs: list[str]) -> bool:
    for out_path in outputs:
        suffix = Path(out_path).suffix.lower()
        if suffix not in POINTCLOUD_SUFFIXES | IMAGE_SUFFIXES:
            logging.error(f"Unsupported output extension '{suffix}' for '{out_path}' "
                           f"(supported: {sorted(POINTCLOUD_SUFFIXES | IMAGE_SUFFIXES)}).")
            return False
    return True


def _dispatch_output(node, result, output_path: str, preview_channel, export_node=None, export_mask=None) -> str:
    from pcseg.io.exporters import export_mask_preview_image, export_to_e57, export_to_las

    suffix = Path(output_path).suffix.lower()

    if suffix in POINTCLOUD_SUFFIXES:
        target_node = export_node if export_node is not None else node
        mask = export_mask if export_mask is not None else result.combined_indices()
        if suffix == ".las":
            return export_to_las(target_node, output_path, mask=mask)
        return export_to_e57(target_node, output_path, mask=mask)
    if suffix in IMAGE_SUFFIXES:
        return export_mask_preview_image(
            node, output_path,
            indices_by_label=result.indices_by_label(),
            colors_by_label=result.color_by_label(),
            base_channel=preview_channel,
        )
    raise ValueError(f"Unsupported output extension '{suffix}' for '{output_path}'.")


def _load_scan_at(path: str, decimation: int):
    from pcseg.io.e57_loader import ScanData

    logging.info(f"Loading {path} (decimation={decimation})...")
    node = ScanData(path, decimation_step=decimation).load()
    logging.info(str(node))
    return node


def _load_scan(args: argparse.Namespace):
    return _load_scan_at(args.input, args.decimation)


def _finish(node, result, args) -> int:
    """Shared tail: filter, classify, (optionally upsample,) export, for whichever command produced `result`."""
    from pcseg.core.classification import assign_classifications

    if not result.instances:
        logging.warning("No objects found. Nothing to export.")
        return 1

    if args.filter != "none":
        filter_fn = FILTERS[args.filter]
        for inst in result.instances:
            inst.indices = filter_fn(node, inst.indices)

    assign_classifications(node, result.indices_by_label())

    export_node, export_mask = None, None
    export_decimation = getattr(args, "export_decimation", None)
    if export_decimation is not None and export_decimation != args.decimation:
        from pcseg.core.classification import upsample_classification

        logging.info(f"Upsampling classification from decimation={args.decimation} to {export_decimation}...")
        export_node = _load_scan_at(args.input, export_decimation)
        export_node.point_classes, export_node.class_mapping = upsample_classification(node, export_node)
        export_mask = np.where(export_node.point_classes != 0)[0]
        logging.info(f"Upsampled: {len(export_mask):,} classified points at decimation={export_decimation} "
                     f"(was {len(result.combined_indices()):,} points at decimation={args.decimation}).")

    preview_channel = _resolve_channel(node, args.preview_channel)
    for out_path in args.outputs:
        logging.info(_dispatch_output(node, result, out_path, preview_channel, export_node=export_node, export_mask=export_mask))

    return 0


def run_command(args: argparse.Namespace) -> int:
    _configure_logging(args.verbose)
    if not _validate_output_extensions(args.outputs):
        return 2

    node = _load_scan(args)

    dino_channel = _resolve_channel(node, args.dino_channel)
    sam_channel = _resolve_channel(node, args.sam_channel)
    logging.info(f"DINO channel: {dino_channel} | SAM2 channel: {sam_channel}")

    from pcseg.services.dino_service import DEFAULT_CHECKPOINT_PATH as DINO_CKPT
    from pcseg.services.dino_service import DEFAULT_CONFIG_PATH as DINO_CFG
    from pcseg.services.dino_service import DinoService
    from pcseg.services.sam_service import DEFAULT_CHECKPOINT_PATH as SAM_CKPT
    from pcseg.services.sam_service import DEFAULT_CONFIG_PATH as SAM_CFG
    from pcseg.services.sam_service import SamService
    from pcseg.services.segmentation_service import SegmentationService

    device = None if args.device == "auto" else args.device
    dino = DinoService(config_path=args.dino_config or DINO_CFG, checkpoint_path=args.dino_checkpoint or DINO_CKPT, device=device)
    sam = SamService(config_path=args.sam_config or SAM_CFG, checkpoint_path=args.sam_checkpoint or SAM_CKPT, device=device)
    logging.info(f"Device: {dino.device}")

    view_grid = _build_coverage_grid(node, args) if args.auto_coverage else None

    result = SegmentationService(dino, sam).run_sweep(
        node, args.prompt,
        step_deg=args.step_deg, fov=args.fov, out_size=args.view_size,
        dino_channel=dino_channel, sam_channel=sam_channel, phi_angles=args.pitch,
        box_threshold=args.dino_box_threshold, text_threshold=args.dino_text_threshold,
        sam_score_threshold=args.sam_score_threshold, min_score=args.min_score,
        merge_duplicates=args.merge_duplicates, merge_iou=args.merge_iou,
        merge_overlap=args.merge_overlap if args.merge_overlap <= 1.0 else None,
        view_grid=view_grid, progress_callback=_log_sweep_progress,
    )
    return _finish(node, result, args)


def _build_coverage_grid(node, args: argparse.Namespace) -> list[tuple[float, float]]:
    from pcseg.core.projection import compute_coverage_grid, real_scan_mask

    valid_mask = real_scan_mask(node)
    phi_range = tuple(args.phi_range) if args.phi_range else None
    grid = compute_coverage_grid(valid_mask, fov=args.fov, overlap=args.overlap, phi_range=phi_range)
    logging.info(f"--auto-coverage: {len(grid)} views (phi_range={phi_range or 'auto'}, overlap={args.overlap})")
    return grid


def _build_indexed_coverage_grid(node, args: argparse.Namespace) -> tuple[list[tuple[int, int, float, float]], int]:
    from pcseg.core.projection import compute_indexed_coverage_grid, real_scan_mask

    valid_mask = real_scan_mask(node)
    phi_range = tuple(args.phi_range) if args.phi_range else None
    entries, n_theta = compute_indexed_coverage_grid(valid_mask, fov=args.fov, overlap=args.overlap, phi_range=phi_range)
    logging.info(f"--auto-coverage: {len(entries)} views (phi_range={phi_range or 'auto'}, overlap={args.overlap})")
    return entries, n_theta


def auto_segment_command(args: argparse.Namespace) -> int:
    _configure_logging(args.verbose)
    if not _validate_output_extensions(args.outputs):
        return 2

    node = _load_scan(args)
    channel = _resolve_channel(node, args.channel)
    logging.info(f"Backend: {args.backend} | channel: {channel}")

    from pcseg.services.auto_segmentation_service import AutoSegmentationService
    from pcseg.services.sam_service import DEFAULT_CHECKPOINT_PATH as SAM_CKPT
    from pcseg.services.sam_service import DEFAULT_CONFIG_PATH as SAM_CFG
    from pcseg.services.sam_service import SamService

    device = None if args.device == "auto" else args.device

    # --backend is the sole extension point: a future model just needs to be built
    # into a plain `image -> list[mask]` callable here (see AutoMaskBackend).
    if args.backend == "sam2":
        import functools

        sam = SamService(config_path=args.sam_config or SAM_CFG, checkpoint_path=args.sam_checkpoint or SAM_CKPT, device=device)
        backend = functools.partial(
            sam.automatic_generate, points_per_side=args.points_per_side,
            pred_iou_thresh=args.pred_iou_thresh, stability_score_thresh=args.stability_score_thresh,
        )
        logging.info(f"Device: {sam.device}")
    else:
        raise ValueError(f"Unknown backend {args.backend!r}")

    surface_merge_distance = args.surface_merge_distance if args.surface_merge_distance > 0 else None
    view_grid = _build_indexed_coverage_grid(node, args) if args.auto_coverage else None
    result = AutoSegmentationService(backend).run_sweep(
        node, step_deg=args.step_deg, fov=args.fov, out_size=args.view_size,
        channel=channel, phi_angles=args.pitch, min_points=args.min_points, merge_iou=args.merge_iou,
        border_merge_overlap=args.border_merge_overlap, surface_merge_distance=surface_merge_distance,
        merge_rounds=args.merge_rounds, progress_callback=_log_sweep_progress, view_grid=view_grid,
    )
    return _finish(node, result, args)


def pick_segment_command(args: argparse.Namespace) -> int:
    _configure_logging(args.verbose)
    if not _validate_output_extensions(args.outputs):
        return 2

    node = _load_scan(args)

    from pcseg.core.projection import find_nearest_point_index

    if args.point_index is not None:
        point_index = args.point_index
    else:
        point_index = find_nearest_point_index(node, tuple(args.point_xyz))
        logging.info(f"Nearest point to {tuple(args.point_xyz)}: index {point_index}")

    channel = _resolve_channel(node, args.channel)

    from pcseg.services.point_segmentation_service import segment_at_point
    from pcseg.services.sam_service import DEFAULT_CHECKPOINT_PATH as SAM_CKPT
    from pcseg.services.sam_service import DEFAULT_CONFIG_PATH as SAM_CFG
    from pcseg.services.sam_service import SamService

    device = None if args.device == "auto" else args.device
    sam = SamService(config_path=args.sam_config or SAM_CFG, checkpoint_path=args.sam_checkpoint or SAM_CKPT, device=device)
    logging.info(f"Device: {sam.device}")

    result = segment_at_point(
        sam, node, point_index, out_size=args.view_size, fov=args.fov,
        channel=channel, keep_all_masks=args.keep_all_masks,
    )
    return _finish(node, result, args)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return run_command(args)
    if args.command == "auto-segment":
        return auto_segment_command(args)
    if args.command == "pick-segment":
        return pick_segment_command(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
