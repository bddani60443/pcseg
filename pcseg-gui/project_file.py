from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np

from layers import Layer, LayerInstance

FORMAT_VERSION = 1


def save_project(
    path: str, *,
    source_path: str,
    load_decimation: int,
    point_count: int,
    view_decimation: int,
    color_mode: str,
    render_mode: int,
    roi_enabled: bool,
    roi_theta_range: tuple[float, float] | None,
    roi_phi_range: tuple[float, float] | None,
    layers: list[Layer],
) -> None:
    """Writes `path` (overwriting if it exists). `source_path` should be
    exactly what reloading it later needs (ScanNode.file_path, as already
    passed to load_point_cloud) - see load_project_manifest's own
    size/mtime fields for the "has this changed since?" fingerprint,
    computed here from the live file so it reflects what was ACTUALLY
    loaded, not just a nominal path string."""
    try:
        stat = Path(source_path).stat()
        source_size, source_mtime = stat.st_size, stat.st_mtime
    except OSError:
        # Source file missing/inaccessible at save time (moved/deleted
        # after loading, e.g.) - still let the project itself save; the
        # fingerprint just can't be captured, so reload-time comparison is
        # skipped rather than blocking the save.
        source_size, source_mtime = None, None

    manifest_layers = []
    arrays: dict[str, np.ndarray] = {}
    for i, layer in enumerate(layers):
        inst_entries = []
        for j, inst in enumerate(layer.instances):
            key = f"layer{i}_inst{j}"
            arrays[key] = np.asarray(inst.indices)
            inst_entries.append({
                "array_key": key,
                "label": inst.label,
                "color": list(inst.color),
                "included": inst.included,
                "score": inst.score,
                "manual": inst.manual,
            })
        manifest_layers.append({
            "name": layer.name,
            "source_mode": layer.source_mode,
            "visible": layer.visible,
            "expanded": layer.expanded,
            "points_hidden": layer.points_hidden,
            "instances": inst_entries,
        })

    manifest = {
        "format_version": FORMAT_VERSION,
        "source_path": source_path,
        "source_size": source_size,
        "source_mtime": source_mtime,
        "load_decimation": load_decimation,
        "point_count": point_count,
        "view_decimation": view_decimation,
        "color_mode": color_mode,
        "render_mode": render_mode,
        "roi_enabled": roi_enabled,
        "roi_theta_range": list(roi_theta_range) if roi_theta_range is not None else None,
        "roi_phi_range": list(roi_phi_range) if roi_phi_range is not None else None,
        "layers": manifest_layers,
    }

    arrays_buf = BytesIO()
    np.savez_compressed(arrays_buf, **arrays)

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        zf.writestr("arrays.npz", arrays_buf.getvalue())


def load_project_manifest(path: str) -> dict:
    """Just the JSON metadata (source path, decimation, layer/instance
    structure) - cheap, no array data touched. What main_window.open_project
    reads FIRST, before deciding whether the source scan needs relocating
    and before spending time on the actual (potentially slow) scan reload."""
    with zipfile.ZipFile(path, "r") as zf:
        return json.loads(zf.read("manifest.json"))


def load_project_layers(path: str, manifest: dict) -> list[Layer]:
    """The actual Layer/LayerInstance objects (with real index arrays) -
    call once the referenced scan has already been reloaded and its
    point_count has been sanity-checked against manifest["point_count"]
    (see main_window.open_project) - an index array from a DIFFERENT point
    count is meaningless, silently pointing at the wrong points."""
    with zipfile.ZipFile(path, "r") as zf:
        arrays_buf = BytesIO(zf.read("arrays.npz"))
    with np.load(arrays_buf) as npz:
        layers = []
        for layer_data in manifest["layers"]:
            instances = [
                LayerInstance(
                    indices=npz[inst_data["array_key"]],
                    label=inst_data["label"],
                    color=tuple(inst_data["color"]),
                    included=inst_data["included"],
                    score=inst_data["score"],
                    manual=inst_data["manual"],
                )
                for inst_data in layer_data["instances"]
            ]
            layers.append(Layer(
                name=layer_data["name"],
                source_mode=layer_data["source_mode"],
                instances=instances,
                visible=layer_data["visible"],
                expanded=layer_data["expanded"],
                # .get with a default - a project file saved before this
                # field existed simply had no hidden layers at load time.
                points_hidden=layer_data.get("points_hidden", False),
            ))
        return layers
