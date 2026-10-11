from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LayerInstance:
    """One detected object within a layer. `included=False` drops it from the
    composed stack without discarding it - e.g. temporarily exclude one
    Run detection you're not sure about, reversibly, within the session."""
    indices: np.ndarray
    label: str
    color: tuple
    included: bool = True
    # DINO's box confidence (Run) or SAM2's own mask score (a pick) - GUI-only
    # display data (the 3D instance-box overlay's label text); not used by
    # composition/export, which only ever needs label+indices+color.
    score: float | None = None
    # True for an instance added via the Layers panel's "Pick into this
    # layer..." (see viewer_widget.py's _commit_pick_candidate) - lets a
    # hand-picked object stay visually distinguishable even after landing in
    # an otherwise fully-automatic layer, now that picking targets an
    # arbitrary EXISTING layer instead of always creating its own new one.
    manual: bool = False


@dataclass
class Layer:
    name: str
    source_mode: str  # "Run Prompt" | "Manual" (an empty layer created from the Layers panel)
    instances: list[LayerInstance] = field(default_factory=list)
    visible: bool = True
    expanded: bool = False
    points_hidden: bool = False

    def active_entries(self) -> list[tuple[str, np.ndarray, tuple]]:
        """(label, indices, color) for this layer's included instances - empty if the
        whole layer is hidden, regardless of individual instances' included flags."""
        if not self.visible:
            return []
        return [(inst.label, inst.indices, inst.color) for inst in self.instances if inst.included]

    def combined_indices(self) -> np.ndarray:
        if not self.instances:
            return np.array([], dtype=np.int64)
        return np.unique(np.concatenate([inst.indices for inst in self.instances]))

    def hidable_indices(self) -> np.ndarray:
        """Points this layer would remove from the 3D view if `points_hidden`
        is True - the same "included" instances active_entries() would color
        if `visible`, but independent of `visible` itself (see
        `points_hidden`'s own docstring for why the two toggles are separate)."""
        included = [inst.indices for inst in self.instances if inst.included]
        if not included:
            return np.array([], dtype=np.int64)
        return np.unique(np.concatenate(included))
