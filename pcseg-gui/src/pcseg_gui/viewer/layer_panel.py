"""
Reorderable layer-stack panel: one compact QGIS-style row per Layer (expand
triangle, visibility checkbox, color swatch, elided name), reordered by
dragging rows rather than ▲/▼ buttons, with per-instance detail nested
INLINE under a layer once its triangle is expanded - not a separate popup or
a shared "Instance details" section below the whole list.

Convention: `self.layers[0]` is the TOP of the stack (most recently added,
displayed at the top of the panel, wins any point/pixel a lower layer also
claims) and `self.layers[-1]` is the BOTTOM (applied first). See `compose_entries`.

Design history: an early version had this same inline-expand idea, reported
live as overwhelming with many instances (a word-wrapped multi-line row
header made the list feel heavy) - replaced with a master-detail layout (one
shared "Instance details" section below the whole list, showing whichever
layer's select-radio was active). Reworked back to inline expand (design
review 2026-09-10, reference: a QGIS layers-panel screenshot) now that the
row header itself is properly compact (one line, real eliding) and expansion
is opt-in per layer (`Layer.expanded`, off by default) rather than always-on -
the earlier complaint was about weight/wrapping, not about inline placement
itself. Reordering also moved from dedicated ▲/▼ buttons to dragging a row's
header (QDrag/QMimeData, matching this QGIS reference too) - deliberately
NOT a QListWidget's built-in drag-and-drop, since rows are custom widgets
(checkboxes, an expand button, nested instance rows) that don't fit a
QListWidgetItem's model cleanly; a plain Python list + explicit
insert/pop stays simpler to get right and to verify programmatically.
"""
from __future__ import annotations

from PyQt6.QtCore import QMimeData, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QDrag, QFontMetrics, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pcseg_gui.layers import Layer, LayerInstance


class _LayerHeader(QWidget):
    """The single-line, always-visible part of a LayerRow - owns click-to-
    select (a plain press+release with no real movement) and drag-to-reorder
    (a press+move past Qt's own drag-start threshold) as ONE gesture, the
    same click-vs-drag disambiguation PickableGLView already uses for
    click-vs-box-select in the 3D view, applied here to click-vs-drag
    instead."""

    clicked = pyqtSignal()
    reorder_requested = pyqtSignal(int, int)  # (from_index, to_index)

    def __init__(self, index: int) -> None:
        super().__init__()
        self.index = index
        self.setAcceptDrops(True)
        self._press_pos = None

    def mousePressEvent(self, ev) -> None:
        if ev.button() == Qt.MouseButton.LeftButton:
            self._press_pos = ev.pos()
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev) -> None:
        if self._press_pos is None or not (ev.buttons() & Qt.MouseButton.LeftButton):
            return
        if (ev.pos() - self._press_pos).manhattanLength() < QApplication.startDragDistance():
            return
        self._press_pos = None  # this press is now a drag, not a pending click - see mouseReleaseEvent
        drag = QDrag(self)
        mime = QMimeData()
        mime.setText(str(self.index))
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.MoveAction)

    def mouseReleaseEvent(self, ev) -> None:
        if self._press_pos is not None:
            self._press_pos = None
            self.clicked.emit()
        super().mouseReleaseEvent(ev)

    def dragEnterEvent(self, ev) -> None:
        if ev.mimeData().hasText():
            ev.acceptProposedAction()

    def dragMoveEvent(self, ev) -> None:
        # QWidget's own default dragMoveEvent ignores the event regardless
        # of what dragEnterEvent decided - without overriding this too, the
        # cursor would show "not allowed" throughout the drag and dropEvent
        # might never fire.
        if ev.mimeData().hasText():
            ev.acceptProposedAction()

    def dropEvent(self, ev) -> None:
        try:
            source_index = int(ev.mimeData().text())
        except ValueError:
            return
        ev.acceptProposedAction()
        self.reorder_requested.emit(source_index, self.index)


class _InstanceSubRow(QWidget):
    """One instance, nested under its layer's expanded header - included
    checkbox, color swatch, label/count/score text. Clicking the row (not
    its checkbox) toggles this instance into/out of the SAME 3D
    instance-selection Viewer3DWidget's own click/box-select in the 3D view
    already drives (`_selected_instance_ids`) - one selection concept, two
    entry points, not a second, separate "selected in the panel" idea."""

    included_toggled = pyqtSignal(bool)
    clicked = pyqtSignal()

    def __init__(self, inst: LayerInstance, index: int, total: int, selected: bool) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 1, 4, 1)
        layout.setSpacing(4)

        self.check = QCheckBox()
        self.check.setChecked(inst.included)
        self.check.setToolTip(
            "Untick to exclude just this object from the layer, without deleting the whole layer."
        )
        self.check.toggled.connect(self.included_toggled.emit)
        layout.addWidget(self.check)

        pix = QPixmap(10, 10)
        pix.fill(QColor(*[int(c) for c in inst.color[:3]]))
        swatch = QLabel()
        swatch.setPixmap(pix)
        layout.addWidget(swatch)

        # See the matching comment in the old _populate_instance_list (git
        # history) for why '' is real GroundingDINO output, not a display
        # bug, and why label text itself is never touched here.
        label_text = inst.label if inst.label else "(unlabeled)"
        manual_marker = "  [manual]" if inst.manual else ""
        score_txt = f", score {inst.score:.2f}" if inst.score is not None else ""
        text = QLabel(f"{label_text} #{index + 1}  ({len(inst.indices):,} pts){score_txt}{manual_marker}")
        text.setStyleSheet("font-size: 11px;")
        layout.addWidget(text, stretch=1)

        tooltip_label = inst.label if inst.label else "(none - see the label above)"
        manual_txt = "\nAdded via \"Pick into this layer...\" (manual), not this layer's own run." \
            if inst.manual else ""
        self.setToolTip(
            f"Label: {tooltip_label}  (instance #{index + 1} of {total}{score_txt})\n"
            "Click to select/deselect (highlights blue in the 3D view, same as clicking it there). "
            f"Untick the box to exclude it from the layer.{manual_txt}"
        )
        # WA_StyledBackground is what actually makes a plain QWidget subclass
        # paint a stylesheet background at all - without it, background-color
        # silently does nothing on a custom QWidget, only on widgets Qt
        # already paints a background for by default. Applied UNCONDITIONALLY
        # (not just when selected) - measured live that applying a stylesheet
        # only in the selected branch changes this widget's own sizeHint()
        # (Qt's CSS box-model layout path computes different effective
        # padding than the native style's), which visibly shrank/resized the
        # row the moment it got selected. Both states now go through the
        # identical styled path, varying only the background color - a
        # transparent background reads the same as "no stylesheet" visually,
        # but keeps the geometry stable across selection changes.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bg = "#3a5a8a" if selected else "transparent"
        self.setStyleSheet(f"background-color: {bg}; border-radius: 3px;")

    def mousePressEvent(self, ev) -> None:
        if ev.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(ev)


class LayerRow(QFrame):
    """One layer: a compact `_LayerHeader` (expand triangle, visibility
    checkbox, color swatch, elided name) plus, only while `layer.expanded`,
    its instances nested inline as `_InstanceSubRow`s directly below. Purely
    presentational - all state lives in the `Layer`/`LayerInstance`s it's
    given; every interaction re-emits through the row's own signals rather
    than mutating anything itself, so LayerPanel stays the single source of
    truth for `self.layers` and can freely rebuild rows."""

    visibility_toggled = pyqtSignal(bool)
    select_requested = pyqtSignal()
    pick_into_requested = pyqtSignal()
    delete_requested = pyqtSignal()
    expand_toggled = pyqtSignal()
    reorder_requested = pyqtSignal(int, int)  # (from_index, to_index)
    instance_toggled = pyqtSignal(int, bool)  # (instance_index, included)
    instance_clicked = pyqtSignal(int)  # instance_index
    points_hidden_toggled = pyqtSignal(bool)  # "Hide points from 3D view" context-menu action

    def __init__(self, layer: Layer, index: int, selected: bool, highlighted_instance_ids: set[int]) -> None:
        super().__init__()
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.layer = layer

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = _LayerHeader(index)
        header.clicked.connect(self.select_requested.emit)
        header.reorder_requested.connect(self.reorder_requested.emit)
        header.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        header.customContextMenuRequested.connect(self._show_context_menu)
        header.setToolTip(
            "Click to select this layer (its 3D bounding-box overlay and instances become the "
            "active ones). Drag to reorder - higher in the stack wins any point/pixel a lower "
            "layer also claims. Right-click for more actions."
        )
        # Only visual cue that ties this row back to "this is the layer the
        # 3D view's instance-box overlay is currently showing" - easy to
        # miss at a glance otherwise, now that there's no select radio.
        # WA_StyledBackground/unconditional-stylesheet: see the matching
        # comment in _InstanceSubRow - a stylesheet applied only when
        # selected measurably changed this widget's own sizeHint(), which
        # shrank the whole row the moment it got selected. Always styled
        # now, varying only the color, keeps the row's size stable either
        # way.
        header.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bg = "#3a5a8a" if selected else "transparent"
        header.setStyleSheet(f"background-color: {bg}; border-radius: 3px;")

        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(4, 3, 4, 3)
        header_layout.setSpacing(4)

        self.expand_btn = QPushButton("▼" if layer.expanded else "▶")
        self.expand_btn.setFixedWidth(18)
        self.expand_btn.setFlat(True)
        self.expand_btn.setEnabled(bool(layer.instances))
        self.expand_btn.setToolTip("Show/hide this layer's individual instances.")
        self.expand_btn.clicked.connect(self.expand_toggled.emit)
        header_layout.addWidget(self.expand_btn)

        self.visible_check = QCheckBox()
        self.visible_check.setChecked(layer.visible)
        self.visible_check.setToolTip("Show/hide this whole layer")
        self.visible_check.toggled.connect(self.visibility_toggled.emit)
        header_layout.addWidget(self.visible_check)

        # A small swatch of the layer's FIRST instance color (a rough visual
        # cue, not a precise legend - a layer can hold many differently
        # colored instances).
        if layer.instances:
            pix = QPixmap(12, 12)
            pix.fill(QColor(*[int(c) for c in layer.instances[0].color[:3]]))
            swatch = QLabel()
            swatch.setPixmap(pix)
            header_layout.addWidget(swatch)

        included = sum(1 for i in layer.instances if i.included)
        full_text = f"{layer.name} · {layer.source_mode} ({included}/{len(layer.instances)})"
        name_label = QLabel()
        # A QLabel's minimumSizeHint() normally equals its sizeHint() (full
        # unelided text width) for single-line text - setMinimumWidth(0) is
        # a NO-OP against that (0 doesn't override anything already >= 0);
        # an explicit small nonzero minimum actually overrides it, letting
        # the layout shrink the label down to it (with elided text standing
        # in) instead of forcing the whole row/panel wider than it should be.
        name_label.setMinimumWidth(20)
        self._full_text = full_text
        self._name_label = name_label
        header_layout.addWidget(name_label, stretch=1)

        outer.addWidget(header)

        if layer.expanded and layer.instances:
            sub_container = QWidget()
            sub_layout = QVBoxLayout(sub_container)
            sub_layout.setContentsMargins(22, 0, 2, 2)  # indent under the expand triangle
            sub_layout.setSpacing(0)
            n = len(layer.instances)
            for i, inst in enumerate(layer.instances):
                sub_row = _InstanceSubRow(inst, i, n, selected=(id(inst) in highlighted_instance_ids))
                sub_row.included_toggled.connect(lambda checked, i=i: self.instance_toggled.emit(i, checked))
                sub_row.clicked.connect(lambda i=i: self.instance_clicked.emit(i))
                sub_layout.addWidget(sub_row)
            outer.addWidget(sub_container)

        self._elide_name_label()

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        self._elide_name_label()

    def _elide_name_label(self) -> None:
        fm = QFontMetrics(self._name_label.font())
        elided = fm.elidedText(self._full_text, Qt.TextElideMode.ElideRight, self._name_label.width())
        self._name_label.setText(elided)

    def _show_context_menu(self, pos) -> None:
        menu = QMenu(self)
        pick_action = menu.addAction("Pick into this layer...")
        pick_action.triggered.connect(self.pick_into_requested.emit)
        menu.addSeparator()
        # Separate from the header's own visibility checkbox (which only
        # reverts this layer's points to unclassified/base coloring, still
        # rendered) - this removes them from the 3D view entirely, folded
        # into node.hidden_mask (see Layer.points_hidden's own docstring and
        # Viewer3DWidget._recompute_composition). A checkable menu item
        # rather than a second always-visible row checkbox, to keep the
        # compact header from growing a second checkbox that would be easy
        # to confuse with the first at a glance.
        hide_action = menu.addAction("Hide points from 3D view")
        hide_action.setCheckable(True)
        hide_action.setChecked(self.layer.points_hidden)
        hide_action.toggled.connect(self.points_hidden_toggled.emit)
        menu.addSeparator()
        delete_action = menu.addAction("Delete Layer")
        delete_action.triggered.connect(self.delete_requested.emit)
        menu.exec(self.sender().mapToGlobal(pos))


class LayerPanel(QWidget):
    """Owns the layer list itself (not just its display) - Viewer3DWidget adds
    layers and reads `compose_entries()`/`combined_indices()` off this, and reacts
    to `changed` to recompute the 3D view's colors and node.point_classes."""

    changed = pyqtSignal()
    # Fired whenever `selected_layer` itself changes identity (add/select/delete) -
    # separate from `changed` so Viewer3DWidget can refresh the (cheap) instance-box
    # overlay without over-triggering it on every single unrelated mutation... though
    # in practice it listens to both, since a reorder/visibility/include change can
    # also affect what the SELECTED layer's overlay should currently show.
    selectionChanged = pyqtSignal()
    # A layer's right-click menu chose "Pick into this layer..." - Viewer3DWidget
    # connects this to _start_pick_into_layer, which owns everything past this
    # point (settings dialog, arming, running SAM2, adding the result back in).
    pick_into_requested = pyqtSignal(object)  # Layer
    # An instance sub-row was clicked (see _InstanceSubRow) - Viewer3DWidget
    # connects this to toggle the SAME `_selected_instance_ids`/3D-highlight
    # system a 3D-view click or box-select already drives.
    instance_clicked = pyqtSignal(object, object)  # (Layer, LayerInstance)

    def __init__(self) -> None:
        super().__init__()
        self.layers: list[Layer] = []
        self.selected_layer: Layer | None = None
        # Set by Viewer3DWidget (see set_highlighted_instance_ids) whenever
        # `_selected_instance_ids` changes, purely so expanded instance
        # sub-rows can show the same highlight state - this panel doesn't
        # otherwise know or care about 3D selection.
        self.highlighted_instance_ids: set[int] = set()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        self.empty_label = QLabel("No layers yet - run a segmentation to add one.")
        self.empty_label.setWordWrap(True)
        self.empty_label.setStyleSheet("color: #888;")
        outer.addWidget(self.empty_label)

        btn_new_empty = QPushButton("+ New empty layer...")
        btn_new_empty.setToolTip(
            "Creates an empty layer (you name it) with nothing in it yet, then immediately "
            "offers to arm picking into it - same as right-clicking any existing layer and "
            "choosing \"Pick into this layer...\", just starting from a fresh one instead of "
            "an existing run's result."
        )
        btn_new_empty.clicked.connect(self._on_new_empty_layer_clicked)
        outer.addWidget(btn_new_empty)

        self.rows_layout = QVBoxLayout()
        self.rows_layout.setSpacing(2)
        outer.addLayout(self.rows_layout)

        outer.addStretch()

    # ── Mutation entry points (Viewer3DWidget calls these) ─────────────────

    def add_layer(self, layer: Layer) -> None:
        self.layers.insert(0, layer)  # new layer -> top of stack, top of panel
        # The just-added layer becomes the instance-box overlay's target by
        # default - "show me what I just got" is the common case; the user
        # can always click a different layer afterward.
        self.selected_layer = layer
        self._rebuild()
        self.changed.emit()
        self.selectionChanged.emit()

    def add_empty_layer(self, name: str, source_mode: str = "Manual") -> Layer:
        """Creates and adds a Layer with no instances yet - the "+ New empty
        layer..." entry point for single-point picking that isn't targeting
        an existing run's result."""
        layer = Layer(name=name, source_mode=source_mode, instances=[])
        self.add_layer(layer)
        return layer

    def add_instance(self, layer: Layer, instance: LayerInstance) -> None:
        """Appends `instance` to `layer` (must already be in `self.layers`) -
        the single-point-picking counterpart of add_layer, for adding ONE
        object into an EXISTING layer instead of creating a new one. Selects
        `layer` too (same "show me what I just got" default as add_layer),
        which matters when picking into a layer other than the currently
        selected one."""
        layer.instances.append(instance)
        self.selected_layer = layer
        self._rebuild()
        self.changed.emit()
        self.selectionChanged.emit()

    def select(self, layer: Layer) -> None:
        if layer is self.selected_layer:
            return
        self.selected_layer = layer
        self._rebuild()
        self.selectionChanged.emit()

    def set_highlighted_instance_ids(self, ids: set[int]) -> None:
        """Viewer3DWidget calls this after any change to its own
        `_selected_instance_ids` so expanded instance sub-rows can reflect
        the same selection - see the `highlighted_instance_ids` attribute."""
        ids = set(ids)
        if ids == self.highlighted_instance_ids:
            return
        self.highlighted_instance_ids = ids
        self._rebuild()

    def _on_new_empty_layer_clicked(self) -> None:
        name, ok = QInputDialog.getText(self, "New empty layer", "Layer name:")
        name = name.strip()
        if not ok or not name:
            return
        layer = self.add_empty_layer(name)
        self.pick_into_requested.emit(layer)

    def clear(self) -> None:
        self.layers = []
        self.selected_layer = None
        self._rebuild()
        self.changed.emit()
        self.selectionChanged.emit()

    def refresh(self) -> None:
        """For a caller (Viewer3DWidget's point-deletion/undo) that mutated
        a Layer's/LayerInstance's data directly rather than through one of
        this class's own methods - e.g. deleting points shrinks an
        instance's `.indices` in place, which this panel has no way to
        observe on its own. Re-renders rows from the current
        self.layers/self.selected_layer and notifies listeners, same as any
        other mutation here does."""
        self._rebuild()
        self.changed.emit()

    # ── Composition (Viewer3DWidget reads these) ────────────────────────────

    def compose_entries(self) -> list[tuple[str, "object", tuple]]:
        """Bottom-of-stack first (painted first) -> top-of-stack last (wins),
        matching self.layers[0] = top-of-stack/top-of-panel."""
        entries = []
        for layer in reversed(self.layers):
            entries.extend(layer.active_entries())
        return entries

    def combined_indices(self):
        import numpy as np
        arrays = [indices for _, indices, _ in self.compose_entries()]
        if not arrays:
            return np.array([], dtype=np.int64)
        return np.unique(np.concatenate(arrays))

    # ── Internal: rebuild rows from self.layers ─────────────────────────────

    def _rebuild(self) -> None:
        while self.rows_layout.count():
            child = self.rows_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        self.empty_label.setVisible(not self.layers)

        for i, layer in enumerate(self.layers):
            row = LayerRow(
                layer, index=i, selected=(layer is self.selected_layer),
                highlighted_instance_ids=self.highlighted_instance_ids,
            )
            row.visibility_toggled.connect(lambda checked, i=i: self._set_visibility(i, checked))
            row.select_requested.connect(lambda layer=layer: self.select(layer))
            row.pick_into_requested.connect(lambda layer=layer: self.pick_into_requested.emit(layer))
            row.delete_requested.connect(lambda i=i: self._delete(i))
            row.points_hidden_toggled.connect(lambda checked, i=i: self._set_points_hidden(i, checked))
            row.expand_toggled.connect(lambda layer=layer: self._toggle_expanded(layer))
            row.reorder_requested.connect(self._reorder)
            row.instance_toggled.connect(lambda inst_i, checked, i=i: self._set_instance_included(i, inst_i, checked))
            row.instance_clicked.connect(
                lambda inst_i, layer=layer: self.instance_clicked.emit(layer, layer.instances[inst_i])
            )
            self.rows_layout.addWidget(row)

    def _toggle_expanded(self, layer: Layer) -> None:
        layer.expanded = not layer.expanded
        self._rebuild()

    def _reorder(self, from_i: int, to_i: int) -> None:
        if from_i == to_i or not (0 <= from_i < len(self.layers)) or not (0 <= to_i < len(self.layers)):
            return
        layer = self.layers.pop(from_i)
        self.layers.insert(to_i, layer)
        self._rebuild()
        self.changed.emit()

    def _delete(self, i: int) -> None:
        deleted = self.layers[i]
        del self.layers[i]
        if deleted is self.selected_layer:
            # Fall back to the new top of the stack, same "most recent"
            # default add_layer picks - None only once the stack is empty.
            self.selected_layer = self.layers[0] if self.layers else None
            self._rebuild()
            self.changed.emit()
            self.selectionChanged.emit()
        else:
            self._rebuild()
            self.changed.emit()

    def _set_visibility(self, i: int, checked: bool) -> None:
        self.layers[i].visible = checked
        self.changed.emit()

    def _set_points_hidden(self, i: int, checked: bool) -> None:
        """See Layer.points_hidden's own docstring - independent of
        `visible`, this removes the layer's points from the 3D view
        entirely rather than just reverting their coloring. `changed` is
        the SAME signal `_set_visibility` already emits - Viewer3DWidget's
        own handler (_recompute_composition) recomputes both the
        composed coloring AND node.hidden_mask from the current layer
        stack every time, so one signal already covers both concerns."""
        self.layers[i].points_hidden = checked
        self.changed.emit()

    def _set_instance_included(self, layer_i: int, inst_i: int, checked: bool) -> None:
        self.layers[layer_i].instances[inst_i].included = checked
        self._rebuild()  # refreshes the row's "(included/total)" count label
        self.changed.emit()
