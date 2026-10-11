from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

import model_sizes
from log_panel import LogPanel
from workers import run_in_background


class ModelsDialog(QDialog):
    """`viewer` is the Viewer3DWidget whose _ensure_dino_service/_ensure_sam_service
    should actually use the selected size for the NEXT run - selection is applied
    immediately (no separate "Apply"/"OK" step), since it only takes effect the
    next time a DinoService/SamService gets lazily constructed anyway."""

    def __init__(self, parent, viewer, log_panel: LogPanel) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self.log_panel = log_panel
        self.setWindowTitle("Manage Models")
        self.resize(560, 420)

        self._busy = False
        self._rows: dict[tuple[str, str], tuple[QLabel, QPushButton]] = {}  # (kind, size) -> (status, download btn)
        self._task_thread = None
        self._task_worker = None

        layout = QVBoxLayout(self)

        active_box = QGroupBox("Active for the next run")
        active_form = QGridLayout(active_box)
        active_form.addWidget(QLabel("GroundingDINO size:"), 0, 0)
        self.dino_combo = QComboBox()
        self.dino_combo.setModel(QStandardItemModel(self.dino_combo))
        self.dino_combo.currentTextChanged.connect(self._on_dino_size_changed)
        active_form.addWidget(self.dino_combo, 0, 1)
        active_form.addWidget(QLabel("SAM2 size:"), 1, 0)
        self.sam_combo = QComboBox()
        self.sam_combo.setModel(QStandardItemModel(self.sam_combo))
        self.sam_combo.currentTextChanged.connect(self._on_sam_size_changed)
        active_form.addWidget(self.sam_combo, 1, 1)
        layout.addWidget(active_box)

        dino_box = QGroupBox("GroundingDINO weights")
        dino_grid = QGridLayout(dino_box)
        for row, size in enumerate(model_sizes.DINO_SIZES):
            self._build_row(dino_grid, row, "dino", size)
        layout.addWidget(dino_box)

        sam_box = QGroupBox("SAM2 weights")
        sam_grid = QGridLayout(sam_box)
        for row, size in enumerate(model_sizes.SAM_SIZES):
            self._build_row(sam_grid, row, "sam", size)
        layout.addWidget(sam_box)

        layout.addWidget(QLabel("Download output also appears in the Log panel below the 3D view."))

        close_row = QHBoxLayout()
        close_row.addStretch()
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        close_row.addWidget(btn_close)
        layout.addLayout(close_row)

        self.refresh_all()
        self.dino_combo.setCurrentText(getattr(self.viewer, "dino_size", "tiny"))
        self.sam_combo.setCurrentText(getattr(self.viewer, "sam_size", "tiny"))

    def _build_row(self, grid: QGridLayout, row: int, kind: str, size: str) -> None:
        grid.addWidget(QLabel(size), row, 0)
        status = QLabel("...")
        status.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        grid.addWidget(status, row, 1)
        btn = QPushButton("Download")
        btn.clicked.connect(lambda: self._start_download(kind, size))
        grid.addWidget(btn, row, 2)
        self._rows[(kind, size)] = (status, btn)

    # ── Status refresh ──────────────────────────────────────────────────────

    def _row_status(self, kind: str, size: str) -> tuple[bool, int]:
        if kind == "dino":
            return model_sizes.dino_checkpoint_status(size)
        return model_sizes.sam_checkpoint_status(size)

    def refresh_row(self, kind: str, size: str) -> None:
        status_label, btn = self._rows[(kind, size)]
        present, n_bytes = self._row_status(kind, size)
        if present:
            status_label.setText(f"present - {model_sizes.human_size(n_bytes)}")
            status_label.setStyleSheet("color: #4caf50;")
            btn.setVisible(False)
        else:
            status_label.setText("missing")
            status_label.setStyleSheet("color: #e05555;")
            btn.setVisible(True)
        self._refresh_combo(kind)

    def refresh_all(self) -> None:
        for kind, size in self._rows:
            self.refresh_row(kind, size)

    def _refresh_combo(self, kind: str) -> None:
        combo = self.dino_combo if kind == "dino" else self.sam_combo
        sizes = model_sizes.DINO_SIZES if kind == "dino" else model_sizes.SAM_SIZES
        current = combo.currentText() or list(sizes)[0]
        combo.blockSignals(True)
        model = combo.model()
        model.clear()
        for size in sizes:
            present, _ = self._row_status(kind, size)
            label = size if present else f"{size} (needs download)"
            item = QStandardItem(label)
            item.setData(size, Qt.ItemDataRole.UserRole)
            if not present:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            model.appendRow(item)
        idx = next((i for i in range(model.rowCount()) if model.item(i).data(Qt.ItemDataRole.UserRole) == current), 0)
        combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    # ── Selection ────────────────────────────────────────────────────────────

    def _current_size(self, combo: QComboBox) -> str | None:
        item = combo.model().item(combo.currentIndex())
        return item.data(Qt.ItemDataRole.UserRole) if item is not None else None

    def _on_dino_size_changed(self, _text: str) -> None:
        size = self._current_size(self.dino_combo)
        if size is not None:
            self.viewer.set_dino_size(size)

    def _on_sam_size_changed(self, _text: str) -> None:
        size = self._current_size(self.sam_combo)
        if size is not None:
            self.viewer.set_sam_size(size)

    # ── Download ─────────────────────────────────────────────────────────────

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.dino_combo.setEnabled(not busy)
        self.sam_combo.setEnabled(not busy)
        for _status, btn in self._rows.values():
            btn.setEnabled(not busy)

    def _start_download(self, kind: str, size: str) -> None:
        if self._busy:
            return
        dino_size = size if kind == "dino" else "tiny"
        sam_size = size if kind == "sam" else "tiny"
        # The script always installs ONE dino size + ONE sam size together (it's
        # not per-file) - pin the OTHER slot to "tiny" (cheap, likely already
        # present, and the script skips anything that's already on disk) rather
        # than re-downloading whatever the user currently has active.
        self.log_panel.append_line(f"--- Downloading GroundingDINO={dino_size}, SAM2={sam_size} ---")
        self._set_busy(True)

        def on_success(_result) -> None:
            self._set_busy(False)
            self.refresh_row(kind, size)
            # Switches "Active for the next run" to what was JUST downloaded -
            # downloading a size you didn't already have is virtually always
            # because you want to use it, not so it can sit on disk while the
            # dropdown quietly stays on whatever was active before (easy to
            # miss, since nothing else in this dialog calls that out). Only
            # the row actually downloaded switches - the other kind's slot
            # (pinned to "tiny" above, not the user's real choice) is left
            # untouched.
            combo = self.dino_combo if kind == "dino" else self.sam_combo
            combo.setCurrentText(size)
            self.log_panel.append_line(f"--- Download finished (GroundingDINO={dino_size}, SAM2={sam_size}) ---")

        def on_error(message: str) -> None:
            self._set_busy(False)
            self.log_panel.append_line(f"--- Download FAILED: {message} ---")

        self._task_thread, self._task_worker = run_in_background(
            model_sizes.run_install_script, dino_size, sam_size,
            on_success=on_success, on_error=on_error,
            on_progress=lambda _pct, msg: self.log_panel.append_line(msg),
        )
