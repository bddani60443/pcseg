from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QSpinBox,
)

from crash_log import LOG_FILE
from loaders import load_point_cloud
from log_forwarder import LogForwarder, attach_log_forwarder
from log_panel import LogPanel
from models_dialog import ModelsDialog
from viewer.viewer_widget import Viewer3DWidget
from workers import run_in_background


class _LoadOptionsDialog(QDialog):
    """Two DISTINCT, deliberately separately-labeled decimation controls, so
    "how much real data got loaded" (Load detail) and "how much of it I'm
    currently choosing to look at" (View detail) can't be confused for one
    another - the second one only affects rendering and is also adjustable
    live later via the main toolbar's own "View detail" spinbox."""

    def __init__(self, parent, default_load: int, default_view: int) -> None:
        super().__init__(parent)
        self.setWindowTitle("Open scan options")
        form = QFormLayout(self)

        self.load_spin = QSpinBox()
        self.load_spin.setRange(1, 32)
        self.load_spin.setValue(default_load)
        self.load_spin.setToolTip(
            "E57 grid decimation step (matches the CLI's --decimation): how many REAL "
            "points get loaded from the file at all. Affects segmentation quality - a "
            "coarser value is a genuinely blurrier source, not just a rendering shortcut."
        )
        form.addRow("Load detail (1 = full resolution):", self.load_spin)

        self.view_spin = QSpinBox()
        self.view_spin.setRange(1, 32)
        self.view_spin.setValue(default_view)
        self.view_spin.setToolTip(
            "Rendering-only subsample of the points once loaded, for interactive "
            "performance. Segmentation always runs against the FULL loaded point set "
            "regardless of this - it never affects quality, only how much you're "
            "currently choosing to look at. Adjustable live after loading too."
        )
        form.addRow("View detail (1 = show every loaded point):", self.view_spin)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("pcseg-view")
        self.resize(1280, 800)

        self.viewer = Viewer3DWidget()
        self.setCentralWidget(self.viewer)
        self.viewer.export_full_resolution_requested.connect(self.export_at_full_resolution)

        self.status_label = QLabel("No file loaded.")
        self.statusBar().addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)

        self._last_decimation = 2  # remembered across opens, .e57 only (.las ignores it)
        self._last_view_decimation = 1  # remembered across opens, both .e57 and .las
        self._load_thread = None
        self._load_worker = None
        self._models_dialog = None

        self._build_log_dock()
        self._build_menu()

        # Persistent, app-lifetime log forwarding into the dock - separate from
        # (and in addition to) Viewer3DWidget's own per-task forwarding into its
        # single-line status label (log_forwarder.py's attach/detach is meant to
        # be used exactly this way, more than once, for different consumers).
        # Never detached - the dock's whole point is a durable history of
        # everything logged for the session, not just what happened during an
        # active sweep.
        self._log_forwarder = LogForwarder()
        self._log_forwarder.message.connect(self.log_panel.append_line)
        self._log_handler = attach_log_forwarder(self._log_forwarder)

        # crash_log.install() already logged this to the console + the crash
        # log file itself, but that happened in main() before this dock (or
        # its forwarder) existed - a packaged/double-clicked run has no
        # console, so this is the one copy the user can actually see without
        # already knowing where to look on disk.
        self.log_panel.append_line(f"Crash log: {LOG_FILE}")

    def _build_log_dock(self) -> None:
        self.log_panel = LogPanel()
        dock = QDockWidget("Log", self)
        dock.setObjectName("log_dock")
        dock.setWidget(self.log_panel)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        # Qt's default dock-layout split gave this roughly HALF the window's
        # height with nothing else competing for the bottom area - starving
        # the viewer/settings panel above it (reported live as panels looking
        # crowded/overlapping - confirmed the real cause: this dock, not the
        # settings splitter, was eating most of the vertical space). A log is
        # secondary, reference material - start it modest; still fully
        # resizable/floatable/closable by hand.
        self.resizeDocks([dock], [160], Qt.Orientation.Vertical)
        self.log_dock = dock

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        self.open_action = file_menu.addAction("&Open scan or result...")
        self.open_action.triggered.connect(self.open_file_dialog)

        file_menu.addSeparator()
        open_project_action = file_menu.addAction("Open &Project...")
        open_project_action.setToolTip(
            "Reopens a saved .pcsegproj - reloads the scan it references (by path) and "
            "restores the layer stack, ROI, and color/render mode from where you left off."
        )
        open_project_action.triggered.connect(self.open_project_dialog)
        save_project_action = file_menu.addAction("&Save Project As...")
        save_project_action.setToolTip(
            "Saves the current layer stack (labels, colors, included/manual flags), ROI, "
            "and color/render mode to a .pcsegproj file - references the loaded scan by "
            "its file path rather than duplicating it, so the original file needs to stay "
            "reachable (or get relocated) to reopen this project later."
        )
        save_project_action.triggered.connect(self.save_project_dialog)

        file_menu.addSeparator()
        reload_detail_action = file_menu.addAction("&Reload at Different Detail...")
        reload_detail_action.setToolTip(
            "Reloads the current scan at a different Load detail (e.g. go from a coarse "
            "decimation to full resolution, or the other way) and tries to carry your "
            "current classifications over to the newly loaded points - a spatial transfer "
            "(voxel-based), not exact, since a different decimation samples an entirely "
            "different set of points. Points right at an instance's edge may not carry "
            "over; an instance that ends up with none of its own points among the new ones "
            "is dropped."
        )
        reload_detail_action.triggered.connect(self.reload_at_different_detail_dialog)

        file_menu.addSeparator()
        exit_action = file_menu.addAction("E&xit")
        exit_action.triggered.connect(self.close)

        tools_menu = self.menuBar().addMenu("&Tools")
        tools_menu.setToolTipsVisible(True)
        models_action = tools_menu.addAction("&Manage Models...")
        models_action.triggered.connect(self.open_models_dialog)
        tools_menu.addSeparator()

        coverage_action = tools_menu.addAction("Save &Coverage Grid Preview...")
        coverage_action.setToolTip(
            "Saves a PNG of the scan's panorama showing how the CURRENT Run settings "
            "would sweep it: each scanned pixel tinted by how many views see it (red = "
            "gap, green = 1, amber = 2, purple = 3+), every view's footprint outlined "
            "and numbered in sweep order - check the coverage before spending the time "
            "(or GPU memory) on a real Run."
        )
        coverage_action.triggered.connect(self.viewer.save_coverage_grid_preview)

        debug_action = tools_menu.addAction("Save Debug Images (Runs and Picks)")
        debug_action.setCheckable(True)
        debug_action.setChecked(self.viewer.debug_sweep_images)
        debug_action.setToolTip(
            "Saves an annotated PNG (box + label/score + mask overlay) for every view "
            "DINO+SAM2 processes during a Run - the coarse sweep into its own 'prefilter' "
            "folder, the Refine pass (if on) into its own 'refine' folder - and one PNG per "
            "point-click pick into a 'picks' folder (the view SAM2 saw with the clicked "
            "point(s), then every candidate mask with its score and whether it was kept), "
            "so the whole detection pipeline stays visually traceable. Prompts for a folder "
            "the first time it's turned on; remembered across restarts either way."
        )
        debug_action.toggled.connect(lambda checked: self._on_debug_sweep_images_toggled(debug_action, checked))
        debug_folder_action = tools_menu.addAction("Debug Images Folder...")
        debug_folder_action.setToolTip(
            "Choose or change the base folder debug images are saved into - each Run gets "
            "its own timestamped subfolder and picks share a 'picks' subfolder, so nothing "
            "is ever overwritten."
        )
        debug_folder_action.triggered.connect(self.viewer.set_debug_output_folder)
        tools_menu.addSeparator()

        reset_all_action = tools_menu.addAction("Reset All Settings to Defaults")
        reset_all_action.setToolTip(
            "Resets Region of interest and every Run setting at once - Run's own settings "
            "page instead resets one field at a time (a small icon appears next to a field "
            "once it's been changed). Single-point picking has no persistent settings at "
            "all (always uses the current view channel and a fixed FOV), so it's unaffected."
        )
        reset_all_action.triggered.connect(self.viewer.reset_all_settings)

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.log_dock.toggleViewAction())
        view_menu.addSeparator()
        outline_action = view_menu.addAction("Instance Outline Effect")
        outline_action.setCheckable(True)
        outline_action.setChecked(self.viewer.outline_effect_enabled)
        outline_action.setToolTip(
            "Draws a solid-color outline around the currently hovered/selected instance's "
            "own on-screen shape - a real GPU post-process effect (not everyone wants the "
            "extra render cost or the look). Remembered across restarts."
        )
        outline_action.toggled.connect(self.viewer.set_outline_effect_enabled)
        view_menu.addSeparator()
        bg_action = view_menu.addAction("&Background Color...")
        bg_action.setToolTip("Changes the 3D view's background color. Remembered across restarts.")
        bg_action.triggered.connect(self.viewer.pick_background_color)
        bg_reset_action = view_menu.addAction("Reset Background Color to Default")
        bg_reset_action.triggered.connect(self.viewer.reset_background_color)
        view_menu.addSeparator()
        label_size_action = view_menu.addAction("&Label Size...")
        label_size_action.setToolTip(
            "Font size for the instance-box overlay's billboard labels (Instance labels, "
            "in the toolbar above the 3D view)."
        )
        label_size_action.triggered.connect(self.viewer.pick_label_size)

    def _on_debug_sweep_images_toggled(self, action, checked: bool) -> None:
        """Wired to the Tools menu's checkable 'Save Debug Images (Runs and Picks)'
        action instead of a direct .toggled.connect(self.viewer....) (unlike
        every other checkable action here) because this ONE toggle can fail
        (turning it on with no remembered folder prompts for one - see
        Viewer3DWidget.set_debug_sweep_images - and a cancelled prompt must
        leave it off). blockSignals around the correction avoids re-entering
        this same handler from the setChecked call below."""
        actual = self.viewer.set_debug_sweep_images(checked)
        if actual != checked:
            action.blockSignals(True)
            action.setChecked(actual)
            action.blockSignals(False)

    def open_models_dialog(self) -> None:
        if self._models_dialog is None:
            self._models_dialog = ModelsDialog(self, self.viewer, self.log_panel)
        else:
            self._models_dialog.refresh_all()
        self._models_dialog.show()
        self._models_dialog.raise_()
        self._models_dialog.activateWindow()

    def _has_unsaved_project_state(self) -> bool:
        """A loaded scan with at least one layer is "work that would be lost"
        - an empty/no scan has nothing worth prompting to save."""
        return self.viewer.node is not None and bool(self.viewer.layer_panel.layers)

    def _confirm_discard_current_project(self) -> bool:
        """Gate for every "load a genuinely different scan" entry point
        (open_file_dialog, open_project) - NOT reload_at_different_detail
        (that one transfers the current layers over to the new points
        instead of discarding them, and has its own separate confirmation).
        Returns False if the caller should abandon the load entirely (the
        user hit Cancel here, or backed out of the save-path dialog while
        trying to save first - silently losing work was never the request in
        either case)."""
        if not self._has_unsaved_project_state():
            return True
        reply = QMessageBox.question(
            self, "Unsaved changes",
            "Loading a new scan will discard the current layers unless you save this "
            "project first.\n\nSave the current project before continuing?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if reply == QMessageBox.StandardButton.Cancel:
            return False
        if reply == QMessageBox.StandardButton.Save:
            return self.save_project_dialog()
        return True  # Discard

    def open_file_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open point cloud", "", "Point clouds (*.e57 *.las);;E57 scans (*.e57);;LAS results (*.las)",
        )
        if not path:
            return
        if not self._confirm_discard_current_project():
            return

        decimation = self._last_decimation
        view_decimation = self._last_view_decimation
        if Path(path).suffix.lower() == ".e57":
            dialog = _LoadOptionsDialog(self, self._last_decimation, self._last_view_decimation)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            decimation = dialog.load_spin.value()
            view_decimation = dialog.view_spin.value()
            self._last_decimation = decimation
            self._last_view_decimation = view_decimation
        # .las results were already decimated whenever pcseg exported them, so
        # Load detail doesn't apply - but View detail still can, live via the
        # viewer's own toolbar spinbox after loading, so no dialog is needed here.

        self.open_file(path, decimation=decimation, view_decimation=view_decimation)

    def open_file(self, path: str, decimation: int = 2, view_decimation: int = 1, on_loaded=None) -> None:
        """Loads on a background thread - E57 loading (dequantizing coordinates, computing
        normals from the range image) can take several seconds on a large scan, and blocking
        the UI for that gives no feedback that anything is happening at all.

        `on_loaded(node)`, if given, runs right after _finish_load's own
        load_node() call - open_project's hook to restore the rest of a
        project file's state (layers, ROI, color/render mode) once the scan
        it references has actually finished loading, without open_file
        itself needing to know anything about project files."""
        self.open_action.setEnabled(False)
        self.status_label.setText(f"Loading {Path(path).name}...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # indeterminate - loaders.py reports no fine-grained progress

        def _on_success(node) -> None:
            self._finish_load(node, path, decimation, view_decimation)
            if on_loaded is not None:
                on_loaded(node)

        def _on_error(message: str) -> None:
            logging.exception(f"Failed to load {path}")
            QMessageBox.critical(self, "Load failed", message)
            self.status_label.setText("Load failed.")
            self.progress_bar.setVisible(False)
            self.open_action.setEnabled(True)

        self._load_thread, self._load_worker = run_in_background(
            load_point_cloud, path, decimation=decimation,
            on_success=_on_success, on_error=_on_error,
        )

    def _finish_load(self, node, path: str, decimation: int, view_decimation: int = 1) -> None:
        self.progress_bar.setVisible(False)
        self.open_action.setEnabled(True)
        self.viewer.load_node(node, view_decimation=view_decimation)
        self.status_label.setText(
            f"{node.scan_name}: {node.point_count:,} points "
            f"(load detail={decimation}, view detail={view_decimation})"
        )

    def save_project_dialog(self) -> bool:
        """Returns whether a project was actually saved - False for "nothing
        to save"/a cancelled path dialog/a write failure alike, since
        _confirm_discard_current_project treats all three the same way (not
        safe to proceed with discarding the current state)."""
        if self.viewer.node is None:
            QMessageBox.information(self, "Nothing to save", "Load a scan first.")
            return False
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project As", "", "pcseg projects (*.pcsegproj)",
        )
        if not path:
            return False
        if not path.lower().endswith(".pcsegproj"):
            path += ".pcsegproj"
        try:
            self.viewer.save_project(path)
        except OSError as e:
            logging.exception(f"Failed to save project to {path}")
            QMessageBox.critical(self, "Save failed", str(e))
            return False
        return True

    def open_project_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", "pcseg projects (*.pcsegproj)",
        )
        if not path:
            return
        self.open_project(path)

    def open_project(self, path: str) -> None:
        """Reads the project's manifest (cheap - just JSON, no array data
        yet), makes sure the scan it references is actually reachable
        (prompting to relocate it if the recorded path is gone - the
        standard "external asset moved" recovery every project-file-based
        tool needs, since this format deliberately never embeds the scan
        itself - see project_file.py), then reloads that scan the normal
        way (open_file) and restores the rest of the project's state once
        it's actually finished loading."""
        if not self._confirm_discard_current_project():
            return

        import project_file

        try:
            manifest = project_file.load_project_manifest(path)
        except Exception as e:
            logging.exception(f"Failed to read project {path}")
            QMessageBox.critical(self, "Open project failed", f"Couldn't read '{path}':\n{e}")
            return

        source_path = manifest.get("source_path", "")
        if not source_path or not Path(source_path).exists():
            QMessageBox.information(
                self, "Locate source scan",
                f"This project's original file isn't at its recorded location:\n{source_path}\n\n"
                "Pick where it is now.",
            )
            source_path, _ = QFileDialog.getOpenFileName(
                self, "Locate source scan", str(Path(path).parent),
                "Point clouds (*.e57 *.las)",
            )
            if not source_path:
                return

        expected_size = manifest.get("source_size")
        if expected_size is not None:
            try:
                actual_size = Path(source_path).stat().st_size
            except OSError:
                actual_size = None
            if actual_size is not None and actual_size != expected_size:
                proceed = QMessageBox.warning(
                    self, "Source file may have changed",
                    "The scan file's size doesn't match what this project was saved "
                    "against - if it's genuinely a different file (or was re-exported/"
                    "re-decimated since), the saved selections may no longer line up "
                    "with the right points.\n\nOpen it anyway?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if proceed != QMessageBox.StandardButton.Yes:
                    return

        def _on_loaded(node) -> None:
            expected_count = manifest.get("point_count")
            if expected_count is not None and node.point_count != expected_count:
                QMessageBox.warning(
                    self, "Point count mismatch",
                    f"Loaded {node.point_count:,} points, but this project was saved "
                    f"against {expected_count:,} - the saved selections almost certainly "
                    "won't line up with the right points (a different Load detail, or a "
                    "genuinely different file). Restoring layers anyway, but check the "
                    "result carefully.",
                )
            try:
                layers = project_file.load_project_layers(path, manifest)
            except Exception as e:
                logging.exception(f"Failed to read project layers from {path}")
                QMessageBox.critical(self, "Open project failed", f"Couldn't restore layers:\n{e}")
                return
            self.viewer.restore_project_state(manifest, layers)
            self.status_label.setText(f"Project opened: {Path(path).name}")

        self.open_file(
            source_path,
            decimation=manifest.get("load_decimation", 1),
            view_decimation=manifest.get("view_decimation", 1),
            on_loaded=_on_loaded,
        )

    def reload_at_different_detail_dialog(self) -> None:
        """Reloads the CURRENTLY loaded scan's own file at a different Load
        detail, carrying the current Layers stack over via a spatial (voxel)
        transfer - see viewer.snapshot_for_label_transfer/
        apply_label_transfer and label_transfer.py's own module docstring
        for why this can't just reuse the old indices (a different
        decimation samples an entirely different set of points, so the old
        indices are meaningless against the new ones)."""
        if self.viewer.node is None:
            QMessageBox.information(self, "Nothing loaded", "Load a scan first.")
            return
        if Path(self.viewer.node.file_path).suffix.lower() != ".e57":
            QMessageBox.information(
                self, "Not available",
                "Reloading at a different Load detail only applies to a raw .e57 scan - "
                "a .las result has no Load detail of its own (it was already decimated, "
                "once, whenever pcseg exported it).",
            )
            return

        current = self.viewer.node.decimation_step
        new_decimation, ok = QInputDialog.getInt(
            self, "Reload at Different Detail", "New Load detail (1 = full resolution):",
            current, 1, 32,
        )
        if not ok or new_decimation == current:
            return

        has_instances = any(layer.instances for layer in self.viewer.layer_panel.layers)
        if has_instances:
            proceed = QMessageBox.question(
                self, "Reload at Different Detail",
                "This reloads the scan at Load detail "
                f"{new_decimation} and tries to carry your current classifications over to "
                "the newly loaded points (a spatial transfer, not exact - points right at "
                "an instance's edge may not carry over, and an instance that ends up with "
                "none of its own points among the new ones will be dropped).\n\nContinue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if proceed != QMessageBox.StandardButton.Yes:
                return

        source_path = self.viewer.node.file_path
        view_decimation = self.viewer.point_item.view_decimation if self.viewer.point_item is not None else 1
        layers_meta, owner_map = self.viewer.snapshot_for_label_transfer()

        def _on_loaded(node) -> None:
            new_layers, stats = self.viewer.apply_label_transfer(layers_meta, owner_map)
            self.viewer.layer_panel.layers = new_layers
            self.viewer.layer_panel.selected_layer = new_layers[0] if new_layers else None
            self.viewer.layer_panel.refresh()
            msg = f"Reloaded at Load detail={new_decimation}."
            if stats["kept"] or stats["dropped"]:
                msg += f" Transferred {stats['kept']} instance(s)"
                if stats["dropped"]:
                    msg += f", dropped {stats['dropped']} (no matching points)"
                msg += "."
            self.status_label.setText(msg)

        self.open_file(source_path, decimation=new_decimation, view_decimation=view_decimation, on_loaded=_on_loaded)

    def export_at_full_resolution(self) -> None:
        """"Export at Full Resolution..." (Layers panel, see
        Viewer3DWidget.export_full_resolution_requested): a one-click
        version of reload_at_different_detail_dialog(new_decimation=1) that
        chains straight into export_segmentation() once the transfer lands -
        segmenting can stay fast at a coarse Load detail while what actually
        gets exported is always the full, undecimated point cloud. Reuses
        the exact same reload/transfer machinery (same reasoning: a
        different decimation samples an entirely different set of points, so
        the old indices need the same spatial/voxel transfer, not a reused
        index array) - just fixed at decimation=1 and with no interactive
        "which detail?" prompt, since there's only one answer here."""
        if self.viewer.node is None:
            QMessageBox.information(self, "Nothing loaded", "Load a scan first.")
            return
        if not self.viewer.layer_panel.layers:
            QMessageBox.information(
                self, "Nothing to export", "No layers to export yet - run a segmentation "
                "or pick some points first.",
            )
            return
        if Path(self.viewer.node.file_path).suffix.lower() != ".e57":
            QMessageBox.information(
                self, "Not available",
                "Export at Full Resolution only applies to a raw .e57 scan loaded at a "
                "reduced Load detail - a .las result has no coarser original to expand "
                "back to (it was already decimated once, whenever it was exported). Use "
                "Export Segmentation... directly instead.",
            )
            return

        current = self.viewer.node.decimation_step
        if current == 1:
            # Already full resolution - nothing to reload/transfer, just export as-is.
            self.viewer.export_segmentation()
            return

        proceed = QMessageBox.question(
            self, "Export at Full Resolution",
            f"This reloads the scan at full resolution (Load detail=1, currently "
            f"{current}) and transfers your current classifications onto the newly "
            "loaded points (a spatial transfer, not exact - points right at an "
            "instance's edge may not carry over, and an instance that ends up with none "
            "of its own points among the new ones will be dropped), then opens Export "
            "Segmentation on the result. This can take noticeably longer and use "
            "significantly more memory than the current view, on a large scan.\n\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if proceed != QMessageBox.StandardButton.Yes:
            return

        source_path = self.viewer.node.file_path
        view_decimation = self.viewer.point_item.view_decimation if self.viewer.point_item is not None else 1
        layers_meta, owner_map = self.viewer.snapshot_for_label_transfer()

        def _on_loaded(node) -> None:
            new_layers, stats = self.viewer.apply_label_transfer(layers_meta, owner_map)
            self.viewer.layer_panel.layers = new_layers
            self.viewer.layer_panel.selected_layer = new_layers[0] if new_layers else None
            self.viewer.layer_panel.refresh()
            msg = "Reloaded at full resolution."
            if stats["kept"] or stats["dropped"]:
                msg += f" Transferred {stats['kept']} instance(s)"
                if stats["dropped"]:
                    msg += f", dropped {stats['dropped']} (no matching points)"
                msg += "."
            self.status_label.setText(msg)
            self.viewer.export_segmentation()

        self.open_file(source_path, decimation=1, view_decimation=view_decimation, on_loaded=_on_loaded)
