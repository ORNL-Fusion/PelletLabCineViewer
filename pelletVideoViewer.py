# pelletVideoViewer.py
"""
This program reads cine files and can:
1. Apply various filters to the image(s)
2. Outline the pellet with a polygon to determine size
3. Find the pellet's speed
4. Output rectangles for further analysis


required packages:

pip install -U  napari[all] pycine numpy scipy PyQt5 qtpy magicgui

Optional (for a menu to apply various filters)
pip install -U scikit-image napari-skimage

To run:
python pelletVideoViewer.py

-or, to open a file directly-
python pelletVideoViewer.py "/Users/plh/research/csp/lab_videos/15708.cine"
"""

import sys
import numpy as np
import napari
from napari.layers import Shapes, Points
from napari.utils.notifications import show_info

from pycine.file import read_header
from pycine.raw import read_frames
from pathlib import Path
from qtpy.QtWidgets import (
    QWidget,
    QLineEdit,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QTextEdit,
    QFileDialog,
    QMessageBox,
    QTabWidget,
)
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QFont


# ═══════════════════════════════════════════════════════════════
# Cine Loader, Calibration Classes
# ═══════════════════════════════════════════════════════════════
class CineLoader:
    """
    Thin wrapper around pycine to load frames from a .cine file.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.frames: np.ndarray | None = None
        self.fps: float = 1.0
        self.frame_count: int = 0
        self.width: int = 0
        self.height: int = 0
        self.bpp: int = 8  # bits per pixel
        self._load()

    def _load(self):
        try:
            frame_generator, setup, bpp = read_frames(str(self.path))
            self.fps = float(setup.FrameRate)
            self.bpp = bpp
            self.setup = setup
            self.frames = np.array(list(frame_generator))
            self.header = read_header(str(self.path))

        except Exception as exc:
            raise RuntimeError(
                f"Could not load '{self.path}'.\n"
                "Make sure pycine is installed:  pip install pycine\n"
                f"Original error: {exc}"
            ) from exc


class Calibration:
    """Stores pixels-per-unit scale derived from a user-drawn calibration line."""

    UNIT_FACTORS = {
        "mm": 1e-3,
        "cm": 1e-2,
        "m": 1.0,
        "px": 1.0,
    }

    def __init__(self):
        self.pixels_per_meter: float = 1.0
        self.reference_pixels: float | None = None
        self.reference_real: float | None = None
        self.reference_unit: str = "mm"

    @property
    def is_set(self) -> bool:
        return self.pixels_per_meter is not None

    def set_from_line(self, pixel_length: float, real_length: float, unit: str):
        """pixel_length in px, real_length in 'unit' units."""
        meters = real_length * self.UNIT_FACTORS[unit]
        self.pixels_per_meter = pixel_length / meters
        self.reference_pixels = pixel_length
        self.reference_real = real_length
        self.reference_unit = unit

    def px_to_m(self, px: float) -> float:
        if not self.is_set:
            raise ValueError("Calibration not set.")
        return px / self.pixels_per_meter

    def px2_to_m2(self, px2: float) -> float:
        if not self.is_set:
            raise ValueError("Calibration not set.")
        return px2 / (self.pixels_per_meter**2)

    def format_length(self, meters: float, unit: str = "mm") -> str:
        val = meters / self.UNIT_FACTORS[unit]
        return f"{val:.4f} {unit}"

    def format_area(self, m2: float, unit: str = "mm") -> str:
        factor = self.UNIT_FACTORS[unit] ** 2
        val = m2 / factor
        return f"{val:.4f} {unit}²"


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════


def _poly_area_px(vertices: np.ndarray) -> float:
    """Shoelace formula for polygon area in pixel²."""
    n = len(vertices)
    if n < 3:
        return 0.0
    x, y = vertices[:, 0], vertices[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _line_length_px(pts: np.ndarray) -> float:
    """Total length of a polyline in pixels."""
    if pts.ndim == 1 or len(pts) < 2:
        return 0.0
    diffs = np.diff(pts, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


# ═══════════════════════════════════════════════════════════════
# Main Viewer Widget
# ═══════════════════════════════════════════════════════════════


class CineViewerWidget(QWidget):
    """
    Side-panel dock widget that lives inside napari and provides
    calibration + measurement tools.
    """

    SHAPE_LAYER = "Measurements"
    TRACK_LAYER = "Speed Markers"
    CALIB_LAYER = "Calibration Line"

    def __init__(self, viewer: napari.Viewer, loader: CineLoader | None = None):
        super().__init__()
        self.viewer = viewer
        self.loader = loader
        self.calib = Calibration()
        self._speed_markers: list[tuple[int, np.ndarray]] = []  # (frame_idx, centroid)
        self._playback_timer = QTimer(self)
        self._playback_timer.timeout.connect(self._next_frame)
        self._current_frame = 0

        self._build_ui()
        self._connect_viewer_signals()

    # ──────────────────────────────────────────────────────────────────
    # UI construction
    # ──────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)
        root.setContentsMargins(6, 6, 6, 6)

        # Title
        title = QLabel("🎞  Cine Viewer")
        title.setFont(QFont("Arial", 13, QFont.Bold))  # type: ignore
        title.setAlignment(Qt.AlignCenter)  # type: ignore
        root.addWidget(title)

        # Open button
        open_btn = QPushButton("📂  Open .cine File")
        open_btn.setFixedHeight(32)
        open_btn.clicked.connect(self._open_file)
        root.addWidget(open_btn)

        # File info
        self._file_label = QLabel("No file loaded")
        self._file_label.setWordWrap(True)
        self._file_label.setStyleSheet("color: #888; font-size: 10px;")
        root.addWidget(self._file_label)

        # Playback controls
        root.addWidget(self._build_playback())

        # Tabs for tools
        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.North)  # type: ignore
        tabs.addTab(self._build_calibration_tab(), "📏 Calibrate")
        tabs.addTab(self._build_measure_tab(), "📐 Measure")
        tabs.addTab(self._build_speed_tab(), "⚡️ Speed")
        tabs.addTab(self._build_polygon_tab(), "🔸 Polygons")
        tabs.addTab(self._build_results_tab(), "📋 Results")
        root.addWidget(tabs)

    # ── Playback ──────────────────────────────────────────────────────

    def _build_playback(self) -> QGroupBox:
        grp = QGroupBox("Playback")
        lay = QVBoxLayout(grp)

        row1 = QHBoxLayout()
        self._prev_btn = QPushButton("⏮")
        self._prev_btn.setFixedWidth(36)
        self._prev_btn.clicked.connect(self._prev_frame)

        self._play_btn = QPushButton("▶")
        self._play_btn.setFixedWidth(36)
        self._play_btn.setCheckable(True)
        self._play_btn.toggled.connect(self._toggle_play)

        self._next_btn = QPushButton("⏭")
        self._next_btn.setFixedWidth(36)
        self._next_btn.clicked.connect(self._next_frame)

        self._frame_label = QLabel("Frame: 0 / 0")
        self._frame_label.setAlignment(Qt.AlignCenter)  # type: ignore

        row1.addWidget(self._prev_btn)
        row1.addWidget(self._play_btn)
        row1.addWidget(self._next_btn)
        row1.addWidget(self._frame_label, 1)
        lay.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Speed:"))
        self._fps_spin = QDoubleSpinBox()
        self._fps_spin.setRange(0.5, 1000.0)
        self._fps_spin.setValue(4.0)
        self._fps_spin.setSuffix(" fps")
        self._fps_spin.setDecimals(1)
        self._fps_spin.valueChanged.connect(self._update_timer)
        row2.addWidget(self._fps_spin)

        self._fps_info = QLabel("(native: —)")
        self._fps_info.setStyleSheet("color: #888; font-size: 10px;")
        row2.addWidget(self._fps_info)
        lay.addLayout(row2)

        return grp

    # ── Calibration tab ───────────────────────────────────────────────

    def _build_calibration_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(8)

        info = QLabel(
            "1. Click <b>Add Calibration Line</b><br>"
            "2. Draw a line of known length in the viewer<br>"
            "3. Enter the real-world length below<br>"
            "4. Click <b>Set Calibration</b>"
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #aaa; font-size: 11px;")
        lay.addWidget(info)

        self._add_calib_btn = QPushButton("+ Add Calibration Line")
        self._add_calib_btn.clicked.connect(self._add_calibration_layer)
        lay.addWidget(self._add_calib_btn)

        row = QHBoxLayout()
        row.addWidget(QLabel("Real length:"))
        self._calib_len = QDoubleSpinBox()
        self._calib_len.setRange(0.001, 1e9)
        self._calib_len.setValue(1.0)
        self._calib_len.setDecimals(4)
        row.addWidget(self._calib_len)

        self._calib_unit = QComboBox()
        self._calib_unit.addItems(["mm", "cm", "m"])
        row.addWidget(self._calib_unit)
        lay.addLayout(row)

        self._set_calib_btn = QPushButton("✔  Set Calibration")
        self._set_calib_btn.clicked.connect(self._set_calibration)
        lay.addWidget(self._set_calib_btn)

        self._calib_status = QLabel("Status: not calibrated")
        self._calib_status.setWordWrap(True)
        self._calib_status.setStyleSheet("color: #e8a000; font-weight: bold;")
        lay.addWidget(self._calib_status)

        lay.addStretch()
        return w

    # ── Measure tab ───────────────────────────────────────────────────

    def _build_measure_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(8)

        info = QLabel(
            "Draw shapes in the <b>Measurements</b> layer, then click a button to measure."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #aaa; font-size: 11px;")
        lay.addWidget(info)

        self._add_shapes_btn = QPushButton("+ Add Measurement Layer")
        self._add_shapes_btn.clicked.connect(self._add_shapes_layer)
        lay.addWidget(self._add_shapes_btn)

        grp_len = QGroupBox("Line / Polyline Length")
        g1 = QVBoxLayout(grp_len)
        r = QHBoxLayout()
        r.addWidget(QLabel("Output unit:"))
        self._len_unit = QComboBox()
        self._len_unit.addItems(["mm", "cm", "m"])
        r.addWidget(self._len_unit)
        g1.addLayout(r)
        m_line_btn = QPushButton("Measure Selected Line(s)")
        m_line_btn.clicked.connect(self._measure_lines)
        g1.addWidget(m_line_btn)
        lay.addWidget(grp_len)

        grp_area = QGroupBox("Polygon Area")
        g2 = QVBoxLayout(grp_area)
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Output unit:"))
        self._area_unit = QComboBox()
        self._area_unit.addItems(["mm", "cm", "m"])
        r2.addWidget(self._area_unit)
        g2.addLayout(r2)
        m_area_btn = QPushButton("Measure Selected Polygon(s)")
        m_area_btn.clicked.connect(self._measure_areas)
        g2.addWidget(m_area_btn)
        lay.addWidget(grp_area)

        lay.addStretch()
        return w

    # ── Speed tab ─────────────────────────────────────────────────────

    def _build_speed_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(8)

        info = QLabel(
            "Mark the same object in <b>two different frames</b>:<br>"
            "1. Navigate to the first frame<br>"
            "2. Click <b>Mark Position A</b> then click the object in the viewer<br>"
            "3. Navigate to the second frame<br>"
            "4. Click <b>Mark Position B</b> then click the object<br>"
            "5. Click <b>Calculate Speed</b>"
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #aaa; font-size: 11px;")
        lay.addWidget(info)

        self._mark_a_btn = QPushButton("🔵  Mark Position A  (frame A)")
        self._mark_a_btn.setCheckable(True)
        self._mark_a_btn.toggled.connect(self._toggle_mark_a)
        lay.addWidget(self._mark_a_btn)

        self._mark_b_btn = QPushButton("🔴  Mark Position B  (frame B)")
        self._mark_b_btn.setCheckable(True)
        self._mark_b_btn.toggled.connect(self._toggle_mark_b)
        lay.addWidget(self._mark_b_btn)

        self._speed_a_lbl = QLabel("A: not set")
        self._speed_b_lbl = QLabel("B: not set")
        lay.addWidget(self._speed_a_lbl)
        lay.addWidget(self._speed_b_lbl)

        r = QHBoxLayout()
        r.addWidget(QLabel("Speed unit:"))
        self._speed_out_unit = QComboBox()
        self._speed_out_unit.addItems(["m/s", "cm/s", "mm/s", "km/h", "mph"])
        r.addWidget(self._speed_out_unit)
        lay.addLayout(r)

        calc_btn = QPushButton("⚡️ Calculate Speed")
        calc_btn.clicked.connect(self._calculate_speed)
        lay.addWidget(calc_btn)

        clear_btn = QPushButton("🗑️ Clear Markers")
        clear_btn.clicked.connect(self._clear_speed_markers)
        lay.addWidget(clear_btn)

        lay.addStretch()
        return w

    # ── Polygons tab ─────────────────────────────────────────────

    def _build_polygon_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(8)

        info = QLabel(
            "Create complex polygons around pellets, print out the verticies for further "
            "analysis with other programs"
        )
        info.setStyleSheet("color: #aaa; font-size: 11px;")
        lay.addWidget(info)

        row = QHBoxLayout()
        self._poly_name = QLineEdit()
        self._poly_name.setPlaceholderText("Layer name, e.g. CSP Core")
        row.addWidget(self._poly_name)
        btn_add = QPushButton("Add Layer")
        btn_add.clicked.connect(self._add_polygon_layer)
        row.addWidget(btn_add)
        lay.addLayout(row)

        self._poly_log = QTextEdit()
        self._poly_log.setReadOnly(True)
        self._poly_log.setPlaceholderText("Output verticies are listed here:")
        lay.addWidget(self._poly_log)

        row = QHBoxLayout()
        print_btn = QPushButton("🖨️ Print Verticies")
        print_btn.clicked.connect(self._print_verticies)
        copy_btn = QPushButton("📋 Copy All")
        copy_btn.clicked.connect(self._copy_polygons)
        clear_btn = QPushButton("🗑 Clear")
        clear_btn.clicked.connect(self._poly_log.clear)
        row.addWidget(print_btn)
        row.addWidget(copy_btn)
        row.addWidget(clear_btn)
        lay.addLayout(row)

        return w

    # ── Polygon logic ────────────────────────────────────────────

    def _add_polygon_layer(self):
        name = self._poly_name.text().strip() or f"ROI {len(self.viewer.layers)}"
        layer = self.viewer.add_shapes(
            name=name,
            edge_color="#4488ffff",
            edge_width=2,
            face_color="transparent",
        )
        self.viewer.layers.selection.active = layer
        layer.mode = "add_polygon"
        self._poly_name.clear()

    def _print_verticies(self):

        for layer in self.viewer.layers:
            # Check if the layer is a Shapes layer

            if isinstance(layer, Shapes):
                print(f"--- Layer: {layer.name} ---")

                # Zip data with shape types to find polygons
                for shape_data, shape_type in zip(layer.data, layer.shape_type):
                    print(shape_type)
                    if shape_type in ("polygon", "rectangle", "ellipse", "path"):
                        # shape_data is already a numpy array of vertices
                        msg = f"{layer.name} = np.{repr(shape_data)}"
                        print(msg)
                        self._poly_log.append(msg)

    def _copy_polygons(self):
        from qtpy.QtWidgets import QApplication

        QApplication.clipboard().setText(self._poly_log.toPlainText())  # type: ignore
        show_info("Results copied to clipboard.")

    # ── Results tab ───────────────────────────────────────────────────

    def _build_results_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        self._results = QTextEdit()
        self._results.setReadOnly(True)
        self._results.setPlaceholderText("Measurement results will appear here…")
        lay.addWidget(self._results)

        row = QHBoxLayout()
        copy_btn = QPushButton("📋 Copy All")
        copy_btn.clicked.connect(self._copy_results)
        clear_btn = QPushButton("🗑 Clear")
        clear_btn.clicked.connect(self._results.clear)
        row.addWidget(copy_btn)
        row.addWidget(clear_btn)
        lay.addLayout(row)

        return w

    # ──────────────────────────────────────────────────────────────────
    # File loading
    # ──────────────────────────────────────────────────────────────────

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open .cine file", "", "Cine files (*.cine);;All files (*)"
        )
        if not path:
            return
        self._load_cine(path)

    def _load_cine(self, path: str):
        try:
            loader = CineLoader(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))
            return

        self.loader = loader
        self._current_frame = 0

        # Remove old image layers
        for lyr in list(self.viewer.layers):
            if (
                hasattr(lyr, "data")
                and isinstance(lyr.data, np.ndarray)
                and lyr.data.ndim in (3, 4)
            ):
                try:
                    self.viewer.layers.remove(lyr)
                except Exception:
                    pass

        self.viewer.add_image(
            loader.frames,
            name=Path(path).name,
            colormap="gray",
        )
        self.viewer.dims.current_step = (0,) + self.viewer.dims.current_step[1:]

        # Update UI
        self._file_label.setText(
            f"{Path(path).name}\n"
            f"{loader.frame_count} frames · {loader.width}✖️{loader.height} · {loader.fps:.1f} fps"
        )
        self._fps_info.setText(f"(native: {loader.fps:.1f})")
        self._fps_spin.setValue(min(loader.fps, 4.0))
        self._update_frame_label()
        self._log(f"Opened: {path}")
        self._log(
            f"  {loader.frame_count} frames, {loader.fps} fps, {loader.width}✖️{loader.height} px"
        )

    # ──────────────────────────────────────────────────────────────────
    # Playback
    # ──────────────────────────────────────────────────────────────────

    def _toggle_play(self, checked: bool):
        if checked:
            self._play_btn.setText("⏸")
            self._update_timer()
        else:
            self._play_btn.setText("▶")
            self._playback_timer.stop()

    def _update_timer(self):
        if self._play_btn.isChecked():
            interval = int(1000 / max(self._fps_spin.value(), 0.5))
            self._playback_timer.start(interval)

    def _next_frame(self):
        if self.loader is None:
            return
        step = list(self.viewer.dims.current_step)
        nf = self.loader.frame_count
        step[0] = (step[0] + 1) % nf
        self.viewer.dims.current_step = tuple(step)
        self._current_frame = step[0]
        self._update_frame_label()

    def _prev_frame(self):
        if self.loader is None:
            return
        step = list(self.viewer.dims.current_step)
        nf = self.loader.frame_count
        step[0] = (step[0] - 1) % nf
        self.viewer.dims.current_step = tuple(step)
        self._current_frame = step[0]
        self._update_frame_label()

    def _update_frame_label(self):
        if self.loader:
            n = self.loader.frame_count
            f = self.viewer.dims.current_step[0]
            self._frame_label.setText(f"Frame: {f} / {n-1}")
        else:
            self._frame_label.setText("Frame: 0 / 0")

    def _connect_viewer_signals(self):
        self.viewer.dims.events.current_step.connect(
            lambda e: self._update_frame_label()
        )

    # ──────────────────────────────────────────────────────────────────
    # Calibration
    # ──────────────────────────────────────────────────────────────────

    def _add_calibration_layer(self):
        existing = [l for l in self.viewer.layers if l.name == self.CALIB_LAYER]
        if existing:
            self.viewer.layers.selection.active = existing[0]
            show_info("Calibration layer already exists — select it and draw a line.")
            return
        layer = self.viewer.add_shapes(
            name=self.CALIB_LAYER,
            shape_type="line",
            edge_color="yellow",
            edge_width=2,
            face_color="transparent",
        )
        layer.mode = "add_line"
        show_info("Draw a single line of known length, then click 'Set Calibration'.")

    def _set_calibration(self):
        layers = [l for l in self.viewer.layers if l.name == self.CALIB_LAYER]
        if not layers:
            QMessageBox.warning(self, "No layer", "Add a calibration line layer first.")
            return
        layer = layers[0]
        if len(layer.data) == 0:
            QMessageBox.warning(
                self, "No line", "Draw a line in the Calibration Layer first."
            )
            return

        # Take the last drawn shape
        pts = layer.data[-1]  # shape (N,2) or (N,D)
        # use only the last two spatial dims
        pts_2d = pts[:, -2:]  # (row, col) = (y, x)
        px_len = _line_length_px(pts_2d)
        real_len = self._calib_len.value()
        unit = self._calib_unit.currentText()

        self.calib.set_from_line(px_len, real_len, unit)

        self._calib_status.setText(
            f"✔ Calibrated: {px_len:.2f} px = {real_len} {unit}\n"
            f"   Scale: {self.calib.pixels_per_meter:.2f} px/m"
        )
        self._calib_status.setStyleSheet("color: #00cc66; font-weight: bold;")
        self._log(
            f"Calibration set: {px_len:.2f} px → {real_len} {unit} "
            f"({self.calib.pixels_per_meter:.4f} px/m)"
        )

    # ──────────────────────────────────────────────────────────────────
    # Measurement – lines
    # ──────────────────────────────────────────────────────────────────

    def _add_shapes_layer(self):
        existing = [l for l in self.viewer.layers if l.name == self.SHAPE_LAYER]
        if existing:
            self.viewer.layers.selection.active = existing[0]
            show_info("Measurement layer already active.")
            return
        self.viewer.add_shapes(
            name=self.SHAPE_LAYER,
            edge_color="lawngreen",
            edge_width=2,
            face_color="transparent",
        )
        show_info("Draw lines, polygons, or rectangles in the Measurements layer.")

    def _get_shapes_layer(self) -> Shapes | None:
        layers = [l for l in self.viewer.layers if l.name == self.SHAPE_LAYER]
        if not layers:
            QMessageBox.warning(
                self, "No layer", "Add a Measurement Layer first and draw some shapes."
            )
            return None
        return layers[0]  # type: ignore

    def _require_calibration(self) -> bool:
        if not self.calib.is_set:
            QMessageBox.warning(
                self, "Not calibrated", "Please calibrate first (Calibrate tab)."
            )
            return False
        return True

    def _measure_lines(self):
        if not self._require_calibration():
            return
        layer = self._get_shapes_layer()
        if layer is None:
            return

        selected = list(layer.selected_data)
        if not selected:
            selected = list(range(len(layer.data)))

        unit = self._len_unit.currentText()
        measured = 0
        for idx in selected:
            stype = layer.shape_type[idx]
            pts = layer.data[idx][:, -2:]
            if stype in ("line", "path"):
                px_len = _line_length_px(pts)
                real_m = self.calib.px_to_m(px_len)
                result = self.calib.format_length(real_m, unit)
                self._log(f"Shape {idx} ({stype}): {px_len:.2f} px → {result}")
                measured += 1

        if measured == 0:
            self._log("No lines/paths found in selection.")
        show_info(f"Measured {measured} line(s) — see Results tab.")

    def _measure_areas(self):
        if not self._require_calibration():
            return
        layer = self._get_shapes_layer()
        if layer is None:
            return

        selected = list(layer.selected_data)
        if not selected:
            selected = list(range(len(layer.data)))

        unit = self._area_unit.currentText()
        measured = 0
        for idx in selected:
            shape_data = layer.data[idx]
            stype = layer.shape_type[idx]
            pts = layer.data[idx][:, -2:]
            real_height, real_width, angle_deg = 0, 0, None
            height, width = 0, 0

            if stype in ("polygon", "rectangle", "ellipse"):
                if stype == "ellipse":
                    # napari stores ellipse as 4 corner-like control points
                    # approximate as bounding-box ellipse area
                    height = np.linalg.norm(pts[0] - pts[2]) / 2
                    width = np.linalg.norm(pts[1] - pts[3]) / 2
                    real_height = self.calib.px_to_m(float(height))
                    real_width = self.calib.px_to_m(float(width))
                    px_area = np.pi * height * width
                else:
                    px_area = _poly_area_px(pts)
                if stype == "rectangle":
                    edge_vector = shape_data[1] - shape_data[0]
                    angle_rad = np.arctan2(edge_vector[1], edge_vector[0])
                    angle_deg = np.rad2deg(angle_rad)

                    height = np.linalg.norm(pts[1] - pts[2])
                    width = np.linalg.norm(pts[0] - pts[1])

                    real_height = self.calib.px_to_m(float(height))
                    real_width = self.calib.px_to_m(float(width))
                real_m2 = self.calib.px2_to_m2(float(px_area))
                result = self.calib.format_area(real_m2, unit)

                msg = (
                    f"── Polygon Measurement ────────────────\n"
                    f"  Shape {idx}:         {stype}\n"
                    f"  Area:            {px_area:.2f} px² \n"
                    f"  Area:            {result} \n"
                    f"  Rotation Angle:  {angle_deg:.2f} degrees \n"
                    f"  height:          {self.calib.format_length(real_height, 'px')}\n"
                    f"  height:          {self.calib.format_length(real_height, unit)}\n"
                    f"  width:           {self.calib.format_length(real_width, 'px')}\n"
                    f"  width:           {self.calib.format_length(real_width, unit)}\n"
                    "────────────────────────────────────────"
                )

                self._log(msg)
                measured += 1

        if measured == 0:
            self._log("No polygons/rectangles/ellipses found in selection.")
        show_info(f"Measured {measured} polygon(s) — see Results tab.")

    # ──────────────────────────────────────────────────────────────────
    # Speed measurement
    # ──────────────────────────────────────────────────────────────────

    _speed_point_a: tuple[int, np.ndarray] | None = None  # (frame, [y, x])
    _speed_point_b: tuple[int, np.ndarray] | None = None

    def _get_or_add_track_layer(self) -> Points:
        layers = [l for l in self.viewer.layers if l.name == self.TRACK_LAYER]
        if layers:
            return layers[0]  # type: ignore
        return self.viewer.add_points(
            name=self.TRACK_LAYER,
            size=12,
            face_color="red",
            border_color="white",
            border_width=0.1,
        )

    def _toggle_mark_a(self, checked: bool):
        if not checked:
            return
        self._mark_b_btn.setChecked(False)
        layer = self._get_or_add_track_layer()
        self.viewer.layers.selection.active = layer
        layer.mode = "add"
        layer.events.data.connect(self._on_track_data_changed)
        self._pending_mark = "A"
        show_info("Click on the object to mark Position A.")

    def _toggle_mark_b(self, checked: bool):
        if not checked:
            return
        self._mark_a_btn.setChecked(False)
        layer = self._get_or_add_track_layer()
        self.viewer.layers.selection.active = layer
        layer.mode = "add"
        layer.events.data.connect(self._on_track_data_changed)
        self._pending_mark = "B"
        show_info("Click on the object to mark Position B.")

    def _on_track_data_changed(self, event):
        layer = self._get_or_add_track_layer()
        if len(layer.data) == 0:
            return
        # Grab the most recently added point
        pt = layer.data[-1]  # could be (2,) or (3,) depending on dims
        frame = int(self.viewer.dims.current_step[0])

        if getattr(self, "_pending_mark", None) == "A":
            self._speed_point_a = (frame, pt[-2:].copy())  # [y, x]
            self._speed_a_lbl.setText(
                f"A: frame {frame}, pos ({pt[-1]:.1f}, {pt[-2]:.1f})"
            )
            self._mark_a_btn.setChecked(False)
            layer.face_color = [[0.2, 0.4, 1.0, 1.0]] * len(layer.data)
            layer.events.data.disconnect(self._on_track_data_changed)
            self._pending_mark = None

        elif getattr(self, "_pending_mark", None) == "B":
            self._speed_point_b = (frame, pt[-2:].copy())
            self._speed_b_lbl.setText(
                f"B: frame {frame}, pos ({pt[-1]:.1f}, {pt[-2]:.1f})"
            )
            layer.face_color = [["red"]] * len(layer.data)
            self._mark_b_btn.setChecked(False)
            layer.events.data.disconnect(self._on_track_data_changed)
            self._pending_mark = None

    def _calculate_speed(self):
        if not self._require_calibration():
            return
        if self._speed_point_a is None or self._speed_point_b is None:
            QMessageBox.warning(
                self, "Missing markers", "Set both Position A and Position B first."
            )
            return
        if self.loader is None:
            QMessageBox.warning(self, "No video", "Load a .cine file first.")
            return

        frame_a, pos_a = self._speed_point_a
        frame_b, pos_b = self._speed_point_b

        if frame_a == frame_b:
            QMessageBox.warning(
                self, "Same frame", "Positions A and B must be in different frames."
            )
            return

        # Pixel distance
        dpy = float(pos_b[0] - pos_a[0])
        dpx = float(pos_b[1] - pos_a[1])
        px_dist = np.sqrt(dpy**2 + dpx**2)

        # Real distance in metres
        real_m = self.calib.px_to_m(px_dist)

        # Time between frames
        dt = abs(frame_b - frame_a) / self.loader.fps  # seconds

        speed_ms = real_m / dt  # m/s

        # Convert to chosen output unit
        unit = self._speed_out_unit.currentText()
        CONV = {
            "m/s": (speed_ms, "m/s"),
            "cm/s": (speed_ms * 100, "cm/s"),
            "mm/s": (speed_ms * 1000, "mm/s"),
            "km/h": (speed_ms * 3.6, "km/h"),
            "mph": (speed_ms * 2.23694, "mph"),
        }
        val, lbl = CONV[unit]

        msg = (
            f"── Speed Measurement ──────────────────\n"
            f"  Frame A:    {frame_a}\n"
            f"  Frame B:    {frame_b}\n"
            f"  Δt:         {dt*1000:.3f} ms  ({dt:.6f} s)\n"
            f"  Pixel dist: {px_dist:.2f} px\n"
            f"  Real dist:  {real_m*1000:.4f} mm  ({real_m:.6f} m)\n"
            f"  Speed:      {val:.4f} {lbl}\n"
            f"  Also:       {speed_ms*2.23694:.4f} mph, {speed_ms*3.6:.4f} km/h\n"
            "────────────────────────────────────────"
        )
        self._log(msg)
        show_info(f"Speed: {val:.4f} {lbl}")

    def _clear_speed_markers(self):
        self._speed_point_a = None
        self._speed_point_b = None
        self._speed_a_lbl.setText("A: not set")
        self._speed_b_lbl.setText("B: not set")
        layers = [l for l in self.viewer.layers if l.name == self.TRACK_LAYER]
        for l in layers:
            l.data = np.empty((0, 2))

    # ──────────────────────────────────────────────────────────────────
    # Results log
    # ──────────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        self._results.append(msg)

    def _copy_results(self):
        from qtpy.QtWidgets import QApplication

        QApplication.clipboard().setText(self._results.toPlainText())  # type: ignore
        show_info("Results copied to clipboard.")


def launch(cine_path: str | None = None):
    """Launch the viewer.  Pass a .cine path or leave None for demo mode."""
    viewer = napari.Viewer(title="Pellet Lab Cine Viewer — Measurement Tools")

    widget = CineViewerWidget(viewer)

    if cine_path:
        widget._load_cine(cine_path)

    viewer.window.add_dock_widget(
        widget,
        name="Cine Viewer Tools",
        area="right",
    )
    napari.run()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    launch(path)
