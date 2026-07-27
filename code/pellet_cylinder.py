"""
pellet_cylinder.py
==================

A parametric *cylinder* overlay for PelletLabCineViewer (napari).

napari has no built-in "cylinder object with a properties panel", so this module
builds one: a Shapes layer holding a wireframe cylinder — two elliptical end
caps joined by two side lines — regenerated from six numbers

    center (row, col), diameter, length, in-plane angle, tilt

Under orthographic projection a tilted cylinder's silhouette is a rectangle
capped by two half-ellipses, so the tilt parameter lets you recover the *true*
length even when the pellet is tipped toward or away from the camera:

    width across the axis = D                      (independent of tilt)
    projected length      = L*cos(t) + D*|sin(t)|
    end caps              = ellipses with semi-axes (D/2)*sin(t) along the axis
                                                    (D/2)        across it

At t = 0 the caps collapse to straight lines and the wireframe becomes a plain
rectangle, which is the correct edge-on view of a flat-ended pellet.  Volume is
pi*(D/2)^2*L in calibrated units, independent of tilt.

Usage inside pelletVideoViewer.py
---------------------------------
    from pellet_cylinder import CylinderTab
    ...
    tabs.addTab(CylinderTab(self.viewer, self.calib), "\U0001f9ea Cylinder")

(`self.calib` is the existing `Calibration` instance; pass None to work in px.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# ═══════════════════════════════════════════════════════════════════════
# Geometry (pure numpy — no napari/Qt needed, easy to unit-test)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class CylinderParams:
    """A right circular cylinder as seen by a camera (orthographic)."""

    cy: float = 100.0          # center, row (px)
    cx: float = 100.0          # center, column (px)
    diameter: float = 40.0     # px
    length: float = 80.0       # TRUE length along the axis (px)
    angle_deg: float = 0.0     # in-plane rotation of the axis
    tilt_deg: float = 0.0      # tilt out of the image plane; 0 = axis in plane,
    #                            positive/negative = far end away/toward camera

    # ── derived quantities ────────────────────────────────────────────
    @property
    def radius(self) -> float:
        return self.diameter / 2.0

    @property
    def projected_length(self) -> float:
        """Tip-to-tip extent of the silhouette along the axis (px)."""
        t = math.radians(self.tilt_deg)
        return self.length * math.cos(t) + self.diameter * abs(math.sin(t))

    def volume_px3(self) -> float:
        return math.pi * self.radius**2 * self.length

    # ── drawing ───────────────────────────────────────────────────────
    def _to_image(self, uv: np.ndarray) -> np.ndarray:
        """Local (u along axis, v across) -> image (row, col)."""
        phi = math.radians(self.angle_deg)
        c, s = math.cos(phi), math.sin(phi)
        col = uv[:, 0] * c - uv[:, 1] * s + self.cx
        row = uv[:, 0] * s + uv[:, 1] * c + self.cy
        return np.column_stack((row, col))

    def caps(self, n: int = 64) -> list[np.ndarray]:
        """The two end-cap ellipses, as closed vertex loops (row, col)."""
        t = math.radians(self.tilt_deg)
        a = 0.5 * self.length * math.cos(t)   # half projected axis
        b = self.radius * math.sin(t)         # cap semi-axis along the axis
        r = self.radius

        psi = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
        u = b * np.cos(psi)
        v = r * np.sin(psi)
        return [
            self._to_image(np.column_stack((a + u, v))),
            self._to_image(np.column_stack((-a + u, v))),
        ]

    def sides(self) -> list[np.ndarray]:
        """The two side lines tangent to both caps, as 2-point paths."""
        t = math.radians(self.tilt_deg)
        a = 0.5 * self.length * math.cos(t)
        r = self.radius
        return [
            self._to_image(np.array([[a, r], [-a, r]], float)),
            self._to_image(np.array([[a, -r], [-a, -r]], float)),
        ]

    def wireframe(self, n: int = 64) -> tuple[list[np.ndarray], list[str]]:
        """All wireframe pieces plus the matching napari shape types."""
        caps = self.caps(n)
        sides = self.sides()
        return caps + sides, ["polygon", "polygon", "path", "path"]

    def silhouette(self, n_cap: int = 48) -> np.ndarray:
        """Filled outline (rectangle + half-ellipse caps) as one loop."""
        t = math.radians(self.tilt_deg)
        a = 0.5 * self.length * math.cos(t)
        b = self.radius * abs(math.sin(t))
        r = self.radius

        psi = np.linspace(-math.pi / 2, math.pi / 2, n_cap)
        right = np.column_stack((a + b * np.cos(psi), r * np.sin(psi)))
        left = np.column_stack((-a - b * np.cos(psi), -r * np.sin(psi)))
        return self._to_image(np.vstack((right, left)))


def min_area_rect(points: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Rotating-calipers minimum-area rectangle.

    Returns (center_rowcol, long_side, short_side, angle_deg) where the angle is
    that of the long side, measured the same way as CylinderParams.angle_deg.
    """
    from scipy.spatial import ConvexHull

    pts = np.asarray(points, float)[:, ::-1]  # -> (x=col, y=row)
    hull = pts[ConvexHull(pts).vertices]
    best = None
    for i in range(len(hull)):
        edge = hull[(i + 1) % len(hull)] - hull[i]
        theta = math.atan2(edge[1], edge[0])
        c, s = math.cos(-theta), math.sin(-theta)
        rot = hull @ np.array([[c, -s], [s, c]]).T
        w = float(np.ptp(rot[:, 0]))
        h = float(np.ptp(rot[:, 1]))
        if best is None or w * h < best[0]:
            ctr_rot = np.array(
                [
                    (rot[:, 0].min() + rot[:, 0].max()) / 2,
                    (rot[:, 1].min() + rot[:, 1].max()) / 2,
                ]
            )
            back = np.array([[math.cos(theta), -math.sin(theta)],
                             [math.sin(theta), math.cos(theta)]])
            best = (w * h, back @ ctr_rot, w, h, theta)

    _, ctr, w, h, theta = best
    if w >= h:
        long_side, short_side, ang = w, h, theta
    else:
        long_side, short_side, ang = h, w, theta + math.pi / 2
    return np.array([ctr[1], ctr[0]]), long_side, short_side, math.degrees(ang)


# ═══════════════════════════════════════════════════════════════════════
# Qt tab
# ═══════════════════════════════════════════════════════════════════════


class CylinderTab(QWidget):
    """Side-panel tab that drives a live wireframe cylinder in napari."""

    def __init__(self, viewer, calib=None, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.calib = calib
        self.params = CylinderParams()
        self._layer_name: str | None = None
        self._updating = False
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────

    def _spin(self, val, lo, hi, step, suffix=""):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setDecimals(2)
        s.setValue(val)
        if suffix:
            s.setSuffix(suffix)
        s.valueChanged.connect(self._on_param_changed)
        return s

    def _build_ui(self):
        root = QVBoxLayout(self)

        info = QLabel(
            "Overlay a wireframe cylinder and match it to the pellet. "
            "Tilt corrects for foreshortening, so length and volume stay true "
            "even when the pellet leans toward or away from the camera."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #aaa; font-size: 11px;")
        root.addWidget(info)

        # ── layer row ────────────────────────────────────────────────
        row = QHBoxLayout()
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Layer name, e.g. Pellet A")
        row.addWidget(self._name_edit)
        btn_add = QPushButton("+ Add Cylinder Layer")
        btn_add.clicked.connect(self._add_cylinder_layer)
        row.addWidget(btn_add)
        root.addLayout(row)

        self._layer_label = QLabel("Active layer: none")
        self._layer_label.setStyleSheet("color: #888; font-size: 10px;")
        root.addWidget(self._layer_label)

        # ── parameters ───────────────────────────────────────────────
        grp = QGroupBox("Cylinder parameters (px / deg)")
        form = QFormLayout(grp)
        self.s_cy = self._spin(self.params.cy, -1e5, 1e5, 1)
        self.s_cx = self._spin(self.params.cx, -1e5, 1e5, 1)
        self.s_dia = self._spin(self.params.diameter, 0.1, 1e5, 1)
        self.s_len = self._spin(self.params.length, 0.1, 1e5, 1)
        self.s_ang = self._spin(self.params.angle_deg, -180, 180, 0.5, "\u00b0")
        self.s_tilt = self._spin(self.params.tilt_deg, -89, 89, 0.5, "\u00b0")
        form.addRow("Center row (y)", self.s_cy)
        form.addRow("Center col (x)", self.s_cx)
        form.addRow("Diameter (width)", self.s_dia)
        form.addRow("Length (true)", self.s_len)
        form.addRow("In-plane rotation", self.s_ang)
        form.addRow("Tilt", self.s_tilt)
        root.addWidget(grp)

        row = QHBoxLayout()
        b_show = QPushButton("Update overlay")
        b_show.clicked.connect(self._redraw)
        b_from = QPushButton("Fit from selected shape")
        b_from.clicked.connect(self._fit_from_shape)
        b_copy = QPushButton("\U0001f4cb Copy")
        b_copy.clicked.connect(self._copy)
        for b in (b_show, b_from, b_copy):
            row.addWidget(b)
        root.addLayout(row)

        self.readout = QTextEdit()
        self.readout.setReadOnly(True)
        self.readout.setPlaceholderText(
            "Add a cylinder layer to start; measurements appear here."
        )
        root.addWidget(self.readout)

        self._report()

    # ── state <-> widgets ─────────────────────────────────────────────

    def _read_widgets(self) -> CylinderParams:
        return CylinderParams(
            cy=self.s_cy.value(),
            cx=self.s_cx.value(),
            diameter=self.s_dia.value(),
            length=self.s_len.value(),
            angle_deg=self.s_ang.value(),
            tilt_deg=self.s_tilt.value(),
        )

    def _write_widgets(self, p: CylinderParams):
        self._updating = True
        try:
            self.s_cy.setValue(p.cy)
            self.s_cx.setValue(p.cx)
            self.s_dia.setValue(p.diameter)
            self.s_len.setValue(p.length)
            self.s_ang.setValue(p.angle_deg)
            self.s_tilt.setValue(p.tilt_deg)
        finally:
            self._updating = False
        self.params = p

    def _on_param_changed(self, *_):
        if self._updating:
            return
        self.params = self._read_widgets()
        self._redraw()

    # ── napari layer ──────────────────────────────────────────────────

    def _next_name(self) -> str:
        existing = {layer.name for layer in self.viewer.layers}
        i = 1
        while f"Cylinder {i}" in existing:
            i += 1
        return f"Cylinder {i}"

    def _add_cylinder_layer(self):
        """Create a fresh Shapes layer and make it the active cylinder."""
        name = self._name_edit.text().strip() or self._next_name()
        layer = self.viewer.add_shapes(
            name=name,
            edge_color="#00ff88",
            edge_width=2,
            face_color="transparent",
            opacity=0.9,
        )
        self._layer_name = layer.name  # napari may de-duplicate the name
        self._name_edit.clear()
        self.viewer.layers.selection.active = layer

        # start centred on the current field of view so it is easy to find
        try:
            cy, cx = (float(v) for v in self.viewer.camera.center[-2:])
            self._write_widgets(replace(self.params, cy=cy, cx=cx))
        except Exception:
            pass

        self._redraw()

    def _active_layer(self):
        if self._layer_name and self._layer_name in self.viewer.layers:
            return self.viewer.layers[self._layer_name]
        return None

    def _redraw(self):
        layer = self._active_layer()
        if layer is None:
            self._layer_label.setText("Active layer: none")
            self._report(note="No cylinder layer — click '+ Add Cylinder Layer'.")
            return

        self._layer_label.setText(f"Active layer: {layer.name}")
        shapes, kinds = self.params.wireframe()
        layer.data = []
        layer.add(shapes, shape_type=kinds)
        layer.mode = "pan_zoom"
        self._report()

    def _fit_from_shape(self):
        """Take a shape you have drawn and back out the parameters."""
        from napari.layers import Shapes

        for layer in self.viewer.layers.selection:
            if isinstance(layer, Shapes) and len(layer.data):
                idx = next(iter(layer.selected_data), 0)
                verts = np.asarray(layer.data[idx])[:, -2:]
                ctr, long_side, short_side, ang = min_area_rect(verts)
                self._write_widgets(
                    replace(
                        self.params,
                        cy=float(ctr[0]),
                        cx=float(ctr[1]),
                        diameter=float(short_side),
                        length=float(long_side),
                        angle_deg=float(ang),
                    )
                )
                self._redraw()
                return
        self._report(note="Select a shape in a Shapes layer first.")

    # ── reporting ─────────────────────────────────────────────────────

    def _report(self, note: str | None = None):
        p = self.params
        lines = [
            "── Cylinder fit ──",
            f"center      : ({p.cy:.1f}, {p.cx:.1f}) px",
            f"angle/tilt  : {p.angle_deg:.1f}\u00b0 / {p.tilt_deg:+.1f}\u00b0",
            f"diameter    : {p.diameter:.2f} px",
            f"length      : {p.length:.2f} px  "
            f"(projected {p.projected_length:.2f} px)",
        ]
        if self.calib is not None and getattr(self.calib, "is_set", False):
            d_m = self.calib.px_to_m(p.diameter)
            l_m = self.calib.px_to_m(p.length)
            v_m3 = math.pi * (d_m / 2) ** 2 * l_m
            lines += [
                f"diameter    : {d_m * 1e3:.4f} mm",
                f"length      : {l_m * 1e3:.4f} mm",
                f"volume      : {v_m3 * 1e9:.4f} mm\u00b3",
            ]
        else:
            lines.append(
                f"volume      : {p.volume_px3():.1f} px\u00b3 "
                "(calibrate for mm\u00b3)"
            )
        if note:
            lines += ["", note]
        self.readout.setPlainText("\n".join(lines))

    def _copy(self):
        from qtpy.QtWidgets import QApplication

        QApplication.clipboard().setText(self.readout.toPlainText())
