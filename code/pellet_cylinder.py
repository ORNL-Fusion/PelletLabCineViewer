"""
pellet_cylinder.py
==================

A parametric *cylinder* overlay for PelletLabCineViewer (napari).

napari has no built-in "cylinder object with a properties panel", so this module
builds one: a Shapes layer whose polygon is regenerated from five numbers

    center (row, col), diameter, length, in-plane angle, out-of-plane tilt

Because a cylinder viewed in orthographic projection has a silhouette that is a
rectangle capped by two half-ellipses, the tilt parameter lets you recover the
*true* length even when the pellet is tipped toward or away from the camera:

    projected length  = L*cos(tilt) + D*sin(tilt)
    apparent end caps = ellipses with semi-axes (D/2)*sin(tilt) x (D/2)

Volume is then pi*(D/2)^2*L in calibrated units, independent of tilt.

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
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
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
    tilt_deg: float = 0.0      # out-of-plane tilt; 0 = axis lies in image plane
    flat_ends: bool = False    # True -> draw square ends (ignore cap ellipses)

    # ── derived quantities ────────────────────────────────────────────
    @property
    def radius(self) -> float:
        return self.diameter / 2.0

    @property
    def projected_length(self) -> float:
        """Tip-to-tip extent of the silhouette along the axis (px)."""
        t = math.radians(self.tilt_deg)
        return self.length * math.cos(t) + self.diameter * math.sin(t)

    def volume_px3(self) -> float:
        return math.pi * self.radius**2 * self.length

    def silhouette(self, n_cap: int = 48) -> np.ndarray:
        """Outline vertices as an (N, 2) array of (row, col) — napari order."""
        t = math.radians(self.tilt_deg)
        a = 0.5 * self.length * math.cos(t)          # half projected axis
        b = 0.0 if self.flat_ends else self.radius * math.sin(t)  # cap bulge
        r = self.radius

        psi = np.linspace(-math.pi / 2, math.pi / 2, n_cap)
        # right cap (u = +a side), then left cap, walking around the outline
        right = np.column_stack((a + b * np.cos(psi), r * np.sin(psi)))
        left = np.column_stack((-a - b * np.cos(psi), -r * np.sin(psi)))
        uv = np.vstack((right, left))

        phi = math.radians(self.angle_deg)
        c, s = math.cos(phi), math.sin(phi)
        # local (u along axis, v across) -> image (col, row)
        col = uv[:, 0] * c - uv[:, 1] * s + self.cx
        row = uv[:, 0] * s + uv[:, 1] * c + self.cy
        return np.column_stack((row, col))


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
    center_rowcol = np.array([ctr[1], ctr[0]])
    return center_rowcol, long_side, short_side, math.degrees(ang)


def auto_fit_frame(image: np.ndarray) -> CylinderParams | None:
    """Threshold the brightest/darkest blob and return a starting guess."""
    from skimage.filters import threshold_otsu
    from skimage.measure import label, regionprops

    img = np.asarray(image, float)
    if img.ndim == 3:  # RGB
        img = img.mean(axis=-1)
    thr = threshold_otsu(img)
    mask = img > thr
    if mask.mean() > 0.5:  # pellet is dark on a bright background
        mask = ~mask
    lab = label(mask)
    if lab.max() == 0:
        return None
    region = max(regionprops(lab), key=lambda r: r.area)
    ctr, long_side, short_side, ang = min_area_rect(
        np.argwhere(lab == region.label)
    )
    return CylinderParams(
        cy=float(ctr[0]),
        cx=float(ctr[1]),
        diameter=float(short_side),
        length=float(long_side),
        angle_deg=float(ang),
        tilt_deg=0.0,
    )


# ═══════════════════════════════════════════════════════════════════════
# Qt tab
# ═══════════════════════════════════════════════════════════════════════


class CylinderTab(QWidget):
    """Side-panel tab that drives a live cylinder overlay in napari."""

    LAYER = "Cylinder Fit"

    def __init__(self, viewer, calib=None, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.calib = calib
        self.params = CylinderParams()
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
            "Overlay a parametric cylinder and match it to the pellet. "
            "Tilt corrects for foreshortening, so length and volume stay true "
            "even when the pellet leans toward the camera."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #aaa; font-size: 11px;")
        root.addWidget(info)

        grp = QGroupBox("Cylinder parameters (px / deg)")
        form = QFormLayout(grp)
        self.s_cy = self._spin(self.params.cy, -1e5, 1e5, 1)
        self.s_cx = self._spin(self.params.cx, -1e5, 1e5, 1)
        self.s_dia = self._spin(self.params.diameter, 0.1, 1e5, 1)
        self.s_len = self._spin(self.params.length, 0.1, 1e5, 1)
        self.s_ang = self._spin(self.params.angle_deg, -180, 180, 0.5, "\u00b0")
        self.s_tilt = self._spin(self.params.tilt_deg, 0, 89, 0.5, "\u00b0")
        form.addRow("Center row (y)", self.s_cy)
        form.addRow("Center col (x)", self.s_cx)
        form.addRow("Diameter (width)", self.s_dia)
        form.addRow("Length (true)", self.s_len)
        form.addRow("In-plane rotation", self.s_ang)
        form.addRow("Tilt out of plane", self.s_tilt)
        self.cb_flat = QCheckBox("Draw flat ends (no cap ellipses)")
        self.cb_flat.stateChanged.connect(self._on_param_changed)
        form.addRow(self.cb_flat)
        root.addWidget(grp)

        row = QHBoxLayout()
        b_show = QPushButton("Show / Update overlay")
        b_show.clicked.connect(self._redraw)
        b_auto = QPushButton("Auto-fit frame")
        b_auto.clicked.connect(self._auto_fit)
        b_from = QPushButton("Fit from selected shape")
        b_from.clicked.connect(self._fit_from_shape)
        for b in (b_show, b_auto, b_from):
            row.addWidget(b)
        root.addLayout(row)

        self.readout = QTextEdit()
        self.readout.setReadOnly(True)
        self.readout.setPlaceholderText("Measurements appear here.")
        root.addWidget(self.readout)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Density (g/cm\u00b3)"))
        self.s_rho = QDoubleSpinBox()
        self.s_rho.setRange(0.0, 30.0)
        self.s_rho.setDecimals(4)
        self.s_rho.setValue(0.2)  # solid D2 ~0.2 g/cm^3
        self.s_rho.valueChanged.connect(self._report)
        row2.addWidget(self.s_rho)
        b_copy = QPushButton("\U0001f4cb Copy")
        b_copy.clicked.connect(self._copy)
        row2.addWidget(b_copy)
        root.addLayout(row2)

    # ── state <-> widgets ─────────────────────────────────────────────

    def _read_widgets(self) -> CylinderParams:
        return CylinderParams(
            cy=self.s_cy.value(),
            cx=self.s_cx.value(),
            diameter=self.s_dia.value(),
            length=self.s_len.value(),
            angle_deg=self.s_ang.value(),
            tilt_deg=self.s_tilt.value(),
            flat_ends=self.cb_flat.isChecked(),
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
            self.cb_flat.setChecked(p.flat_ends)
        finally:
            self._updating = False
        self.params = p

    def _on_param_changed(self, *_):
        if self._updating:
            return
        self.params = self._read_widgets()
        self._redraw()

    # ── napari layer ──────────────────────────────────────────────────

    def _layer(self):
        if self.LAYER in self.viewer.layers:
            return self.viewer.layers[self.LAYER]
        return self.viewer.add_shapes(
            name=self.LAYER,
            edge_color="#00ff88",
            edge_width=2,
            face_color="transparent",
            opacity=0.9,
        )

    def _redraw(self):
        p = self.params
        layer = self._layer()
        layer.data = []
        layer.add_polygons(p.silhouette())
        # axis line, useful for eyeballing the alignment
        phi = math.radians(p.angle_deg)
        half = 0.5 * p.length * math.cos(math.radians(p.tilt_deg))
        d = np.array([math.sin(phi), math.cos(phi)])  # (row, col)
        c = np.array([p.cy, p.cx])
        layer.add_paths(np.vstack((c - half * d, c + half * d)))
        layer.mode = "pan_zoom"
        self._report()

    def _fit_from_shape(self):
        """Take the selected shape (any type) and back out the parameters."""
        from napari.layers import Shapes

        for layer in self.viewer.layers.selection:
            if isinstance(layer, Shapes) and len(layer.data):
                idx = next(iter(layer.selected_data), 0)
                verts = np.asarray(layer.data[idx])[:, -2:]
                ctr, long_side, short_side, ang = min_area_rect(verts)
                p = replace(
                    self.params,
                    cy=float(ctr[0]),
                    cx=float(ctr[1]),
                    diameter=float(short_side),
                    length=float(long_side),
                    angle_deg=float(ang),
                )
                self._write_widgets(p)
                self._redraw()
                return
        self.readout.append("No shape selected.")

    def _auto_fit(self):
        for layer in self.viewer.layers:
            if layer.__class__.__name__ == "Image":
                frame = np.asarray(layer.data)
                if frame.ndim >= 3 and frame.shape[0] > 4:
                    frame = frame[int(self.viewer.dims.current_step[0])]
                guess = auto_fit_frame(frame)
                if guess is None:
                    self.readout.append("Auto-fit found nothing.")
                    return
                guess.tilt_deg = self.params.tilt_deg
                guess.flat_ends = self.params.flat_ends
                self._write_widgets(guess)
                self._redraw()
                return
        self.readout.append("No image layer found.")

    # ── reporting ─────────────────────────────────────────────────────

    def _report(self):
        p = self.params
        lines = [
            "── Cylinder fit ──",
            f"center      : ({p.cy:.1f}, {p.cx:.1f}) px",
            f"angle/tilt  : {p.angle_deg:.1f}\u00b0 / {p.tilt_deg:.1f}\u00b0",
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
            rho = self.s_rho.value() * 1000.0  # g/cm^3 -> kg/m^3
            if rho > 0:
                lines.append(f"mass @ {self.s_rho.value():g} g/cm\u00b3 : "
                             f"{v_m3 * rho * 1e6:.4f} mg")
        else:
            lines.append(f"volume      : {p.volume_px3():.1f} px\u00b3 "
                         "(calibrate for mm\u00b3)")
        self.readout.setPlainText("\n".join(lines))

    def _copy(self):
        from qtpy.QtWidgets import QApplication

        QApplication.clipboard().setText(self.readout.toPlainText())
