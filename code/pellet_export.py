"""
pellet_export.py
================

Publication-ready figure export for PelletLabCineViewer.

The raster comes straight out of napari's own renderer, so whatever colormap,
contrast limits, gamma, opacity, blending and layer visibility you have dialled
in on screen is exactly what lands in the file — this module never re-implements
the display pipeline, it just captures it and dresses it up.

On top of that raster it composites, as *vector* elements via matplotlib:

    * a calibrated scale bar (auto-rounded to a 1/2/5 x 10^n length)
    * a frame / timestamp stamp
    * a panel letter (a, b, c ...) for multi-panel figures
    * a caption below the image

and writes PNG / TIFF / PDF / SVG at a chosen physical width and DPI, which is
how journals actually specify figures ("single column, 300 dpi") rather than in
raw pixels.

Usage inside pelletVideoViewer.py
---------------------------------
    from pellet_export import ExportTab
    ...
    tabs.addTab(ExportTab(self.viewer, self.calib, host=self), "\U0001f5bc Export")
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

try:  # matplotlib does the page layout and the vector overlays
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    HAVE_MPL = True
except ImportError:  # pragma: no cover
    HAVE_MPL = False


MM_PER_IN = 25.4

# Common journal column widths, in millimetres.
PRESETS = {
    "Single column (90 mm)": 90.0,
    "1.5 column (140 mm)": 140.0,
    "Double column (180 mm)": 180.0,
    "Slide / poster (240 mm)": 240.0,
    "Custom": None,
}


# ═══════════════════════════════════════════════════════════════════════
# Helpers (pure — no Qt, no napari)
# ═══════════════════════════════════════════════════════════════════════


def nice_length(raw: float) -> float:
    """Round a length down to the nearest 1, 2 or 5 times a power of ten."""
    if raw <= 0 or not math.isfinite(raw):
        return 0.0
    exp = math.floor(math.log10(raw))
    mant = raw / 10**exp
    for step in (5.0, 2.0, 1.0):
        if mant >= step:
            return step * 10**exp
    return 10**exp


def auto_scale_bar(meters_per_px: float, width_px: float, frac: float = 0.2):
    """Pick a tidy scale-bar length covering ~`frac` of the image width.

    Returns (length_in_meters, label_string).
    """
    if meters_per_px <= 0:
        return 0.0, ""
    target_m = frac * width_px * meters_per_px
    for unit, factor, fmt in (
        ("mm", 1e-3, "{:g} mm"),
        ("\u00b5m", 1e-6, "{:g} \u00b5m"),
        ("nm", 1e-9, "{:g} nm"),
    ):
        if target_m >= factor:
            val = nice_length(target_m / factor)
            if val > 0:
                return val * factor, fmt.format(val)
    val = nice_length(target_m / 1e-9)
    return val * 1e-9, f"{val:g} nm"


def current_frame_index(viewer, host=None) -> int:
    """Which frame is on screen right now.

    ``viewer.dims.current_step`` is the source of truth: it moves whether the
    frame changed via the slider, the playback timer, or the step buttons.  The
    host's ``_current_frame`` is only a cache that the step buttons write, so it
    is used purely as a fallback for viewers with no slider dimension.
    """
    try:
        step = viewer.dims.current_step
        if len(step) >= 3:          # (frame, row, col) or deeper
            return int(step[0])
    except Exception:
        pass
    if host is not None:
        try:
            return int(getattr(host, "_current_frame", 0))
        except Exception:
            pass
    return 0


def format_time(frame_idx: int, fps: float) -> str:
    """Frame index + fps -> a human-sized timestamp."""
    if not fps or fps <= 0:
        return f"frame {frame_idx}"
    t = frame_idx / fps
    if t < 1e-3:
        return f"{t * 1e6:.1f} \u00b5s"
    if t < 1.0:
        return f"{t * 1e3:.2f} ms"
    return f"{t:.3f} s"


ANCHORS = {
    # name: (x, y, ha, va) in axes fraction, with a 0.035 inset
    "Bottom right": (0.965, 0.045, "right", "bottom"),
    "Bottom left": (0.035, 0.045, "left", "bottom"),
    "Top right": (0.965, 0.955, "right", "top"),
    "Top left": (0.035, 0.955, "left", "top"),
}


def compose_figure(
    rgba: np.ndarray,
    out_path: str | Path,
    *,
    width_mm: float = 90.0,
    dpi: int = 300,
    facecolor: str = "white",
    smooth: bool = False,
    scale_bar_m: float = 0.0,
    scale_bar_label: str = "",
    meters_per_px: float = 0.0,
    scale_bar_pos: str = "Bottom right",
    show_bar_label: bool = True,
    stamp: str = "",
    stamp_pos: str = "Top right",
    panel_label: str = "",
    caption: str = "",
    fg: str = "white",
    font_size: float = 8.0,
    font_family: str = "sans-serif",
    border: bool = False,
) -> Path:
    """Lay the captured raster out as a figure and write it to disk."""
    if not HAVE_MPL:
        raise RuntimeError("matplotlib is required for figure export.")

    out_path = Path(out_path)
    arr = np.asarray(rgba)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]  # the canvas is opaque; drop the alpha channel
    h, w = arr.shape[:2]

    width_in = width_mm / MM_PER_IN
    img_h_in = width_in * h / w

    caption_lines = caption.count("\n") + 1 if caption.strip() else 0
    cap_in = 0.0
    if caption_lines:
        cap_in = 0.10 + caption_lines * (font_size * 1.45 / 72.0)
    fig_h_in = img_h_in + cap_in

    with plt.rc_context(
        {"font.family": font_family, "font.size": font_size, "svg.fonttype": "none"}
    ):
        fig = plt.figure(figsize=(width_in, fig_h_in), dpi=dpi, facecolor=facecolor)
        ax = fig.add_axes([0.0, cap_in / fig_h_in, 1.0, img_h_in / fig_h_in])
        ax.imshow(
            arr,
            interpolation="antialiased" if smooth else "nearest",
            aspect="auto",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(border)
            spine.set_color(fg)
            spine.set_linewidth(0.6)

        # ── scale bar ────────────────────────────────────────────────
        if scale_bar_m > 0 and meters_per_px > 0:
            bar_px = scale_bar_m / meters_per_px
            bar_frac = bar_px / w
            if bar_frac > 0.9:
                raise ValueError(
                    "Scale bar is wider than the image; pick a shorter length."
                )
            x, y, ha, va = ANCHORS.get(scale_bar_pos, ANCHORS["Bottom right"])
            bar_h = max(0.010, 2.0 / h)
            x0 = x - bar_frac if ha == "right" else x
            y0 = y if va == "bottom" else y - bar_h
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    bar_frac,
                    bar_h,
                    transform=ax.transAxes,
                    facecolor=fg,
                    edgecolor="none",
                    zorder=5,
                )
            )
            if show_bar_label:
                above = va == "bottom"
                ax.text(
                    x0 + bar_frac / 2,
                    y0 + bar_h + 0.012 if above else y0 - 0.012,
                    scale_bar_label,
                    transform=ax.transAxes,
                    color=fg,
                    ha="center",
                    va="bottom" if above else "top",
                    fontsize=font_size,
                    zorder=5,
                )

        # ── stamp ────────────────────────────────────────────────────
        if stamp.strip():
            x, y, ha, va = ANCHORS.get(stamp_pos, ANCHORS["Top right"])
            if panel_label.strip() and stamp_pos == "Top left":
                # the panel letter owns that corner; slide the stamp clear of it
                x += 0.035 + 0.022 * len(panel_label.strip())
            ax.text(
                x,
                y,
                stamp,
                transform=ax.transAxes,
                color=fg,
                ha=ha,
                va=va,
                fontsize=font_size,
                zorder=5,
            )

        # ── panel letter ─────────────────────────────────────────────
        if panel_label.strip():
            ax.text(
                0.012,
                0.985,
                panel_label,
                transform=ax.transAxes,
                color=fg,
                ha="left",
                va="top",
                fontsize=font_size * 1.35,
                fontweight="bold",
                zorder=6,
            )

        # ── caption ──────────────────────────────────────────────────
        if caption_lines:
            fig.text(
                0.0,
                0.012,
                caption,
                ha="left",
                va="bottom",
                fontsize=font_size,
                color="black" if facecolor not in ("black", "none") else "white",
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out_path,
            dpi=dpi,
            facecolor=facecolor,
            edgecolor="none",
            transparent=(facecolor == "none"),
        )
        plt.close(fig)
    return out_path


# ═══════════════════════════════════════════════════════════════════════
# Qt tab
# ═══════════════════════════════════════════════════════════════════════


class ExportTab(QWidget):
    """Side-panel tab that renders the current view as a figure file."""

    def __init__(self, viewer, calib=None, host=None, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.calib = calib
        self.host = host          # the CineViewerWidget, for fps / frame index
        self._last_dir = str(Path.home())
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)

        info = QLabel(
            "Renders the current frame with every visible layer on top, using "
            "the colormap, contrast limits and gamma set in the viewer."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #aaa; font-size: 11px;")
        root.addWidget(info)

        if not HAVE_MPL:
            warn = QLabel(
                "matplotlib is not installed, so figure export is disabled.\n"
                "Install it with:  pip install matplotlib"
            )
            warn.setWordWrap(True)
            warn.setStyleSheet("color: #e88; font-size: 11px;")
            root.addWidget(warn)
            root.addStretch(1)
            return

        # ── page setup ───────────────────────────────────────────────
        page = QGroupBox("Page")
        f = QFormLayout(page)

        self.c_preset = QComboBox()
        self.c_preset.addItems(PRESETS.keys())
        self.c_preset.currentTextChanged.connect(self._preset_changed)
        f.addRow("Width preset", self.c_preset)

        self.s_width = QDoubleSpinBox()
        self.s_width.setRange(10.0, 1000.0)
        self.s_width.setValue(90.0)
        self.s_width.setSuffix(" mm")
        self.s_width.valueChanged.connect(self._update_estimate)
        f.addRow("Figure width", self.s_width)

        self.s_dpi = QSpinBox()
        self.s_dpi.setRange(72, 1200)
        self.s_dpi.setSingleStep(50)
        self.s_dpi.setValue(300)
        self.s_dpi.valueChanged.connect(self._update_estimate)
        f.addRow("DPI", self.s_dpi)

        self.c_region = QComboBox()
        self.c_region.addItems(["Full frame (recommended)", "Canvas view (as shown)"])
        self.c_region.currentTextChanged.connect(self._update_estimate)
        f.addRow("Region", self.c_region)

        self.s_capture = QSpinBox()
        self.s_capture.setRange(0, 16)
        self.s_capture.setValue(0)
        self.s_capture.setSpecialValueText("auto")
        self.s_capture.setPrefix("\u00d7")
        self.s_capture.valueChanged.connect(self._update_estimate)
        f.addRow("Capture scale", self.s_capture)

        self.cb_smooth = QCheckBox("Smooth interpolation (off = crisp pixels)")
        f.addRow(self.cb_smooth)

        self._estimate = QLabel("")
        self._estimate.setWordWrap(True)
        self._estimate.setStyleSheet("color: #888; font-size: 10px;")
        f.addRow(self._estimate)
        root.addWidget(page)

        # ── annotations ──────────────────────────────────────────────
        ann = QGroupBox("Annotations")
        f2 = QFormLayout(ann)

        bar_row = QHBoxLayout()
        self.cb_bar = QCheckBox("Scale bar")
        self.cb_bar.setChecked(True)
        bar_row.addWidget(self.cb_bar)
        self.s_bar = QDoubleSpinBox()
        self.s_bar.setRange(0.0, 1e6)
        self.s_bar.setDecimals(3)
        self.s_bar.setSpecialValueText("auto")
        self.s_bar.setSuffix(" mm")
        bar_row.addWidget(self.s_bar)
        self.c_bar_pos = QComboBox()
        self.c_bar_pos.addItems(ANCHORS.keys())
        bar_row.addWidget(self.c_bar_pos)
        f2.addRow(bar_row)

        self.cb_bar_label = QCheckBox("Label the scale bar")
        self.cb_bar_label.setChecked(True)
        f2.addRow(self.cb_bar_label)

        stamp_row = QHBoxLayout()
        self.cb_stamp = QCheckBox("Frame / time stamp")
        self.cb_stamp.setChecked(True)
        stamp_row.addWidget(self.cb_stamp)
        self.c_stamp_pos = QComboBox()
        self.c_stamp_pos.addItems(ANCHORS.keys())
        self.c_stamp_pos.setCurrentText("Top right")
        stamp_row.addWidget(self.c_stamp_pos)
        f2.addRow(stamp_row)

        self.e_panel = QLineEdit()
        self.e_panel.setPlaceholderText("(a)")
        f2.addRow("Panel label", self.e_panel)

        self.e_caption = QLineEdit()
        self.e_caption.setPlaceholderText("Optional caption printed below")
        f2.addRow("Caption", self.e_caption)

        style_row = QHBoxLayout()
        self.c_fg = QComboBox()
        self.c_fg.addItems(["white", "black", "yellow", "red", "#00ff88"])
        style_row.addWidget(QLabel("Ink"))
        style_row.addWidget(self.c_fg)
        self.c_bg = QComboBox()
        self.c_bg.addItems(["white", "black", "none"])
        style_row.addWidget(QLabel("Page"))
        style_row.addWidget(self.c_bg)
        self.s_font = QDoubleSpinBox()
        self.s_font.setRange(4.0, 24.0)
        self.s_font.setValue(8.0)
        self.s_font.setSuffix(" pt")
        style_row.addWidget(QLabel("Font"))
        style_row.addWidget(self.s_font)
        f2.addRow(style_row)

        self.c_family = QComboBox()
        self.c_family.addItems(["sans-serif", "serif"])
        self.cb_border = QCheckBox("Thin border")
        fam_row = QHBoxLayout()
        fam_row.addWidget(self.c_family)
        fam_row.addWidget(self.cb_border)
        f2.addRow("Typeface", fam_row)
        root.addWidget(ann)

        # ── actions ──────────────────────────────────────────────────
        btns = QHBoxLayout()
        b_prev = QPushButton("\U0001f441 Preview")
        b_prev.clicked.connect(self._preview)
        b_exp = QPushButton("\U0001f5bc Export figure\u2026")
        b_exp.clicked.connect(self._export)
        btns.addWidget(b_prev)
        btns.addWidget(b_exp)
        root.addLayout(btns)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: #888; font-size: 10px;")
        root.addWidget(self._status)
        root.addStretch(1)

        self._update_estimate()

    def _preset_changed(self, name: str):
        mm = PRESETS.get(name)
        if mm:
            self.s_width.setValue(mm)
        self._update_estimate()

    # ── capture ───────────────────────────────────────────────────────

    def _image_layer(self):
        for lyr in self.viewer.layers:
            if lyr.__class__.__name__ == "Image" and lyr.visible:
                return lyr
        return None

    def _frame_size(self) -> tuple[int, int]:
        """(height, width) of one frame, in data pixels."""
        lyr = self._image_layer()
        if lyr is None:
            return (0, 0)
        shape = np.asarray(lyr.data).shape
        return (int(shape[-2]), int(shape[-1]))

    def _needed_scale(self) -> int:
        """Capture upscale that keeps the raster at or above the target DPI."""
        if self.s_capture.value() > 0:
            return int(self.s_capture.value())
        h, w = self._frame_size()
        if w == 0:
            return 1
        target_px = self.s_width.value() / MM_PER_IN * self.s_dpi.value()
        return max(1, min(16, math.ceil(target_px / w)))

    def _update_estimate(self, *_):
        h, w = self._frame_size()
        if w == 0:
            self._estimate.setText("No image layer loaded.")
            return
        s = self._needed_scale()
        target_px = self.s_width.value() / MM_PER_IN * self.s_dpi.value()
        eff_dpi = w * s / (self.s_width.value() / MM_PER_IN)
        msg = (
            f"Frame {w}\u00d7{h} px \u2192 capture \u00d7{s} = {w * s}\u00d7{h * s} px "
            f"for a {target_px:.0f} px wide page ({eff_dpi:.0f} dpi effective)."
        )
        if eff_dpi < self.s_dpi.value() * 0.99:
            msg += "  Upscaling beyond the sensor adds no real detail."
        self._estimate.setText(msg)

    @contextmanager
    def _clean_canvas(self):
        """Hide edit handles, selections and napari's own overlays briefly."""
        viewer = self.viewer
        modes, sels = {}, {}
        saved_selection = list(viewer.layers.selection)
        for lyr in viewer.layers:
            if hasattr(lyr, "mode"):
                modes[lyr] = lyr.mode
                try:
                    lyr.mode = "pan_zoom"
                except Exception:
                    pass
            if hasattr(lyr, "selected_data"):
                try:
                    sels[lyr] = set(lyr.selected_data)
                    lyr.selected_data = set()
                except Exception:
                    pass
        overlays = {}
        for name in ("scale_bar", "axes", "text_overlay"):
            ov = getattr(viewer, name, None)
            if ov is not None and hasattr(ov, "visible"):
                overlays[ov] = ov.visible
                ov.visible = False
        try:
            viewer.layers.selection.clear()
        except Exception:
            pass
        try:
            yield
        finally:
            for ov, vis in overlays.items():
                ov.visible = vis
            for lyr, mode in modes.items():
                try:
                    lyr.mode = mode
                except Exception:
                    pass
            for lyr, sel in sels.items():
                try:
                    lyr.selected_data = sel
                except Exception:
                    pass
            try:
                viewer.layers.selection.clear()
                for lyr in saved_selection:
                    viewer.layers.selection.add(lyr)
            except Exception:
                pass

    def _capture(self) -> tuple[np.ndarray, float]:
        """Grab the canvas. Returns (rgba, output px per data px)."""
        scale = self._needed_scale()
        full = self.c_region.currentText().startswith("Full")
        with self._clean_canvas():
            if full:
                rgba = self._export_figure(scale)
                px_per_data = float(scale)
            else:
                rgba = self.viewer.screenshot(
                    canvas_only=True, scale=scale, flash=False
                )
                px_per_data = self._canvas_px_per_data(rgba, scale)
        return np.asarray(rgba), px_per_data

    def _export_figure(self, scale: int) -> np.ndarray:
        """Call export_figure across the napari versions that renamed its arg."""
        import inspect

        fn = self.viewer.export_figure
        params = inspect.signature(fn).parameters
        kw = {"flash": False}
        if "scale_factor" in params:
            kw["scale_factor"] = scale
        elif "scale" in params:
            kw["scale"] = scale
        return fn(**kw)

    def _canvas_px_per_data(self, rgba, scale: int) -> float:
        """Zoom-dependent pixel ratio for a canvas screenshot.

        Returns 0.0 when it cannot be established, which suppresses the scale
        bar rather than drawing one at the wrong length.
        """
        try:
            zoom = float(self.viewer.camera.zoom)
            canvas = self.viewer.window._qt_viewer.canvas
            size = getattr(canvas, "size", None)
            canvas_h = float(size[1] if not callable(size) else size().height())
            if canvas_h <= 0:
                return 0.0
            dpr = np.asarray(rgba).shape[0] / (canvas_h * scale)
            return zoom * scale * dpr
        except Exception:
            return 0.0

    # ── figure assembly ───────────────────────────────────────────────

    def _stamp_text(self) -> str:
        idx = current_frame_index(self.viewer, self.host)
        loader = getattr(self.host, "loader", None) if self.host else None
        try:
            fps = float(getattr(loader, "fps", 0.0) or 0.0)
        except Exception:
            fps = 0.0
        return (
            f"frame {idx}"
            if fps <= 0
            else f"frame {idx} \u00b7 {format_time(idx, fps)}"
        )

    def _build(self, out_path: Path) -> Path:
        rgba, px_per_data = self._capture()
        if rgba is None or rgba.size == 0:
            raise RuntimeError("The canvas returned an empty image.")

        # metres per output pixel, for the scale bar
        m_per_px = 0.0
        if (
            self.calib is not None
            and getattr(self.calib, "is_set", False)
            and px_per_data > 0
        ):
            m_per_px = self.calib.px_to_m(1.0) / px_per_data

        bar_m, bar_label = 0.0, ""
        if self.cb_bar.isChecked() and m_per_px > 0:
            if self.s_bar.value() > 0:
                bar_m = self.s_bar.value() * 1e-3
                bar_label = f"{self.s_bar.value():g} mm"
            else:
                bar_m, bar_label = auto_scale_bar(m_per_px, rgba.shape[1])

        return compose_figure(
            rgba,
            out_path,
            width_mm=self.s_width.value(),
            dpi=self.s_dpi.value(),
            facecolor=self.c_bg.currentText(),
            smooth=self.cb_smooth.isChecked(),
            scale_bar_m=bar_m,
            scale_bar_label=bar_label,
            meters_per_px=m_per_px,
            scale_bar_pos=self.c_bar_pos.currentText(),
            show_bar_label=self.cb_bar_label.isChecked(),
            stamp=self._stamp_text() if self.cb_stamp.isChecked() else "",
            stamp_pos=self.c_stamp_pos.currentText(),
            panel_label=self.e_panel.text(),
            caption=self.e_caption.text(),
            fg=self.c_fg.currentText(),
            font_size=self.s_font.value(),
            font_family=self.c_family.currentText(),
            border=self.cb_border.isChecked(),
        )

    def _warn_if_uncalibrated(self):
        if self.cb_bar.isChecked() and not getattr(self.calib, "is_set", False):
            self._status.setText(
                "No calibration set, so the scale bar was skipped. "
                "Set one on the Calibrate tab."
            )

    # ── actions ───────────────────────────────────────────────────────

    def _preview(self):
        import tempfile

        try:
            tmp = Path(tempfile.mkdtemp()) / "preview.png"
            self._build(tmp)
        except Exception as exc:
            QMessageBox.critical(self, "Preview failed", str(exc))
            return

        from qtpy.QtGui import QPixmap

        dlg = QDialog(self)
        dlg.setWindowTitle("Figure preview")
        lay = QVBoxLayout(dlg)
        lbl = QLabel()
        pix = QPixmap(str(tmp))
        if pix.width() > 900:
            pix = pix.scaledToWidth(900)
        lbl.setPixmap(pix)
        lay.addWidget(lbl)
        close = QPushButton("Close")
        close.clicked.connect(dlg.accept)
        lay.addWidget(close)
        self._warn_if_uncalibrated()
        dlg.exec_()

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export figure",
            str(Path(self._last_dir) / "figure.png"),
            "PNG (*.png);;TIFF (*.tif *.tiff);;PDF (*.pdf);;SVG (*.svg)",
        )
        if not path:
            return
        try:
            out = self._build(Path(path))
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self._last_dir = str(out.parent)
        self._status.setText(f"Wrote {out}")
        self._warn_if_uncalibrated()
        if self.host is not None and hasattr(self.host, "_log"):
            self.host._log(f"Exported figure: {out}")
