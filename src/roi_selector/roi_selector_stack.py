"""A native PyQt6 ROI selector for ``(n, height, width)`` image stacks.

Coordinates are ``(x, y)`` = ``(column, row)``.  The dialog displays one
frame while :func:`crop_stack` crops the final two dimensions of an array.
Zoom percentages use Qt logical pixels, so high-DPI displays are handled by
Qt's normal automatic scaling.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Sequence

import numpy as np


DEFAULT_BEAM_CENTER_XY: tuple[float, float] | None = None
_ROI_QT_APP = None

ZOOM_LEVELS: tuple[float, ...] = (
    0.125, 0.25, 0.333, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    8.0, 12.0, 16.0, 24.0, 32.0,
)
MAX_ZOOM = 32.0
SENSITIVITY = 1.0015


def _image_size(image_shape: Sequence[int]) -> tuple[int, int]:
    """Return ``(height, width)`` from an image or stack shape."""
    if len(image_shape) < 2:
        raise ValueError("Expected an image shape ending in (height, width).")
    height, width = int(image_shape[-2]), int(image_shape[-1])
    if height < 1 or width < 1:
        raise ValueError("Image dimensions must be positive.")
    return height, width


def _round_half_up(value: float) -> int:
    return math.floor(value + 0.5)


def roi_center_from_bounds(left: int, top: int, right: int, bottom: int) -> tuple[float, float]:
    """Return the centre of integer image bounds."""
    return (left + right) / 2, (top + bottom) / 2


def centre_ranges(image_shape: Sequence[int], width: int, height: int) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return valid ``((min_x, max_x), (min_y, max_y))`` ROI centre ranges."""
    image_height, image_width = _image_size(image_shape)
    if not (1 <= width <= image_width and 1 <= height <= image_height):
        raise ValueError("ROI size does not fit image.")
    return ((width / 2, image_width - width / 2), (height / 2, image_height - height / 2))


def fit_scale(viewport_width: float, viewport_height: float, image_width: float, image_height: float) -> float:
    """Compute the exact scale which fits an image in a viewport."""
    if min(viewport_width, viewport_height, image_width, image_height) <= 0:
        return 0.0
    return min(viewport_width / image_width, viewport_height / image_height)


def clamp_zoom(scale: float, minimum: float, maximum: float = MAX_ZOOM) -> float:
    """Clamp a zoom scale to the fit and maximum limits."""
    return min(maximum, max(minimum, scale))


def zoom_in_step(current: float, fit: float) -> float:
    """Return the next discrete zoom level above ``current``."""
    current = clamp_zoom(current, fit)
    return next((level for level in ZOOM_LEVELS if level > current + 1e-9), MAX_ZOOM) if current < MAX_ZOOM else MAX_ZOOM


def zoom_out_step(current: float, fit: float) -> float:
    """Return the previous discrete zoom level, never below fit."""
    current = clamp_zoom(current, fit)
    if current <= fit + 1e-9:
        return fit
    levels = [level for level in ZOOM_LEVELS if level < current - 1e-9 and level >= fit]
    return levels[-1] if levels else fit


@dataclass(frozen=True)
class ROI:
    """A rectangular ROI expressed by its centre and size in image pixels."""

    center_x: float
    center_y: float
    width: int
    height: int

    def bounds(self, image_shape: Sequence[int]) -> tuple[int, int, int, int]:
        """Return clipped integer ``(left, top, right, bottom)`` bounds."""
        image_height, image_width = _image_size(image_shape)
        if not (1 <= self.width <= image_width and 1 <= self.height <= image_height):
            raise ValueError(f"ROI size ({self.width}, {self.height}) does not fit image size ({image_width}, {image_height}).")
        left = min(max(_round_half_up(self.center_x - self.width / 2), 0), image_width - self.width)
        top = min(max(_round_half_up(self.center_y - self.height / 2), 0), image_height - self.height)
        return left, top, left + self.width, top + self.height

    def clipped(self, image_shape: Sequence[int]) -> "ROI":
        """Return this ROI shifted inside the image, with canonical centre."""
        return ROI(*roi_center_from_bounds(*self.bounds(image_shape)), self.width, self.height)


def save_beam_center(path: str | Path, center_xy: Sequence[float]) -> Path:
    """Persist a beam centre as UTF-8 JSON."""
    if len(center_xy) != 2:
        raise ValueError("center_xy must contain (x, y).")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"beam_center_xy": [float(center_xy[0]), float(center_xy[1])]}, indent=2) + "\n", encoding="utf-8")
    return destination


def load_beam_center(path: str | Path) -> tuple[float, float]:
    """Load a centre created by :func:`save_beam_center`."""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))["beam_center_xy"]
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError
        return float(value[0]), float(value[1])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Settings file must contain 'beam_center_xy': [x, y].") from error


def crop_stack(stack: np.ndarray, roi: ROI) -> np.ndarray:
    """Crop the final two dimensions of a 2-D image or image stack."""
    array = np.asarray(stack)
    left, top, right, bottom = roi.bounds(array.shape)
    return array[..., top:bottom, left:right]


def _display_u8(image: np.ndarray) -> np.ndarray:
    """Make the specified full-range 8-bit display representation."""
    values = np.asarray(image, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = (np.nan_to_num(values, nan=low) - low) * (255.0 / (high - low))
    return np.ascontiguousarray(np.clip(scaled, 0, 255).astype(np.uint8))


class ROISelector:
    """Blocking, subprocess-backed selector suitable for Jupyter notebooks."""

    def __init__(self, stack: np.ndarray, display_index: int = 0, reference_center_xy: Sequence[float] | None = None,
                 settings_path: str | Path | None = None, input_path: str | Path | None = None, auto_show: bool = True) -> None:
        stack = np.asarray(stack)
        if stack.ndim != 3:
            raise ValueError("ROISelector expects a 3-D stack with shape (n, h, w).")
        if not 0 <= display_index < stack.shape[0]:
            raise IndexError(f"display_index {display_index} is out of range for a stack containing {stack.shape[0]} images.")
        self.image = stack[display_index]
        self.image_height, self.image_width = _image_size(self.image.shape)
        self.input_path = Path(input_path) if input_path else None
        self.settings_path = Path(settings_path) if settings_path else None
        chosen = (reference_center_xy if reference_center_xy is not None else
                  load_beam_center(self.settings_path) if self.settings_path and self.settings_path.exists() else
                  DEFAULT_BEAM_CENTER_XY if DEFAULT_BEAM_CENTER_XY is not None else
                  (self.image_width / 2, self.image_height / 2))
        if len(chosen) != 2:
            raise ValueError("reference_center_xy must contain (x, y).")
        self.reference_center_xy = float(chosen[0]), float(chosen[1])
        self.roi = ROI(*self.reference_center_xy, min(512, self.image_width), min(512, self.image_height)).clipped(self.image.shape)
        self.accepted = False
        if auto_show:
            self.show()

    @property
    def selected_roi(self) -> ROI:
        """The selected ROI (or the pre-dialog ROI after cancellation)."""
        return self.roi

    def show(self) -> "ROISelector":
        """Run the Qt dialog in a child interpreter and consume its result."""
        if importlib.util.find_spec("PyQt6") is None:
            raise ImportError("ROISelector requires PyQt6. Install it in the notebook environment with: pip install PyQt6")
        try:
            module_path = Path(__file__).resolve()
        except NameError as error:
            raise RuntimeError("ROISelector must be imported from a .py file on disk.") from error
        if not module_path.is_file():
            raise RuntimeError("ROISelector must be imported from a .py file on disk.")
        initial = self.roi
        with tempfile.TemporaryDirectory(prefix="roi_selector_", ignore_cleanup_errors=True) as folder:
            directory = Path(folder); image_path = directory / "image.npy"; result_path = directory / "result.json"; config_path = directory / "config.json"
            np.save(image_path, self.image, allow_pickle=False)
            config_path.write_text(json.dumps({"image_path": str(image_path), "result_path": str(result_path), "reference_center_xy": list(self.reference_center_xy), "settings_path": str(self.settings_path) if self.settings_path else None, "input_path": str(self.input_path) if self.input_path else None, "initial_roi": initial.__dict__}), encoding="utf-8")
            env = os.environ.copy(); env["PYTHONUTF8"] = "1"; env["PYTHONIOENCODING"] = "utf-8"
            completed = subprocess.run([sys.executable, str(module_path), "--roi-worker", str(config_path)], stdin=subprocess.DEVNULL, capture_output=True, check=False, encoding="utf-8", errors="replace", env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if completed.returncode:
                details = completed.stderr.strip() or completed.stdout.strip()
                raise RuntimeError("The ROI selector subprocess failed" + (":\n" + details if details else "."))
            if not result_path.exists():
                raise RuntimeError("The ROI selector process exited without returning a result.")
            result = json.loads(result_path.read_text(encoding="utf-8"))
        self.accepted = bool(result["accepted"])
        if self.accepted:
            self.roi = ROI(**result["roi"]).clipped(self.image.shape)
        else:
            self.roi = initial
        return self

    def _show_dialog(self) -> "ROISelector":
        """Build and run the worker-owned native Qt dialog."""
        from PyQt6.QtCore import QPointF, QRectF, Qt
        from PyQt6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap
        from PyQt6.QtWidgets import (QApplication, QButtonGroup, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QGraphicsItem, QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel, QPushButton, QRadioButton, QSpinBox, QVBoxLayout, QWidget)
        global _ROI_QT_APP
        _ROI_QT_APP = QApplication.instance() or QApplication([])
        selector = self

        class ImageView(QGraphicsView):
            def __init__(self) -> None:
                super().__init__(); self.fit_mode = True; self.fit = 1.0
                self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            def minimum_scale(self) -> float:
                return fit_scale(self.viewport().width(), self.viewport().height(), selector.image_width, selector.image_height)
            def set_scale_at(self, scale: float, anchor: Any | None = None) -> None:
                self.fit = self.minimum_scale(); scale = clamp_zoom(scale, self.fit)
                anchor = anchor or self.viewport().rect().center(); point = self.mapToScene(anchor)
                if not self.viewport().rect().contains(anchor):
                    anchor = self.viewport().rect().center(); point = self.mapToScene(anchor)
                self.resetTransform(); self.scale(scale, scale)
                delta = self.mapFromScene(point) - anchor
                self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + delta.x()); self.verticalScrollBar().setValue(self.verticalScrollBar().value() + delta.y())
                self.fit_mode = abs(scale - self.fit) < 1e-8; update_zoom()
            def fit_image(self) -> None:
                self.fit_mode = True; self.set_scale_at(self.minimum_scale())
            def current_scale(self) -> float: return self.transform().m11()
            def zoom_in(self, anchor: Any | None = None) -> None: self.set_scale_at(zoom_in_step(self.current_scale(), self.minimum_scale()), anchor)
            def zoom_out(self, anchor: Any | None = None) -> None: self.set_scale_at(zoom_out_step(self.current_scale(), self.minimum_scale()), anchor)
            def wheelEvent(self, event: Any) -> None:
                delta = event.angleDelta().y()
                if delta:
                    factor = min(3.0, max(1 / 3.0, SENSITIVITY ** delta)); self.set_scale_at(self.current_scale() * factor, event.position().toPoint()); event.accept()
                else: event.ignore()
            def keyPressEvent(self, event: Any) -> None:
                key = event.key()
                if key in (Qt.Key.Key_Up, Qt.Key.Key_Plus, Qt.Key.Key_Equal): self.zoom_in(self.mapFromGlobal(self.cursor().pos())); event.accept()
                elif key in (Qt.Key.Key_Down, Qt.Key.Key_Minus): self.zoom_out(self.mapFromGlobal(self.cursor().pos())); event.accept()
                else: super().keyPressEvent(event)
            def mousePressEvent(self, event: Any) -> None: self.setFocus(); super().mousePressEvent(event)
            def resizeEvent(self, event: Any) -> None:
                super().resizeEvent(event)
                if self.fit_mode: self.fit_image()
                else: self.set_scale_at(self.current_scale())

        class Frame(QGraphicsRectItem):
            def __init__(self, width: int, height: int) -> None:
                super().__init__(0, 0, width, height); self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True); self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True); self.setZValue(1)
                pen = QPen(QColor("#ff5a36"), 3); pen.setCosmetic(True); self.setPen(pen); self.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            def itemChange(self, change: Any, value: Any) -> Any:
                if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
                    return QPointF(min(max(value.x(), 0), selector.image_width - self.rect().width()), min(max(value.y(), 0), selector.image_height - self.rect().height()))
                if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged: update_from_frame()
                return super().itemChange(change, value)

        data = _display_u8(self.image); image = QImage(data.data, self.image_width, self.image_height, data.strides[0], QImage.Format.Format_Grayscale8).copy()
        scene = QGraphicsScene(0, 0, self.image_width, self.image_height); scene.addItem(QGraphicsPixmapItem(QPixmap.fromImage(image)))
        frame = Frame(self.roi.width, self.roi.height); scene.addItem(frame)
        dialog = QDialog(); dialog.setWindowTitle("Select region of interest" + (f" — {self.input_path.name}" if self.input_path else "")); dialog.resize(1280, 900)
        layout = QHBoxLayout(dialog); view = ImageView(); view.setScene(scene); view.setRenderHint(QPainter.RenderHint.Antialiasing); view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag); layout.addWidget(view, 1)
        panel = QWidget(); panel.setMinimumWidth(300); controls = QVBoxLayout(panel); layout.addWidget(panel)
        controls.addWidget(QLabel("ROI size")); b512, b1024, custom = QRadioButton("512 × 512"), QRadioButton("1024 × 1024"), QRadioButton("Custom"); b512.setEnabled(min(self.image_width, self.image_height) >= 512); b1024.setEnabled(min(self.image_width, self.image_height) >= 1024)
        group = QButtonGroup(dialog)
        for button in (b512, b1024, custom): group.addButton(button); controls.addWidget(button)
        (b512 if b512.isEnabled() else custom).setChecked(True)
        form = QFormLayout(); width_box, height_box = QSpinBox(), QSpinBox(); width_box.setRange(1, self.image_width); height_box.setRange(1, self.image_height); width_box.setValue(min(512, self.image_width)); height_box.setValue(min(512, self.image_height)); form.addRow("Width (px)", width_box); form.addRow("Height (px)", height_box)
        x_box, y_box = QDoubleSpinBox(), QDoubleSpinBox()
        for box in (x_box, y_box): box.setDecimals(1); box.setSingleStep(1.0); box.setKeyboardTracking(False)
        form.addRow("Centre X (px)", x_box); form.addRow("Centre Y (px)", y_box); controls.addLayout(form)
        centre_button, save_button = QPushButton("Centre on beam"), QPushButton("Save current as beam centre"); controls.addWidget(centre_button); controls.addWidget(save_button)
        controls.addWidget(QLabel("Zoom")); zoom_row = QHBoxLayout(); minus, fit_button, one, plus = QPushButton("−"), QPushButton("Fit"), QPushButton("1:1"), QPushButton("+")
        for button in (minus, fit_button, one, plus): zoom_row.addWidget(button)
        controls.addLayout(zoom_row); zoom_label = QLabel(); controls.addWidget(zoom_label); minus.setToolTip("Zoom out (Down or -)"); plus.setToolTip("Zoom in (Up, + or =)"); fit_button.setToolTip("Fit image"); one.setToolTip("100 % zoom")
        controls.addSpacing(12); readout, status = QLabel(), QLabel(); readout.setWordWrap(True); status.setWordWrap(True); controls.addWidget(readout); controls.addWidget(status); controls.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel); buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Use ROI"); controls.addWidget(buttons)
        syncing = False
        def update_zoom() -> None:
            scale, minimum = view.current_scale(), view.minimum_scale(); zoom_label.setText(f"Zoom: {scale * 100:.0f} %"); minus.setEnabled(scale > minimum + 1e-8); plus.setEnabled(scale < MAX_ZOOM - 1e-8)
        def set_roi(roi: ROI) -> None:
            nonlocal syncing
            syncing = True
            selector.roi = roi.clipped(selector.image.shape); left, top, right, bottom = selector.roi.bounds(selector.image.shape); frame.setRect(QRectF(0, 0, right-left, bottom-top)); frame.setPos(left, top)
            (xr, yr) = centre_ranges(selector.image.shape, selector.roi.width, selector.roi.height); x_box.setRange(*xr); y_box.setRange(*yr); x_box.setValue(selector.roi.center_x); y_box.setValue(selector.roi.center_y); syncing = False
            readout.setText(f"ROI centre: ({selector.roi.center_x:.1f}, {selector.roi.center_y:.1f})\nBounds: x={left}:{right}, y={top}:{bottom}\nOutput size: {selector.roi.width} × {selector.roi.height} px")
        def update_from_frame() -> None:
            if not syncing: set_roi(ROI(frame.pos().x()+frame.rect().width()/2, frame.pos().y()+frame.rect().height()/2, int(frame.rect().width()), int(frame.rect().height())))
        def apply_size() -> None:
            if b512.isChecked(): size = (512, 512)
            elif b1024.isChecked(): size = (1024, 1024)
            else: size = (width_box.value(), height_box.value())
            set_roi(ROI(selector.roi.center_x, selector.roi.center_y, *size))
        def select_custom(checked: bool) -> None:
            width_box.setEnabled(checked); height_box.setEnabled(checked)
            if checked: apply_size()
        b512.toggled.connect(lambda checked: checked and apply_size()); b1024.toggled.connect(lambda checked: checked and apply_size()); custom.toggled.connect(select_custom); width_box.valueChanged.connect(lambda _ : custom.isChecked() and apply_size()); height_box.valueChanged.connect(lambda _ : custom.isChecked() and apply_size())
        x_box.valueChanged.connect(lambda value: not syncing and set_roi(ROI(value, selector.roi.center_y, selector.roi.width, selector.roi.height))); y_box.valueChanged.connect(lambda value: not syncing and set_roi(ROI(selector.roi.center_x, value, selector.roi.width, selector.roi.height)))
        centre_button.clicked.connect(lambda: set_roi(ROI(*selector.reference_center_xy, selector.roi.width, selector.roi.height)))
        def save_centre() -> None:
            selector.reference_center_xy = (selector.roi.center_x, selector.roi.center_y)
            status.setText(f"Saved beam centre to:\n{save_beam_center(selector.settings_path, selector.reference_center_xy)}" if selector.settings_path else "Beam centre updated for this selector. Supply settings_path to persist it.")
        save_button.clicked.connect(save_centre); minus.clicked.connect(lambda: (view.zoom_out(), view.setFocus())); plus.clicked.connect(lambda: (view.zoom_in(), view.setFocus())); fit_button.clicked.connect(lambda: (view.fit_image(), view.setFocus())); one.clicked.connect(lambda: (view.set_scale_at(1.0), view.setFocus())); buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject)
        set_roi(self.roi); width_box.setEnabled(custom.isChecked()); height_box.setEnabled(custom.isChecked()); view.fit_image(); view.setFocus()
        try:
            dialog.show(); dialog.raise_(); dialog.activateWindow(); self.accepted = dialog.exec() == QDialog.DialogCode.Accepted
        finally:
            dialog.close(); dialog.deleteLater(); _ROI_QT_APP.processEvents()
        return self


def _roi_worker_main(config_path: str | Path) -> int:
    """Worker entry point; return zero only after writing a result file."""
    try:
        config = json.loads(Path(config_path).read_text(encoding="utf-8")); image = np.load(config["image_path"], allow_pickle=False)
        selector = ROISelector(image[np.newaxis], 0, config["reference_center_xy"], config["settings_path"], config["input_path"], auto_show=False)
        selector.roi = ROI(**config["initial_roi"]).clipped(image.shape); selector._show_dialog()
        Path(config["result_path"]).write_text(json.dumps({"accepted": selector.accepted, "roi": selector.roi.__dict__}), encoding="utf-8")
        return 0
    except Exception as error:
        print(f"ROI selector worker error: {error}", file=sys.stderr); return 1


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--roi-worker":
    raise SystemExit(_roi_worker_main(sys.argv[2]))
