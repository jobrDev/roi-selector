# ROI Selector for NumPy Image Stacks

A native, pixel-accurate PyQt6 region-of-interest (ROI) selector for microscopy image stacks. It is designed for use from Jupyter notebooks on macOS and Windows.

The selector displays a single frame from a stack, lets you move and resize an ROI, and returns it for cropping every frame in the stack. The GUI runs in a separate Python process, so it does not interfere with the Jupyter kernel's event loop.

## Features

- Works with NumPy stacks shaped `(n, height, width)`
- Fixed 512 × 512 and 1024 × 1024 presets, plus custom ROI dimensions
- Pixel-accurate draggable ROI frame and centre-coordinate controls
- ImageJ/Fiji-style zoom from the fitted image size up to 3200%
- Full-range min/max display scaling for dim microscopy images; scientific input data is never changed
- Optional persistent beam-centre setting in JSON
- Crop a whole stack using the selected ROI
- One Python module; no TIFF or other third-party image I/O dependency

## Installation

Requires Python 3.10 or later. Install the latest version directly from GitHub:

```bash
python -m pip install "git+https://github.com/jobrDev/roi-selector.git"
```

Optionally, create and activate a virtual environment before running the install command:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install "git+https://github.com/jobrDev/roi-selector.git"
```

Verify the installation from the same notebook kernel or Python interpreter:

```python
import numpy
import PyQt6
from roi_selector_stack import ROISelector, crop_stack
```

## Notebook usage

```python
from roi_selector_stack import ROISelector, crop_stack

# datr has shape (number_of_frames, height, width)
selector = ROISelector(stack=datr, display_index=50)

# The notebook cell resumes after “Use ROI” or “Cancel”.
roi = selector.selected_roi
cropped_stack = crop_stack(datr, roi)
```

`cropped_stack` is a NumPy view where possible, with shape `(n, roi.height, roi.width)`. Coordinates are `(x, y)` = `(column, row)`.

## Controls

| Action | Control |
| --- | --- |
| Move ROI | Drag the orange frame, or edit Centre X / Centre Y |
| Resize ROI | Select a preset or choose Custom and edit Width / Height |
| Pan image | Drag the background |
| Zoom in | Up, `+`, or `=` |
| Zoom out | Down or `-` |
| Fit image | **Fit** button |
| Show actual pixels | **1:1** button |
| Centre ROI on beam position | **Centre on beam** button |
| Store current beam position | **Save current as beam centre** button |

Zoom starts fitted to the available window, never zooms out beyond that fit, and is capped at 3200%. Zooming changes only the view—not the ROI size or the data.

## Optional beam-centre persistence

Pass a JSON settings path to reuse a known beam centre between sessions:

```python
selector = ROISelector(
    stack=datr,
    display_index=50,
    settings_path="roi_settings.json",
)
```

Clicking **Save current as beam centre** writes the selected centre to that file. An explicitly supplied `reference_center_xy` takes priority over a saved value.

```python
selector = ROISelector(
    stack=datr,
    display_index=50,
    reference_center_xy=(868.0, 525.0),
)
```

## Notes for Jupyter and VS Code

The file must be imported from disk; it is not intended to be pasted into a notebook cell. The selection dialog deliberately runs in a child Python process, keeping the notebook kernel free of Qt event-loop handling.

If VS Code/Pylance reports that it cannot resolve `PyQt6`, ensure that the Python interpreter selected in VS Code is the same environment where PyQt6 is installed:

```python
import sys
import PyQt6

print(sys.executable)
print(PyQt6.__file__)
```

Use **Python: Select Interpreter** in VS Code to select that interpreter, then reload the window if necessary.

## Tests

Run the pure-logic test suite with:

```bash
python -m pytest test_roi_selector_stack.py
```

The tests do not require a visible display for the core ROI, cropping, zoom, persistence, and display-scaling logic.

## License

Add the license appropriate for your project.
