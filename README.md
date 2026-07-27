# PelletLabCineViewer

Program to view Phantom `.cine` files from the pellet lab and make calibrated
measurements — lengths, areas, pellet speed, and cylinder fits — inside
[napari](https://napari.org).

![Example GUI](images/Overview_Image.png)

---

## Contents

- [What it does](#what-it-does)
- [Environment setup](#environment-setup)
  - [Option A — pyenv (recommended)](#option-a--pyenv-recommended)
  - [Option B — plain venv](#option-b--plain-venv)
  - [Option C — conda](#option-c--conda-apple-silicon-friendly)
- [Running](#running)
- [Using the tools](#using-the-tools)
- [The cylinder fit](#the-cylinder-fit)
- [Repository layout](#repository-layout)
- [Troubleshooting](#troubleshooting)

---

## What it does

1. Reads `.cine` files (Vision Research Phantom cameras) via `pycine`.
2. Calibrates pixels → millimetres from a line of known length.
3. Measures lengths and areas from shapes drawn on the video.
4. Tracks the pellet between two frames to get its speed.
5. Fits a parametric cylinder to a tilted, rotated pellet to recover its true
   diameter, length, volume, and mass.
6. Exports polygon vertices for downstream analysis.

---

## Environment setup

Requirements: **Python 3.11 or 3.12**, and a machine with a working OpenGL
stack (napari renders through Qt + vispy, so it needs a real display — not a
bare SSH session).

> Python 3.10 is no longer supported by napari ≥ 0.8, and 3.13 support across
> the whole dependency stack is still patchy. 3.11 is the safe choice, and is
> what `.python-version` pins.

### Option A — pyenv (recommended)

`pyenv` keeps the lab's Python version separate from the system Python, so an
OS update can't quietly break the viewer.

**Install pyenv**

macOS:

```bash
brew update
brew install pyenv pyenv-virtualenv
```

Linux:

```bash
curl -fsSL https://pyenv.run | bash
```

Then add it to your shell — for zsh (the macOS default), append to `~/.zshrc`;
for bash, append to `~/.bashrc`:

```bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

Restart the terminal, then confirm with `pyenv --version`.

**Build the environment**

```bash
git clone https://github.com/ORNL-Fusion/PelletLabCineViewer.git
cd PelletLabCineViewer

pyenv install 3.11.9              # the version pinned in .python-version
pyenv virtualenv 3.11.9 pellet    # a named virtualenv called "pellet"
pyenv local pellet                # auto-activates whenever you cd in here

python -m pip install --upgrade pip
pip install -e .                  # installs deps + the pellet-viewer command
```

`pyenv local pellet` writes the environment name into `.python-version`, so the
right interpreter activates automatically every time you enter the directory —
no `source activate` to forget.

> On macOS, `pyenv install` compiles CPython from source and needs the command
> line tools: `xcode-select --install`.

### Option B — plain venv

If you already have Python 3.11+ installed and don't want another version
manager:

```bash
git clone https://github.com/ORNL-Fusion/PelletLabCineViewer.git
cd PelletLabCineViewer

python3.11 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
pip install -e .
```

To leave the environment: `deactivate`.

### Option C — conda (Apple Silicon friendly)

napari's own docs recommend conda-forge on arm64 Macs, because the Qt builds
there are better behaved:

```bash
conda create -n pellet -c conda-forge python=3.11 napari pyqt
conda activate pellet
pip install -e .                  # picks up pycine and the rest
```

---

## Running

Any of these work once the environment is active:

```bash
pellet-viewer                                   # console script (after pip install -e .)
pellet-viewer /research/csp/lab_videos/15708.cine

python code/pelletVideoViewer.py                # plain script, no install needed
python code/pelletVideoViewer.py "/path/to/file.cine"
```

If you skipped `pip install -e .`, install the dependencies directly with
`pip install -r requirements.txt` first.

---

## Using the tools

The dock widget on the right has one tab per job:

| Tab | What it's for |
| --- | --- |
| 📏 **Calibrate** | Draw a line across an object of known size, type its real length, hit *Set calibration*. **Do this first** — every other measurement depends on it. |
| 📐 **Measure** | Lines and areas from shapes drawn on the current frame. |
| ⚡️ **Speed** | Mark the pellet in two frames; speed comes from the displacement, frame gap, and frame rate. |
| 🔸 **Polygons** | Free-hand outlines; prints vertices for analysis elsewhere. |
| 🧪 **Cylinder** | Parametric cylinder overlay — see below. |
| 📋 **Results** | Running log; copy to clipboard. |

---

## The cylinder fit

A pellet is a cylinder, but it is almost never square to the camera. Under
orthographic projection a tilted cylinder's silhouette is **not** a rectangle —
it is a rectangle capped by two half-ellipses:

```
width across the axis  = D                    (unaffected by tilt)
projected length       = L·cos(t) + D·sin(t)
end caps               = ellipses, semi-axes (D/2)·sin(t) along the axis
                                             (D/2)        across it
```

Fitting a plain rectangle to a tilted pellet therefore gets the length wrong,
and the volume error grows roughly as `1/cos(t)`. The **Cylinder** tab drives
an overlay from six numbers — centre row/col, diameter, true length, in-plane
rotation, out-of-plane tilt — and reports diameter, length, volume, and mass
(volume × density; defaults to 0.2 g/cm³ for solid D₂).

Workflow:

1. Calibrate.
2. **Auto-fit frame** — Otsu threshold plus a minimum-area rectangle gets
   centre, diameter, length, and angle close on the first try.
3. Nudge **tilt** until the drawn end-cap curvature matches the pellet's.
4. Read the numbers; **Copy** to paste into the log.

`Fit from selected shape` does the reverse: select any shape you have already
drawn and it back-solves the parameters from its minimum-area rectangle.

Two caveats: the auto-fit assumes the pellet is the largest blob in the frame,
and tilt is genuinely unrecoverable if the ends are occluded or motion-blurred.
In that case treat the fitted length as a lower bound — the diameter stays
reliable regardless.

---

## Repository layout

```
PelletLabCineViewer/
├── code/
│   ├── pelletVideoViewer.py   # napari viewer + dock widget (entry point)
│   └── pellet_cylinder.py     # cylinder geometry + Cylinder tab
├── images/
├── pyproject.toml             # packaging + dependencies
├── requirements.txt           # same deps, for a plain pip install
├── .python-version            # pyenv pin (committed on purpose)
└── README.md
```

`code/` is deliberately **not** a Python package — a package named `code` would
shadow the standard library's `code` module. Instead `pelletVideoViewer.py`
puts its own directory on `sys.path` at import time, so sibling modules resolve
whether you run it as a script, as an installed console command, or import it
from a notebook. `pyproject.toml` exposes both files as top-level modules via
`package-dir = {"" = "code"}`.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'pellet_cylinder'`**
You're running an old copy of `pelletVideoViewer.py`, or the two files aren't
in the same directory. They must sit side by side in `code/`. The viewer still
launches without it — the Cylinder tab simply disappears and a warning prints.

**Qt backend errors / blank window**
napari 0.8 defaults to PyQt6 and drops PyQt5 (PyQt5 is end-of-life and slated
for removal from napari in Q4 2026). If you need Qt5 while migrating:

```bash
pip uninstall PyQt6 PyQt6-Qt6 PyQt6-sip
pip install "napari[pyqt5]<0.8"
```

`qtpy` smooths over most Qt5-vs-Qt6 differences, including the unscoped enum
access this code uses (`Qt.AlignCenter`, `QTabWidget.North`). If you do hit an
`AttributeError` on one of those, switch to the scoped form
(`Qt.AlignmentFlag.AlignCenter`).

**`.cine` file won't open**
`pycine` is lightly maintained and can trip over NumPy 2. If `read_header` or
`read_frames` raises, try `pip install "numpy<2"` in the environment, or the
alternative reader `cinereader`.

**Nothing renders over SSH**
napari needs a GPU/display. Use X11 forwarding with indirect GLX, a VNC
session, or just run it on the workstation.

**`.DS_Store` keeps reappearing in `git status`**
It is now in `.gitignore`, but one copy is already tracked in the repo history.
Untrack it once:

```bash
git rm --cached .DS_Store
git commit -m "Stop tracking .DS_Store"
```

To stop macOS creating them on network shares altogether:

```bash
defaults write com.apple.desktopservices DSDontWriteNetworkStores -bool true
```
