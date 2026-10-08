# pcseg
A command-line pipeline for segmenting structured terrestrial laser scanning (TLS) / LiDAR point clouds.

- `pcseg run` - prompt-driven: given a text prompt (e.g. `"door, window"`),
  [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) detects
  matching objects across a full 360-degree sweep and
  [SAM2](https://github.com/facebookresearch/sam2) segments them.
- `pcseg auto-segment` - prompt-free: an automatic mask backend (SAM2 by
  default) proposes every object it sees across the sweep, with no prompt
  required; duplicate detections of the same physical object from
  overlapping views are merged in 3D.
- `pcseg pick-segment` - single-point: segments whatever SAM2 finds at one
  given 3D location, the scriptable counterpart of clicking a point in a
  viewer.

Every 2D mask, regardless of which mode produced it, gets back-projected to
3D point indices and the result is exported to a point cloud and/or
preview image.

## Statement of need

Manually segmenting individual objects (doors, windows, furniture, ...) out
of raw TLS point clouds is tedious, and most point cloud viewers (CloudCompare,
Potree, ...) offer no way to drive that segmentation from a text prompt.
`pcseg` combines the equirectangular-panorama projection used by many TLS
scanners with recent open-vocabulary detection (GroundingDINO) and
segmentation (SAM2) models, wrapped in a scriptable, non-interactive CLI
that fits into a batch/automation workflow rather than requiring a GUI
session per scan.

This package extracts the segmentation pipeline that already runs inside a
larger interactive desktop application; it intentionally leaves out that
application's GUI/wizard layer to keep the reusable, testable, scriptable
core small and installable on its own.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate    # Windows (cmd): .venv\Scripts\activate.bat
pip install -e .
./scripts/install_segmentation_backends.sh
```

The virtual environment isn't strictly required, but is recommended,
especially if more than one Python version/install is on the machine
(common on Windows) - installing directly into whichever Python happens to
be first on PATH can mix packages or native DLLs between interpreters (see
the Windows troubleshooting section for a concrete failure mode this
causes). `pcseg-gui` (the companion viewer, separate repo) must be
installed into this SAME environment, not a different one of its own - it
imports `pcseg` in-process (no subprocess), so the two aren't independently
relocatable.

The base `pip install -e .` covers the pure point-cloud pipeline (E57
loading, projection, filtering, LAS/E57/image export) - no GPU or ML
framework required. The segmentation backends (GroundingDINO + SAM2) are a
separate, heavier step handled by `install_segmentation_backends.sh`,
because SAM2 isn't published on PyPI and has to be built from source. The
script:

1. installs the pip-installable segmentation deps (torch, pillow, `groundingdino-py`),
2. pre-fetches GroundingDINO's HuggingFace text encoder (`bert-base-uncased`) so it never needs the network again,
3. fetches GroundingDINO's model config + checkpoint directly (no repo clone needed),
4. clones SAM2 at a pinned commit and installs it with `pip install --no-build-isolation -e .`,
5. downloads the SAM2 checkpoint.

Verified end-to-end (load -> DINO+SAM2 sweep, auto-segment, or single-point
pick -> filter -> LAS/PNG export) on a real outdoor TLS scan, GPU and CPU,
2026-09-06.

`pyproject.toml`'s `segmentation` extra pins upper bounds on `torch`,
`torchvision`, `timm`, `transformers`, and `huggingface_hub` - these are the
libraries GroundingDINO's own (effectively unmaintained) code reaches into
via internal/deprecated APIs rather than stable public ones, so an unrelated
major release of any of them can silently break model loading with no
warning at install time (this actually happened with `transformers` 5.0 -
see the troubleshooting entry below). The rest of pcseg's dependencies stay
unbounded on purpose.

### Windows

`install_segmentation_backends.sh` is a bash script; on Windows, run it under
**WSL2** (Windows Subsystem for Linux) rather than natively. This isn't just
about the script itself - SAM2 is built from source and includes a CUDA C++
extension, which needs a working MSVC + CUDA Toolkit setup to compile natively
on Windows (SAM2 itself only treats this as best-effort, with a slower
Python-only fallback if it fails); under WSL2 it builds the same way it does
on any other Linux machine, and NVIDIA's WSL2 CUDA support passes the GPU
through with no separate driver inside the WSL environment. Install WSL2
(`wsl --install` in an admin PowerShell, one-time), then inside it:

```bash
git clone <this repo> && cd pcseg-cli   # or copy the repo into the WSL filesystem
pip install -e .
./scripts/install_segmentation_backends.sh
```

The base, GPU-free `pip install -e .` (no segmentation backends) does work
natively on Windows - `pye57`, `numpy`, `opencv-python`, `laspy`, etc. all
ship prebuilt Windows wheels - but running the actual `run`/`auto-segment`/
`pick-segment` commands needs the segmentation backends, so WSL2 is the
supported path end-to-end. Not independently verified on Windows/WSL2 (this
project was developed and tested on Linux) - if you hit a WSL2-specific
issue, please open an issue.

**Native Windows (no WSL)**: if you want to try installing directly on Windows
instead - SAM2's own docs "strongly recommend" against this, though it has
been exercised successfully on a real Windows machine as part of this
project's own development - `scripts/install_segmentation_backends.ps1`
mirrors the `.sh` script (same pinned commits/checkpoints) using
`Invoke-WebRequest` instead of `curl`:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate.bat
pip install -e .
.\scripts\install_segmentation_backends.bat
```

If more than one Python version is installed, pin the venv to a specific
one explicitly rather than trusting whichever `python`/`py` resolves to on
PATH by default - `py -3.12` above, or call that version's `python.exe`
directly by its full path if the `py` launcher isn't installed. See the
troubleshooting entry below for what can go wrong otherwise.


`pcseg` itself is MIT-licensed (see `LICENSE`), but it builds directly on
[GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) and
[SAM2](https://github.com/facebookresearch/sam2), both released by their
authors under the Apache License 2.0 - `install_segmentation_backends.sh`
downloads their code/weights separately from this repository, and using
them remains subject to their own license, not `pcseg`'s.

### Using a different model size

By default (and what's verified above) `pcseg` uses the smallest, fastest
release of each model - GroundingDINO SwinT and SAM2.1 Hiera-tiny. Both
projects publish larger, more accurate (but slower, more GPU memory)
alternatives:

| | sizes available | what changes |
|---|---|---|
| GroundingDINO | `tiny` (SwinT, default), `base` (SwinB) | detection accuracy on the text prompt |
| SAM2.1 | `tiny` (default), `small`, `base_plus`, `large` | mask boundary quality |

Fetch a different size's weights with:

```bash
./scripts/install_segmentation_backends.sh --dino-size base --sam-size large
```

This only downloads the weights - it prints the exact
`--dino-config`/`--dino-checkpoint`/`--sam-config`/`--sam-checkpoint` flags
to pass to `pcseg run` / `auto-segment` / `pick-segment` afterward, since
`pcseg` itself keeps defaulting to the tiny weights unless told otherwise
(verified end-to-end with `--sam-size small`, 2026-09-06). There's no way to
know which size is "enough" without trying it on your own scans - bigger
isn't automatically better for a given time budget.

## Usage

```bash
pcseg run \
  --input scan.e57 \
  --prompt "door, window" \
  --dino-channel intensity \
  --sam-channel normals \
  --out result.las \
  --out preview.png
```

The output format is inferred from each `--out` path's extension, and
`--out` can be repeated to produce several outputs from one run:

| Extension        | Produces                                                        |
|-------------------|------------------------------------------------------------------|
| `.las`, `.e57`    | The segmented point cloud, with a per-label `classification` field |
| `.png`, `.jpg`     | A panorama preview with every detected instance highlighted     |

`--dino-channel` and `--sam-channel` independently select which rendered
channel (`rgb`, `intensity`, or `normals`) each model sees - useful when, say,
intensity separates a building facade from vegetation better than RGB does,
while normals make SAM2's mask boundaries cleaner. Both default to `auto`
(the best available channel on the loaded scan).

`--min-score` drops a FINAL instance (after outlier-filtering and merging)
whose score - GroundingDINO's own text-match confidence, averaged across
fragments if `--merge-duplicates` merged more than one into it - is below
this (default `0.0` = keep everything). This is the semantic-confidence
counterpart to `--sam-score-threshold`'s mask-quality one: a low `--min-score`
instance is one DINO itself wasn't confident actually matches the prompt.

Every instance is always cleaned with a per-instance outlier filter first (each
instance judged against only its own points, never the whole scan) - trims
back a mask that leaked onto a spatially disjoint surface (e.g. SAM2 bleeding
from a window onto the wall behind it) to its real object. This is unconditional,
not tied to any flag.

By default, an object visible in more than one sweep view's overlap gets
detected - and counted - once per view. Pass `--merge-duplicates` to fuse
same-label instances that are really overlapping fragments of one physical
object back into a single instance (a union over 3D point-index IoU, same
idea `auto-segment` already uses for its own duplicate proposals - see
below, not a pick-a-winner dedup); off by default since merging can hide
exactly the kind of segmentation mistake worth seeing raw. `--merge-iou`
tunes the overlap threshold (default `0.5`). `--merge-overlap` (default
`0.3`) adds a second, looser rule on top of it (either being satisfied is
enough): merges once the SMALLER fragment is this fraction covered by the
other, regardless of size mismatch - catches the same real object seen
near-complete in one view and only a sliver of it in another, which
`--merge-iou` alone misses since its union balloons relative to the
intersection. **Not yet validated against a real scan**

Run `pcseg run --help` for the full flag reference (sweep resolution,
pitch angles, DINO thresholds, post-filtering).

### Prompt-free: `auto-segment`

```bash
pcseg auto-segment \
  --input scan.e57 \
  --channel rgb \
  --out result.las \
  --out preview.png
```

Proposes every object SAM2's automatic mask generator finds on each sweep
view - no text prompt needed - then merges detections of the same physical
object across overlapping and adjacent views in 3D. This includes stitching
one large surface (e.g. a stretch of pavement) that a single view's crop
boundary cut into fragments, without swallowing compact fully-visible
objects (a window) into whatever bigger surface happens to sit near it -
see `pcseg auto-segment --help` for the merge-tuning flags
(`--merge-iou`, `--border-merge-overlap`, `--surface-merge-distance`,
`--merge-rounds`) and their docstrings in `core/merge.py` for the full
reasoning behind each one.

### Single point: `pick-segment`

```bash
pcseg pick-segment \
  --input scan.e57 \
  --point-xyz 3.0 3.0 -1.0 \
  --keep-all-masks \
  --out result.las \
  --out preview.png
```

There's no mouse in a non-interactive pipeline, so instead of clicking a
point in a viewer, name a real-world 3D coordinate with `--point-xyz`
(the nearest actual scan point is used) or a raw point-cloud index with
`--point-index`, if already known. A single point prompt to SAM2 is often
ambiguous (e.g. a click on a door could mean the door, the door panel, or
the whole wall) - by default only SAM2's best-scoring candidate mask is
kept, but `--keep-all-masks` exports all 3 ranked candidates as separate
classes instead, so you can inspect and pick the right one offline.

## Decimation, view size, and export resolution

Three related-looking flags that each control something different:

- **`--decimation`** sets how many *real* points get loaded from the E57 in
  the first place. This is the only one of the three that affects actual
  optical detail: a coarser decimation means fewer points to build the
  panorama from, so DINO/SAM2 are working from a genuinely blurrier source -
  no other flag can compensate for this, because the detail was never loaded.
- **`--view-size`** sets how large (in pixels) `pcseg` renders each sweep
  view before handing it to the models. It's rarely worth changing:
  GroundingDINO always internally resizes to ~800px and SAM2 (with the
  default config) to exactly 1024x1024, regardless of `--view-size` - so a
  value below ~1024 doesn't hide any detail from the models (they upscale it
  right back), and a value above ~1024-1333 doesn't add any either (they
  downscale it back down) - it only costs extra render time either way. The
  default (1024) already matches SAM2's own working resolution.
- **`--export-decimation`** (see below) only affects the *exported point
  cloud's* density, after segmentation is already done - it cannot recover
  detail `--decimation` didn't capture either.

In short: if detection quality looks poor, lower `--decimation` (more real
points), not `--view-size`.

### Exporting at a finer resolution than you segmented at

Segmentation (especially `auto-segment`'s sweep) gets slower as `--decimation`
gets finer, largely because more points means more work for mask
back-projection and merging - not because the models take longer on a bigger
image (see above). `--export-decimation N` lets you segment fast on a coarse
`--decimation` and still export a denser point cloud: after segmentation,
`pcseg` re-loads the same file at decimation `N` and propagates each point's
classification from the coarse grid cell it falls into (exact nearest-grid-cell
lookup, not a spatial approximation - see `core/classification.py`).

```bash
pcseg auto-segment --input scan.e57 --decimation 4 --export-decimation 2 \
  --out result.las
```

This buys real speed (measured on a 58M-point outdoor scan: decimation 2
end-to-end took ~76s, decimation 4→2 ~41s, decimation 8→2 ~30s, decimation
16→2 ~27s - gains shrink because fixed costs like model loading and the
re-load itself start to dominate). It does **not** buy back detection
quality: a coarser `--decimation` still finds fewer, coarser objects, and at
very heavy decimation (e.g. 16 on that scan) the upsampled classification
shows visible wedge-shaped artifacts radiating from the scan origin - an
artifact of the panorama's azimuth axis being decimated that coarsely, not
random noise. Decimation 4 already looks reasonably clean upsampled;
decimation 16 does not. Treat `--export-decimation` as a way to get a denser
*output* file cheaply, not a way to get a better *segmentation* cheaply.

## Project layout

```
src/pcseg/
  core/        # pure geometry, filtering, classification, mask-merge logic - no Qt, no ML models
  io/          # E57 loading, LAS/E57/image export
  services/    # DINO / SAM2 wrappers and the three pipeline orchestrators (prompted, auto-segment, pick-segment)
  cli.py       # argparse entry point (run, auto-segment, pick-segment)
tests/         # unit tests for the pure core/ logic (no GPU or model weights needed)
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

# pcseg-gui

A 3D viewer for `pcseg` point clouds that can also run `pcseg`'s
prompt-driven detection (DINO+SAM2) or single-point picking directly against
the loaded scan and draw the result back onto the same view - in-process,
via `pcseg.services`, no subprocess and no export/reopen roundtrip. Results
accumulate as a reorderable stack of layers (see
"Layers" below) that you can edit and export. Built for producing figures
and quick interactive exploration; not a persistent-project tool - the
layer stack lives only in this session, until you export it.

## Install

```bash
python -m venv .venv
source .venv/bin/activate    # Windows (cmd): .venv\Scripts\activate.bat
pip install -e ../pcseg-cli   # pcseg itself, if not already installed
pip install -e .
```

**Both must go into the SAME environment** - this app imports `pcseg`
in-process (`pcseg_gui.viewer.viewer_widget` calls straight into
`pcseg.services`, no subprocess), so `pcseg` and `pcseg-gui` living in two
different venvs simply won't see each other. The venv itself isn't strictly
required, but is recommended, especially with more than one Python
version/install on the machine (common on Windows) - see
`pcseg-cli/README.md`'s own Installation section and Windows
troubleshooting entry for why.

On Windows, see `pcseg-cli/README.md`'s "Windows" section for the
segmentation backends (GroundingDINO + SAM2) - WSL2 is the primary
supported path, but a native Windows install has also been exercised
successfully as part of this project's own development (see "Native
Windows (no WSL)" there). PyQt6 itself installs fine natively either way,
so the plain viewer (no segmentation) always works there regardless - just
run `pip install -e .` without the segmentation backends.

## Usage

```bash
pcseg-view scan.e57 --decimation 4
pcseg-view result.las
```

Or launch with no argument and use *File > Open* to pick a file - for an
`.e57` you'll be prompted for two DISTINCT decimation settings, deliberately
kept separate so they can't be confused for one another:

- **Load detail** (1 = full resolution) - the CLI's own `--decimation`: how
  many REAL points get loaded from the file at all. Affects segmentation
  quality - a coarser value is a genuinely blurrier source, not a shortcut.
- **View detail** (1 = show every loaded point) - a purely cosmetic,
  rendering-only subsample of whatever got loaded, for interactive
  performance on a large scan. Segmentation always runs against the FULL
  loaded point set regardless of this value. Also adjustable **live**, at
  any time after loading, via the "View detail" control in the main toolbar
  - no reload needed. Density-adaptive (design review 2026-09-11), not a
  uniform stride: real scans are wildly non-uniform in local density (heavy
  sweep-view overlap can pack far more points into one small area than
  elsewhere), so this thins out the already-dense areas much more than the
  already-sparse ones, which stay close to full detail instead of losing
  the same fraction everywhere a plain "every Nth point" stride would lose.
  Recomputing this can take several real seconds on a very large scan (tens
  of millions of points) - the cursor shows a wait spinner for that
  stretch, and only a genuine change to this value triggers it, not
  Region-of-interest or Frustum culling changing what's currently visible.

`.las` results were already decimated whenever `pcseg` exported them, so
there's no Load-detail prompt for those (View detail still applies).

- **Color**: RGB / intensity / normals for a raw `.e57` scan (whichever the
  scan actually has); a `.las` result shows its true RGB if present, or is
  colored by its per-point `classification` field otherwise (one color per
  segmented object/class, matching `pcseg`'s own PNG previews).
- **Render mode**: *Solid* / *Transparent* / *Clipped* - fades or hides
  points whose normal faces away from the camera, so you can see "through"
  a surface toward what's behind it. Only has an effect where real normals
  exist (a raw `.e57` scan); LAS results have none, so this stays a no-op there.
- **Save Screenshot...** grabs the current 3D view framebuffer and saves it as a PNG.
- **Instance labels** (checkbox, next to Frustum culling) draws a wireframe
  box and a billboarded text label over each instance of whichever layer is
  currently selected in the Layers panel - see [Layers](#layers) below for
  the full explanation and scope. Font size: *View > Label Size...*.
- **ViewCube** (top-right corner of the 3D view) always shows the same rotation as
  the main camera, at a fixed size independent of pan/zoom - each of its 6
  differently-colored, outlined faces (Right/Left/Back/Front/Top/Bottom) highlights
  on hover and, when clicked (press AND release on the same face - a press-drag off
  the cube cancels instead of orbiting the main view through it), snaps the main
  camera to that standard orthogonal view (the cube itself re-orients to match,
  including a clean axis-aligned roll for Top/Bottom rather than whatever
  arbitrary heading was active before the click). Edge/corner clicking (a
  halfway or isometric-style view) isn't implemented - a deliberate scope cut,
  not an oversight; only the 6 faces are clickable. Drawn
  smaller than a naive "fill the corner box" size on purpose - a cube's projected
  extent is larger corner-on than face-on, so it's scaled down enough to stay fully
  inside its allotted region at every rotation, not just face-on.

*View > Background Color...* opens a color picker for the 3D view's
background (remembered across restarts); *View > Reset Background Color to
Default*, right below it, restores the built-in default (a dark neutral gray)
if you want it back. *View > Label Size...* prompts for the Instance labels
overlay's font size (6-48pt, default 16) and applies it live - not persisted
across restarts, unlike background color, since it's a much lower-stakes,
rarely-touched cosmetic tweak.

Run's settings page resets per-field rather than all at once (design review
2026-09-11 - a single page-wide button reset everything even when only one
or two fields had actually changed): a small **↺** icon appears right next
to a field the moment its value differs from its default, and resets just
that one field back. The typed prompt text itself has no default to reset
to - it's user content, not a parameter. The Region of interest box has its
own matching reset button (still page-wide there, since the whole box is
one on/off unit). *Tools > Reset All Settings to Defaults* resets every Run
field and ROI at once, regardless of which per-field icons are currently
showing. Single-point picking (see [Layers](#layers)) has no persistent
settings to reset at all - it always uses the 3D view's own current color
channel and a fixed FOV.

Any "auto" channel choice (the DINO/SAM2 channel dropdowns) shows what it
actually resolves to for the currently loaded scan right next to it
(e.g. "auto (rgb)") - same resolution rule the CLI itself uses (RGB > intensity
> normals, whichever the scan actually has), so the GUI and CLI never disagree
about what "auto" means for a given file. It updates every time a new file loads,
since different scans can resolve differently depending on what they actually have.

### Project files

*File > Save Project As...* saves the current EDITING state - the layer
stack (labels, colors, per-instance included/manual flags, visibility, and
whether a layer's points are hidden from the 3D view - see [Layers](#layers)
below), ROI, and color/render mode - to a `.pcsegproj` file, so a session of
manual curation
(deleting false positives, picking points, merging duplicates) survives
closing the app. Deliberately does NOT duplicate the raw point cloud into
the project file - it references the loaded scan by its file path and the
exact Load detail it was loaded with instead (the standard "project file
points at an external asset" pattern most editing tools use for anything
large), which keeps project files small (typically kilobytes to low
megabytes, dominated by the saved per-instance point index arrays, not the
scan itself).

*File > Open Project...* reloads the referenced scan (through the normal
loading path, in the background, same as *Open scan or result...*) and then
restores everything else. If the recorded file path isn't reachable any
more (moved, renamed, a different machine), it prompts you to locate it -
the project itself still opens once you do. Also sanity-checks the
relocated/reloaded file's size against what was recorded at save time, and
the reloaded point count against what the saved indices actually expect -
both are just warnings (you can proceed anyway), but they exist because a
genuinely different file, or the same file loaded with a different Load
detail, means the saved selections no longer point at the right points.

**Loading a new scan or project over an unsaved one**: *Open scan or
result...* and *Open Project...* both check first whether the currently
loaded scan has any layers on it - if it does, you're asked whether to save
the current project before it gets replaced (Save / Discard / Cancel).
Choosing Save opens the normal *Save Project As...* dialog right there;
cancelling THAT dialog cancels the whole load too, rather than silently
discarding your layers anyway. Nothing is asked when there's nothing to lose
(no scan loaded yet, or a scan with an empty layer stack). *Reload at
Different Detail...* is NOT gated by this - it carries the current layers
over to the newly loaded points instead of discarding them (see below), and
already has its own separate confirmation for that.

Loading a genuinely new scan (confirmed past the prompt above, or with
nothing to lose) always starts from a clean slate: every layer, any
in-progress pick session (armed layer, pending picks), any in-progress
multi-point accumulation, the current 3D selection/hover state, the focus
("Isolate" a selection down to just it, from a layer row's own menu) and the
one-level undo-deletion buffer are all reset - none of it carries any
meaning against a different scan's own points.

### Reloading at a different detail

*File > Reload at Different Detail...* reloads the CURRENT scan's own file
at a different Load detail (e.g. classify at a coarse decimation first,
then switch to full resolution once you like what you're seeing - or the
other way, drop to a coarser load to work faster) and tries to carry the
current Layers stack over to the newly loaded points. Unlike Project files
(which always reopen at the EXACT same decimation the saved indices were
computed against, so they just work), a different decimation samples an
entirely different set of points from the source file - the old point
indices are meaningless against the new set, so there's no valid way to
just reuse them here.

Instead this does a spatial (voxel-grid) transfer: every OLD classified
point gets binned into a voxel remembering which instance it belonged to,
then every NEW point inherits whatever instance owned ITS voxel, if any.
Two instances whose old points shared a voxel resolve the same way overlaps
already do everywhere else in this app - the smaller/more specific instance
wins. This is coarse, not exact: a genuinely fine boundary between two
instances can lose or misassign a thin fringe of points right at the edge,
and an instance can end up with zero transferred points (dropped, not kept
empty - the status bar reports how many transferred vs. were dropped) if
none of its old voxels happen to contain any of the newly loaded points.
Only available for a raw `.e57` scan (a `.las` result has no Load detail of
its own - it was already decimated once, whenever `pcseg` exported it).

## Segmentation

The window is three independent regions side by side, one horizontal
splitter: **Layers** pinned to the far-left edge (what past runs produced -
see [Layers](#layers) below), the **3D view** in the center, and **Run's
settings** on the right (what the next run will do) - reviewing results and
configuring a run stay visually separate from each other AND from the 3D
view itself, rather than either one sharing a container with it. Region of
interest lives as its own floating, collapsible panel on the 3D view itself,
below the ViewCube - see [Region of interest](#region-of-interest) below.
Layers and Settings each keep a fixed-ish width; any
extra window width beyond that goes entirely to the 3D view in the middle,
so a wider window means a bigger 3D view, not wider side panels.

The right-hand panel is **always** Run (DINO+SAM2 prompt-driven detection,
matching `pcseg run`) - no Mode selector, no other segmentation approach to
switch to (design review 2026-09-10: Auto-Segment was dropped from the GUI
entirely, and single-point picking moved out to the Layers panel - see
below - leaving Run as pcseg-gui's one standard, dedicated workflow).
Orbit/pan/zoom/select always work regardless of anything here - they were
never gated by a mode to begin with. Every field has a tooltip explaining it
(sourced from the CLI's own `--help` text).

Prompt sits up front, with a **Sweep geometry** row right below it - a gear
button (⚙) opens its own small settings popup with just **Overlap** (default
20%, the practical equivalent of `--step-deg`: how much neighboring views
overlap) and **FOV** (design review 2026-09-14: these are sensible left at
their defaults day-to-day, so they no longer permanently occupy page space -
the popup keeps them one click away instead). The sweep itself is always the
automatically-computed, overlap-guaranteed view grid `pcseg run
--auto-coverage` uses - there's no manual step/pitch grid mode or separate
coverage toggle to configure in the GUI; scoped directly to the active
Region of interest wedge when one is set (see below), or the scan's own
real-data extent otherwise. DINO/SAM2
channels, the DINO box/text/SAM2-score/min-score thresholds, and the manual
category-order override live in a collapsed **Advanced settings** section
below (click to expand - rarely need changing). **Min score** there drops a
FINAL detection (after outlier-filtering and merging) below a given DINO
text-match confidence (matches `--min-score`, default 0 = off) - the
semantic-confidence counterpart to **SAM2 score threshold** right above it
(that one catches a messy mask BOUNDARY; this one catches DINO itself not
being confident the detection matches the prompt at all). Click **Run** to
sweep and segment every match - **one layer PER DETECTED LABEL**, named
after it: a multi-category prompt (e.g. `window, building`) makes a
"window" layer and a "building" layer, never one mixed-label layer. If a
layer with that exact name already exists (an earlier run, or a manual
"+ New empty layer..." - see [Layers](#layers)), the new instances are
appended into it instead of creating a duplicate. Which label's LAYER ends
up highest in the stack (and so wins any overlap) defaults to whichever has
the most total points - check **Manual order** (in Advanced settings) to
rank the parsed categories yourself instead (top of the list wins).

**Merge overlapping detections** (checkbox, own row below Advanced
settings, above the Run button - deliberately not hidden) fuses same-label
instances that are really overlapping fragments of one physical object seen
from multiple overlapping sweep views back into a single instance (matches
`--merge-duplicates`/`--merge-iou`, same 3D point-index-IoU mechanism
`pcseg auto-segment` uses internally). **On by default in the GUI**
(IoU 0.10) - unlike the CLI's own `--merge-duplicates`, which defaults off
so a first, unfamiliar look at a sweep's raw output isn't already merged;
by the time this reached the GUI it had been tuned and verified through
several rounds of real-data testing, so defaulting to on here matches how
it's actually used day to day. Checking it reveals an **IoU score** field
right below (hidden otherwise, since it does nothing until merging is on)
to tune the overlap threshold. Turn merging off if you're debugging what
the pipeline is finding raw - it can hide exactly the kind of mistake worth
seeing unmerged (two classes fighting over the same points, a mask leaking
onto a neighbor).

**Multi-scale coverage** (checkbox, off by default): splits the normal sweep
grid into THREE tiers at different FOVs, each covering its own share of the
scan and no one else's - every direction is swept exactly once, by whichever
tier fits how far its real points actually are from the scanner. A plain
on/off toggle (design review 2026-09-14) - its own far/near percentile and
FOV-scale settings don't expand inline under the checkbox any more (the
defaults already work well day-to-day); a gear button (⚙) right next to the
checkbox opens them in their own small popup instead, independent of the
checkbox's own state, so tuning them never requires turning the feature off
and back on. A **far** tier, restricted to the scan's own farthest points
(**Far percentile**, default 75 - the farthest quarter) at a smaller FOV
(**Far FOV scale**, default 0.5 - half Sweep geometry's own FOV): a distant
object subtends fewer panorama pixels than a near one, so a tighter zoom
recovers a usable pixel size for DINO to actually resolve it. A **near**
tier, restricted to the scan's own closest points (**Near percentile**,
default 25) at a LARGER FOV (**Near FOV scale**, default 1.3): a close
object is already big/legible, so trading some resolution for more
surrounding context - which often helps disambiguate what an object even IS
- costs comparatively little (not pushed much higher by default: a
rectilinear crop distorts increasingly toward its own edges as FOV grows, so
an overly wide near pass can lose more to that distortion than it gains). A
**middle** tier at Sweep geometry's own plain FOV covers whatever's left
once the far and near thirds are carved out. Percentile-based rather than a
fixed distance, so it scales automatically across very different scan sizes
(a small room vs. a large outdoor site) with no per-scan tuning. Refine
boundary-cut detections (below), if also on, still runs afterward on
everything any tier finds, exactly like it does for the normal sweep. Costs
extra sweep views over a single plain pass - noticeably longer on a scan
with a lot of genuinely distant or close material - but no longer sweeps any
single direction twice.

**Refine boundary-cut detections** (checkbox, right below Multi-scale
coverage, off by default): turns the coarse sweep into a **DINO-only**
box-finding pass - SAM2 never runs there at all, since a coarse-pass mask
would only ever get thrown away the instant the same object gets re-segmented
from a better view anyway. Every box the coarse pass finds gets exactly ONE
real DINO+SAM2 pass instead, on a fresh view re-centered on that box's own
middle and sized to just cover its extent (plus a **Margin** fraction of
padding, floored at **Min FOV** degrees so a small object's re-view still
keeps enough surrounding context, and capped at the sweep's own FOV - a box
already that large still gets re-centered, just not zoomed in further, which
alone still fixes the coarse sweep's own tile-boundary problem: an object
that landed right at a view's edge only showed a cut-off sliver there, but a
fresh, properly centered view gives SAM2 a clean, uncut boundary to trace
regardless of zoom level). A box the refine pass can't confirm at its own
(stricter) bar simply produces nothing - there's no coarse-pass mask left to
fall back on any more, so "not confirmed" means "absent," not "kept at its
old, lower-confidence quality." One real DINO+SAM2 call per detected box, not
two - still costs one full extra pass overall (the coarse pass's own DINO
calls plus this), so a sweep with many detections takes noticeably longer
with this on. Its own **Refine box threshold** (default 0.45, vs. the coarse
pass's DINO box threshold default of 0.3) is what actually enforces that
stricter bar: a zoomed-in, re-centered view naturally raises DINO's own
confidence for the same real object (it fills more of the frame, is less
occluded by whatever sat at the coarse view's own edge), so the refine pass
is judged against its OWN, stricter threshold rather than quietly inheriting
the coarse one and letting more low-quality boxes through than the coarse
pass itself would have.

**Save Coverage Grid Preview...** (button, right above Run): saves a PNG of
the scan's own full 360-degree panorama with the CURRENT sweep geometry
settings' own coverage grid drawn on top - a colored rectangle per tile,
grouped by FOV (the plain sweep's own single FOV, plus Multi-scale
coverage's far/near tiers if that's on, each in its own color with a legend
in the corner) - so you can see exactly what a Run would actually sweep
before spending the time, or GPU memory, running it for real. Respects the
active Region of interest and every sweep-geometry setting (Overlap/FOV in
their own popup, Multi-scale coverage and its own percentile/FOV-scale
popup) -
always a preview of what clicking Run right now would do, never a
separately-tuned computation of its own. Renders on a background thread
(reconstructing the full panorama can take a moment on a large scan); the
save location is asked for FIRST, so cancelling that dialog costs nothing.

Loading a file and running a segmentation both happen on a background
thread, so the UI stays responsive - a progress bar and status text track
what's happening: model-loading messages during first use (e.g. "Loading
Grounding DINO model...") before a sweep even starts, then real
percentage/view-by-view progress for Run (an indeterminate bar for loading
and for a single point-pick's SAM2 call). DINO/SAM2 stay loaded in memory
after their first use in a session so later runs are faster. Only one
load/segmentation runs at a time - the Run button and pick-clicks are
disabled while one is in flight.

**Cancel** (visible only during a running Run sweep) stops it cooperatively:
it takes effect at the next view boundary, not instantly (DINO/SAM2
inference itself can't be interrupted mid-call), and no layer is added for
a cancelled run. A single point-pick's SAM2 call isn't sweep-based, so it
has no Cancel hook.

### Sweep debug images

**Tools > Save Sweep Debug Images** (off by default): saves one annotated
PNG per view processed during a Run - box rectangles with their label and
score, plus a translucent colored fill for each kept mask wherever a mask
actually exists, in the SAME colors those detections get in the 3D view, so
a debug image and the object it became are recognizably linked. A view that
found nothing still gets saved (plain, unannotated) - seeing which parts of
the sweep genuinely came up empty is as useful as seeing what they did find.

Turning it on for the first time (**Tools > Debug Images Folder...** also
opens this same picker any time after) asks for a base folder; every
following Run creates its own timestamped subfolder inside it, so repeated
runs never overwrite each other's images. Each run's folder splits into
**prefilter/** (the coarse pass's own views - "prefilter" as in: this pass
is what decides roughly where objects are, before anything closer) and, only
if **Refine boundary-cut detections** was also on for that run, **refine/**
(its own follow-up views) - kept apart so the two stages of the pipeline
stay independently traceable rather than one folder of mixed, harder-to-
follow images. With Refine on, **prefilter/** images show boxes ONLY, no
mask overlay - the coarse pass never runs SAM2 in that mode (see Refine's
own section above) - while **refine/** shows the real box+mask result, same
as always. The chosen folder is remembered across restarts; the on/off
toggle itself is not, so it starts off each session rather than silently
resuming.

### Region of interest

Restricts a sweep to one angular wedge (theta = horizontal, phi = vertical)
instead of the full scan - e.g. crop to just one building so unrelated
classes never mix in from elsewhere. Applies to Run only, not single-point
picking.

Lives as its own floating, collapsible panel drawn directly on the 3D view
(design review 2026-09-11 - it used to sit at the bottom of Run's settings
panel, but it's really a "what's currently visible/swept" control, not
"what to run next," so it moved next to the other view-level control in
that corner: the ViewCube). A small arrow tab, anchored top-right just below
the ViewCube, expands/collapses the controls (the arrow flips direction to
show which state it's in) - collapsed by default so it doesn't sit
permanently over the point cloud. No text label on the tab itself - the
expanded panel's own title ("Crop") already says what it is. The tab keeps
the same small, explicit-colored dark backdrop the expanded panel itself
uses (color parity between the two, by design) rather than a fully
transparent/native-styled button - a plain system-colored version was tried
first and turned out to only be visible against a light 3D-view background,
disappearing against the app's own actual default (a dark gray), since a
native control's color follows the OS theme, not the arbitrary scene
color it's floating over. Its tooltip notes when the crop is actually
active, so collapsing the panel never fully hides that fact even though the
tab looks the same either way. Expanding/collapsing the tab is purely a
display toggle, independent of the panel's own checkbox (see below) - one
controls whether you can SEE the controls, the other whether
the crop is APPLIED.

Each axis is one compact row - a drag-handle range slider flanked by two
small min/max fields, kept in sync both ways - type an exact number or drag
either handle, whichever's more convenient (the fields are plain text
entry, no up/down stepper arrows - the slider is the primary way to adjust
them). Checking the box does two things together: crops what's drawn in the
3D view to that wedge, and - the actual point of the feature - restricts
what Run's own sweep actually covers, via `pcseg.core.projection`'s
`theta_range`/`phi_range`. A yellow outline traced through real scanned
points shows the current wedge.

The sweep itself only restricts where each view is *centered* - a view
centered right at the wedge's edge still sees a bit beyond it (the same
"overlap between neighboring views" behavior the sweep always has) - but
the RESULT doesn't inherit that spillover: every detected instance is hard-
clipped to the exact requested box before it becomes a layer, so "region of
interest" always means nothing outside the box, full stop, regardless of
which view happened to catch a given point.

## Log panel

A dockable "Log" panel (bottom of the window by default, toggle via *View >
Log*) shows the full scrolling history of everything logged this session -
model loading, sweep progress, cancellations, weight downloads - not just the
single current line the status bar shows. It auto-scrolls to the newest
message; scrolling up to read something pauses that until you scroll back
down to the bottom yourself.

### If something crashes

Every session writes to two files under `~/.pcseg-view/logs/` (same path on
Windows - `Path.home()` resolves to `C:\Users\<you>\.pcseg-view\logs\`),
regardless of whether you're running from a console or a packaged/
double-clicked build - the exact path is also the first line the Log panel
above shows at startup, and is printed to the console if one is attached:

- **`pcseg-view.log`** - every logged message for the session, INFO and up,
  including the FULL traceback of any uncaught Python exception (main thread
  or a background Run/load worker). Rotates at 5MB, keeps
  3 backups, so it never grows unbounded across many sessions.
- **`native_crash.log`** - only written to if the app crashes at a level no
  Python exception can see at all (a segfault, a driver/Qt/OpenGL crash) -
  captured via Python's `faulthandler`. Empty in the normal case.

**If you hit a crash, please attach both files** (or whichever one has
something in it - `native_crash.log` when the app just vanished with no
error dialog, `pcseg-view.log` for anything that raised a Python exception)
when reporting it.

## Models

*Tools > Manage Models...* replaces running `install_segmentation_backends.sh`/
`.ps1` in a terminal: shows whether each GroundingDINO size (tiny/base) and
SAM2 size (tiny/small/base_plus/large) is present on disk, with its file size,
and which one of each is currently active for the next Run/single-point-pick
call (default: tiny/tiny, matching the CLI's own default). A
**Download** button next to a missing size runs the install script itself as a
background subprocess (never a reimplementation of its fetch logic) and
streams its output live into the Log panel above; the dialog's controls are
disabled while a download is running, and the row's status refreshes once it
finishes. The active-size dropdowns only allow selecting a size whose weights
are already present - a missing one shows as "(needs download)" and stays
unselectable until fetched.

## Layers

Every Run result becomes its own layer in a Photoshop-style
stack (its own panel on the window's far-left edge), **added on top of
whatever was there before, not replacing it** - running "flag" right after
an earlier "tree" run leaves both layers active and both colors showing at
once, by design (a multi-run composite), until you either delete the old
layer or click **Clear All Layers**.

Each layer is a single compact row, QGIS-style (▶ triangle, visibility
checkbox, color swatch, elided name - hover for the full text):

- **Click** a row to select it - makes it the target of the **instance box
  overlay** (below), of single-point picking, and of anything else scoped to
  "the active layer." Adding a layer auto-selects it, so it defaults to
  "whatever I just got." The selected row gets a highlighted background so
  it's obvious at a glance.
- **Drag** a row to reorder the stack - higher wins any point/pixel a lower
  layer also claims. Click-vs-drag is told apart purely by how far the mouse
  actually moves before release, same convention the 3D view's own
  click-vs-box-select already uses.
- **▶** (only enabled if the layer has instances) expands the row INLINE to
  show its individual instances directly beneath it - see **Instance
  details** below - instead of a separate panel section; expand state is
  remembered per layer.
- **Show/hide** a whole layer with its checkbox, without deleting it - this
  only reverts its points to unclassified/base coloring, they're still
  rendered.
- **Right-click** a row for **"Pick into this layer..."** (single-point
  picking - see below), **"Hide points from 3D view"** (checkable - unlike
  the visibility checkbox above, this removes the layer's own points from
  the 3D view entirely, e.g. to see what's behind/underneath a segmented
  object - the classic "hide by classification" a point-cloud viewer
  usually offers; only takes effect while the layer's own visibility
  checkbox is ALSO on - turning the layer off makes its points reappear as
  unclassified regardless, but the checked state itself is remembered, so
  turning the layer back on restores whichever hide state it had before),
  or **"Delete Layer"** (falls back to selecting the new top of the stack).
- **+ New empty layer...** (button above the list) creates an empty,
  nameable layer with nothing in it yet, then immediately arms picking into
  it - the entry point for starting a brand new hand-picked class rather
  than adding to an existing run's result.

**Single-point picking**: right-click a layer (or **+ New empty layer...**)
-> **"Pick into this layer..."**, or use the **Add Points by Clicking** box
above Run's own Prompt field (right-hand panel) - a more visible entry point
for the exact same workflow, its own AI segmentation mode distinct from Run
Prompt's DINO+SAM2 sweep (plain SAM2 off a clicked point, no text prompt
involved). Its own **Start Picking...** button targets whichever layer is
currently selected; with none selected it asks for a new layer's name first,
same as **+ New empty layer...** does. Either entry point arms IMMEDIATELY -
no settings dialog:
it uses the 3D view's own currently-displayed color channel and a fixed
90-degree FOV, the same standard settings `pcseg pick-segment` defaults to.
Right-click an object anywhere in the 3D view to segment the point under the
cursor - SAM2 always returns its 3 ranked candidate masks for a single point
(a point alone is often ambiguous: door vs. door panel vs. wall), shown as a
small menu right at the click position, one row per candidate with its score
and point count. **Hover** a row to preview it live - a faint colored
overlay on its exact points in the 3D view, so you can compare before
committing; **click** one to queue THAT candidate as a **pending pick** (the
other 2 are discarded, not kept). Pending picks are NOT added to the layer
yet - keep right-clicking to pick several objects into one batch. They show
up highlighted in **orange** in the 3D view (distinct from the blue
instance-selection highlight) and as their own list in a **"Picking into
'...'".** section of the Layers panel, each row removable with its own
**[x]** if you picked the wrong thing.

Two modifier keys change what a right-click does:

- **Shift+right-click** runs the same SAM2 segmentation, but instead of
  queuing the chosen candidate as a new pending pick, it SUBTRACTS that
  candidate's points from every pending pick it overlaps - an eraser for
  over-segmentation. If an earlier pick swallowed more than one object (a
  wall along with its window, say), shift-click the excess region and let
  SAM2 trace its own boundary there rather than trimming it by hand. A
  pending pick left with zero points after the subtraction is dropped
  entirely.
- **Ctrl+right-click** accumulates points for one shared SAM2 multi-point
  prompt instead of segmenting immediately - useful when a single click
  can't disambiguate the object (two touching, similarly-colored panels,
  say). The first Ctrl+right-click anchors the same fixed-FOV view a normal
  pick uses and draws its boundary as a **cyan outline** in the 3D view,
  since every point of one SAM2 prompt has to land on that same rendered
  view; further Ctrl+right-clicks must land inside that outline (rejected
  with a status message otherwise) and just add another point. A plain
  right-click then adds ITS point too and actually runs SAM2 with
  everything accumulated, showing the result the same way a single-point
  pick does. Escape cancels an in-progress multi-point buffer on its own,
  without discarding pending picks already accepted earlier in the session.

- **Finalize** - cleans each pending pick on its own (the same noise filter
  Run always applies: voxel-connectivity + Local Outlier Factor) - never
  merged together, since each pick is its own deliberately-chosen object.
  Lets you preview the cleaned-up result before committing. Greyed out with
  nothing pending; optional otherwise - Accept below runs this itself either
  way.
- **Accept** - finalizes (if not already) and adds every pending pick into
  the target layer for real, under the layer's own most common existing
  label (or its name, for a still-empty layer) and matching its existing
  color, each marked **`[manual]`** once expanded (see **Instance details**
  below) so a hand-picked object stays distinguishable even mixed into an
  otherwise fully-automatic layer - then ends the pick session.
- **Cancel** (or **Escape**) - discards every pending pick and ends the
  session; nothing is added.

Any layer works as a pick target regardless of how it started - picking
into a Run layer works the same as an empty one - and multiple layers are
allowed to share the same name (nothing here treats a layer's name as a
unique key).

**Instance details**: expanding a layer's **▶** lists its individual
detected instances, indented directly beneath it, each with its own
checkbox - uncheck one to temporarily drop just that instance's contribution
without discarding the whole layer. **Click a row** (not its checkbox) to
select/deselect that ONE instance - the exact same selection a 3D-view click
or box-select drives (see **Instance selection** below): it highlights
blue in the 3D view AND the row itself highlights in the panel, whichever
end you click it from. Each row shows its label, a `#n` index, its point
count (e.g. `flag #2 (1,234 pts)`), and **`[manual]`** if it was added via
"Pick into this layer..." rather than being part of the layer's own original
run - a Run layer's instances all share the same label text otherwise (the
raw detected phrase), so without the index/count they'd be identical rows
distinguishable only by a small color swatch. A row can show
**`(unlabeled)`** instead of a real word - this is real GroundingDINO output,
not a display glitch: a box can clear a lowered `--dino-box-threshold` while
no word in the prompt clears `--dino-text-threshold` for it, so DINO itself
returns no phrase for that detection. Doesn't happen at the CLI's own default
thresholds; more likely the lower `--dino-box-threshold` is pushed.

**Instance labels** (checkbox in the main toolbar above the 3D view, next to
Frustum culling - NOT in the Layers box itself, since it's a rendering/
display toggle like Frustum culling and View detail, not layer data) draws
a wireframe box and a billboarded text
label ("label #n - score") over every instance of the *selected* layer only,
in the 3D view - never the whole stack at once, to keep it readable with
many objects. Each box is the axis-aligned extent of that instance's own
real points (not a re-projection of the original 2D detection box, which
has no 3D pose of its own once the sweep view that produced it is gone).
Switching the selected layer swaps which instances are boxed; toggling the
checkbox off removes them entirely. Labels are always white text with a
black outline, not a single fixed color or the instance's own (possibly
similarly-toned) color - stays legible against any background or point
color, dark or light. Font size: *View > Label Size...*.

**Hover preview**: moving the mouse over the 3D view (no button held) tints
whichever instance of the selected layer is under the cursor a light BLUE -
its wireframe box (if **Instance labels** is on) switches to the same blue
too - "if I clicked now, this is what I'd select," live, before you commit
to a click. It also gets a solid-color OUTLINE traced around its own
on-screen shape (a post-process effect, not per-point - see below).
Separate from the (deeper-blue) selection/orange pending-pick highlight
(layers on top without disturbing either), and from the pick-candidate
menu's own per-row hover preview (same mechanism, just fed one specific
candidate instead of whatever's under the live cursor).

**Instance selection** (always on, no toggle needed; Blender-style controls) -
manual cleanup for false-positive detections: grab a bad box, delete it.
LEFT button is selection, ALWAYS - never navigation: click a detection to
select its WHOLE instance, or drag a box to select every instance it touches
(Windows-style rubber-band) - the two are told apart purely by how far the
mouse actually moves before release (a few pixels of tolerance). Clicking an
instance's row in the Layers panel's expanded **Instance details** (above)
selects/deselects it the exact same way - one selection, driven from either
end. **All camera navigation moved to the MIDDLE button instead**, no
modifier needed for the everyday case: plain middle-drag orbits,
Shift+middle-drag pans in the screen plane, Ctrl+middle-drag pans in the
other (mostly-vertical) plane. Double-clicking the middle button snaps the
view's orbit pivot back to the scan's own center - panning drags that pivot
along with the view, so after enough panning/zooming rotation can start
pivoting around a point that's drifted well away from the data; this is the
same "recenter" reset the toolbar's center button already does, just
reachable without leaving the mouse. 3D-view clicks/drags only search whichever
layer is currently selected - same scope as the box overlay and Instance
details. A plain click/drag ADDS the touched instance(s) to the
selection; Shift+click or Shift+drag REMOVES them instead. Clicking or
dragging over empty space (nothing touched) leaves the current selection
untouched - only Escape clears it explicitly, so a stray miss next to what
you're grabbing doesn't cost you the selection you already built up. Selected
instances' points get a blue tint blended in (not a flat overwrite) - they
still read as recognizably their own color underneath. This is
bounding-box/instance-level selection, not point-wise - picking apart
individual stray points within an otherwise-good instance isn't what this is
for (the always-on per-instance outlier filter already handles that during
segmentation itself).

Selected instances also get a solid-color **outline** traced around the
shape their points actually form on screen - a real-time post-process
effect (an offscreen render of just that instance's points, oversized so
neighboring points merge into one near-solid silhouette instead of a dotted
mask, then a single-pass edge-detect draws the outline around wherever that
silhouette's boundary lands), not a per-point outline (tried first, reported
as reading noisy/messy rather than as one coherent object) or an
edit to the point cloud's own geometry. Selection's outline and hover's own
(above) can both show at once, in their respective colors, if a different
instance is being hovered while something else stays selected; if the SAME
instance is both, the selection color wins. Standard, fully portable OpenGL
(shaders + framebuffer objects, same techniques Instance labels/box overlay
and GPU point-picking already use elsewhere in this app) - no
platform-specific dependency, works identically on Linux and Windows.

Toggleable: *View > Instance Outline Effect* - it's a real extra GPU render
pass, and not everyone wants the look or the cost either. Off leaves the
plain hover/selection point-tint (and box overlay, if on) untouched; only
the traced-line effect itself turns off. Remembered across restarts, and
carried over to whatever scan is loaded next (each load creates a fresh
point-cloud renderer internally, which otherwise wouldn't know this
choice was made).

- **Escape** clears the current selection (no data change).
- **Delete** (or Backspace) removes the selected instance(s) from that
  layer entirely - for real, after a confirmation dialog naming how many.
  The points themselves aren't erased from the scan, only that
  instance/layer's classification - they revert to unclassified (or
  whatever a lower layer in the stack still claims them as).
- **Undo Last Deletion** (button, enabled only right after a deletion)
  restores exactly the instance(s) that deletion removed - one level only,
  overwritten by the next deletion. This app has no general Undo yet; this
  is specific to instance deletion, added because a destructive, one-way
  action needs its own safety net regardless.
- **Right-click** in the 3D view with one or more instances selected for
  **"Focus on Selected"** - isolates them, hiding every other point in the
  scan (the same "isolate selection" most 3D tools offer, e.g. to inspect
  an object with nothing else in the way). Stays in effect regardless of
  what you click/select afterward, until you right-click again for **"Show
  All"** to clear it. Independent of, and takes priority over, any layer's
  own "Hide points from 3D view" (above) while active. Right-click is idle
  here otherwise - the same button drives single-point picking instead
  while a pick session is armed (see **Single-point picking** below), never
  both at once.

The 3D view's colors and the exported classification are always the SAME
composed result: on any change (new layer, reorder, visibility, instance
include/exclude), the whole stack is recomputed from scratch via
`pcseg.core.classification.compose_layers` - a point not covered by any
currently-active layer/instance always reverts to unclassified, it's never
left over from a previous state.

**Clear All Layers** removes every layer and reverts to the plain color
mode. **Export Segmentation...** saves the FULL composed stack (not just the
topmost layer) to a `.las`, with a per-point `classification` field - same
as the CLI's own export.

**Export at Full Resolution...** is for a scan currently loaded at a
REDUCED Load detail (see *Open scan or result...*'s own dialog) - a
deliberate tradeoff to keep segmenting fast/light on a large scan, at the
cost of working against fewer of the file's own real points. One click:
reloads the SAME source file at full resolution (Load detail=1), transfers
the current layer stack onto the newly loaded points (the same spatial/
voxel transfer *File > Reload at Different Detail...* uses - not exact,
see [Reloading at a different detail](#reloading-at-a-different-detail)),
then opens the normal Export Segmentation dialog on the result - so the
`.las` you actually save is always the full, undecimated point cloud,
regardless of what you were working against while segmenting. Only applies
to a raw `.e57` scan (a `.las` result has no coarser original to expand
back to); if the scan's already at full resolution, this just exports
directly, no reload needed. Can take noticeably longer and use
significantly more memory than the current view, on a large scan - asks
for confirmation first. The reloaded, full-resolution scan (with its
transferred layers) becomes the new working state afterward, same as a
manual Reload at Different Detail would - it isn't reverted back to the
coarser detail once the export finishes.
