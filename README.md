# photo-pipe

End-to-end raw pipeline: **`.ARW` → DxO PureRAW → recipe → JPEG**, plus a
comparison contact sheet.

Nothing here consumes pre-made DxO files. The pipeline drives PureRAW itself,
so a run starts from the camera raw and ends at a finished JPEG in one command.

## Install

```bash
pip install -e '.[fit]'        # 'fit' adds SciPy, needed only by fit-camera
```

Also needs [DxO PureRAW 6](https://www.dxo.com/dxo-pureraw/) and `exiftool`
(`brew install exiftool`). On macOS, HEIF decoding uses the built-in `sips`.

## Use

```bash
photo-pipe recipes                                  # list looks
photo-pipe run ~/Images/2026-08-14 --collage        # the whole shoot, all 5 looks
photo-pipe run ~/Images/2026-08-14 --recipes velvia,acros --preview
photo-pipe run photo.ARW --denoise none             # A/B: skip DxO entirely
photo-pipe fit-camera ~/Images/2026-07-30           # camera settings for a look
```

As a library:

```python
from photopipe import develop, recipe

base = develop.develop("DSC0001.dng")            # scene-linear float RGB
look = recipe.load_named(["classic-chrome"])[0]
out  = recipe.apply_recipe(base, look)           # display-referred [0,1]
develop.save_jpeg(out, "out.jpg", exif_from="DSC0001.ARW")
```

Your originals are only ever read. Raws are hardlinked into a cache outside
your photo library (`~/.cache/photo-pipe`, override with `PHOTOPIPE_WORK` or
`--work`), PureRAW writes there, and exports land in `--out` (default `./out`).

---

## Driving DxO PureRAW headlessly

PureRAW ships no documented CLI, but its Lightroom plugin drives the app over
one, and that interface works standalone. The plugin (recovered from
`DxO PureRAW 6.app/Contents/Resources/LightroomPluginData/`) writes a
newline-separated list of source files and runs:

```bash
open -n -b com.dxo-labs.PureRAWv6.standalone --args \
    --as-lightroom-last-settings-plugin \
    --lr-version="14.0" \
    --batch-file="/path/to/list.txt"
```

PureRAW processes the whole batch unattended and writes
`<stem>-DxO_<method>.dng` **next to each source file** — which is exactly why
the pipeline stages hardlinks into `work/01_dxo/` first, instead of letting it
write into your photo library.

Three modes exist; all take the same arguments:

| flag | PureRAW menu equivalent |
|---|---|
| `--as-lightroom-last-settings-plugin` | Process with last settings *(default here)* |
| `--as-lightroom-plugin` | Process instantly |
| `--as-lightroom-preview-plugin` | Process with preview *(opens the UI)* |

The denoise method (DeepPRIME, DeepPRIME XD3, …) comes from PureRAW's own
saved processing preset, not from the command line. Set it once in the PureRAW
UI and every later pipeline run follows it. Output naming reflects whichever
was used, e.g. `DSC06491-DxO_DeepPRIME 3.dng`.

The stage is cached: a source that already has a `-DxO_*` output in `work/`
is skipped, so re-running to iterate on a look costs nothing.

---

## The four stages

1. **Denoise / demosaic / optics** — DxO PureRAW, as above. `--denoise none`
   skips it and develops the ARW directly, which is the honest A/B for judging
   what DxO is actually contributing.
2. **Develop** (`photopipe/develop.py`) — LibRaw decode to **scene-linear**
   RGB. No tone mapping happens here on purpose: every recipe starts from the
   same neutral base, so a comparison is a real comparison. Auto-exposure
   blends a highlight anchor with a log-average key anchor; see `--lift`.
   It deliberately aims *above* white (`highlight_target` 1.6) and lets each
   recipe's shoulder bring the highlights back, which is what keeps daylight
   frames from rendering dark.
3. **Recipe** (`photopipe/recipe.py`) — the look, from a YAML file.
4. **Export + collage** — JPEG with EXIF carried over, then a contact sheet.

Exports are **full resolution** by default; `--preview` does a fast 1600px
pass for iterating on a look, and `--max-dim N` downscales to a fixed long
edge. The collage defaults to lossless PNG (`--collage-out comparison.jpg`
for a smaller, lossy one).

Useful flags: `--max-dim N`, `--preview`, `--quality` (default 97),
`--lift 0..1` (0 protects highlights only, 1 pushes everything to a midtone
key — raise it if night frames render too dark), `--tile` for collage size.

### The camera reference column

If a frame was shot RAW+HEIF, the camera's own `.HIF` is added as the leftmost
collage column, untouched by any recipe — the honest baseline for judging what
the pipeline is adding. HEIF is decoded with macOS's built-in `sips`, which
applies EXIF orientation (so portrait frames come back upright) and is cached
in `work/02_sooc/`.

`--reference auto` (default) uses it when it exists, `hif` requires it, `none`
skips it.

---

## Writing your own recipe

Recipes are plain YAML in `recipes/`. Any `*.yaml` there is picked up
automatically and the filename is the slug you pass to `--recipes`. Copy
[`recipes/_template.yaml.example`](recipes/_template.yaml.example) — it
documents every key with its default — or start from `provia.yaml`.

Every key is optional, so a real recipe can be this short:

```yaml
name: Faded
curve:
  - [0.00, 0.06]     # lifted, matte blacks
  - [0.50, 0.50]
  - [1.00, 0.94]
saturation: 0.85
```

The order of operations is fixed by the engine; a recipe only supplies
numbers. It runs:

| # | stage | keys |
|---|---|---|
| 1 | scene-linear | `exposure`, `temp`, `tint`, `highlight_rolloff` |
| 2 | → display space (sRGB encode) | |
| 3 | tone | `curve`, `rgb_curves` |
| 4 | colour | `hsl`, then `saturation`/`vibrance` **or** `monochrome` |
| 5 | tone split | `split_tone` |
| 6 | detail | `clarity`, `sharpen`, `grain` |

Notes that matter when dialling numbers in:

- **`highlight_rolloff: 1.0` is an exact identity**, so leaving it alone
  really is a no-op. It names the linear value that becomes pure white, so
  larger numbers rescue more overbright detail. It is a true shoulder:
  everything below `highlight_knee` (default 0.6) passes through untouched,
  so raising the rolloff never darkens your midtones.
- **Curves are monotone cubic** (Fritsch–Carlson), so they cannot overshoot
  and invert between your control points.
- **`hsl` bands fade out on near-grey pixels**, which stops shadow noise from
  picking up colour casts.
- **`monochrome` runs after `hsl`**, so `hsl` luminance tweaks behave like
  screwing a coloured filter onto the lens — that is how `acros.yaml` darkens
  the sky.
- **`split_tone` fades out at pure black and white**, keeping blacks clean.

### The five shipped looks

| slug | what it is |
|---|---|
| `provia` | Neutral reference. Gentle S-curve, honest colour. |
| `velvia` | Slide-film punch — deep shadows, saturated greens and blues. |
| `astia` | Softer contrast, warm and forgiving on skin. |
| `classic-chrome` | Muted documentary: lifted dipped blacks, reds to brick, greens to olive. |
| `acros` | B&W, panchromatic response, strong micro-contrast, fine grain. |

## Getting a look in-camera

`fit-camera` solves the inverse problem: which **Sony Creative Look sliders**
come closest to a recipe. It has both ends already — your camera's SOOC HEIF
and the pipeline's render of the same frame — so it models each slider and
fits the values that bridge them, reporting deltas from your camera's current
settings plus a verification strip.

```bash
./pipe.py fit-camera ~/Images/2026-07-30 --recipe classic-chrome
```

By default it fits against the pipeline's own neutral render rather than the
SOOC (`--reference sooc` for the other), which keeps Sony-vs-LibRaw colour
science out of an answer that can only be expressed in slider units.

Worked example with measured values for the A6700:
[docs/a6700-classic-chrome.md](docs/a6700-classic-chrome.md).

---

## Layout

```
src/photopipe/
  cli.py                 the photo-pipe command
  dxo.py                 stage 1 — PureRAW batch driver
  develop.py             stage 2 — raw -> scene-linear, JPEG export, EXIF
  recipe.py              stage 3 — YAML loading + fixed order of operations
  imageops.py            the primitives (curves, HSL, split tone, grain, ...)
  collage.py             stage 4 — contact sheet
  sonylook.py            Creative Look model + solver behind fit-camera
  recipes/*.yaml         bundled default looks
recipes/*.yaml           your looks — these win over the bundled ones
tests/                   pytest suite
~/.cache/photo-pipe/     hardlinked raws + PureRAW output (shared cache)
out/                     JPEGs + comparison.png
```

Recipes are looked up in this order, so editing a look never means touching an
installed package: `$PHOTOPIPE_RECIPES` → `./recipes` → the bundled defaults.

```bash
pytest        # 34 tests, no photos required
```
