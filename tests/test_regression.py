"""Regression snapshots for the rendering pipeline.

The unit tests check properties — monotone, identity, in-range. Those pass
just as happily after a change that quietly shifts every rendered photo, which
is exactly the failure this project has already had twice (a rolloff that
darkened midtones by 20%, a mono mix that did nothing). These tests pin the
actual numbers.

Each snapshot is a small downsampled render plus summary statistics, compared
with a tolerance of about one 8-bit level, so ordinary float and library
jitter passes and a real change in the look does not.

To re-bless after an intentional change, check the diff is what you meant and
run:

    PHOTOPIPE_UPDATE_GOLDEN=1 pytest tests/test_regression.py
"""

import json
import os

import numpy as np
import pytest

from photopipe import recipe as R, imageops as ops, matchlook

GOLDEN = os.path.join(os.path.dirname(__file__), "golden", "render.json")
UPDATING = os.environ.get("PHOTOPIPE_UPDATE_GOLDEN") == "1"
TOL = 1.5 / 255.0          # ~one 8-bit level
THUMB = 6                  # snapshot grid per side


def synthetic_scene(h=72, w=96):
    """A deterministic scene-linear frame that exercises the whole engine.

    Built rather than loaded so the suite needs no photos: a horizontal
    exposure ramp reaching well past white for the shoulder, vertical hue
    sweep for the colour stages, and a dark corner plus a bright corner so
    vignette and split-tone have something to bite on.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x = xx / (w - 1)
    y = yy / (h - 1)
    ramp = 0.02 + 2.2 * x ** 2                    # 0.02 .. 2.2 linear
    r = ramp * (0.55 + 0.45 * np.cos(2 * np.pi * y))
    g = ramp * (0.55 + 0.45 * np.cos(2 * np.pi * (y - 1 / 3)))
    b = ramp * (0.55 + 0.45 * np.cos(2 * np.pi * (y - 2 / 3)))
    return np.stack([r, g, b], axis=2).astype(np.float32)


def digest(img):
    """Downsampled grid plus statistics — small enough to eyeball in a diff."""
    h, w = img.shape[:2]
    ys = np.linspace(0, h - 1, THUMB).astype(int)
    xs = np.linspace(0, w - 1, THUMB).astype(int)
    grid = np.clip(img[np.ix_(ys, xs)], 0, 1)
    return {
        "grid": [[[round(float(v), 4) for v in px] for px in row] for row in grid],
        "mean": [round(float(img[:, :, c].mean()), 5) for c in range(3)],
        "std": [round(float(img[:, :, c].std()), 5) for c in range(3)],
        "p01": round(float(np.percentile(img, 1)), 5),
        "p99": round(float(np.percentile(img, 99)), 5),
    }


def compare(name, got, golden):
    assert name in golden, f"no snapshot for {name}; re-bless with PHOTOPIPE_UPDATE_GOLDEN=1"
    want = golden[name]
    a = np.asarray(got["grid"], dtype=np.float64)
    b = np.asarray(want["grid"], dtype=np.float64)
    assert a.shape == b.shape, f"{name}: snapshot shape changed"
    worst = float(np.abs(a - b).max())
    assert worst <= TOL, (
        f"{name}: render moved by {worst:.4f} (> {TOL:.4f}). "
        f"mean {got['mean']} vs {want['mean']}. "
        "If this was intentional, re-bless with PHOTOPIPE_UPDATE_GOLDEN=1")
    for key in ("mean", "std"):
        assert np.allclose(got[key], want[key], atol=TOL), f"{name}: {key} moved"


def build_snapshots():
    scene = synthetic_scene()
    snaps = {}

    # Every shipped recipe, end to end.
    for r in R.load_all(R.PACKAGED_RECIPE_DIR):
        snaps[f"recipe/{r.slug}"] = digest(R.apply_recipe(scene, r, seed=1234))

    # Individual primitives, so a failure points at the operator rather than
    # leaving five recipes red at once.
    srgb = ops.linear_to_srgb(np.clip(scene, 0, 1))
    snaps["op/rolloff"] = digest(ops.highlight_rolloff(scene, 1.75, knee=0.6) / 2.2)
    snaps["op/auto_exposure"] = digest(np.clip(ops.auto_exposure(scene) / 3.0, 0, 1))
    snaps["op/curve"] = digest(ops.apply_curve(
        srgb, [(0, 0.03), (0.25, 0.22), (0.5, 0.52), (0.75, 0.8), (1, 0.98)]))
    snaps["op/hsl"] = digest(ops.apply_hsl(
        srgb, {"blue": {"hue": -12, "sat": 20, "lum": -8},
               "green": {"sat": -25}}))
    snaps["op/matrix"] = digest(ops.color_matrix(
        srgb, [[1.2, -0.3, 0.1], [0.1, 0.55, 0.35], [-0.02, 0.11, 0.91]]))
    snaps["op/vignette"] = digest(ops.vignette(srgb, 0.09, -0.33))
    snaps["op/split_tone"] = digest(ops.split_tone(
        srgb, shadows=(-5, 0, 7), highlights=(6, 2, -4)))
    snaps["op/monochrome"] = digest(ops.monochrome(srgb, mix=(0.45, 0.45, 0.10)))
    snaps["op/clarity"] = digest(ops.clarity(srgb, 0.25))
    snaps["op/grain"] = digest(ops.grain(srgb, 0.55, size=1.4, seed=7))

    # The measurement path: a known transform must be recovered the same way
    # every time, or a measured recipe silently changes meaning.
    target = ops.color_matrix(
        ops.apply_curve(srgb, [(0, 0.02), (0.5, 0.55), (1, 0.97)]),
        [[1.1, -0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.05, 0.95]])
    curves = matchlook.measure_curves([(srgb, target)])
    snaps["measure/curves"] = {
        "grid": [[[round(float(v), 4) for v in p] for p in curves[c]]
                 for c in ("red", "green", "blue")],
        "mean": [round(float(np.mean([p[1] for p in curves[c]])), 5)
                 for c in ("red", "green", "blue")],
        "std": [round(float(np.std([p[1] for p in curves[c]])), 5)
                for c in ("red", "green", "blue")],
        "p01": 0.0, "p99": 0.0,
    }
    return snaps


@pytest.fixture(scope="module")
def snapshots():
    return build_snapshots()


@pytest.fixture(scope="module")
def golden(snapshots):
    if UPDATING:
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        with open(GOLDEN, "w") as fh:
            json.dump(snapshots, fh, indent=1, sort_keys=True)
        pytest.skip(f"re-blessed {len(snapshots)} snapshots -> {GOLDEN}")
    if not os.path.exists(GOLDEN):
        pytest.fail(f"{GOLDEN} missing; create it with PHOTOPIPE_UPDATE_GOLDEN=1")
    with open(GOLDEN) as fh:
        return json.load(fh)


def test_no_snapshot_was_silently_dropped(snapshots, golden):
    assert set(snapshots) == set(golden), (
        f"snapshot set changed: added {sorted(set(snapshots) - set(golden))}, "
        f"removed {sorted(set(golden) - set(snapshots))}")


@pytest.mark.parametrize("name", sorted(build_snapshots()))
def test_render_matches_snapshot(name, snapshots, golden):
    compare(name, snapshots[name], golden)


def test_scene_itself_is_stable():
    """If the fixture drifts, every other snapshot is meaningless."""
    s = synthetic_scene()
    assert s.shape == (72, 96, 3)
    assert float(s.min()) == pytest.approx(0.002, abs=1e-4)     # deep shadow
    assert float(s.max()) == pytest.approx(2.220, abs=1e-3)     # well over white
    assert float(s.mean()) == pytest.approx(0.41646, abs=1e-4)
