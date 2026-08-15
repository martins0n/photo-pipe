"""Fit Sony Creative Look slider values that move the camera's own rendering
toward a recipe.

The idea: you already have both ends of the problem on disk — the camera's
SOOC HIF, and the pipeline's render of the same frame. Anything Sony's sliders
can do is a fairly simple tone/colour transform, so we model those sliders,
then solve for the values that best bridge SOOC -> recipe.

This is a *model* of the sliders, not Sony's firmware. The numbers it returns
are a well-founded starting point measured on your own frames rather than a
guess, and the reported residual says how much of the look the camera simply
cannot reach.
"""

import numpy as np
import cv2

from . import imageops as ops

# SciPy is only needed by fit(); importing it lazily keeps it an optional
# dependency so `import photopipe` works with the base install.

# Slider order used throughout; matches the camera's parameter list.
PARAM_NAMES = ["contrast", "highlights", "shadows", "fade",
               "saturation", "clarity", "wb_amber", "wb_green"]

# (low, high) in the camera's own units.
BOUNDS = {
    "contrast": (-9, 9), "highlights": (-9, 9), "shadows": (-9, 9),
    "fade": (0, 9), "saturation": (-9, 9), "clarity": (-9, 9),
    "wb_amber": (-7, 7), "wb_green": (-7, 7),
}

# How much one slider step is worth. These constants are the modelling
# assumption — they set the scale that maps image-space change to slider units.
STEP = {
    "contrast": 1 / 9.0, "highlights": 1 / 9.0, "shadows": 1 / 9.0,
    "fade": 1 / 9.0, "saturation": 1 / 9.0, "clarity": 1 / 9.0,
}
WB_AMBER_GAIN = 0.018   # per step, applied +R / -B in linear light
WB_GREEN_GAIN = 0.014   # per step, applied +G


def _contrast(x, c):
    s = c * STEP["contrast"]
    if s >= 0:
        return (1 - s) * x + s * (x * x * (3 - 2 * x))
    return np.clip(0.5 + (x - 0.5) * (1 + s), 0.0, 1.0)


def _highlights(x, h):
    w = np.clip((x - 0.45) / 0.55, 0.0, 1.0) ** 2
    return np.clip(x + h * STEP["highlights"] * 0.25 * w * (1.0 - x), 0.0, 1.0)


def _shadows(x, s):
    # Weight vanishes at pure black so this opens shadows without lifting the
    # black point — that job belongs to Fade, and keeping them separable is
    # what lets the fit tell them apart.
    w = np.clip((0.55 - x) / 0.55, 0.0, 1.0) ** 2 * np.clip(x * 4.0, 0.0, 1.0)
    return np.clip(x + s * STEP["shadows"] * 0.25 * w, 0.0, 1.0)


def _fade(x, f):
    lift = f * STEP["fade"] * 0.12
    return lift + (1.0 - lift) * x


def apply_look(img, params):
    """Apply modelled Creative Look sliders to display-referred RGB in [0,1]."""
    p = dict(zip(PARAM_NAMES, params))
    out = np.clip(img, 0.0, 1.0)

    # White balance shift, done in linear light where a gain is meaningful.
    a, g = p["wb_amber"], p["wb_green"]
    if a or g:
        lin = ops.srgb_to_linear(out)
        lin[:, :, 0] *= 1.0 + a * WB_AMBER_GAIN
        lin[:, :, 2] *= 1.0 - a * WB_AMBER_GAIN
        lin[:, :, 1] *= 1.0 + g * WB_GREEN_GAIN
        out = ops.linear_to_srgb(np.clip(lin, 0.0, None))

    out = _contrast(out, p["contrast"])
    out = _highlights(out, p["highlights"])
    out = _shadows(out, p["shadows"])
    out = _fade(out, p["fade"])
    out = ops.saturation(out, amount=1.0 + p["saturation"] * STEP["saturation"] * 0.5)
    if p["clarity"]:
        out = ops.clarity(out, p["clarity"] * STEP["clarity"] * 0.30)
    return np.clip(out, 0.0, 1.0)


def align_exposure(img, reference):
    """Rescale `img` to `reference`'s overall exposure.

    Without this the fit is unsolvable: the pipeline's auto-exposure and the
    camera's metering rarely agree, and no Creative Look slider can change
    exposure — so the optimiser slams Highlights and Shadows to their limits
    trying to express a brightness offset, and reports a bad fit for a look
    that actually matches. Exposure is set on the camera separately, so it is
    correct to factor it out here.

    Uses the geometric mean of luminance (the standard key measure), which is
    far less sensitive to a bright sky or a dark night frame than the mean.
    """
    a = ops.srgb_to_linear(img)
    b = ops.srgb_to_linear(reference)
    ka = float(np.exp(np.mean(np.log(ops.luminance(a) + 1e-4))))
    kb = float(np.exp(np.mean(np.log(ops.luminance(b) + 1e-4))))
    return ops.linear_to_srgb(np.clip(a * (kb / max(ka, 1e-6)), 0.0, None))


def _lab(img):
    return cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)


def _loss(params, pairs):
    total = 0.0
    for sooc, target_lab in pairs:
        got = _lab(apply_look(sooc, params))
        # L is 0..100 and a/b are roughly -128..127; scaling a/b down keeps the
        # fit from chasing colour at the expense of tone.
        d = got - target_lab
        total += float(np.mean(d[:, :, 0] ** 2)
                       + 0.5 * np.mean(d[:, :, 1] ** 2 + d[:, :, 2] ** 2))
    return total / max(len(pairs), 1)


def fit(pairs, verbose=True):
    """pairs: list of (sooc_rgb, target_rgb) float arrays, same shape.

    Returns (rounded_params_dict, report_dict).
    """
    try:
        from scipy.optimize import minimize
    except ImportError:  # pragma: no cover
        raise SystemExit("fit-camera needs SciPy — install with: pip install 'photo-pipe[fit]'")

    prepared = [(s, _lab(t)) for s, t in pairs]
    x0 = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    bounds = [BOUNDS[n] for n in PARAM_NAMES]

    baseline = _loss(x0, prepared)
    res = minimize(_loss, x0, args=(prepared,), method="Powell", bounds=bounds,
                   options={"maxiter": 4000, "xtol": 0.05, "ftol": 0.05})

    # The camera only accepts integers, so the honest answer is the rounded
    # one — and it must be scored, not assumed equal to the continuous optimum.
    rounded = np.array([int(round(v)) for v in res.x], dtype=float)
    for i, n in enumerate(PARAM_NAMES):
        lo, hi = BOUNDS[n]
        rounded[i] = float(np.clip(rounded[i], lo, hi))

    report = {
        "baseline_loss": baseline,
        "fitted_loss": _loss(res.x, prepared),
        "rounded_loss": _loss(rounded, prepared),
        "continuous": dict(zip(PARAM_NAMES, res.x)),
    }
    report["closed_pct"] = (1.0 - report["rounded_loss"] / baseline) * 100.0 if baseline else 0.0
    if verbose:
        print(f"    loss {baseline:.1f} -> {report['rounded_loss']:.1f} "
              f"({report['closed_pct']:.0f}% of the gap closed)")
    return dict(zip(PARAM_NAMES, rounded.astype(int))), report
