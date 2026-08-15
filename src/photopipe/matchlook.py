"""Measure a camera's rendering and write it out as a recipe.

The inverse of `sonylook`: instead of asking which camera sliders approximate a
recipe, this asks what recipe reproduces the camera. Both ends are already on
disk — the raw and the camera's own JPEG/HEIF of the same frame — so the look
can be measured rather than guessed at.

Method: develop the raw to a neutral rendering, then for each channel measure
where every input level actually lands in the camera's output. Binned medians
across many frames give a transfer curve per channel, which captures tone and
colour balance together. A residual pass in HSL picks up what per-channel
curves structurally cannot: hue-dependent saturation, e.g. muted foliage
alongside an untouched sky.
"""

import numpy as np
import cv2

from . import imageops as ops

# Where control points land. Denser in the shadows, where a curve bends most
# and where an evenly spaced set would smear the toe.
CONTROL_X = [0.0, 0.04, 0.10, 0.18, 0.28, 0.40, 0.55, 0.72, 0.88, 1.0]

_BANDS = ["red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta"]


def neutral_render(linear_rgb, reference):
    """A no-look rendering of the raw, exposure-matched to the camera's output.

    Returns (neutral_srgb, gain). Exposure has to be taken out before fitting
    or it leaks into the curves as a brightness offset and every channel picks
    up the same spurious bend.

    The gain matters just as much as the curves: the pipeline's auto-exposure
    deliberately runs brighter than the camera's metering, so a recipe that
    ignores this feeds the curves inputs shifted up from where they were
    measured. In practice that lands the sky too high on the curve and
    desaturates it — the look's hue comes out right and its colour does not.
    The caller stores the median gain as the recipe's `exposure`.
    """
    lin = np.clip(linear_rgb, 0.0, None)
    ref_lin = ops.srgb_to_linear(reference)
    key = lambda x: float(np.exp(np.mean(np.log(ops.luminance(x) + 1e-4))))
    gain = key(ref_lin) / max(key(lin), 1e-9)
    return ops.linear_to_srgb(lin * gain), gain


def _binned_median(src, dst, bins=96, min_count=40):
    """Median output level for each input level; None where there's no data."""
    idx = np.clip((src * bins).astype(np.int32), 0, bins - 1).ravel()
    flat = dst.ravel()
    order = np.argsort(idx, kind="stable")
    idx_s, flat_s = idx[order], flat[order]
    edges = np.searchsorted(idx_s, np.arange(bins + 1))
    out = np.full(bins, np.nan, dtype=np.float64)
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        if hi - lo >= min_count:
            out[b] = np.median(flat_s[lo:hi])
    return out


def _curve_from_bins(medians, bins=96):
    """Turn binned medians into monotone control points."""
    xs = (np.arange(bins) + 0.5) / bins
    ok = ~np.isnan(medians)
    if ok.sum() < 4:
        return [[0.0, 0.0], [1.0, 1.0]]
    xs, ys = xs[ok], medians[ok]

    # Anchor the ends so the curve spans the full range even when the frames
    # never contained true black or true white. Read the originals into locals
    # first: assigning xs before using xs[0] silently anchors to the new 0.0
    # and flattens the toe at the first populated level, clipping the shadows.
    if xs[0] > 0.02:
        x0, y0 = float(xs[0]), float(ys[0])
        xs = np.r_[0.0, xs]
        ys = np.r_[max(0.0, y0 - x0), ys]      # extend with slope 1 toward black
    if xs[-1] < 0.98:
        x1, y1 = float(xs[-1]), float(ys[-1])
        xs = np.r_[xs, 1.0]
        ys = np.r_[ys, min(1.0, y1 + (1.0 - x1))]

    ys = np.maximum.accumulate(ys)          # enforce monotone
    pts = np.interp(CONTROL_X, xs, ys)
    pts = np.maximum.accumulate(np.clip(pts, 0.0, 1.0))
    return [[round(float(x), 4), round(float(y), 4)] for x, y in zip(CONTROL_X, pts)]


def measure_curves(pairs, bins=96):
    """pairs: [(neutral_rgb, target_rgb)] -> {"red"/"green"/"blue": points}."""
    acc = {c: [] for c in range(3)}
    for neutral, target in pairs:
        for c in range(3):
            acc[c].append(_binned_median(neutral[:, :, c], target[:, :, c], bins))
    curves = {}
    for c, name in enumerate(("red", "green", "blue")):
        stack = np.vstack(acc[c])
        with np.errstate(invalid="ignore"):
            merged = np.nanmedian(stack, axis=0)
        curves[name] = _curve_from_bins(merged, bins)
    return curves


def _apply_rgb_curves(img, curves):
    out = img.copy()
    for idx, name in enumerate(("red", "green", "blue")):
        pts = [(p[0], p[1]) for p in curves[name]]
        out[:, :, idx] = ops.apply_curve(out[:, :, idx], pts)
    return out


def measure_vignette(pairs, curves, bins=24):
    """Fit the residual radial brightness difference left by the curves.

    Least squares of (ratio - 1) on r^2 and r^4, where ratio is the camera's
    luminance over ours, binned by radius. Fitted after the curves because
    that is where it is applied, which keeps the fit self-consistent.
    """
    num = np.zeros(bins); den = np.zeros(bins)
    for neutral, target in pairs:
        got = _apply_rgb_curves(neutral, curves)
        lg = ops.luminance(got); lt = ops.luminance(target)
        h, w = lg.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r2 = (((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2) / 2.0
        # Mid-tones only: near black the ratio explodes, near white it clips.
        m = (lg > 0.12) & (lg < 0.88) & (lt > 0.12) & (lt < 0.88)
        if m.sum() < 500:
            continue
        idx = np.clip((r2[m] * bins).astype(np.int32), 0, bins - 1)
        ratio = lt[m] / np.maximum(lg[m], 1e-4)
        np.add.at(num, idx, ratio)
        np.add.at(den, idx, 1.0)

    ok = den > 200
    if ok.sum() < 6:
        return {}
    x = (np.arange(bins)[ok] + 0.5) / bins
    y = num[ok] / den[ok] - 1.0
    A = np.stack([x, x * x], axis=1)
    try:
        a1, a2 = np.linalg.lstsq(A, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {}
    if abs(a1) < 0.01 and abs(a2) < 0.01:
        return {}
    return {"a1": round(float(np.clip(a1, -0.6, 0.6)), 4),
            "a2": round(float(np.clip(a2, -0.6, 0.6)), 4)}


def _staged(neutral, curves, vignette=None, matrix=None):
    """Our rendering up to the point a given term is fitted."""
    got = _apply_rgb_curves(neutral, curves)
    if vignette:
        got = ops.vignette(got, vignette.get("a1", 0.0), vignette.get("a2", 0.0))
    if matrix is not None:
        got = ops.color_matrix(got, matrix)
    return got


def measure_matrix(pairs, curves, vignette=None, sample=60000, seed=0):
    """Least-squares 3x3 mapping our colours onto the camera's.

    Each row is constrained to sum to 1 so neutrals are untouched, which
    also drops the fit to two free parameters per row:

        target - B = a*(R - B) + b*(G - B)

    Without that constraint the matrix quietly absorbs exposure and white
    balance, and then fights the curves that already handle them.
    """
    rng = np.random.default_rng(seed)
    src, dst = [], []
    for neutral, target in pairs:
        got = _staged(neutral, curves, vignette)
        g = got.reshape(-1, 3); t = target.reshape(-1, 3)
        # Skip clipped pixels: they carry no information about the mapping.
        m = (g.max(axis=1) < 0.97) & (g.min(axis=1) > 0.02) & (t.max(axis=1) < 0.97)
        idx = np.flatnonzero(m)
        if idx.size == 0:
            continue
        if idx.size > sample:
            idx = rng.choice(idx, sample, replace=False)
        src.append(g[idx]); dst.append(t[idx])
    if not src:
        return None

    S = np.concatenate(src).astype(np.float64)
    D = np.concatenate(dst).astype(np.float64)
    A = np.stack([S[:, 0] - S[:, 2], S[:, 1] - S[:, 2]], axis=1)
    rows = []
    for c in range(3):
        y = D[:, c] - S[:, 2]
        try:
            a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None
        rows.append([a, b, 1.0 - a - b])
    m = np.asarray(rows)
    if not np.all(np.isfinite(m)) or np.abs(m).max() > 4.0:
        return None
    if np.abs(m - np.eye(3)).max() < 0.01:      # nothing worth storing
        return None
    return [[round(float(v), 4) for v in row] for row in m]


def measure_hsl(pairs, curves, min_pixels=2000, vignette=None, matrix=None):
    """Per-hue saturation/luminance residual left over after the curves."""
    sums = {b: [0.0, 0.0, 0.0] for b in _BANDS}   # sat ratio, lum ratio, weight
    # Hue is an angle, so it accumulates as a vector — averaging degrees
    # directly breaks across the 0/360 wrap.
    hue_vec = {b: [0.0, 0.0] for b in _BANDS}
    for neutral, target in pairs:
        got = _staged(neutral, curves, vignette, matrix)
        g = cv2.cvtColor(np.clip(got, 0, 1), cv2.COLOR_RGB2HSV)
        t = cv2.cvtColor(np.clip(target, 0, 1), cv2.COLOR_RGB2HSV)
        for band in _BANDS:
            centre = ops.HSL_BANDS[band]
            w = ops._band_weight(g[:, :, 0], centre) * np.clip(g[:, :, 1] * 3, 0, 1)
            m = w > 0.35
            if m.sum() < min_pixels:
                continue
            gs, ts = g[:, :, 1][m], t[:, :, 1][m]
            gv, tv = g[:, :, 2][m], t[:, :, 2][m]
            if gs.mean() > 0.02:
                sums[band][0] += float(ts.mean() / gs.mean()) * m.sum()
            if gv.mean() > 0.02:
                sums[band][1] += float(tv.mean() / gv.mean()) * m.sum()
            sums[band][2] += m.sum()

            dh = np.deg2rad(((t[:, :, 0][m] - g[:, :, 0][m]) + 180.0) % 360.0 - 180.0)
            hue_vec[band][0] += float(np.sin(dh).sum())
            hue_vec[band][1] += float(np.cos(dh).sum())

    hsl = {}
    for band, (s, v, n) in sums.items():
        if n == 0:
            continue
        sat = (s / n - 1.0) * 100.0
        lum = (v / n - 1.0) * 100.0
        hue = np.rad2deg(np.arctan2(hue_vec[band][0], hue_vec[band][1]))
        entry = {}
        if abs(hue) >= 1.5:
            entry["hue"] = int(round(float(np.clip(hue, -30, 30))))
        if abs(sat) >= 3:
            entry["sat"] = int(round(np.clip(sat, -60, 60)))
        if abs(lum) >= 3:
            entry["lum"] = int(round(np.clip(lum, -40, 40)))
        if entry:
            hsl[band] = entry
    return hsl


def error(pairs, curves, hsl=None, vignette=None, matrix=None):
    """Mean Lab error of the measured look against the camera."""
    total = 0.0
    for neutral, target in pairs:
        got = _staged(neutral, curves, vignette, matrix)
        if hsl:
            got = ops.apply_hsl(got, hsl)
        d = (cv2.cvtColor(np.clip(got, 0, 1), cv2.COLOR_RGB2LAB)
             - cv2.cvtColor(np.clip(target, 0, 1), cv2.COLOR_RGB2LAB))
        total += float(np.sqrt(np.mean(d ** 2)))
    return total / max(len(pairs), 1)


def to_recipe(curves, hsl, name, description, order=50, exposure=0.0,
              vignette=None, matrix=None):
    """Assemble the measured pieces into a recipe dict ready to dump as YAML."""
    data = {
        "name": name,
        "description": description,
        "order": order,
    }
    if abs(exposure) >= 0.02:
        data["exposure"] = round(float(exposure), 3)
    data["rgb_curves"] = {k: v for k, v in curves.items()}
    if vignette:
        data["vignette"] = vignette
    if matrix:
        data["matrix"] = matrix
    if hsl:
        data["hsl"] = hsl
    return data
