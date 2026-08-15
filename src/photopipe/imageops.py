"""Colour and tone primitives for the look engine.

Everything here takes and returns float32 RGB. Two working spaces are used:
scene-linear (unbounded, straight off the raw) for exposure-like operations,
and display-referred sRGB in [0,1] for everything a curve touches.
"""

import numpy as np
import cv2


# --- transfer functions ----------------------------------------------------

def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92,
                    1.055 * np.power(x, 1.0 / 2.4) - 0.055).astype(np.float32)


def srgb_to_linear(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92,
                    np.power((x + 0.055) / 1.055, 2.4)).astype(np.float32)


def luminance(rgb):
    return (0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1]
            + 0.0722 * rgb[:, :, 2]).astype(np.float32)


# --- scene-linear operations -----------------------------------------------

def auto_exposure(lin, headroom_pct=99.5, highlight_target=1.60,
                  key=0.10, key_weight=0.25, max_lift=2.5, max_gain=256.0):
    """Normalise a scene-linear image to a sane starting exposure.

    Two anchors, blended geometrically:

    * a highlight anchor (percentile -> just over white) which alone renders a
      night scene as a black frame with a few bright lamps, because a large
      dark sky drags nothing but the lamps into range;
    * a log-average key anchor, the classic Reinhard estimate, which alone
      blows a night scene apart — on a real frame here it asked for 90x gain.

    Weighting mostly toward the highlight anchor and capping the extra lift at
    `max_lift` keeps daylight honest while opening night frames enough to see
    the subject. Values above 1.0 are intentional: the recipe's highlight
    rolloff is what brings them back, so specular lamps stay bright.
    """
    lum = luminance(lin)
    p = float(np.percentile(lum, headroom_pct))
    if p <= 1e-9:
        return lin

    gain_hi = highlight_target / p
    logavg = float(np.exp(np.mean(np.log(lum + 1e-6))))
    gain_key = key / max(logavg, 1e-9)

    w = float(np.clip(key_weight, 0.0, 1.0))
    gain = (gain_hi ** (1.0 - w)) * (gain_key ** w)
    gain = float(np.clip(gain, gain_hi / max_lift, gain_hi * max_lift))
    return (lin * min(gain, max_gain)).astype(np.float32)


def exposure(lin, stops):
    if stops == 0:
        return lin
    return (lin * (2.0 ** stops)).astype(np.float32)


def temp_tint(lin, temp=0.0, tint=0.0):
    """Cheap creative white balance. temp>0 warms, tint>0 pushes magenta.

    Applied as per-channel gains in linear light, which is where a real WB
    change happens, so it stays neutral through the rest of the chain.
    """
    if temp == 0.0 and tint == 0.0:
        return lin
    r = 1.0 + 0.30 * temp
    b = 1.0 - 0.30 * temp
    g = 1.0 - 0.20 * tint
    out = lin.copy()
    out[:, :, 0] *= r
    out[:, :, 1] *= g
    out[:, :, 2] *= b
    return out.astype(np.float32)


def highlight_rolloff(lin, white=1.0, knee=0.6):
    """A real shoulder: identity below `knee`, compressed above it.

    `white` is the linear value that lands on 1.0.

    The obvious implementation — plain extended Reinhard over the whole range —
    is not a shoulder at all but a global tone compressor: at white=1.6 it
    darkens a 0.5 midtone by 20%, which silently underexposes every recipe that
    asks for any rolloff. Anchoring the curve at a knee keeps the midtones
    exactly where the tone curve put them and spends the compression only on
    the highlights that actually need it.

    Above the knee the remaining range is mapped with a rational curve whose
    slope is 1 at the knee, so there is no visible crease where it takes over.
    """
    if white <= 1.0:
        return np.clip(lin, 0.0, None)

    x = np.clip(lin, 0.0, None)
    k = float(np.clip(knee, 0.0, 0.95))
    span_in = white - k          # input headroom to compress
    span_out = 1.0 - k           # output room available for it
    if span_in <= 0:
        return x

    c = span_in / span_out - 1.0   # slope-matching constant; c > -1 stays monotone
    # Clamp at 0 before dividing: below the knee we discard the result anyway,
    # and an unclamped negative u hits the pole at u = -1/c for real pixel
    # values (x = 0.03 when white = 1.95), producing inf.
    u = np.clip((x - k) / span_in, 0.0, None)
    shoulder = k + span_out * (u * (1.0 + c) / (1.0 + c * u))
    return np.where(x > k, shoulder, x).astype(np.float32)


# --- curves ----------------------------------------------------------------

def _pchip(xs, ys, xq):
    """Monotone cubic Hermite (Fritsch-Carlson) — a curve that never overshoots."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    n = len(xs)
    h = np.diff(xs)
    delta = np.diff(ys) / h

    d = np.zeros(n)
    d[0], d[-1] = delta[0], delta[-1]
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0:
            d[i] = 0.0
        else:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])

    idx = np.clip(np.searchsorted(xs, xq) - 1, 0, n - 2)
    hh = h[idx]
    t = (xq - xs[idx]) / hh
    t2, t3 = t * t, t * t * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    return (h00 * ys[idx] + h10 * hh * d[idx]
            + h01 * ys[idx + 1] + h11 * hh * d[idx + 1])


def build_curve_lut(points, size=4096):
    """Sample a control-point curve into a dense LUT for np.interp."""
    pts = sorted(points)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    grid = np.linspace(0.0, 1.0, size)
    return grid.astype(np.float32), np.clip(_pchip(xs, ys, grid), 0.0, 1.0).astype(np.float32)


def apply_curve(img, points):
    """Apply an RGB curve in display space."""
    if not points:
        return img
    gx, gy = build_curve_lut(points)
    return np.interp(np.clip(img, 0.0, 1.0), gx, gy).astype(np.float32)


# --- HSL / saturation ------------------------------------------------------

# Fuji-style band centres in degrees, matching the eight-band editors people
# are used to, so recipe numbers read the same way as in a raw developer.
HSL_BANDS = {
    "red": 0.0, "orange": 30.0, "yellow": 60.0, "green": 120.0,
    "aqua": 180.0, "blue": 240.0, "purple": 285.0, "magenta": 320.0,
}
_BAND_WIDTH = 45.0


def _band_weight(hue_deg, centre):
    """Smooth cosine falloff around a hue centre, wrapping at 360."""
    d = np.abs(((hue_deg - centre + 180.0) % 360.0) - 180.0)
    w = np.clip(1.0 - d / _BAND_WIDTH, 0.0, 1.0)
    return (w * w * (3.0 - 2.0 * w)).astype(np.float32)  # smoothstep


def apply_hsl(img, bands):
    """Per-hue-band hue/sat/lum tweaks.

    bands: {"green": {"hue": -12, "sat": -22, "lum": -5}, ...}
    hue in degrees; sat and lum in percent.
    """
    if not bands:
        return img
    hsv = cv2.cvtColor(np.clip(img, 0.0, 1.0), cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    dh = np.zeros_like(h)
    smul = np.ones_like(s)
    vmul = np.ones_like(v)
    for name, adj in bands.items():
        centre = HSL_BANDS[name]
        w = _band_weight(h, centre)
        # Only saturated pixels carry a meaningful hue, so fade the whole band
        # out on near-greys — otherwise noise in shadows picks up colour.
        w = w * np.clip(s * 3.0, 0.0, 1.0)
        if adj.get("hue"):
            dh += w * float(adj["hue"])
        if adj.get("sat"):
            smul *= 1.0 + w * (float(adj["sat"]) / 100.0)
        if adj.get("lum"):
            vmul *= 1.0 + w * (float(adj["lum"]) / 100.0)

    hsv[:, :, 0] = (h + dh) % 360.0
    hsv[:, :, 1] = np.clip(s * smul, 0.0, 1.0)
    hsv[:, :, 2] = np.clip(v * vmul, 0.0, 1.0)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)


def saturation(img, amount=1.0, vibrance=0.0):
    """Global saturation plus a vibrance term that spares already-saturated pixels."""
    if amount == 1.0 and vibrance == 0.0:
        return img
    hsv = cv2.cvtColor(np.clip(img, 0.0, 1.0), cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1]
    if amount != 1.0:
        s = s * amount
    if vibrance != 0.0:
        s = s + vibrance * s * (1.0 - s)
    hsv[:, :, 1] = np.clip(s, 0.0, 1.0)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)


def split_tone(img, shadows=(0, 0, 0), highlights=(0, 0, 0), balance=0.5, strength=1.0):
    """Tint shadows and highlights toward two colours.

    shadows/highlights are signed RGB pushes in roughly -100..100.
    """
    sh = np.asarray(shadows, dtype=np.float32) / 100.0
    hi = np.asarray(highlights, dtype=np.float32) / 100.0
    if strength == 0 or (not sh.any() and not hi.any()):
        return img
    lum = luminance(img)
    # balance slides the crossover point between the two tints
    t = np.clip((lum - balance) / max(balance, 1e-3) * 0.5 + 0.5, 0.0, 1.0)
    w_hi = (t * t * (3.0 - 2.0 * t))[:, :, None]
    w_sh = 1.0 - w_hi
    # Fade both tints out at the extremes so blacks and whites stay clean.
    rolloff = (4.0 * lum * (1.0 - lum))[:, :, None]
    out = img + strength * rolloff * (w_sh * sh + w_hi * hi) * 0.5
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# --- detail ----------------------------------------------------------------

def clarity(img, amount=0.0, radius=None):
    """Large-radius local contrast on luminance only, so colour stays put."""
    if amount == 0.0:
        return img
    h, w = img.shape[:2]
    if radius is None:
        radius = max(3.0, min(h, w) / 200.0)
    lab = cv2.cvtColor(np.clip(img, 0.0, 1.0), cv2.COLOR_RGB2LAB)
    l = lab[:, :, 0]
    blur = cv2.GaussianBlur(l, (0, 0), radius)
    detail = l - blur
    # Taper the boost where L is already near 0 or 100 to avoid halo clipping.
    guard = np.clip(1.0 - np.abs(l / 50.0 - 1.0) ** 3, 0.0, 1.0)
    lab[:, :, 0] = np.clip(l + detail * amount * 2.0 * guard, 0.0, 100.0)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32)


def sharpen(img, amount=0.0, radius=0.8, threshold=0.0):
    """Fine-radius unsharp mask on L, edge-gated to keep flat areas noise-free."""
    if amount == 0.0:
        return img
    lab = cv2.cvtColor(np.clip(img, 0.0, 1.0), cv2.COLOR_RGB2LAB)
    l = lab[:, :, 0]
    blur = cv2.GaussianBlur(l, (0, 0), radius)
    detail = l - blur
    if threshold > 0:
        gate = np.clip((np.abs(detail) - threshold) / max(threshold, 1e-3), 0.0, 1.0)
        detail = detail * gate
    lab[:, :, 0] = np.clip(l + detail * amount, 0.0, 100.0)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32)


def monochrome(img, mix=(0.30, 0.59, 0.11), filter_strength=1.0):
    """Channel-mixer B&W. `mix` is the panchromatic response of the emulsion."""
    m = np.asarray(mix, dtype=np.float32)
    m = m / max(m.sum(), 1e-6)
    lum = luminance(img)
    mixed = (img[:, :, 0] * m[0] + img[:, :, 1] * m[1] + img[:, :, 2] * m[2])
    grey = lum + (mixed - lum) * filter_strength
    return np.repeat(np.clip(grey, 0.0, 1.0)[:, :, None], 3, axis=2).astype(np.float32)


def vignette(img, a1=0.0, a2=0.0):
    """Radial gain: 1 + a1*r^2 + a2*r^4, with r=1 at the frame corner.

    Needed when reproducing a camera's rendering, because the raw developer
    and the camera rarely agree on how much lens falloff to remove — DxO
    strips more of it than the A6700 does, which shows up as a systematic
    edge-versus-centre brightness difference (measured at +5.3 L) that no
    global tone curve can express.
    """
    if a1 == 0.0 and a2 == 0.0:
        return img
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = (((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2) / 2.0
    gain = (1.0 + a1 * r2 + a2 * r2 * r2)[:, :, None]
    return np.clip(img * gain, 0.0, 1.0).astype(np.float32)


def grain(img, amount=0.0, size=1.0, seed=0):
    """Luminance grain, strongest in the midtones like real film."""
    if amount == 0.0:
        return img
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    if size > 1.0:
        small = rng.standard_normal((max(1, int(h / size)), max(1, int(w / size)))).astype(np.float32)
        noise = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        noise = rng.standard_normal((h, w)).astype(np.float32)
    lum = luminance(img)
    weight = (4.0 * lum * (1.0 - lum)) ** 0.5
    out = img + (noise * weight * amount * 0.06)[:, :, None]
    return np.clip(out, 0.0, 1.0).astype(np.float32)
