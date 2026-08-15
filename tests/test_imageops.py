import numpy as np
import pytest

from photopipe import imageops as ops


def px(*values):
    """A 1x1xN image, the smallest thing the ops accept."""
    return np.array([[list(values)]], dtype=np.float32)


def grey(v):
    return px(v, v, v)


# --- highlight_rolloff -----------------------------------------------------
#
# These pin down the bug that made every recipe render ~2/3 stop dark: the
# rolloff was extended Reinhard over the whole range, which compresses
# midtones as well as highlights.

def test_rolloff_off_is_exact_identity():
    for v in (0.05, 0.2, 0.5, 0.9, 1.0):
        assert ops.highlight_rolloff(grey(v), 1.0)[0, 0, 0] == pytest.approx(v)


def test_rolloff_leaves_midtones_untouched():
    """The regression that mattered: a shoulder must not darken the midtones."""
    for v in (0.05, 0.2, 0.35, 0.5, 0.59):
        out = ops.highlight_rolloff(grey(v), 1.6, knee=0.6)[0, 0, 0]
        assert out == pytest.approx(v, abs=1e-6)


def test_rolloff_maps_white_to_one():
    for white in (1.3, 1.6, 1.95, 2.5):
        out = ops.highlight_rolloff(grey(white), white, knee=0.6)[0, 0, 0]
        assert out == pytest.approx(1.0, abs=1e-4)


def test_rolloff_is_monotone():
    xs = np.linspace(0, 4, 400, dtype=np.float32)
    ys = ops.highlight_rolloff(xs.reshape(-1, 1, 1).repeat(3, axis=2), 1.95)[:, 0, 0]
    assert np.all(np.diff(ys) >= -1e-6)


def test_rolloff_has_no_crease_at_the_knee():
    """Slope must stay ~1 across the knee or the join shows as a band."""
    k = 0.6
    a = ops.highlight_rolloff(grey(k - 0.001), 1.6, knee=k)[0, 0, 0]
    b = ops.highlight_rolloff(grey(k + 0.001), 1.6, knee=k)[0, 0, 0]
    assert (b - a) / 0.002 == pytest.approx(1.0, abs=0.02)


def test_rolloff_survives_the_pole():
    """u = -1/c lands on real pixel values (x~0.03 at white=1.95) -> inf."""
    xs = np.linspace(0, 0.2, 200, dtype=np.float32)
    out = ops.highlight_rolloff(xs.reshape(-1, 1, 1).repeat(3, axis=2), 1.95, knee=0.6)
    assert np.all(np.isfinite(out))


# --- transfer functions ----------------------------------------------------

def test_srgb_roundtrip():
    xs = np.linspace(0, 1, 64, dtype=np.float32).reshape(-1, 1, 1).repeat(3, axis=2)
    assert np.allclose(ops.srgb_to_linear(ops.linear_to_srgb(xs)), xs, atol=1e-5)


# --- curves ----------------------------------------------------------------

def test_curve_identity_points_are_identity():
    img = np.linspace(0, 1, 32, dtype=np.float32).reshape(-1, 1, 1).repeat(3, axis=2)
    out = ops.apply_curve(img, [(0, 0), (0.5, 0.5), (1, 1)])
    assert np.allclose(out, img, atol=1e-3)


def test_curve_never_overshoots():
    """Monotone cubic must not ring between control points."""
    img = np.linspace(0, 1, 256, dtype=np.float32).reshape(-1, 1, 1).repeat(3, axis=2)
    out = ops.apply_curve(img, [(0, 0.03), (0.12, 0.105), (0.3, 0.275),
                                (0.5, 0.495), (0.75, 0.775), (1, 0.985)])
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.all(np.diff(out[:, 0, 0]) >= -1e-6)


# --- monochrome ------------------------------------------------------------

def test_monochrome_output_is_neutral():
    out = ops.monochrome(px(0.4, 0.6, 0.9))[0, 0]
    assert out[0] == pytest.approx(out[1]) == pytest.approx(out[2])


def test_red_weighted_mix_darkens_blue_sky():
    """Acros rendered as haze because its mix merely tracked luminance."""
    sky = px(0.40, 0.60, 0.90)
    luminance_like = ops.monochrome(sky, mix=(0.30, 0.60, 0.10))[0, 0, 0]
    yellow_filter = ops.monochrome(sky, mix=(0.45, 0.45, 0.10))[0, 0, 0]
    assert yellow_filter < luminance_like - 0.02


# --- auto exposure ---------------------------------------------------------

def test_auto_exposure_brightens_a_dark_frame():
    rng = np.random.default_rng(0)
    dark = (rng.random((64, 64, 3)) * 0.01).astype(np.float32)
    assert ops.luminance(ops.auto_exposure(dark)).mean() > ops.luminance(dark).mean()


def test_auto_exposure_is_stable_on_a_black_frame():
    black = np.zeros((16, 16, 3), dtype=np.float32)
    assert np.all(np.isfinite(ops.auto_exposure(black)))


# --- no-op guards ----------------------------------------------------------

@pytest.mark.parametrize("fn,kwargs", [
    (ops.clarity, {"amount": 0.0}),
    (ops.sharpen, {"amount": 0.0}),
    (ops.grain, {"amount": 0.0}),
    (ops.saturation, {"amount": 1.0, "vibrance": 0.0}),
])
def test_zero_amount_is_a_noop(fn, kwargs):
    rng = np.random.default_rng(1)
    img = rng.random((16, 16, 3)).astype(np.float32)
    assert np.allclose(fn(img, **kwargs), img)


# --- vignette --------------------------------------------------------------

def test_vignette_zero_is_a_noop():
    rng = np.random.default_rng(3)
    img = rng.random((32, 48, 3)).astype(np.float32)
    assert np.allclose(ops.vignette(img, 0.0, 0.0), img)


def test_vignette_leaves_the_centre_alone_and_moves_the_corners():
    img = np.full((64, 64, 3), 0.5, dtype=np.float32)
    out = ops.vignette(img, -0.2, 0.0)
    assert out[32, 32, 0] == pytest.approx(0.5, abs=0.01)   # centre
    assert out[0, 0, 0] < 0.5 - 0.01                        # corner darkened


# --- colour matrix ---------------------------------------------------------

def test_matrix_identity_is_a_noop():
    rng = np.random.default_rng(5)
    img = rng.random((16, 16, 3)).astype(np.float32)
    assert np.allclose(ops.color_matrix(img, np.eye(3)), img, atol=1e-6)


def test_matrix_rows_are_normalised_so_grey_stays_grey():
    """Rows summing to something other than 1 would shift exposure."""
    out = ops.color_matrix(grey(0.5), [[2, 0, 0], [0, 2, 0], [0, 0, 2]])[0, 0]
    assert out[0] == pytest.approx(0.5, abs=1e-5)
    assert out[0] == pytest.approx(out[1]) == pytest.approx(out[2])


def test_matrix_can_trade_between_channels():
    """The thing per-channel curves cannot do."""
    out = ops.color_matrix(px(0.8, 0.2, 0.2), [[1.3, -0.3, 0.0], [0, 1, 0], [0, 0, 1]])
    assert out[0, 0, 0] > 0.8
