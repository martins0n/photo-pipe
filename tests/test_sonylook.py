import numpy as np
import pytest

from photopipe import sonylook, imageops as ops

scipy = pytest.importorskip("scipy", reason="fit() needs the [fit] extra")


def frame(seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(rng.random((40, 60, 3)).astype(np.float32) * 0.8 + 0.1, 0, 1)


def zeros():
    return [0] * len(sonylook.PARAM_NAMES)


def test_all_zero_sliders_are_a_no_op():
    img = frame()
    assert np.allclose(sonylook.apply_look(img, zeros()), img, atol=1e-6)


def test_output_stays_in_range():
    img = frame()
    for params in ([9, 9, 9, 9, 9, 9, 7, 7], [-9, -9, -9, 0, -9, -9, -7, -7]):
        out = sonylook.apply_look(img, params)
        assert np.all(np.isfinite(out)) and out.min() >= 0 and out.max() <= 1


def test_fade_lifts_the_black_point():
    img = frame() * 0.2
    p = zeros(); p[sonylook.PARAM_NAMES.index("fade")] = 9
    assert sonylook.apply_look(img, p).min() > img.min()


def test_saturation_slider_moves_saturation():
    import cv2
    img = frame()
    def sat(v):
        p = zeros(); p[sonylook.PARAM_NAMES.index("saturation")] = v
        return cv2.cvtColor(sonylook.apply_look(img, p), cv2.COLOR_RGB2HSV)[:, :, 1].mean()
    assert sat(-9) < sat(0) < sat(9)


def test_align_exposure_matches_overall_brightness():
    """Exposure must be factored out, or the fit spends its sliders on it."""
    img = frame()
    dim = ops.linear_to_srgb(ops.srgb_to_linear(img) * 0.4)
    aligned = sonylook.align_exposure(dim, img)
    key = lambda x: np.exp(np.mean(np.log(ops.luminance(ops.srgb_to_linear(x)) + 1e-4)))
    assert key(aligned) == pytest.approx(key(img), rel=0.05)


def test_fit_recovers_known_sliders():
    """Round trip: apply known sliders, then solve for them."""
    img = frame()
    truth = zeros()
    truth[sonylook.PARAM_NAMES.index("saturation")] = -6
    truth[sonylook.PARAM_NAMES.index("contrast")] = 3
    target = sonylook.apply_look(img, truth)

    params, report = sonylook.fit([(img, target)], verbose=False)
    assert report["rounded_loss"] < report["baseline_loss"]
    assert params["saturation"] < -2      # right direction, meaningful size
    assert params["contrast"] > 0


def test_fit_returns_integers_within_bounds():
    params, _ = sonylook.fit([(frame(), frame(1))], verbose=False)
    for name, value in params.items():
        lo, hi = sonylook.BOUNDS[name]
        assert isinstance(value, (int, np.integer))
        assert lo <= value <= hi
