import numpy as np
import pytest

from photopipe import matchlook, imageops as ops, recipe as R


def scene(seed=0, shape=(120, 160)):
    rng = np.random.default_rng(seed)
    img = rng.random((*shape, 3)).astype(np.float32)
    # a smooth ramp so every input level is populated, plus some colour
    ramp = np.linspace(0, 1, shape[1], dtype=np.float32)[None, :, None]
    return np.clip(img * 0.35 + ramp * 0.65, 0, 1)


def test_neutral_render_matches_reference_exposure():
    lin = ops.srgb_to_linear(scene())
    ref = scene(1)
    neutral, gain = matchlook.neutral_render(lin, ref)
    key = lambda x: np.exp(np.mean(np.log(ops.luminance(ops.srgb_to_linear(x)) + 1e-4)))
    assert key(neutral) == pytest.approx(key(ref), rel=0.05)
    assert gain > 0


def test_identity_pair_measures_an_identity_curve():
    """Same image on both sides must not invent a look."""
    img = scene()
    curves = matchlook.measure_curves([(img, img)])
    for name in ("red", "green", "blue"):
        for x, y in curves[name]:
            assert y == pytest.approx(x, abs=0.06)


def test_measured_curves_recover_a_known_transform():
    """Apply a known per-channel change, then measure it back."""
    img = scene()
    target = img.copy()
    target[:, :, 2] = np.clip(target[:, :, 2] * 0.75, 0, 1)   # pull the blues down
    curves = matchlook.measure_curves([(img, target)])
    mid_blue = dict(map(tuple, curves["blue"]))[0.55]
    mid_red = dict(map(tuple, curves["red"]))[0.55]
    assert mid_blue < mid_red - 0.05
    assert matchlook.error([(img, target)], curves) < matchlook.error(
        [(img, target)], {c: [[0, 0], [1, 1]] for c in ("red", "green", "blue")})


def test_curves_are_monotone_and_in_range():
    curves = matchlook.measure_curves([(scene(), scene(2))])
    for name in ("red", "green", "blue"):
        ys = [p[1] for p in curves[name]]
        assert all(b >= a - 1e-6 for a, b in zip(ys, ys[1:]))
        assert 0.0 <= min(ys) and max(ys) <= 1.0


def test_to_recipe_round_trips_through_the_engine():
    """Whatever match-look writes must be a recipe the engine can apply."""
    img = scene()
    curves = matchlook.measure_curves([(img, img)])
    data = matchlook.to_recipe(curves, {"blue": {"sat": -10}},
                               name="T", description="d", exposure=-0.85)
    assert data["exposure"] == pytest.approx(-0.85)
    out = R.apply_recipe(ops.srgb_to_linear(img), R.Recipe(data))
    assert np.all(np.isfinite(out)) and out.min() >= 0 and out.max() <= 1


def test_negligible_exposure_is_omitted():
    curves = matchlook.measure_curves([(scene(), scene())])
    assert "exposure" not in matchlook.to_recipe(curves, {}, "T", "d", exposure=0.001)


def test_measured_hsl_recovers_a_hue_shift():
    """The blue sky sat ~5 degrees off until hue was measured at all."""
    import cv2
    img = scene(4)
    hsv = cv2.cvtColor(np.clip(img, 0, 1), cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + 0.35, 0, 1)     # make the hue meaningful
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    hsv[:, :, 0] = (hsv[:, :, 0] + 12.0) % 360.0          # rotate every hue
    target = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    flat = {c: [[0.0, 0.0], [1.0, 1.0]] for c in ("red", "green", "blue")}
    hsl = matchlook.measure_hsl([(img, target)], flat)
    shifts = [v["hue"] for v in hsl.values() if "hue" in v]
    assert shifts, "no hue shift measured at all"
    assert np.median(shifts) > 5


def test_measure_matrix_recovers_a_known_mix():
    from photopipe import imageops as ops
    img = scene(6)
    truth = np.array([[1.15, -0.10, -0.05], [0.0, 1.0, 0.0], [-0.05, 0.05, 1.0]])
    target = ops.color_matrix(img, truth)
    flat = {c: [[0.0, 0.0], [1.0, 1.0]] for c in ("red", "green", "blue")}
    m = np.array(matchlook.measure_matrix([(img, target)], flat))
    assert np.allclose(m, truth, atol=0.06)


def test_measure_matrix_rows_sum_to_one():
    flat = {c: [[0.0, 0.0], [1.0, 1.0]] for c in ("red", "green", "blue")}
    m = matchlook.measure_matrix([(scene(7), scene(8))], flat)
    if m is not None:
        assert np.allclose(np.array(m).sum(axis=1), 1.0, atol=1e-3)


def test_identical_pair_needs_no_matrix():
    img = scene(9)
    flat = {c: [[0.0, 0.0], [1.0, 1.0]] for c in ("red", "green", "blue")}
    assert matchlook.measure_matrix([(img, img)], flat) is None


def test_detail_ratio_falls_when_an_image_is_blurred():
    import cv2
    img = scene(11)
    g = cv2.cvtColor(np.clip(img, 0, 1), cv2.COLOR_RGB2GRAY) * 255
    soft = cv2.GaussianBlur(g, (0, 0), 1.2)
    assert matchlook._detail_ratio(soft) < matchlook._detail_ratio(g)


def test_measure_sharpen_recovers_lost_detail():
    """A softened render must be given sharpening back, not left alone."""
    from photopipe import imageops as ops
    import cv2
    rng = np.random.default_rng(3)
    target = np.clip(rng.random((256, 256, 3)).astype(np.float32) * 0.6 + 0.2, 0, 1)
    soft = cv2.GaussianBlur(target, (0, 0), 0.9)
    flat = {c: [[0.0, 0.0], [1.0, 1.0]] for c in ("red", "green", "blue")}

    got = matchlook.measure_sharpen([(soft, target)], flat)
    assert got is not None and got["amount"] > 0.05
    # and it must actually close the gap
    before = matchlook._detail_ratio(ops.luminance(soft) * 255)
    after = matchlook._detail_ratio(ops.luminance(
        ops.sharpen(soft, **{k: got[k] for k in ("amount", "radius", "threshold")})) * 255)
    assert after > before


def test_measure_sharpen_is_skipped_when_already_crisp():
    """No sharpening term when the render already matches — do not oversharpen."""
    img = scene(12)
    flat = {c: [[0.0, 0.0], [1.0, 1.0]] for c in ("red", "green", "blue")}
    assert matchlook.measure_sharpen([(img, img)], flat) is None
