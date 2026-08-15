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
