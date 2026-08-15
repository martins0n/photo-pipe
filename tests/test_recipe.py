import numpy as np
import pytest

from photopipe import recipe as R


@pytest.fixture
def scene():
    """A small scene-linear frame with genuine highlight overrange."""
    rng = np.random.default_rng(7)
    img = rng.random((24, 32, 3)).astype(np.float32) * 1.4
    img[:6] *= 2.0            # a bright sky band, above 1.0
    return img


def test_packaged_recipes_all_load():
    recipes = R.load_all(R.PACKAGED_RECIPE_DIR)
    slugs = {r.slug for r in recipes}
    assert {"provia", "velvia", "astia", "classic-chrome", "acros"} <= slugs


def test_template_is_not_loaded_as_a_recipe():
    """_template.yaml.example must stay out of the recipe list."""
    assert "_template" not in {r.slug for r in R.load_all(R.PACKAGED_RECIPE_DIR)}


def test_every_packaged_recipe_produces_valid_output(scene):
    for r in R.load_all(R.PACKAGED_RECIPE_DIR):
        out = R.apply_recipe(scene, r)
        assert out.shape == scene.shape
        assert np.all(np.isfinite(out)), f"{r.slug} produced non-finite pixels"
        assert out.min() >= 0.0 and out.max() <= 1.0, f"{r.slug} left [0,1]"


def test_empty_recipe_is_a_valid_no_op_look(scene):
    """Every key optional means a bare recipe must still render."""
    out = R.apply_recipe(scene, R.Recipe({"name": "bare"}))
    assert np.all(np.isfinite(out)) and out.max() <= 1.0


def test_acros_carries_no_real_colour(scene):
    """Acros is toned, not strictly neutral: split_tone runs after monochrome,
    so a faint cool/warm cast is intentional. What must hold is that no actual
    hue survives — the channel spread stays tiny."""
    acros = R.load_named(["acros"], R.PACKAGED_RECIPE_DIR)[0]
    out = R.apply_recipe(scene, acros)
    spread = out.max(axis=2) - out.min(axis=2)
    assert spread.max() < 0.02


def test_monochrome_before_toning_is_exactly_neutral(scene):
    """The mono conversion itself must be strictly grey."""
    mono = R.Recipe({"monochrome": {"mix": [0.45, 0.45, 0.10]}})
    out = R.apply_recipe(scene, mono)
    assert np.allclose(out[:, :, 0], out[:, :, 1], atol=1e-6)
    assert np.allclose(out[:, :, 1], out[:, :, 2], atol=1e-6)


def test_velvia_is_more_saturated_than_classic_chrome(scene):
    import cv2
    def sat(slug):
        r = R.load_named([slug], R.PACKAGED_RECIPE_DIR)[0]
        out = R.apply_recipe(scene, r)
        return cv2.cvtColor(np.clip(out, 0, 1), cv2.COLOR_RGB2HSV)[:, :, 1].mean()
    assert sat("velvia") > sat("classic-chrome")


def test_unknown_recipe_names_are_rejected():
    with pytest.raises(SystemExit):
        R.load_named(["no-such-look"], R.PACKAGED_RECIPE_DIR)


def test_grain_is_deterministic_for_a_seed(scene):
    acros = R.load_named(["acros"], R.PACKAGED_RECIPE_DIR)[0]
    assert np.allclose(R.apply_recipe(scene, acros, seed=42),
                       R.apply_recipe(scene, acros, seed=42))


def test_curve_points_accept_both_spellings(scene):
    as_pairs = R.Recipe({"curve": [[0, 0], [0.5, 0.6], [1, 1]]})
    as_dicts = R.Recipe({"curve": [{"x": 0, "y": 0}, {"x": 0.5, "y": 0.6}, {"x": 1, "y": 1}]})
    assert np.allclose(R.apply_recipe(scene, as_pairs), R.apply_recipe(scene, as_dicts))


def test_jpeg_export_is_444_with_an_icc_profile(tmp_path):
    """Exports went out 4:2:0 for a while because a cv2 flag was requested
    conditionally and silently dropped on older builds."""
    from PIL import Image, JpegImagePlugin
    from photopipe import develop
    rng = np.random.default_rng(2)
    img = rng.random((64, 96, 3)).astype(np.float32)
    path = str(tmp_path / "out.jpg")
    develop.save_jpeg(img, path, quality=97)

    im = Image.open(path)
    assert JpegImagePlugin.get_sampling(im) == 0, "chroma is subsampled"
    assert im.info.get("icc_profile"), "no colour profile embedded"
    assert im.size == (96, 64)
