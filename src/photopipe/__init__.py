"""photo-pipe — an end-to-end raw photo pipeline.

ARW (or any raw) -> DxO PureRAW -> a YAML-defined look -> JPEG, plus a
comparison contact sheet and a solver for matching a look in-camera.

Typical library use:

    from photopipe import develop, recipe

    base = develop.develop("DSC0001.dng")          # scene-linear float RGB
    look = recipe.load_named(["classic-chrome"])[0]
    out  = recipe.apply_recipe(base, look)         # display-referred [0,1]
    develop.save_jpeg(out, "out.jpg", exif_from="DSC0001.ARW")

The command line equivalent is `photo-pipe run <dir> --collage`.
"""

from . import collage, develop, dxo, imageops, recipe, sonylook
from .recipe import Recipe, apply_recipe, load_all, load_named, default_recipe_dir

__version__ = "0.1.0"

__all__ = [
    "collage", "develop", "dxo", "imageops", "recipe", "sonylook",
    "Recipe", "apply_recipe", "load_all", "load_named", "default_recipe_dir",
    "__version__",
]
