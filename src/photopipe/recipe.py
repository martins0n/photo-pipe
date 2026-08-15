"""Recipe loading and application.

A recipe is a plain YAML file in recipes/. Every key is optional; anything you
leave out is a no-op, so a new look can start as three lines and grow. The
order of operations below is fixed and is the whole contract — a recipe only
supplies numbers, never sequencing.
"""

import os
import glob

import numpy as np
import yaml

from . import imageops as ops

#: Recipes bundled with the package — the defaults you get after a plain install.
PACKAGED_RECIPE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recipes")


def default_recipe_dir():
    """Where recipes are read from, most specific source first.

    The point of the search order is that editing a look never requires
    touching an installed package: drop a `recipes/` directory next to
    wherever you run from, or point PHOTOPIPE_RECIPES at one, and it wins.
    """
    env = os.environ.get("PHOTOPIPE_RECIPES")
    if env and os.path.isdir(env):
        return env
    local = os.path.join(os.getcwd(), "recipes")
    if os.path.isdir(local) and (glob.glob(os.path.join(local, "*.yaml"))
                                 or glob.glob(os.path.join(local, "*.yml"))):
        return local
    return PACKAGED_RECIPE_DIR


# Kept as a module attribute for convenience; call default_recipe_dir() when
# the working directory may have changed since import.
RECIPE_DIR = default_recipe_dir()


class Recipe:
    def __init__(self, data, path=None):
        self.data = data or {}
        self.path = path
        self.slug = (os.path.splitext(os.path.basename(path))[0] if path
                     else self.data.get("name", "untitled"))
        self.name = self.data.get("name", self.slug)
        self.description = self.data.get("description", "")

    def get(self, key, default=None):
        return self.data.get(key, default)

    def __repr__(self):
        return f"<Recipe {self.slug}>"


def load_recipe(path):
    with open(path) as fh:
        return Recipe(yaml.safe_load(fh), path)


def load_all(recipe_dir=None):
    """Every *.yaml in the recipe dir, ordered by optional `order:` then name."""
    recipe_dir = recipe_dir or default_recipe_dir()
    paths = sorted(glob.glob(os.path.join(recipe_dir, "*.yaml"))
                   + glob.glob(os.path.join(recipe_dir, "*.yml")))
    recipes = [load_recipe(p) for p in paths]
    recipes.sort(key=lambda r: (r.get("order", 999), r.slug))
    return recipes


def load_named(names, recipe_dir=None):
    """Resolve a list of slugs, erroring with the available set if one is unknown."""
    available = {r.slug: r for r in load_all(recipe_dir)}
    out = []
    for n in names:
        key = os.path.splitext(os.path.basename(n))[0]
        if key not in available:
            raise SystemExit(
                f"unknown recipe '{n}'. available: {', '.join(sorted(available))}")
        out.append(available[key])
    return out


def _points(raw):
    """Accept [[x,y], ...] or [{x: , y: }, ...]."""
    if not raw:
        return None
    pts = []
    for p in raw:
        if isinstance(p, dict):
            pts.append((float(p["x"]), float(p["y"])))
        else:
            pts.append((float(p[0]), float(p[1])))
    return pts


def apply_recipe(linear_rgb, recipe, seed=0):
    """Scene-linear float32 RGB in, display-referred float32 sRGB in [0,1] out."""
    r = recipe
    img = linear_rgb

    # --- scene-linear stage ---
    img = ops.exposure(img, float(r.get("exposure", 0.0)))
    img = ops.temp_tint(img, float(r.get("temp", 0.0)), float(r.get("tint", 0.0)))
    img = ops.highlight_rolloff(img, float(r.get("highlight_rolloff", 1.0)),
                                knee=float(r.get("highlight_knee", 0.6)))

    # --- to display space ---
    img = ops.linear_to_srgb(img)

    # --- tone ---
    img = ops.apply_curve(img, _points(r.get("curve")))
    rgb_curves = r.get("rgb_curves") or {}
    for idx, channel in enumerate(("red", "green", "blue")):
        pts = _points(rgb_curves.get(channel))
        if pts:
            img[:, :, idx] = ops.apply_curve(img[:, :, idx], pts)

    # Lens falloff difference, fitted where it is applied: after the tone
    # curves, before colour.
    vig = r.get("vignette")
    if vig:
        img = ops.vignette(img, float(vig.get("a1", 0.0)), float(vig.get("a2", 0.0)))

    # --- colour ---
    img = ops.apply_hsl(img, r.get("hsl") or {})

    mono = r.get("monochrome")
    if mono:
        img = ops.monochrome(img,
                             mix=mono.get("mix", (0.30, 0.59, 0.11)),
                             filter_strength=float(mono.get("filter", 1.0)))
    else:
        img = ops.saturation(img,
                             amount=float(r.get("saturation", 1.0)),
                             vibrance=float(r.get("vibrance", 0.0)))

    st = r.get("split_tone")
    if st:
        img = ops.split_tone(img,
                             shadows=st.get("shadows", (0, 0, 0)),
                             highlights=st.get("highlights", (0, 0, 0)),
                             balance=float(st.get("balance", 0.5)),
                             strength=float(st.get("strength", 1.0)))

    # --- detail ---
    img = ops.clarity(img, float(r.get("clarity", 0.0)))

    sh = r.get("sharpen")
    if sh:
        img = ops.sharpen(img,
                          amount=float(sh.get("amount", 0.0)),
                          radius=float(sh.get("radius", 0.8)),
                          threshold=float(sh.get("threshold", 0.0)))

    gr = r.get("grain")
    if gr:
        img = ops.grain(img,
                        amount=float(gr.get("amount", 0.0)),
                        size=float(gr.get("size", 1.0)),
                        seed=seed)

    return np.clip(img, 0.0, 1.0)
