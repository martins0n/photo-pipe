"""Stage 2 — raw file to scene-linear RGB.

Deliberately does no tone mapping: the whole point is that every recipe starts
from the same neutral, linear base so a comparison is actually a comparison.
"""

import os
import subprocess
import json

import numpy as np
import cv2
import rawpy

from . import imageops as ops

# re-exported so the CLI can reach the primitives through this module



def develop(path, max_dim=None, half_size=False, **exposure_kw):
    """Decode a raw (ARW or PureRAW DNG) to scene-linear float32 RGB.

    Output is scene-linear and may exceed 1.0 in the highlights; the recipe's
    rolloff decides how that headroom is spent.
    """
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(
            gamma=(1, 1),                       # stay linear
            no_auto_bright=True,                # we do our own, predictably
            output_bps=16,
            use_camera_wb=True,
            output_color=rawpy.ColorSpace.sRGB,
            half_size=half_size,
            highlight_mode=rawpy.HighlightMode.Blend,
        )

    img = rgb.astype(np.float32) / 65535.0
    if max_dim:
        img = resize_max(img, max_dim)
    return ops.auto_exposure(img, **exposure_kw)


SIBLING_EXT = (".HIF", ".HEIC", ".HEIF", ".JPG", ".JPEG")


def find_sibling_rendered(raw_path):
    """The camera's own rendering of a frame, if it was shot RAW+JPEG/HEIF."""
    stem = os.path.splitext(raw_path)[0]
    for ext in SIBLING_EXT:
        for cand in (stem + ext, stem + ext.lower()):
            if os.path.exists(cand):
                return cand
    return None


def load_rendered(path, max_dim=None, cache_dir=None):
    """Load a camera-rendered file as display-referred float RGB in [0,1].

    HEIF/HIF goes through `sips`, which is built into macOS and — unlike a
    plain decode — already applies EXIF orientation, so portrait frames come
    back the right way up. The conversion is cached because it is the slow part.
    """
    ext = os.path.splitext(path)[1].lower()
    src = path
    if ext in (".hif", ".heic", ".heif"):
        stem = os.path.splitext(os.path.basename(path))[0]
        cache_dir = cache_dir or os.path.join(os.path.dirname(path), ".rendered")
        os.makedirs(cache_dir, exist_ok=True)
        src = os.path.join(cache_dir, stem + "_sooc.jpg")
        if not os.path.exists(src):
            subprocess.run(["sips", "-s", "format", "jpeg", "-s", "formatOptions", "best",
                            path, "--out", src], check=True, capture_output=True)

    bgr = cv2.imread(src, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"could not decode {path}")
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if max_dim:
        img = resize_max(img, max_dim)
    return img


def resize_max(img, max_dim):
    h, w = img.shape[:2]
    scale = max_dim / float(max(h, w))
    if scale >= 1.0:
        return img
    return cv2.resize(img, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                      interpolation=cv2.INTER_AREA)


def save_jpeg(img_float, path, quality=97, exif_from=None):
    """Write display-referred float [0,1] RGB as 8-bit JPEG, carrying EXIF over."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    height, width = img_float.shape[:2]
    img8 = np.clip(img_float * 255.0 + 0.5, 0, 255).astype(np.uint8)
    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    # 4:4:4 chroma keeps saturated edges clean, but the flag is missing on
    # older OpenCV builds (absent in 4.6), so only ask for it when it exists.
    if hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR"):
        params += [int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR),
                   int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444)]
    cv2.imwrite(path, cv2.cvtColor(img8, cv2.COLOR_RGB2BGR), params)
    if exif_from:
        try:
            subprocess.run(
                ["exiftool", "-overwrite_original", "-TagsFromFile", exif_from,
                 # Don't inherit the source's geometry: LibRaw has already
                 # rotated the pixels upright and we may have resized them.
                 "-all:all", "--Orientation", "--ExifImageWidth", "--ExifImageHeight",
                 "-ColorSpace=sRGB",
                 # The '#' forces the numeric value. Without it exiftool matches
                 # "1" against the *descriptions* and lands on "Rotate 180",
                 # which makes every viewer flip the image.
                 "-Orientation#=1",
                 f"-ExifImageWidth#={width}", f"-ExifImageHeight#={height}",
                 path],
                check=True, capture_output=True)
        except Exception:
            pass  # metadata is a nicety; never fail an export over it
    return path


def read_exif(path):
    """Small EXIF subset used for collage captions."""
    try:
        out = subprocess.run(
            ["exiftool", "-j", "-Model", "-LensModel", "-FocalLength",
             "-FNumber", "-ExposureTime", "-ISO", path],
            capture_output=True, check=True, text=True).stdout
        return json.loads(out)[0]
    except Exception:
        return {}


LOOK_TAGS = ["CreativeStyle", "Contrast", "Saturation", "Sharpness", "Shadows",
             "Highlights", "Fade", "Clarity", "SharpnessRange", "WBShiftAB",
             "WBShiftGM", "DynamicRangeOptimizer"]

# Every maker names its picture-style setting differently; try them in turn and
# take the first that answers. Order matters only in that a more specific tag
# should precede a more generic one for the same brand.
LOOK_NAME_TAGS = [
    "CreativeStyle",        # Sony  (Creative Look: FL, VV, IN, ...)
    "CreativeLook",         # Sony, newer bodies
    "FilmMode",             # Fujifilm (Provia / Velvia / Astia / Classic Chrome)
    "SaturationSetting",    # Fujifilm, when FilmMode is absent
    "PictureControlName",   # Nikon
    "PictureStyle",         # Canon
    "PictureMode",          # Olympus / OM System
    "PhotoStyle",           # Panasonic
    "FilmSimulation",       # some Fujifilm exports
]


def camera_look_name(path, default="camera"):
    """The camera's picture-style name, whatever the maker calls it.

    Returns something like "FL" (Sony) or "Classic Chrome" (Fujifilm), falling
    back to the model name so a measured recipe is still identifiable.
    """
    try:
        args = ["exiftool", "-j", "-Model"] + [f"-{t}" for t in LOOK_NAME_TAGS]
        data = json.loads(subprocess.run(args + [path], capture_output=True,
                                         check=True, text=True).stdout)[0]
    except Exception:
        return default
    for tag in LOOK_NAME_TAGS:
        value = data.get(tag)
        if value not in (None, "", "n/a", "Off", "None"):
            # Fuji writes things like "F2/Fujichrome (Velvia)" — keep the name
            # a human would use.
            text = str(value)
            if "(" in text and text.endswith(")"):
                text = text[text.rindex("(") + 1:-1]
            return text.strip()
    return str(data.get("Model") or default).strip()


def read_look_settings(path):
    """The camera's Creative Look state, as numbers, from Sony's MakerNotes.

    Needed because a fitted slider value is a *delta* from wherever the camera
    already is — reporting absolutes without this would be wrong.
    """
    try:
        args = ["exiftool", "-j", "-n"] + [f"-{t}" for t in LOOK_TAGS] + [path]
        raw = json.loads(subprocess.run(args, capture_output=True, check=True,
                                        text=True).stdout)[0]
    except Exception:
        return {}
    # -n leaves CreativeStyle numeric; the readable name needs a second pass.
    try:
        raw["CreativeStyleName"] = subprocess.run(
            ["exiftool", "-s3", "-CreativeStyle", path],
            capture_output=True, check=True, text=True).stdout.strip()
    except Exception:
        pass
    return raw


def exif_caption(meta):
    bits = []
    if meta.get("FocalLength"):
        bits.append(str(meta["FocalLength"]).replace(" mm", "mm"))
    if meta.get("FNumber"):
        bits.append(f"f/{meta['FNumber']}")
    if meta.get("ExposureTime"):
        bits.append(f"{meta['ExposureTime']}s")
    if meta.get("ISO"):
        bits.append(f"ISO {meta['ISO']}")
    return "  ".join(bits)
