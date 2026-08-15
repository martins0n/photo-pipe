"""Stage 4 — the comparison sheet.

Rows are photos, columns are recipes, so scanning down a column shows how one
look behaves across a shoot and scanning across a row shows the five looks on
one frame.
"""

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG = (18, 18, 20)
FG = (238, 238, 240)
DIM = (140, 140, 148)
RULE = (52, 52, 58)

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Helvetica Neue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/SFNS.ttf",
]


def _font(size, bold=False):
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size, index=1 if bold else 0)
            except Exception:
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    continue
    return ImageFont.load_default()


def _to_pil(img_float):
    return Image.fromarray(np.clip(img_float * 255.0 + 0.5, 0, 255).astype(np.uint8))


def _scaled(im, budget):
    """Scale so the long edge is `budget`, whatever the orientation.

    Never upscales — a tile bigger than the source would just add soft pixels,
    so the sheet silently caps at whatever `--max-dim` produced.
    """
    scale = budget / float(max(im.width, im.height))
    if scale >= 1.0:
        return im
    return im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))),
                     Image.LANCZOS)


def build(cells, row_labels, col_labels, out_path,
          tile_w=560, gutter=180, pad=18, title=None, subtitle=None,
          row_sublabels=None):
    """cells[row][col] -> float RGB image. Writes a labelled contact sheet.

    Columns sit at fixed x so a look can be followed straight down the sheet,
    but row height follows the frames in that row. A shoot that mixes portrait
    and landscape therefore wastes no space, and nothing gets letterboxed.
    """
    n_rows, n_cols = len(cells), len(col_labels)

    f_title = _font(34, bold=True)
    f_head = _font(23, bold=True)
    f_label = _font(19, bold=True)
    f_small = _font(15)

    # Every frame gets the same long edge, so no orientation is favoured.
    tiles = [[_scaled(_to_pil(img), tile_w) for img in row] for row in cells]
    row_heights = [max(t.height for t in row) for row in tiles]

    head_h = 52
    title_h = 96 if title else 0
    width = gutter + n_cols * (tile_w + pad) + pad
    height = title_h + head_h + sum(h + pad for h in row_heights) + pad

    sheet = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(sheet)

    if title:
        draw.text((pad + 6, 26), title, font=f_title, fill=FG)
        if subtitle:
            draw.text((pad + 6, 66), subtitle, font=f_small, fill=DIM)

    for c, name in enumerate(col_labels):
        x = gutter + c * (tile_w + pad)
        draw.text((x, title_h + 16), name, font=f_head, fill=FG)

    y = title_h + head_h
    for r in range(n_rows):
        draw.text((pad + 6, y + 8), row_labels[r], font=f_label, fill=FG)
        if row_sublabels and row_sublabels[r]:
            for i, line in enumerate(row_sublabels[r].split("  ")):
                draw.text((pad + 6, y + 34 + i * 20), line, font=f_small, fill=DIM)
        for c in range(n_cols):
            tile = tiles[r][c]
            x = gutter + c * (tile_w + pad) + (tile_w - tile.width) // 2
            sheet.paste(tile, (x, y + (row_heights[r] - tile.height) // 2))
        y += row_heights[r] + pad
        if r < n_rows - 1:
            draw.line([(pad, y - pad // 2), (width - pad, y - pad // 2)], fill=RULE, width=1)

    return _save(sheet, out_path)


def _save(sheet, out_path):
    """PNG is lossless; JPEG gets max quality and no chroma subsampling.

    A contact sheet is exactly the wrong subject for lossy compression — it is
    judged by fine tonal and colour differences between adjacent tiles, which
    is what JPEG throws away first.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if out_path.lower().endswith(".png"):
        sheet.save(out_path, format="PNG", compress_level=6)
    else:
        sheet.save(out_path, quality=97, subsampling=0)
    return out_path


def build_strip(images, labels, out_path, tile_w=760, pad=16, title=None, subtitle=None):
    """One photo across all recipes — the detail view for a single frame."""
    cells = [images]
    return build(cells, [""], labels, out_path, tile_w=tile_w, gutter=pad,
                 pad=pad, title=title, subtitle=subtitle)
