#!/usr/bin/env python3
"""photo-pipe — end-to-end raw pipeline: ARW -> DxO PureRAW -> recipe -> JPEG.

    ./pipe.py recipes
    ./pipe.py run ~/Images/2026-08-14 --collage
    ./pipe.py run ~/Images/2026-08-14 --recipes velvia,acros --full
    ./pipe.py run photo.ARW --denoise none          # A/B without DxO

Nothing is ever written next to your originals: raws are hardlinked into
work/ and every output lands in out/.
"""

import argparse
import os
import sys
import time

import numpy as np

from . import dxo, develop, recipe as recipes_mod, collage, sonylook

RAW_EXT = (".arw", ".dng", ".nef", ".cr2", ".cr3", ".raf", ".orf", ".rw2")


def collect_raws(inputs):
    """Expand directories to their raws, skipping anything DxO already made."""
    out = []
    for item in inputs:
        item = os.path.expanduser(item)
        if os.path.isdir(item):
            for f in sorted(os.listdir(item)):
                if f.lower().endswith(RAW_EXT) and "-DxO_" not in f and not f.startswith("."):
                    out.append(os.path.join(item, f))
        elif os.path.isfile(item):
            out.append(item)
        else:
            raise SystemExit(f"no such input: {item}")
    if not out:
        raise SystemExit("no raw files found")
    return out


def cmd_recipes(args):
    for r in recipes_mod.load_all():
        desc = " ".join(r.description.split())
        print(f"  {r.slug:<16} {r.name}")
        if desc:
            print(f"  {'':<16} {desc}")
    print(f"\n  reading from {recipes_mod.default_recipe_dir()}")
    print("  any *.yaml there is picked up; set PHOTOPIPE_RECIPES to point elsewhere")


def cmd_run(args):
    t0 = time.time()
    raws = collect_raws(args.inputs)
    picked = (recipes_mod.load_named(args.recipes.split(","))
              if args.recipes else recipes_mod.load_all())
    if not picked:
        raise SystemExit(f"no recipes found in {recipes_mod.default_recipe_dir()}")

    work = os.path.abspath(args.work)
    out_dir = os.path.abspath(args.out)
    stage_dir = os.path.join(work, "01_dxo")
    os.makedirs(out_dir, exist_ok=True)

    print(f"photo-pipe: {len(raws)} raw(s) x {len(picked)} recipe(s)")
    print(f"  denoise : {args.denoise}")
    print(f"  recipes : {', '.join(r.slug for r in picked)}")
    print(f"  out     : {out_dir}\n")

    # --- stage 1: denoise / demosaic ---
    if args.denoise == "dxo":
        print("[1/4] DxO PureRAW")
        sources = dxo.process(raws, stage_dir, log=print)
    else:
        print("[1/4] DxO skipped — developing the ARW directly")
        sources = {r: r for r in raws}

    max_dim = 1600 if args.preview else args.max_dim

    # Only spend a column on the camera rendering if the shoot actually has one.
    siblings = {src: develop.find_sibling_rendered(src) for src in raws}
    use_reference = args.reference != "none" and any(siblings.values())
    if args.reference == "hif" and not use_reference:
        raise SystemExit("--reference hif: no HIF/JPEG found next to any input")

    grid, row_labels, row_subs = [], [], []

    for src in raws:
        developed_from = sources.get(src, src)
        stem = os.path.splitext(os.path.basename(src))[0]
        print(f"\n[2/4] develop {stem}  ({os.path.basename(developed_from)})")
        base = develop.develop(developed_from, max_dim=max_dim, key_weight=args.lift)
        print(f"      {base.shape[1]}x{base.shape[0]} scene-linear")

        meta = develop.read_exif(src)
        row = []

        # The camera's own rendering, as the leftmost reference column. It is
        # already display-referred, so no recipe is applied — that is the point.
        if use_reference:
            sib = siblings[src]
            if sib:
                print(f"[3/4]   reference  ({os.path.basename(sib)})")
                row.append(develop.load_rendered(
                    sib, max_dim=max_dim,
                    cache_dir=os.path.join(work, "02_sooc")))
            else:
                row.append(np.zeros((8, 12, 3), dtype=np.float32))  # frame shot raw-only

        for r in picked:
            print(f"[3/4]   {r.slug}")
            img = recipes_mod.apply_recipe(base, r, seed=abs(hash(stem)) % (2 ** 31))
            path = os.path.join(out_dir, f"{stem}_{r.slug}.jpg")
            develop.save_jpeg(img, path, quality=args.quality, exif_from=src)
            row.append(img)
        grid.append(row)
        row_labels.append(stem)
        row_subs.append(develop.exif_caption(meta))

    # --- stage 4: comparison sheet ---
    if args.collage:
        print("\n[4/4] collage")
        sheet = os.path.join(out_dir, args.collage_out)
        col_labels = [r.name for r in picked]
        if use_reference:
            col_labels = ["Camera SOOC (HIF)"] + col_labels
        collage.build(
            grid, row_labels, col_labels, sheet,
            tile_w=args.tile,
            title="photo-pipe — Fuji recipe comparison",
            subtitle=(f"{'ARW -> DxO PureRAW' if args.denoise == 'dxo' else 'ARW -> LibRaw direct'}"
                      f"  ->  recipe  ->  JPEG   |   {len(raws)} frames x {len(picked)} looks"
                      + ("   |   first column is the camera's own HIF, untouched"
                         if use_reference else "")),
            row_sublabels=row_subs)
        print(f"      {sheet}")

    print(f"\ndone in {time.time() - t0:.0f}s -> {out_dir}")


def cmd_fit_camera(args):
    """Solve for Creative Look sliders that push the SOOC toward a recipe."""
    raws = collect_raws(args.inputs)
    target = recipes_mod.load_named([args.recipe])[0]
    neutral = recipes_mod.load_named([args.from_recipe])[0]
    work = os.path.abspath(args.work)
    out_dir = os.path.abspath(args.out)

    pairs, used, strips = [], [], []
    for src in raws:
        sib = develop.find_sibling_rendered(src)
        if not sib:
            continue
        stem = os.path.splitext(os.path.basename(src))[0]
        print(f"  {stem}")

        source = src
        if args.denoise == "dxo":
            source = dxo.process([src], os.path.join(work, "01_dxo"), log=print)[src]
        base = develop.develop(source, max_dim=args.fit_size, key_weight=args.lift)
        want = recipes_mod.apply_recipe(base, target)
        sooc = develop.load_rendered(sib, max_dim=args.fit_size,
                                     cache_dir=os.path.join(work, "02_sooc"))
        if sooc.shape != want.shape:
            sooc = develop.resize_max(sooc, args.fit_size)
            want = develop.resize_max(want, args.fit_size)
        if sooc.shape != want.shape:
            print(f"    skipped — SOOC {sooc.shape[:2]} vs recipe {want.shape[:2]}")
            continue

        # What we fit *from* decides what the answer means.
        #
        # "neutral": both sides come from our own pipeline, so LibRaw-vs-Sony
        # colour science cancels out and the fit isolates the look itself —
        # the sliders that turn a neutral rendering into Classic Chrome. Those
        # transfer to the camera on top of a neutral Creative Look.
        #
        # "sooc": fits against the camera's actual output, so it also absorbs
        # the difference between Sony's colour and LibRaw's — which no slider
        # can express, and which drags the reported fit down.
        ref = (recipes_mod.apply_recipe(base, neutral) if args.reference == "neutral"
               else sooc)
        # Exposure is a camera setting, not a Creative Look slider.
        want = sonylook.align_exposure(want, ref)
        pairs.append((ref, want))
        used.append((stem, sib, ref, want, sooc))

    if not pairs:
        raise SystemExit("no RAW + HIF/JPEG pairs found to fit against")

    print(f"\nfitting {len(pairs)} frame(s) -> {target.name}")
    params, report = sonylook.fit(pairs)

    current = develop.read_look_settings(used[0][1])
    look = current.get("CreativeStyleName", "?")
    if args.reference == "neutral":
        print(f"\nfitted against our neutral '{neutral.slug}' render — apply these "
              f"on a neutral Creative Look (ST or NT),\nnot on the {look} the "
              f"camera is currently set to.")
    else:
        print(f"\ncamera is currently on Creative Look: {look}")
    print(f"\n{'parameter':<14}{'now':>6}{'delta':>8}{'set to':>9}")
    print("  " + "-" * 35)
    for name in sonylook.PARAM_NAMES:
        tag = {"wb_amber": "WBShiftAB", "wb_green": "WBShiftGM"}.get(
            name, name.capitalize())
        now = current.get(tag, 0) or 0
        lo, hi = sonylook.BOUNDS[name]
        target_val = int(np.clip(now + params[name], lo, hi))
        print(f"  {name:<14}{now:>6}{params[name]:>+8}{target_val:>9}")
    print(f"\n  {report['closed_pct']:.0f}% of the SOOC->recipe gap is reachable "
          f"with these sliders")

    # Visual proof: what the fitted sliders actually do, next to the goal.
    for stem, _, ref, want, sooc in used[:args.strips]:
        got = sonylook.apply_look(ref, [params[n] for n in sonylook.PARAM_NAMES])
        base_label = ("neutral base" if args.reference == "neutral"
                      else "Camera SOOC (now)")
        collage.build_strip(
            [sooc, ref, got, want],
            [f"Camera SOOC ({look})", base_label, "base + fitted sliders",
             f"{target.name} (goal)"],
            os.path.join(out_dir, f"fit_{stem}.png"), tile_w=args.tile,
            title=f"Creative Look fit — {stem}",
            subtitle=", ".join(f"{n} {params[n]:+d}"
                               for n in sonylook.PARAM_NAMES if params[n]))
    print(f"\nwrote {min(len(used), args.strips)} verification strip(s) -> {out_dir}")


def main():
    # Resolved here rather than at import time so the library can be used
    # from anywhere without the working directory being baked in.
    here = os.getcwd()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("recipes", help="list available recipes").set_defaults(func=cmd_recipes)

    run = sub.add_parser("run", help="process raws end to end")
    run.add_argument("inputs", nargs="+", help="raw files or a directory of them")
    run.add_argument("--recipes", help="comma-separated slugs (default: all)")
    run.add_argument("--denoise", choices=["dxo", "none"], default="dxo")
    run.add_argument("--out", default=os.path.join(here, "out"))
    run.add_argument("--work", default=os.path.join(here, "work"))
    run.add_argument("--max-dim", type=int, default=None,
                     help="downscale exports to this long edge (default: full resolution)")
    run.add_argument("--preview", action="store_true",
                     help="fast 1600px pass, for iterating on a recipe")
    run.add_argument("--quality", type=int, default=97)
    run.add_argument("--lift", type=float, default=0.25,
                     help="auto-exposure bias, 0 = protect highlights only, "
                          "1 = push everything to a midtone key (default 0.25)")
    run.add_argument("--collage", action="store_true", help="also build comparison.jpg")
    run.add_argument("--reference", choices=["auto", "hif", "none"], default="auto",
                     help="add the camera's own HIF/JPEG as the first collage "
                          "column (auto: only when one exists)")
    run.add_argument("--tile", type=int, default=900,
                     help="collage tile long edge (capped by --max-dim)")
    run.add_argument("--collage-out", default="comparison.png",
                     help="collage filename; .png is lossless, .jpg is smaller")
    run.set_defaults(func=cmd_run)

    fit = sub.add_parser("fit-camera",
                         help="solve for Creative Look sliders matching a recipe")
    fit.add_argument("inputs", nargs="+", help="raws shot RAW+HEIF")
    fit.add_argument("--recipe", default="classic-chrome")
    fit.add_argument("--from-recipe", default="provia",
                     help="the neutral render the look is measured against")
    fit.add_argument("--reference", choices=["neutral", "sooc"], default="neutral",
                     help="neutral: isolate the look (recommended). "
                          "sooc: fit the camera's actual output, which also "
                          "absorbs Sony-vs-LibRaw colour differences")
    fit.add_argument("--denoise", choices=["dxo", "none"], default="none",
                     help="the fit works on downscaled frames, so denoise "
                          "rarely changes the answer (default: skip, much faster)")
    fit.add_argument("--fit-size", type=int, default=640,
                     help="long edge used for fitting")
    fit.add_argument("--strips", type=int, default=3,
                     help="how many verification strips to write")
    fit.add_argument("--out", default=os.path.join(here, "out"))
    fit.add_argument("--work", default=os.path.join(here, "work"))
    fit.add_argument("--lift", type=float, default=0.25)
    fit.add_argument("--tile", type=int, default=900)
    fit.set_defaults(func=cmd_fit_camera)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
