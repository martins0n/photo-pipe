#!/usr/bin/env python3
"""photo-pipe — end-to-end raw pipeline: ARW -> DxO PureRAW -> recipe -> JPEG.

    photo-pipe recipes
    photo-pipe run ~/Images/2026-08-14 --collage
    photo-pipe run ~/Images/2026-08-14 --recipes velvia,acros --preview
    photo-pipe run photo.ARW --denoise none          # A/B without DxO
    photo-pipe fit-camera ~/Images/2026-07-30        # camera settings for a look

Nothing is ever written next to your originals: raws are hardlinked into a
cache outside the photo library (~/.cache/photo-pipe) and exports go to --out.
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
    # Streamed, not batched: PureRAW is a separate process, so developing a
    # frame the moment its DNG lands overlaps the two expensive stages for
    # free, and a frame exported early survives a later failure.
    if args.denoise == "dxo":
        print("[1/4] DxO PureRAW")
        stream = dxo.process_iter(raws, stage_dir,
                                  timeout=args.dxo_timeout, log=print)
    else:
        print("[1/4] DxO skipped — developing the ARW directly")
        stream = ((r, r) for r in raws)

    max_dim = 1600 if args.preview else args.max_dim

    # Only spend a column on the camera rendering if the shoot actually has one.
    siblings = {src: develop.find_sibling_rendered(src) for src in raws}
    use_reference = args.reference != "none" and any(siblings.values())
    if args.reference == "hif" and not use_reference:
        raise SystemExit("--reference hif: no HIF/JPEG found next to any input")

    # Keyed by source rather than appended: frames arrive in whatever order
    # PureRAW finishes them, while the collage wants the order you asked for.
    rows = {}

    for src, developed_from in stream:
        stem = os.path.splitext(os.path.basename(src))[0]
        print(f"\n[2/4] develop {stem}  ({os.path.basename(developed_from)})")
        base = develop.develop(developed_from, max_dim=max_dim, key_weight=args.lift)
        print(f"      {base.shape[1]}x{base.shape[0]} scene-linear")

        # Borrow the camera's exposure decision for this frame. Sony's
        # multi-segment metering is not reproducible from global statistics
        # (optimising every auto-exposure parameter still leaves 0.42 stops of
        # per-frame spread), but when the frame was shot RAW+HEIF the answer
        # is simply sitting next to the raw. Aligning per frame is what stops
        # a measured look drifting: the sky rides up or down the tone curve
        # with exposure, so a frame rendered too bright comes out desaturated
        # and one too dark comes out oversaturated.
        if args.match_exposure and siblings.get(src):
            ref = develop.load_rendered(siblings[src], max_dim=640,
                                        cache_dir=os.path.join(work, "02_sooc"))
            key = lambda x: float(np.exp(np.mean(np.log(develop.ops.luminance(x) + 1e-4))))
            gain = key(develop.ops.srgb_to_linear(ref)) / max(key(base), 1e-9)
            base = (base * gain).astype(np.float32)
            print(f"      exposure matched to {os.path.basename(siblings[src])} "
                  f"({np.log2(gain):+.2f} stops)")

        row = []

        # The camera's own rendering, as the leftmost reference column. It is
        # already display-referred, so no recipe is applied — that is the point.
        # Decoding it is pure collage work, so it waits for a collage to exist.
        if args.collage and use_reference:
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
            use = r
            if args.match_exposure and siblings.get(src) and r.get("exposure"):
                # Its `exposure` is the median offset from our auto-exposure to
                # the camera's; per-frame alignment already did that job.
                use = recipes_mod.Recipe(
                    {k: v for k, v in r.data.items() if k != "exposure"}, r.path)
            img = recipes_mod.apply_recipe(base, use, seed=abs(hash(stem)) % (2 ** 31))
            path = os.path.join(out_dir, f"{stem}_{r.slug}.jpg")
            develop.save_jpeg(img, path, quality=args.quality, exif_from=src)
            if args.collage:
                row.append(img)
        # Only the collage needs the pixels afterwards. Holding every frame at
        # full resolution otherwise costs hundreds of MB apiece, which a long
        # run turns into tens of GB for a sheet nobody asked for.
        if args.collage:
            # One exiftool call per frame, and only the caption wants it.
            rows[src] = (row, stem,
                         develop.exif_caption(develop.read_exif(src)))

    # --- stage 4: comparison sheet ---
    if args.collage:
        print("\n[4/4] collage")
        sheet = os.path.join(out_dir, args.collage_out)
        grid = [rows[src][0] for src in raws if src in rows]
        row_labels = [rows[src][1] for src in raws if src in rows]
        row_subs = [rows[src][2] for src in raws if src in rows]
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

    if args.denoise == "dxo" and args.cache_gb > 0:
        dxo.prune(stage_dir, int(args.cache_gb * 2 ** 30), log=print)

    print(f"\ndone in {time.time() - t0:.0f}s -> {out_dir}")


def cmd_cache(args):
    """Inspect or trim the PureRAW cache."""
    stage = os.path.join(os.path.abspath(args.work), "01_dxo")
    entries = dxo.cache_entries(stage)
    total = sum(e[1] for e in entries)
    print(f"cache : {stage}")
    print(f"size  : {total / 2**30:.2f} GB in {len(entries)} PureRAW output(s)")
    if entries:
        newest = time.strftime("%Y-%m-%d %H:%M", time.localtime(entries[0][2]))
        oldest = time.strftime("%Y-%m-%d %H:%M", time.localtime(entries[-1][2]))
        print(f"range : {oldest}  ..  {newest}")
        print("        (staged raws are hardlinks and cost no extra space)")

    if args.clear:
        # prune() treats max_bytes<=0 as "no cap", so clearing is its own loop.
        freed = removed = 0
        for path, size, _ in entries:
            stem = os.path.basename(path).split("-DxO_")[0]
            try:
                os.remove(path); freed += size; removed += 1
            except OSError:
                continue
            for f in os.listdir(stage):
                if os.path.splitext(f)[0] == stem:
                    try:
                        os.remove(os.path.join(stage, f))
                    except OSError:
                        pass
        print(f"\ncleared {freed / 2**30:.2f} GB ({removed} file(s))")
    elif args.max_gb is not None:
        freed, removed = dxo.prune(stage, int(args.max_gb * 2**30), log=print)
        if not removed:
            print(f"\nalready under {args.max_gb} GB — nothing removed")


def cmd_match_look(args):
    """Measure the camera's own rendering and write it out as a recipe."""
    import yaml
    from . import matchlook

    raws = collect_raws(args.inputs)
    work = os.path.abspath(args.work)
    pairs, gains, sharp_pairs = [], [], []

    for src in raws[:args.limit]:
        sib = develop.find_sibling_rendered(src)
        if not sib:
            continue
        stem = os.path.splitext(os.path.basename(src))[0]
        source = src
        if args.denoise == "dxo":
            source = dxo.process([src], os.path.join(work, "01_dxo"), log=lambda *a: None)[src]
        base = develop.develop(source, max_dim=args.fit_size)
        sooc = develop.load_rendered(sib, max_dim=args.fit_size,
                                     cache_dir=os.path.join(work, "02_sooc"))
        if sooc.shape != base.shape:
            print(f"  {stem}: skipped (size mismatch)")
            continue
        neutral, gain = matchlook.neutral_render(base, sooc)
        pairs.append((neutral, sooc))
        gains.append(gain)

        # Detail has to be measured at native resolution. The pairs above are
        # downscaled for speed, which is fine for tone and colour but fatal
        # here: resampling low-passes both sides equally, so a render that is
        # visibly softer at 1:1 measures as identical on a 900px proxy.
        if len(sharp_pairs) < args.sharpen_frames:
            full = develop.develop(source)
            ref_full = develop.load_rendered(
                sib, cache_dir=os.path.join(work, "02_sooc"))
            if ref_full.shape == full.shape:
                k = lambda x: float(np.exp(np.mean(
                    np.log(develop.ops.luminance(x) + 1e-4))))
                full = full * (k(develop.ops.srgb_to_linear(ref_full))
                               / max(k(full), 1e-9))
                sharp_pairs.append((develop.ops.linear_to_srgb(full), ref_full))
        print(f"  {stem}")

    if not pairs:
        raise SystemExit("no RAW + HIF/JPEG pairs to measure")

    look = develop.camera_look_name(develop.find_sibling_rendered(raws[0]))
    print(f"\nmeasuring '{look}' from {len(pairs)} frame(s)")

    import math
    exposure = math.log2(float(np.median(gains))) if gains else 0.0
    print(f"  exposure offset vs the pipeline's auto-exposure: {exposure:+.2f} stops")

    curves = matchlook.measure_curves(pairs)
    err_curves = matchlook.error(pairs, curves)
    vig = {} if args.no_vignette else matchlook.measure_vignette(pairs, curves)
    err_vig = matchlook.error(pairs, curves, None, vig) if vig else err_curves
    mat = None if args.no_matrix else matchlook.measure_matrix(pairs, curves, vig)
    err_mat = matchlook.error(pairs, curves, None, vig, mat) if mat else err_vig
    if mat and err_mat > err_vig:
        print("  matrix made it worse — dropping it")
        mat, err_mat = None, err_vig
    hsl = ({} if args.no_hsl else
           matchlook.measure_hsl(pairs, curves, vignette=vig, matrix=mat))
    err_full = matchlook.error(pairs, curves, hsl, vig, mat) if hsl else err_mat

    baseline = matchlook.error(pairs, {c: [[0, 0], [1, 1]] for c in ("red", "green", "blue")})
    print(f"  Lab error  neutral {baseline:6.2f}  ->  curves {err_curves:6.2f}"
          + (f"  ->  +vignette {err_vig:6.2f}" if vig else "")
          + (f"  ->  +matrix {err_mat:6.2f}" if mat else "")
          + (f"  ->  +hsl {err_full:6.2f}" if hsl else ""))
    if vig:
        print(f"  vignette   a1={vig['a1']:+.3f}  a2={vig['a2']:+.3f}")
    if hsl and err_full > err_mat:
        print("  hsl residual made it worse — dropping it")
        hsl, err_full = {}, err_mat

    sharp = (None if args.no_sharpen or not sharp_pairs else
             matchlook.measure_sharpen(sharp_pairs, curves, vig, mat))
    if sharp:
        print(f"  sharpen    amount={sharp['amount']} "
              f"(matched to the camera's detail rendering)")
    else:
        print("  sharpen    not needed — already as crisp as the camera")

    slug = args.name or look.lower().replace(" ", "-")
    data = matchlook.to_recipe(
        curves, hsl, name=args.title or f"{look} (measured)",
        description=(f"{look} camera look, measured from {len(pairs)} raw + "
                     f"camera-JPEG pairs: per-channel transfer curves"
                     + (", lens falloff" if vig else "")
                     + (", a 3x3 channel mix" if mat else "")
                     + (", per-hue residual" if hsl else "")
                     + (", capture sharpening" if sharp else "")
                     + ". Regenerate with `photo-pipe match-look`."),
        order=args.order, exposure=exposure, vignette=vig, matrix=mat,
        sharpen=sharp)

    dest = args.out_recipe or os.path.join(recipes_mod.default_recipe_dir(), f"{slug}.yaml")
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    with open(dest, "w") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, default_flow_style=None, width=100)
    print(f"\nwrote {dest}")
    print(f"  use it:  photo-pipe run <photos> --recipes {slug}")


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


def default_work_dir():
    """Where the DxO cache lives.

    Deliberately *not* the working directory: the command is on PATH, so it
    gets run from inside photo folders, and a work/ full of staged hardlinks
    has no business appearing in a photo library. A fixed cache also means the
    expensive PureRAW stage is reused no matter where you run from.
    """
    return os.environ.get("PHOTOPIPE_WORK") or os.path.expanduser("~/.cache/photo-pipe")


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
    run.add_argument("--dxo-timeout", type=int, default=1800,
                     help="give up if PureRAW produces nothing for this many "
                          "seconds (per frame, not per batch; default 1800)")
    run.add_argument("--out", default=os.path.join(here, "out"))
    run.add_argument("--work", default=default_work_dir(),
                     help="DxO cache dir (default: ~/.cache/photo-pipe)")
    run.add_argument("--cache-gb", type=float,
                     default=float(os.environ.get("PHOTOPIPE_CACHE_GB", 10)),
                     help="cap the PureRAW cache after a run, oldest evicted "
                          "first; 0 disables (default 10 GB, PHOTOPIPE_CACHE_GB)")
    run.add_argument("--max-dim", type=int, default=None,
                     help="downscale exports to this long edge (default: full resolution)")
    run.add_argument("--preview", action="store_true",
                     help="fast 1600px pass, for iterating on a recipe")
    run.add_argument("--quality", type=int, default=97)
    run.add_argument("--lift", type=float, default=0.25,
                     help="auto-exposure bias, 0 = protect highlights only, "
                          "1 = push everything to a midtone key (default 0.25)")
    run.add_argument("--collage", action="store_true", help="also build comparison.jpg")
    run.add_argument("--match-exposure", action="store_true",
                     help="take each frame's exposure from its sibling HIF/JPEG "
                          "instead of metering it ourselves (RAW+HEIF only)")
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
    fit.add_argument("--work", default=default_work_dir())
    fit.add_argument("--lift", type=float, default=0.25)
    fit.add_argument("--tile", type=int, default=900)
    fit.set_defaults(func=cmd_fit_camera)

    cache = sub.add_parser("cache", help="show or trim the PureRAW cache")
    cache.add_argument("--work", default=default_work_dir())
    cache.add_argument("--max-gb", type=float, help="evict oldest until under this size")
    cache.add_argument("--clear", action="store_true", help="remove everything")
    cache.set_defaults(func=cmd_cache)

    match = sub.add_parser("match-look",
                           help="measure the camera's own look and save it as a recipe")
    match.add_argument("inputs", nargs="+", help="raws shot RAW+HEIF")
    match.add_argument("--name", help="recipe slug (default: the Creative Look name)")
    match.add_argument("--title", help="display name inside the recipe")
    match.add_argument("--out-recipe", help="where to write it")
    match.add_argument("--denoise", choices=["dxo", "none"], default="dxo",
                       help="match the base the recipe will be used on (default dxo)")
    match.add_argument("--fit-size", type=int, default=900)
    match.add_argument("--limit", type=int, default=12, help="max frames to measure")
    match.add_argument("--sharpen-frames", type=int, default=3,
                       help="frames decoded at full resolution to match detail")
    match.add_argument("--no-sharpen", action="store_true",
                       help="skip matching the camera's capture sharpening")
    match.add_argument("--no-matrix", action="store_true",
                       help="skip the 3x3 channel-mix term")
    match.add_argument("--no-vignette", action="store_true",
                       help="skip the lens-falloff term")
    match.add_argument("--no-hsl", action="store_true",
                       help="per-channel curves only, skip the hue residual")
    match.add_argument("--order", type=int, default=50)
    match.add_argument("--work", default=default_work_dir())
    match.set_defaults(func=cmd_match_look)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
