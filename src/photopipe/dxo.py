"""Stage 1 — denoise / demosaic / optical correction via DxO PureRAW.

PureRAW ships no documented CLI, but its Lightroom plugin drives the app over
one, and that interface works standalone. The plugin writes a newline-separated
list of source files and launches:

    open -n -b com.dxo-labs.PureRAWv6.standalone --args \\
        --as-lightroom-last-settings-plugin --lr-version="14.0" \\
        --batch-file="<list>"

PureRAW processes the batch unattended and writes `<stem>-DxO_<method>.dng`
next to each source. That "next to each source" is why we stage: originals are
hardlinked into the work directory first, so the photo library is only ever
read, never written to.
"""

import os
import shutil
import subprocess
import time

BUNDLE_ID = "com.dxo-labs.PureRAWv6.standalone"
APP_PATH = "/Applications/DxO PureRAW 6.app"
# The app only checks that a version was supplied, not what it is.
LR_VERSION = "14.0"

MODE_LAST_SETTINGS = "--as-lightroom-last-settings-plugin"
MODE_INSTANT = "--as-lightroom-plugin"


def available():
    return os.path.isdir(APP_PATH)


def _stage(src, stage_dir):
    """Hardlink the original in, falling back to a copy across filesystems."""
    dst = os.path.join(stage_dir, os.path.basename(src))
    if os.path.exists(dst):
        return dst
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst


def _find_output(stage_dir, stem):
    """PureRAW appends the denoise method to the stem, e.g. `-DxO_DeepPRIME 3`."""
    hits = [f for f in os.listdir(stage_dir)
            if f.startswith(stem + "-DxO_") and f.lower().endswith((".dng", ".jpg", ".tif", ".tiff"))]
    if not hits:
        return None
    # Prefer DNG; among equals take the newest.
    hits.sort(key=lambda f: (not f.lower().endswith(".dng"),
                             -os.path.getmtime(os.path.join(stage_dir, f))))
    return os.path.join(stage_dir, hits[0])


def _settled(path, checks=2, interval=1.5):
    """True once the file size stops changing — PureRAW writes in place."""
    last = -1
    for _ in range(checks):
        try:
            size = os.path.getsize(path)
        except OSError:
            return False
        if size == last and size > 0:
            return True
        last = size
        time.sleep(interval)
    return os.path.getsize(path) == last and last > 0


def cache_entries(stage_dir):
    """PureRAW outputs in the cache, newest first, with their sizes.

    Only the DNGs are counted: the staged sources are hardlinks to your
    originals, so they occupy no extra space.
    """
    if not os.path.isdir(stage_dir):
        return []
    out = []
    for f in os.listdir(stage_dir):
        if "-DxO_" in f and f.lower().endswith((".dng", ".tif", ".tiff", ".jpg")):
            p = os.path.join(stage_dir, f)
            try:
                out.append((p, os.path.getsize(p), os.path.getmtime(p)))
            except OSError:
                pass
    return sorted(out, key=lambda e: -e[2])


def cache_size(stage_dir):
    return sum(e[1] for e in cache_entries(stage_dir))


def prune(stage_dir, max_bytes, log=print):
    """Evict the oldest PureRAW outputs until the cache fits in max_bytes.

    Re-running a look on a recent shoot is the common case, so age is the
    right thing to evict on. Anything removed simply costs one more PureRAW
    pass if you come back to it.
    """
    entries = cache_entries(stage_dir)
    total = sum(e[1] for e in entries)
    if max_bytes <= 0 or total <= max_bytes:
        return 0, 0

    freed = removed = 0
    for path, size, _ in reversed(entries):     # oldest first
        if total - freed <= max_bytes:
            break
        # Drop the staged hardlink too, so a later run re-stages and reprocesses
        # rather than finding a source with no output beside it.
        stem = os.path.basename(path).split("-DxO_")[0]
        try:
            os.remove(path)
            freed += size
            removed += 1
        except OSError:
            continue
        for f in os.listdir(stage_dir):
            if os.path.splitext(f)[0] == stem:
                try:
                    os.remove(os.path.join(stage_dir, f))
                except OSError:
                    pass
    if removed:
        log(f"    cache: freed {freed / 2**30:.1f} GB ({removed} file(s), oldest first)")
    return freed, removed


def _scan_outputs(stage_dir, stems):
    """Map stem -> PureRAW output for the stems we are waiting on.

    One listdir per sweep rather than one per stem: with a few hundred frames
    queued the per-stem version spends more time walking the directory than
    waiting on the app.
    """
    found = {}
    for f in os.listdir(stage_dir):
        if "-DxO_" not in f or not f.lower().endswith((".dng", ".jpg", ".tif", ".tiff")):
            continue
        stem = f.split("-DxO_")[0]
        if stem not in stems:
            continue
        path = os.path.join(stage_dir, f)
        best = found.get(stem)
        # Prefer DNG; among equals take the newest.
        if best is None or (not best.lower().endswith(".dng") and path.lower().endswith(".dng")) \
                or (best.lower().endswith(".dng") == path.lower().endswith(".dng")
                    and os.path.getmtime(path) > os.path.getmtime(best)):
            found[stem] = path
    return found


def process_iter(raw_paths, stage_dir, mode=MODE_LAST_SETTINGS, timeout=1800, log=print):
    """Yield (source_path, output_path) as each frame comes out of PureRAW.

    Sources already carrying a PureRAW output in stage_dir are yielded straight
    away, so a re-run is cheap and the expensive stage is effectively cached.

    `timeout` is a *stall* budget, not a budget for the whole batch: the clock
    restarts every time a frame lands. A batch of five and a batch of five
    hundred therefore get the same generous allowance for any single frame,
    while an app sitting on a modal prompt still fails instead of hanging. The
    old whole-batch deadline meant a large enough queue could not finish no
    matter how healthy the run was — and because this stage ran to completion
    before anything was exported, that failure threw away every frame it had
    already denoised.
    """
    if not available():
        raise RuntimeError(f"DxO PureRAW not found at {APP_PATH}")

    os.makedirs(stage_dir, exist_ok=True)
    staged, cached, todo = {}, [], []
    for src in raw_paths:
        stem = os.path.splitext(os.path.basename(src))[0]
        dst = _stage(src, stage_dir)
        staged[stem] = src
        existing = _find_output(stage_dir, stem)
        if existing:
            cached.append((src, existing))
        else:
            todo.append(dst)

    if todo:
        batch_file = os.path.join(stage_dir, ".dxo_batch.txt")
        with open(batch_file, "w") as fh:
            fh.write("\n".join(todo))

        log(f"    launching PureRAW on {len(todo)} file(s)...")
        subprocess.run(
            ["open", "-n", "-b", BUNDLE_ID, "--args",
             mode, f"--lr-version={LR_VERSION}", f"--batch-file={batch_file}"],
            check=True)

    # Hand back what is already on disk before waiting on anything, so a
    # resumed run goes straight to developing.
    for src, out in cached:
        log(f"    cached  {os.path.basename(out)}")
        yield src, out

    pending = {os.path.splitext(os.path.basename(p))[0] for p in todo}
    while pending:
        deadline = time.time() + timeout
        fresh = []
        while not fresh:
            # Look before judging: a frame that landed during the last wait
            # counts as progress, however long that wait happened to be.
            for stem, out in _scan_outputs(stage_dir, pending).items():
                if _settled(out):
                    fresh.append((stem, out))
            if fresh:
                break
            if time.time() >= deadline:
                raise TimeoutError(
                    f"PureRAW produced nothing for {timeout}s with "
                    f"{len(pending)} frame(s) left: {sorted(pending)}. "
                    "If its window is showing a prompt, answer it once and "
                    "re-run — the setting is remembered.")
            time.sleep(2.0)
        for stem, out in fresh:
            log(f"    done    {os.path.basename(out)}")
            yield staged[stem], out
            pending.discard(stem)


def process(raw_paths, stage_dir, mode=MODE_LAST_SETTINGS, timeout=1800, log=print):
    """Run a batch through PureRAW. Returns {source_path: output_dng_path}.

    The eager form of process_iter, for callers that want the whole batch
    before they start.
    """
    return dict(process_iter(raw_paths, stage_dir, mode=mode,
                             timeout=timeout, log=log))
