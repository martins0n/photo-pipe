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


def process(raw_paths, stage_dir, mode=MODE_LAST_SETTINGS, timeout=1800, log=print):
    """Run a batch through PureRAW. Returns {source_path: output_dng_path}.

    Sources already carrying a PureRAW output in stage_dir are skipped, so a
    re-run is cheap and the expensive stage is effectively cached.
    """
    if not available():
        raise RuntimeError(f"DxO PureRAW not found at {APP_PATH}")

    os.makedirs(stage_dir, exist_ok=True)
    staged, results, todo = {}, {}, []
    for src in raw_paths:
        stem = os.path.splitext(os.path.basename(src))[0]
        dst = _stage(src, stage_dir)
        staged[stem] = src
        existing = _find_output(stage_dir, stem)
        if existing:
            log(f"    cached  {os.path.basename(existing)}")
            results[src] = existing
        else:
            todo.append(dst)

    if not todo:
        return results

    batch_file = os.path.join(stage_dir, ".dxo_batch.txt")
    with open(batch_file, "w") as fh:
        fh.write("\n".join(todo))

    log(f"    launching PureRAW on {len(todo)} file(s)...")
    subprocess.run(
        ["open", "-n", "-b", BUNDLE_ID, "--args",
         mode, f"--lr-version={LR_VERSION}", f"--batch-file={batch_file}"],
        check=True)

    pending = {os.path.splitext(os.path.basename(p))[0] for p in todo}
    deadline = time.time() + timeout
    while pending and time.time() < deadline:
        for stem in sorted(pending):
            out = _find_output(stage_dir, stem)
            if out and _settled(out):
                log(f"    done    {os.path.basename(out)}")
                results[staged[stem]] = out
                pending.discard(stem)
        if pending:
            time.sleep(2.0)

    if pending:
        raise TimeoutError(
            f"PureRAW did not finish {sorted(pending)} within {timeout}s. "
            "If its window is showing a prompt, answer it once and re-run — "
            "the setting is remembered.")
    return results
