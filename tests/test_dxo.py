"""The PureRAW stage, with a stand-in for the app itself.

PureRAW is a GUI application with no test hook, so what these exercise is the
contract photo-pipe relies on: a frame is handed back the moment its DNG lands,
and the timeout measures a stall rather than the length of the queue. That
distinction is the whole point — the batch deadline it replaced meant a long
enough run could not finish however healthy it was, and because nothing was
exported until the stage completed, the failure discarded every frame it had
already paid for.
"""

import os

import pytest

from photopipe import dxo


@pytest.fixture
def stage(tmp_path, monkeypatch):
    """A stage dir with the app faked out and no real launch."""
    monkeypatch.setattr(dxo, "available", lambda: True)
    monkeypatch.setattr(dxo.subprocess, "run", lambda *a, **k: None)
    # _settled() polls the file size twice; nothing here is written twice.
    monkeypatch.setattr(dxo, "_settled", lambda path, **k: True)
    d = tmp_path / "01_dxo"
    d.mkdir()
    return d


def raws(tmp_path, *stems):
    out = []
    for stem in stems:
        p = tmp_path / f"{stem}.ARW"
        p.write_bytes(b"raw")
        out.append(str(p))
    return out


def finish(stage, stem):
    """Stand in for PureRAW writing one output."""
    (stage / f"{stem}-DxO_DeepPRIME 3.dng").write_bytes(b"dng")


def test_cached_frames_are_yielded_before_anything_is_awaited(tmp_path, stage):
    srcs = raws(tmp_path, "A", "B")
    for stem in ("A", "B"):
        finish(stage, stem)

    # timeout=0 would fail instantly if the generator waited on anything.
    got = list(dxo.process_iter(srcs, str(stage), timeout=0, log=lambda *a: None))
    assert [s for s, _ in got] == srcs
    assert all(out.endswith(".dng") for _, out in got)


def test_a_frame_is_handed_back_before_the_rest_are_denoised(tmp_path, stage):
    """The stage streams: one output present is enough to start developing."""
    srcs = raws(tmp_path, "A", "B", "C")
    finish(stage, "A")

    stream = dxo.process_iter(srcs, str(stage), timeout=5, log=lambda *a: None)
    src, out = next(stream)

    assert src == srcs[0]
    assert os.path.basename(out).startswith("A-DxO_")
    # B and C are still in the queue, and we got A without waiting for them.
    assert not os.path.exists(str(stage / "B-DxO_DeepPRIME 3.dng"))


def test_timeout_is_a_stall_budget_not_a_batch_budget(tmp_path, stage, monkeypatch):
    """A queue that outlives `timeout` in total still finishes if it progresses.

    The clock is driven forward here by the poll itself: every sweep advances
    time by more than the whole budget, so a batch-wide deadline would fail on
    the second frame no matter what the app did.
    """
    srcs = raws(tmp_path, "A", "B", "C")
    now = [1000.0]
    monkeypatch.setattr(dxo.time, "time", lambda: now[0])

    pending = ["A", "B", "C"]

    def tick(_):
        now[0] += 100.0          # each wait costs 100s, budget below is 60s
        if pending:
            finish(stage, pending.pop(0))

    monkeypatch.setattr(dxo.time, "sleep", tick)

    got = list(dxo.process_iter(srcs, str(stage), timeout=60, log=lambda *a: None))
    assert [os.path.basename(s) for s, _ in got] == ["A.ARW", "B.ARW", "C.ARW"]


def test_a_genuinely_stuck_app_still_fails(tmp_path, stage, monkeypatch):
    srcs = raws(tmp_path, "A")
    now = [1000.0]
    monkeypatch.setattr(dxo.time, "time", lambda: now[0])
    monkeypatch.setattr(dxo.time, "sleep", lambda _: now.__setitem__(0, now[0] + 10.0))

    with pytest.raises(TimeoutError) as e:
        list(dxo.process_iter(srcs, str(stage), timeout=60, log=lambda *a: None))
    assert "A" in str(e.value)


def test_process_still_returns_the_whole_batch_as_a_mapping(tmp_path, stage):
    """fit-camera and match-look use the eager form; keep it working."""
    srcs = raws(tmp_path, "A", "B")
    for stem in ("A", "B"):
        finish(stage, stem)

    result = dxo.process(srcs, str(stage), timeout=0, log=lambda *a: None)
    assert set(result) == set(srcs)
    assert all(v.endswith(".dng") for v in result.values())


def test_originals_are_only_ever_read(tmp_path, stage):
    """Outputs land in the stage dir, never beside the raw."""
    srcs = raws(tmp_path, "A")
    finish(stage, "A")
    list(dxo.process_iter(srcs, str(stage), timeout=0, log=lambda *a: None))

    beside = [f for f in os.listdir(tmp_path) if "-DxO_" in f]
    assert beside == []


def test_scan_prefers_the_dng_over_a_sidecar_jpeg(tmp_path, stage):
    srcs = raws(tmp_path, "A")
    (stage / "A-DxO_DeepPRIME 3.jpg").write_bytes(b"jpg")
    (stage / "A-DxO_DeepPRIME 3.dng").write_bytes(b"dng")

    (_, out), = dxo.process_iter(srcs, str(stage), timeout=0, log=lambda *a: None)
    assert out.endswith(".dng")
