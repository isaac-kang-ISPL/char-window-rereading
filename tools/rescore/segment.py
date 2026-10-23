"""Position segmentation S_i — splits the image into K segments using per-position attention maps and character points.

Composition:
    maps      per-position attention maps scaled to a maximum of 1 (linearly interpolated to the original resolution)
    seeds     discs around the character points (radius SEED_R · position spacing) · an overlapping pixel goes to the
              position with the larger map value · each character-point pixel is pinned to its own index
    splitting `nearest` (default) — every pixel joins the position of the nearest seed pixel in Euclidean distance
              `cut` (earlier window definition, not used in the paper) — seed-controlled watershed on the terrain -(max_k H_k) + a distance term
              (compactness COMPACT)
There is no background — every pixel belongs to some position (labels 0..K-1).

Watershed flooding runs in a C++ kernel (`compact_watershed.cpp`) that ports scikit-image 0.26's compact watershed as is —
on the first call it is built with g++ and placed in `CWR_RESCORE_KERNEL_DIR` (default ~/.cache/cwr_rescore/kernels).
If building or loading fails, scikit-image's `watershed` is used instead (same labels · slower) and this is noted once on
stderr — which one ran is given by `backend()`. With `CWR_SEGMENT_BACKEND=skimage` scikit-image is used from the start;
with `=cpp` it stops when the kernel is unavailable.

Helper functions for window neighbour decisions: `label_at` (segment a point falls in) · `touching` (segments touching in
4-neighbourhood) · `clipped_labels` (labels with pixels outside the mask set to -1).

    lab = nearest(rgb, maps_px, points)      # (H,W) int16 · values 0..K-1 · None if there are no seeds (default)
    lab = cut(rgb, maps_px, points)          # same shape · watershed (earlier window definition)
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.segmentation import _watershed as _wsmod
from skimage.segmentation import watershed as _ws

from tools._io import cache_dir, env

#: Disc seed radius — ratio to the position spacing (`unit_w`).
SEED_R = 0.25
#: Weight of the distance term in the watershed (compactness of skimage watershed).
COMPACT = 1e-4
#: Gaussian blur σ of the ink (Sobel magnitude). 0 means no blur.
INK_BLUR = 1.0

#: C++ watershed source · where the built shared library goes · compile options (-ffp-contract=off: multiply-add is
#: not fused, so the values match the original).
KERNEL_SRC = Path(__file__).with_name('compact_watershed.cpp')
KERNEL_DIR = Path(env('CWR_RESCORE_KERNEL_DIR', os.path.join(cache_dir(), 'kernels')))
CXX_FLAGS = ('-O3', '-std=c++17', '-fPIC', '-shared', '-ffp-contract=off')
#: Value seeds are queued with — 'terrain' (scikit-image 0.26 · terrain value at that pixel) | '-inf' (scikit-image 0.25).
SEED_COST = env('CWR_SEGMENT_SEED', 'terrain')
_FLOOD = None
_BACKEND = None


def unit_w(cs, shape):
    """Position spacing (px) — median distance between neighbouring character points (lower bound 1). With one point,
    the image's long side ÷ number of positions."""
    d = [float(np.hypot(cs[i + 1][0] - cs[i][0], cs[i + 1][1] - cs[i][1]))
         for i in range(len(cs) - 1)]
    return max(float(np.median(d)) if d else max(shape[:2]) / max(len(cs), 1), 1.0)


def ink(rgb: np.ndarray) -> np.ndarray:
    """(H,W) 0~1 stroke evidence — grayscale Sobel magnitude · σ=INK_BLUR blur · divided by the 99th percentile."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    m = np.hypot(gx, gy)
    if INK_BLUR > 0:
        m = cv2.GaussianBlur(m, (0, 0), INK_BLUR)
    hi = float(np.percentile(m, 99.0))
    return np.clip(m / hi, 0, 1) if hi > 0 else np.zeros_like(m)


def _up(maps: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Maps (K,h,w) -> (K,H,W) float32 · linear interpolation if the size differs."""
    h, w = hw
    if maps.shape[1:] == (h, w):
        return maps.astype(np.float32)
    return np.stack([cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
                     for m in maps.astype(np.float32)])


def norm(maps: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Maps (K,h,w) -> (K,H,W), each position divided by its maximum to a maximum of 1 and clipped to [0,1] (unchanged
    if the maximum is ≤ 0)."""
    m = _up(np.asarray(maps, np.float32), hw)
    if not len(m):
        return m
    mx = m.reshape(len(m), -1).max(1)
    mx[mx <= 0] = 1.0
    return np.clip(m / mx[:, None, None], 0.0, 1.0)


def _disc_argmax_markers(m, cs, disc):
    """Disc seeds for abnormal input (NaN · negatives in the maps, NaN radius) — draws all discs, then decides the owner
    by argmax."""
    K, H, W = m.shape
    out = np.zeros((H, W), np.int32)
    if K == 0 or not len(cs):
        return out
    claim = np.zeros((K, H, W), bool)
    yy, xx = np.mgrid[0:H, 0:W]
    for k, (x, y) in enumerate(cs):
        xi = int(min(max(round(x), 0), W - 1))
        yi = int(min(max(round(y), 0), H - 1))
        sel = np.zeros((H, W), bool)
        sel[yi, xi] = True
        if disc > 0:
            sel |= ((xx - x) ** 2 + (yy - y) ** 2) <= disc * disc
        claim[k] = sel
    any_ = claim.any(0)
    if any_.any():
        win = np.where(claim, m, -1.0).argmax(0)
        out[any_] = win[any_] + 1
    for k, (x, y) in enumerate(cs):
        xi = int(min(max(round(x), 0), W - 1))
        yi = int(min(max(round(y), 0), H - 1))
        out[yi, xi] = k + 1
    return out


def disc_markers(m, cs, r: float) -> np.ndarray:
    """Disc seeds -> (H,W) int32 `0`=none · `1..K`.

    `m` normalized maps (K,H,W) · `cs` K character points (px) · `r` radius (px). An overlapping pixel goes to the
    position with the larger map value (on a tie, the earlier position), and at the end each character-point pixel is
    pinned to its own index.
    """
    K, H, W = m.shape
    out = np.zeros((H, W), np.int32)
    if K == 0 or not len(cs):
        return out
    if not np.isfinite(r) or not np.isfinite(m).all() or np.min(m) < 0:
        return _disc_argmax_markers(m, cs, r)
    best = np.full((H, W), -1., dtype=np.result_type(m.dtype, np.float32))
    for k, (x, y) in enumerate(cs):
        xi = int(min(max(round(x), 0), W - 1))
        yi = int(min(max(round(y), 0), H - 1))
        rr = max(float(r), 0.)
        x0 = max(0, min(xi, int(np.floor(x - rr))))
        x1 = min(W, max(xi + 1, int(np.ceil(x + rr)) + 1))
        y0 = max(0, min(yi, int(np.floor(y - rr))))
        y1 = min(H, max(yi + 1, int(np.ceil(y + rr)) + 1))
        yy, xx = np.mgrid[y0:y1, x0:x1]
        claim = ((xx-x)**2 + (yy-y)**2 <= r*r) if r > 0 else np.zeros(xx.shape, bool)
        claim[yi-y0, xi-x0] = True
        values = m[k, y0:y1, x0:x1]
        score, dst = best[y0:y1, x0:x1], out[y0:y1, x0:x1]
        take = claim & (values > score)  # strict comparison — on a tie, the earlier position
        score[take], dst[take] = values[take], k + 1
    for k, (x, y) in enumerate(cs):
        out[int(min(max(round(y), 0), H-1)), int(min(max(round(x), 0), W-1))] = k + 1
    return out


def relief(m: np.ndarray, ink_map: np.ndarray, beta: float = 0.0) -> np.ndarray:
    """Watershed terrain `-(max_k H_k + beta*ink)` — valleys inside characters, ridges between characters."""
    return -(m.max(0) + beta * np.clip(ink_map, 0, 1)) if len(m) else -ink_map * 0


def build(force: bool = False) -> Path:
    """Build the C++ kernel and return the shared library path — the name contains the source sha256, so a changed source
    is rebuilt. It is built under a temporary name and then moved, so concurrent builds by several processes never read a
    half-written file. The compiler is `CXX` (default g++)."""
    lib = KERNEL_DIR / f'compact_watershed_{hashlib.sha256(KERNEL_SRC.read_bytes()).hexdigest()[:16]}.so'
    if force or not lib.exists():
        KERNEL_DIR.mkdir(parents=True, exist_ok=True)
        tmp = lib.with_name(f'{lib.name}.{os.getpid()}.tmp')
        subprocess.run([os.environ.get('CXX', 'g++'), *CXX_FLAGS, str(KERNEL_SRC), '-o', str(tmp)],
                       check=True, capture_output=True, text=True)
        os.replace(tmp, lib)
    return lib


def _skimage_version() -> str:
    import skimage
    return skimage.__version__


def _skimage_026() -> bool:
    """Is this the version the kernel ports (0.26)? — 0.25 queues seeds with -inf, so the fallback path's labels may
    differ."""
    return _skimage_version().split('.')[:2] == ['0', '26']


def _skimage_seed_cost():
    """Seed rule of the installed scikit-image — 0.26 'terrain' · 0.25 '-inf' · otherwise unchecked (None)."""
    return {('0', '26'): 'terrain', ('0', '25'): '-inf'}.get(tuple(_skimage_version().split('.')[:2]))


def _kernel():
    """C++ watershed function (built · loaded once, on first use) — None if unavailable, with the reason left in
    `_BACKEND` and on stderr."""
    global _FLOOD, _BACKEND
    if _BACKEND is not None:
        return _FLOOD
    want = env('CWR_SEGMENT_BACKEND', 'auto')
    if want == 'skimage':
        _BACKEND = dict(name='skimage', reason='CWR_SEGMENT_BACKEND=skimage', library=None)
        return None
    try:
        lib = build()
        fn = ctypes.CDLL(str(lib)).compact_flood
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64,
                       ctypes.c_void_p, ctypes.c_int64, ctypes.c_double, ctypes.c_int, ctypes.c_void_p]
        fn.restype = ctypes.c_int
    except Exception as e:                                         # no compiler · build failed · load failed
        if want == 'cpp':
            raise
        why = (getattr(e, 'stderr', None) or str(e)).strip().splitlines()
        _BACKEND = dict(name='skimage', reason=f'C++ kernel unavailable — {type(e).__name__}: {why[-1] if why else ""}', library=None)
        same = ('same labels' if SEED_COST == _skimage_seed_cost()
                else f'scikit-image {_skimage_version()} seeds differently and will stop')
        print(f"[segment] {_BACKEND['reason']} · using scikit-image watershed instead ({same}; slower)", file=sys.stderr, flush=True)
        return None
    _FLOOD = fn
    _BACKEND = dict(name='cpp', reason='compact_watershed.cpp', library=str(lib))
    return _FLOOD


def backend() -> dict:
    """Implementation used for watershed flooding — {'name': 'cpp' | 'skimage', 'reason', 'library'}."""
    _kernel()
    return dict(_BACKEND)


def _flood(terrain: np.ndarray, mk: np.ndarray) -> np.ndarray:
    """Same labels (int32) as `skimage.segmentation.watershed(terrain, markers=mk, connectivity=2, compactness=COMPACT)`.

    Input conversion · padding · neighbour order use scikit-image's functions as is; only the heap flooding runs in C++.
    As in the original, distances are measured with the strides of the padded terrain (for an F-order terrain, 1-D
    distances — this original behaviour, which arises with rotated attention maps, is kept).
    If the seeds are not C-ordered, the original fills the seeds into a copy and the result differs, so scikit-image is
    called directly."""
    fn = _kernel()
    if fn is None or not np.asarray(mk).flags.c_contiguous:
        if SEED_COST != _skimage_seed_cost():
            raise RuntimeError(f'SEED_COST={SEED_COST!r} cannot be reproduced with scikit-image {_skimage_version()} — '
                               'the C++ kernel is needed')
        return _ws(terrain, markers=mk, connectivity=2, compactness=COMPACT)
    image, markers, mask = _wsmod._validate_inputs(terrain, mk, None, 2)
    conn, offset = _wsmod._validate_connectivity(image.ndim, 2, None)
    pad = [(p, p) for p in offset]
    image = np.pad(image, pad, mode='constant')
    strides = np.asarray(image.strides, np.int64) // image.itemsize
    flat = np.ascontiguousarray(image.ravel(), np.float64)
    valid = np.ascontiguousarray(np.pad(mask, pad, mode='constant').ravel(), np.uint8)
    out = np.ascontiguousarray(np.pad(markers, pad, mode='constant'), np.int32)
    nhood = np.ascontiguousarray(_wsmod._offsets_to_raveled_neighbors(image.shape, conn, center=offset), np.int64)
    seeds = np.flatnonzero(out).astype(np.int64)
    rc = fn(flat.ctypes.data, valid.ctypes.data, seeds.ctypes.data, len(seeds), nhood.ctypes.data, len(nhood),
            strides.ctypes.data, len(strides), float(COMPACT), int(SEED_COST == '-inf'), out.ctypes.data)
    if rc:
        raise RuntimeError('compact_flood failed')
    return out[tuple(slice(p, -p) for p in offset)].copy()


def watershed(maps: np.ndarray, ink_map: np.ndarray, mk: np.ndarray) -> np.ndarray:
    """Watershed with flooding + distance from seeds `mk` -> (H,W) int16 labels 0..K-1. `maps` are scaled once more here."""
    m = norm(maps, ink_map.shape)
    if not len(m):
        return np.zeros(ink_map.shape, np.int16)
    lab = _flood(relief(m, ink_map, 0.0), mk)
    return (lab.astype(np.int16) - 1)


def cut(rgb, maps_px, points):
    """Position segmentation -> `lab` (H,W) int16 with position indices 0..K-1 · None if there are no seeds.

    `rgb` original (H,W,3) uint8 · `maps_px` per-position attention maps (K,h0,w0) (original coordinates ·
    `position.px_maps`) · `points` K character points [(x, y), ...] (original px · p_tl). They are converted to Python
    floats here — using a float32 array as is computes the position spacing in float32 and changes the disc radius.
    """
    cs = [(float(x), float(y)) for x, y in points]
    hw = np.asarray(rgb).shape[:2]
    m = norm(maps_px, hw)
    mk = disc_markers(m, cs, SEED_R * unit_w(cs, m.shape[1:]))
    if mk.max() == 0:
        return None
    # Same values as `watershed(m, ink(rgb), mk)` with less computation — ink has weight 0, so only its shape is used
    # (0·ink = 0), and rescaling the scaled maps leaves them unchanged. Adding the zero array stays: the terrain's array
    # order (C/F) decides the distance computation.
    return (_flood(-(m.max(0) + np.zeros(hw, np.float32)), mk).astype(np.int16) - 1)


def nearest(rgb, maps_px, points):
    """Position segmentation (default) -> `lab` (H,W) int16 with position indices 0..K-1 · None if there are no seeds.

    Seeds are the same discs as in `cut` (`disc_markers`); a non-seed pixel takes the position of the nearest seed pixel
    via the Euclidean distance transform (on equal distance, the pixel the distance transform picks). Attention maps are
    used only to decide the seeds — the path with the same result that does not scale the maps when the discs do not
    overlap is `nearest_lazy`.
    """
    cs = [(float(x), float(y)) for x, y in points]
    hw = np.asarray(rgb).shape[:2]
    m = norm(maps_px, hw)
    return nearest_labels(disc_markers(m, cs, SEED_R * unit_w(cs, m.shape[1:])))


def nearest_labels(mk):
    """Seeds (H,W) int32 (0 = none · 1..K) -> position index of the nearest seed pixel (H,W) int16 · None if there are
    no seeds."""
    from scipy.ndimage import distance_transform_edt
    if not np.any(mk > 0):
        return None
    index = distance_transform_edt(mk == 0, return_distances=False, return_indices=True)
    return mk[tuple(index)].astype(np.int16) - 1


def disjoint_disc_markers(hw, cs, raw_maps, r):
    """The same seeds as `disc_markers` from geometry alone — only when no two discs (character-point pixels included)
    share a pixel, otherwise None. Attention only decides the owner of overlapping pixels (scaled maps are in [0, 1], so
    each disc gets all its pixels), so without overlap these are the same seeds without maps. None (use the current
    computation) if the maps contain NaN · negatives or the radius is not finite."""
    H, W = hw
    maps = np.asarray(raw_maps, dtype=np.float32)
    if (not len(cs) or H < 1 or W < 1 or len(maps) != len(cs) or maps.size == 0 or not np.isfinite(maps).all()
            or np.min(maps) < 0 or not np.isfinite(r) or not np.isfinite(np.asarray(cs, float)).all()):
        return None
    out = np.zeros((H, W), np.int32)
    for k, (x, y) in enumerate(cs):
        xi = int(min(max(round(x), 0), W - 1))
        yi = int(min(max(round(y), 0), H - 1))
        rr = max(float(r), 0.)
        x0 = max(0, min(xi, int(np.floor(x - rr))))
        x1 = min(W, max(xi + 1, int(np.ceil(x + rr)) + 1))
        y0 = max(0, min(yi, int(np.floor(y - rr))))
        y1 = min(H, max(yi + 1, int(np.ceil(y + rr)) + 1))
        yy, xx = np.mgrid[y0:y1, x0:x1]
        claim = ((xx-x)**2 + (yy-y)**2 <= r*r) if r > 0 else np.zeros(xx.shape, bool)
        claim[yi-y0, xi-x0] = True
        dst = out[y0:y1, x0:x1]
        if np.any(dst[claim] > 0):
            return None
        dst[claim] = k + 1
    for k, (x, y) in enumerate(cs):
        out[int(min(max(round(y), 0), H-1)), int(min(max(round(x), 0), W-1))] = k + 1
    return out


def nearest_lazy(rgb, raw_maps, maps_px, points):
    """Same labels as `nearest(rgb, maps_px(), points)` — does not scale the attention maps when the position discs do
    not overlap. raw_maps: grid maps (the record's maps) · maps_px: no-argument function that builds the original-px
    maps (called only on overlap)."""
    cs = [(float(x), float(y)) for x, y in points]
    hw = np.asarray(rgb).shape[:2]
    r = SEED_R * unit_w(cs, hw)
    mk = disjoint_disc_markers(hw, cs, raw_maps, r)
    if mk is None:
        mk = disc_markers(norm(maps_px(), hw), cs, r)
    return nearest_labels(mk)


SEGMENTATIONS = {'nearest': nearest, 'watershed': cut}


def label_at(lab, bg0, cx, cy):
    """Segment the point (cx, cy) falls in -> (label, whether it was background).

    If `bg0` is true and that pixel is background (0), the nearest nonzero segment is used instead — (None, True) if
    there is none.
    """
    h, w = lab.shape
    x = int(np.clip(round(cx), 0, w - 1))
    y = int(np.clip(round(cy), 0, h - 1))
    v = int(lab[y, x])
    if not (bg0 and v == 0):
        return v, False
    ys, xs = np.nonzero(lab != 0)
    if not len(xs):
        return None, True
    j = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
    return int(lab[ys[j], xs[j]]), True


def touching(lab, lid, bg0):
    """Set of segment labels directly touching segment `lid` in 4-neighbourhood (excluding itself · also excluding 0 if
    `bg0`)."""
    if lid is None:
        return set()
    m = lab == lid
    adj = set()
    for a, b in ((m[:, :-1], lab[:, 1:]), (m[:, 1:], lab[:, :-1]),
                 (m[:-1, :], lab[1:, :]), (m[1:, :], lab[:-1, :])):
        if a.any():
            adj.update(int(v) for v in np.unique(b[a]))
    adj.discard(int(lid))
    if bg0:
        adj.discard(0)
    return adj


def clipped_labels(lab, mask, int64=False):
    """Labels with pixels outside `mask` set to -1 (after promotion to int64 if `int64`)."""
    return np.where(mask, lab.astype(np.int64) if int64 else lab, -1)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Build and check the C++ watershed kernel')
    ap.add_argument('--build', action='store_true', help='rebuild even when the source is unchanged')
    a = ap.parse_args()
    if a.build:
        build(force=True)
    print(backend())
