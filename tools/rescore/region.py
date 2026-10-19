"""Text-line base band TR · rbox · W window region (TRER; TRW1.5 when no rbox can be built).

Inputs are the original RGB · the segmentation lab (`segment.cut`) · character points p_tl · the text-line curve tl_4
(`position.derive`) · the attention record rec (keys text · maps · gw · gh · wh0 · rotated). All coordinates are px of
the original image.

    TR     = band between cap height and baseline along the text-line curve (base band before ink expansion).
             Used to decide whether neighbouring segments touch.
    rbox   = per character, a rectangle centred at p_tl and aligned with the tl_4 tangent. Foreground (Sauvola k 0.2 ·
             polarity = band-border log ratio + attention log ratio) ∩ dilated segment S_i: outline of the components
             with an attention share ≥ 0.2 → attach detached dots → margin.
    TRER   = TR ∪ (rboxes that overlap TR). The pixel extent of a W window.
    TRW1.5 = band of text-line centre ± 1.5 x-heights. Replacement for images where no rbox could be built
             (a one-character image has no text line, so the whole image).

Public functions: tr_lines · strip_mask · tr_strip · wide_band · current_boxes · trer_region · window_region.

Text-line curve point thinning (`_spaced_indices`) runs as a C++ loop with the same arithmetic (`curve_spacing.cpp`). On
the first call it is built with g++ in the same place as the segment kernel (`LEGIT_RESCORE_KERNEL_DIR`); if building or
loading fails, a Python loop is used instead (same result · slower · `spacing_backend()`).
With `LEGIT_REGION_BACKEND=python` Python is used from the start; with `=cpp` it stops when the kernel is unavailable.
"""
from __future__ import annotations

import ctypes
import hashlib
import math
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from skimage.filters import threshold_sauvola

from tools.rescore.position import (angles, common_min_maps, corner_prior_maps, grid_points, line,
                                    neighbor_competition_maps, px_maps)

# The old path set OpenCV to a single thread when it called this computation (many small operations make that
# faster) — keep it the same.
cv2.setNumThreads(1)

# ---------------------------------------------------------------- constants

CAP_OVER_XH = 1.333                 # cap height ÷ x-height (median of six synthetic fonts)
BORDER_UP, BORDER_DN = 2.5, 1.5     # character region top · bottom borders — x-heights from the baseline (+2.5 · −1.5)
EXPANDED_PAD = 0.5                  # expanded band for polarity — TR widened along the normal by x-height × 0.5 per side
SAUVOLA_K, SAUVOLA_R = .2, 127.5    # Sauvola threshold (window = x-height × 1, odd)
POL_FLOOR = 1e-3                    # lower bound of the foreground rate in the polarity score
VREACH = .8                         # search window vertical half-width = VREACH × max(decision-map half-height, spacing/2)
VSHIFT = .5                         # shift search window centre vertically by font metric (top+bottom)/2 × spacing × VSHIFT
TAU = .2                            # component selection threshold — attention mass ÷ total mass on foreground
DIL = .05                           # radius for dilating segment S_i — p_tl spacing × DIL
# characters whose detached dot is attached · within x-height × DOT_REACH above the box (below for ! ?)
DOT_CHARS, DOT_REACH = 'ij!?;:', .5
NO_CLIP = 99.                       # no horizontal clip — the search window ends at the neighbouring p_tl, so the box stays in it
# geometric box when no component is picked — neighbour p_tl midpoint × 1.2 · decision-map half-height × 0.55
GEO_KU, GEO_KV = 1.2, .55
# box margin — horizontal, each side height × 0.10 · vertical height × 0.05 · minimum width height × 0.5
PAD_U, PAD_V, PAD_MIN_W = .10, .05, .5
RBOX_MASS = .9                      # mass kept by the decision-map quantile box (for the half-height)
TRW_HALF = 1.5                      # replacement band when rbox fails — text-line centre ± 1.5 x-height
RASTER_SHIFT = 8                    # rbox polygon raster precision 1/256 px

RBOX_VERSION = 'rbox-si-k0.2-dot-logsumpol-v6'
PREPARATION = 'current-rbox-prepared-v1'

# The 10 fonts used for font metrics (per-character top · bottom) — found with fc-match.
FONT_PATTERNS = ('DejaVu Sans:style=Book', 'DejaVu Sans:style=Bold',
                 'DejaVu Serif:style=Book', 'DejaVu Serif:style=Bold', 'DejaVu Sans Mono:style=Book',
                 'Liberation Sans:style=Regular', 'Liberation Sans:style=Bold',
                 'Liberation Serif:style=Regular', 'Liberation Serif:style=Bold',
                 'Liberation Sans Narrow:style=Regular')


# ---------------------------------------------------------------- text lines · bands

SPACING_SRC = Path(__file__).with_name('curve_spacing.cpp')
_SPACING = None
_SPACING_BACKEND = None


def _spacing_kernel():
    """C++ thinning function (built · loaded once, on first use) — None if unavailable, with the reason left in
    `_SPACING_BACKEND` and on stderr."""
    global _SPACING, _SPACING_BACKEND
    if _SPACING_BACKEND is not None:
        return _SPACING
    want = os.environ.get('LEGIT_REGION_BACKEND', 'auto')
    if want == 'python':
        _SPACING_BACKEND = dict(name='python', reason='LEGIT_REGION_BACKEND=python', library=None)
        return None
    try:
        from tools.rescore.segment import CXX_FLAGS, KERNEL_DIR
        lib = KERNEL_DIR / f'curve_spacing_{hashlib.sha256(SPACING_SRC.read_bytes()).hexdigest()[:16]}.so'
        if not lib.exists():
            KERNEL_DIR.mkdir(parents=True, exist_ok=True)
            tmp = lib.with_name(f'{lib.name}.{os.getpid()}.tmp')
            subprocess.run([os.environ.get('CXX', 'g++'), *CXX_FLAGS, str(SPACING_SRC), '-o', str(tmp)],
                           check=True, capture_output=True, text=True)
            os.replace(tmp, lib)
        fn = ctypes.CDLL(str(lib)).spaced_indices
        fn.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_double, ctypes.c_double,
                       ctypes.c_void_p, ctypes.c_void_p]
        fn.restype = ctypes.c_int64
    except Exception as e:                                         # no compiler · build failed · load failed
        if want == 'cpp':
            raise
        why = (getattr(e, 'stderr', None) or str(e)).strip().splitlines()
        _SPACING_BACKEND = dict(name='python', reason=f'C++ kernel unavailable — {type(e).__name__}: {why[-1] if why else ""}', library=None)
        print(f"[region] {_SPACING_BACKEND['reason']} · falling back to the Python loop (same result, slower)", file=sys.stderr, flush=True)
        return None
    _SPACING = fn
    _SPACING_BACKEND = dict(name='cpp', reason='curve_spacing.cpp', library=str(lib))
    return _SPACING


def spacing_backend() -> dict:
    """Implementation used for curve point thinning — {'name': 'cpp' | 'python', 'reason', 'library'}."""
    _spacing_kernel()
    return dict(_SPACING_BACKEND)


def _spaced_indices(points, threshold):
    """Indices of the points kept from the sequence: a point is kept when its distance from the last kept point is
    ≥ threshold (first point included) — C++ loop, or `_spaced_indices_py` when unavailable.
    Near the boundary (|distance − threshold| ≤ 1e-12·max(1, threshold)) and for infinite distances it follows the
    Python rule that decides with numpy norm."""
    fn = _spacing_kernel()
    if fn is None or len(points) < 2:
        return _spaced_indices_py(points, threshold)
    xy = np.ascontiguousarray(points, dtype=np.float64)
    n = len(xy)
    tolerance = 1e-12 * max(1., threshold)
    out = np.empty(n, np.int64); got = np.zeros(1, np.int64)
    keep, start = [0], 1
    while start < n:
        stop = int(fn(xy.ctypes.data, n, start, keep[-1], float(threshold), tolerance, out.ctypes.data, got.ctypes.data))
        keep.extend(out[:int(got[0])].tolist())
        if stop < 0:
            break
        if np.linalg.norm(points[stop] - points[keep[-1]]) >= threshold:
            keep.append(stop)
        start = stop + 1
    return keep


def _spaced_indices_py(points, threshold):
    """Keep the original scalar arithmetic and near-boundary norm fallback."""
    keep = [0]
    if len(points) < 2:
        return keep
    rows = points.tolist()
    last_x, last_y = float(rows[0][0]), float(rows[0][1])
    tolerance = 1e-12 * max(1., threshold)
    sqrt, isfinite = math.sqrt, math.isfinite
    for i, row in enumerate(rows[1:], 1):
        x, y = float(row[0]), float(row[1])
        dx, dy = x - last_x, y - last_y
        distance = sqrt(dx * dx + dy * dy)
        if not isfinite(distance) or abs(distance - threshold) <= tolerance:
            distance = np.linalg.norm(points[i] - points[keep[-1]])
        if distance >= threshold:
            keep.append(i)
            last_x, last_y = x, y
    return keep


def _derive_lines(f_pts, s, up=BORDER_UP, down=BORDER_DN):
    """Six lines from the text-line curve points and x-height s (baseline · text line · x-height · cap height · upper ·
    lower border).

    Shifts are multiples of x-height from the baseline (up is +), applied along the text-line normal. The normal follows
    the reading direction.
    """
    return {k: v.tolist() for k, v in _line_arrays(f_pts, s, up, down).items()}


def _line_arrays(f_pts, s, up=BORDER_UP, down=BORDER_DN, keys=None):
    """The lines of `_derive_lines` as float arrays (n, 2) — same values (rounded to three decimals · short point
    sequences are not rounded).
    If keys is given, only those lines are computed (the two bands use only the upper · lower borders and cap height ·
    baseline)."""
    up_ = {'baseline': 0.0, 'textline': 0.5, 'xheight': 1.0,
           'capheight': CAP_OVER_XH,
           'bordup': up, 'borddn': -down}
    off = {k: 0.5 - v for k, v in up_.items() if keys is None or k in keys}   # from the text line · screen coords
    P = np.asarray(f_pts, float)
    if len(P) < 2:
        return {k: np.asarray([[float(x), float(y + v * s)] for x, y in f_pts], float).reshape(-1, 2) for k, v in off.items()}
    # Thin the points to max(s/4, 0.5px) spacing, then measure tangents (avoids rounding jitter of dense points).
    step_ = max(float(s) * 0.25, 0.5)
    keep = _spaced_indices(P, step_)
    if keep[-1] != len(P) - 1:
        keep.append(len(P) - 1)
    P = P[keep]
    if len(P) < 2:
        P = np.asarray(f_pts, float)[[0, -1]]
    tg = np.gradient(P, axis=0)
    nrm = np.linalg.norm(tg, axis=1, keepdims=True)
    # Fill coincident points (zero gradient) with the overall direction.
    dead = (nrm[:, 0] < 1e-6)
    if dead.any():
        g = P[-1] - P[0]
        gn = float(np.linalg.norm(g))
        g = (g / gn) if gn > 1e-9 else np.array([1.0, 0.0])
        tg[dead] = g
        nrm[dead] = 1.0
    tg = tg / np.maximum(nrm, 1e-9)
    nv = np.stack([-tg[:, 1], tg[:, 0]], 1)
    return {k: (P + nv * (v * s)).round(3) for k, v in off.items()}


def tr_lines(geometry):
    """Six lines from the tl_4 curve samples — scale is geometry['size'] (x-height)."""
    size = float(geometry.get('size', 0) or 0)
    return _derive_lines(line(geometry), size)


def strip_mask(lines, shape):
    """Rasterize the band between the cap-height line and the baseline as a union of adjacent triangles (1/256 px) —
    clipped outside the image."""
    cap = np.asarray(lines['capheight'], float)
    base = np.asarray(lines['baseline'], float)
    assert cap.shape == base.shape and cap.ndim == 2 and cap.shape[1] == 2
    mask = np.zeros(shape, np.uint8)
    # Each vertex was converted repeatedly in adjacent triangles. Convert it
    # once, retaining the exact rint/int32 arithmetic and triangle draw order.
    # Do not batch triangles into fillPoly: overlapping pieces would then use
    # even/odd fill rather than this union of separately filled triangles.
    cap = np.rint(cap * 256).astype(np.int32)
    base = np.rint(base * 256).astype(np.int32)
    triangles = np.empty((2 * max(0, len(cap) - 1), 3, 2), np.int32)
    triangles[0::2, 0], triangles[0::2, 1], triangles[0::2, 2] = cap[:-1], cap[1:], base[1:]
    triangles[1::2, 0], triangles[1::2, 1], triangles[1::2, 2] = cap[:-1], base[1:], base[:-1]
    for triangle in triangles:
        cv2.fillConvexPoly(mask, triangle, 1, shift=8)
    return mask > 0


def tr_strip(shape, geometry):
    """Base band TR (cap height to baseline) — the whole image if there is no text-line curve."""
    size = float(geometry.get('size', 0))
    if geometry.get('samples', 0) < 2 or size <= 0:
        return np.ones(shape, bool)
    return strip_mask(_line_arrays(line(geometry), size, keys=('capheight', 'baseline')), shape)


def wide_band(shape, geometry, half=TRW_HALF):
    """Band of text-line centre (baseline +0.5) ± half x-heights — the whole image if there is no text-line curve."""
    size = float(geometry.get('size', 0))
    if geometry.get('samples', 0) < 2 or size <= 0:
        return np.ones(shape, bool)
    half = float(half)
    lines = _line_arrays(line(geometry), size, up=.5 + half, down=half - .5, keys=('bordup', 'borddn'))
    return strip_mask(dict(capheight=lines['bordup'], baseline=lines['borddn']), shape)


def wide_and_tr_bands(shape, geometry, half=TRW_HALF):
    """TRW and TR from one sampled curve/normal calculation; identical raster rules."""
    size = float(geometry.get('size', 0))
    if geometry.get('samples', 0) < 2 or size <= 0:
        return np.ones(shape, bool), np.ones(shape, bool)
    half = float(half)
    lines = _line_arrays(line(geometry), size, up=.5 + half, down=half - .5, keys=('bordup', 'borddn', 'capheight', 'baseline'))
    wide = strip_mask(dict(capheight=lines['bordup'], baseline=lines['borddn']), shape)
    return wide, strip_mask(lines, shape)


# ---------------------------------------------------------------- foreground · polarity

def _normalized_gray(rgb, reference_mask):
    """Grayscale of the whole image, min-max normalized by the minimum · maximum intensity inside TR (unchanged if the
    band is empty or flat)."""
    gray = cv2.cvtColor(np.asarray(rgb, np.uint8), cv2.COLOR_RGB2GRAY)
    reference_mask = np.asarray(reference_mask, bool)
    if reference_mask.shape != gray.shape:
        raise ValueError('normalization reference mask must match image shape')
    values = gray[reference_mask]
    lo, hi = (float(values.min()), float(values.max())) if values.size else (None, None)
    flat = not values.size or lo == hi
    normalized = gray.copy() if flat else np.rint(
        np.clip((gray.astype(np.float64) - lo) / (hi - lo), 0, 1) * 255
    ).astype(np.uint8)
    return cv2.cvtColor(np.repeat(normalized[..., None], 3, axis=2), cv2.COLOR_RGB2GRAY)


def _sauvola_positive(gray, xheight, ink_high):
    """Sauvola foreground (window = x-height × 1, odd · k 0.2) — if ink_high, light characters (255 − intensity)."""
    size = max(3, int(round(float(xheight) * 1)) | 1)
    source = 255 - gray if ink_high else gray
    threshold = threshold_sauvola(source, window_size=size, k=SAUVOLA_K, r=SAUVOLA_R)
    return source <= threshold


def _full_image_ink_high(dark, light, shape):
    """The side with less foreground on the image border is the character (on a tie, less total foreground · then dark
    characters)."""
    mask = np.ones(shape, bool)
    boundary = mask & ~cv2.erode(mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
        borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    rank = {}
    for name, positive in [('dark', dark), ('light', light)]:
        positive = np.asarray(positive, dtype=bool)
        if positive.shape != mask.shape:
            raise ValueError('Foreground and region shapes must match')
        rank[name] = (int((positive & boundary).sum()), int((positive & mask).sum()))
    return rank['light'] < rank['dark']


def _expanded_band(lines, size, shape):
    """Expanded band for polarity and its border (the two curves · image border inside the band)."""
    cap = np.asarray(lines['capheight'], float)
    base = np.asarray(lines['baseline'], float)
    if (cap.shape != base.shape or cap.ndim != 2 or cap.shape[1] != 2
            or len(cap) < 2 or not np.isfinite([cap, base]).all()
            or not np.isfinite(size) or size <= 0):
        raise ValueError('Expanded band needs paired finite curves and a positive x-height scale')
    distance = np.linalg.norm(base - cap, axis=1)
    if (distance <= 1e-8).any():
        raise ValueError('Cap-height and baseline must define a normal at every point')
    normal = (base - cap) / distance[:, None]
    padding = float(size) * EXPANDED_PAD
    upper, lower = cap - padding * normal, base + padding * normal
    region = strip_mask(dict(capheight=upper.tolist(), baseline=lower.tolist()), shape)
    edges = []
    for curve in (upper, lower):
        edge = np.zeros(shape, np.uint8)
        cv2.polylines(edge, [np.rint(curve * 256).astype(np.int32)],
                      False, 1, thickness=1, lineType=cv2.LINE_8, shift=8)
        edges.append(edge > 0)
    curve_boundary = edges[0] | edges[1]
    region |= curve_boundary
    frame = np.zeros(shape, bool)
    frame[[0, -1], :] = True
    frame[:, [0, -1]] = True
    boundary = curve_boundary | (frame & region)
    return region, boundary


def _band_ink_high(dark, light, region, boundary, shape):
    """The side with less foreground on the expanded-band border is the character (on a tie, band foreground · then dark
    characters). Without a border, the image-border rule."""
    full = _full_image_ink_high(dark, light, shape)
    if not int(boundary.sum()):
        return full
    rank = {}
    for name, positive in (('dark', dark), ('light', light)):
        positive = np.asarray(positive, bool)
        if positive.shape != region.shape or boundary.shape != region.shape:
            raise ValueError('Foreground, region and boundary shapes must match')
        rank[name] = (int((positive & boundary).sum()), int((positive & region).sum()))
    return rank['light'] < rank['dark']


def _logsum_polarity(dark, light, edge, mp):
    """Polarity score band + attn (> 0 means dark characters).

    Attention mask = per-position maps divided by their own maximum, then the per-pixel maximum → Otsu on 0–255.
    attn = ln(dark inside/outside) − ln(light inside/outside) ·
    band = ln(light band-border foreground rate) − ln(dark band-border foreground rate).
    """
    mp = np.asarray(mp, float)
    amap = (mp / np.maximum(mp.reshape(len(mp), -1).max(1), 1e-12)[:, None, None]).max(0)
    v = np.clip(amap * 255, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    Am = v > t
    rate = lambda m, M: max(float(m[M].mean()) if M.any() else 0., POL_FLOOR)
    attn = (np.log(rate(dark, Am)) - np.log(rate(dark, ~Am))) - (np.log(rate(light, Am)) - np.log(rate(light, ~Am)))
    band = np.log(rate(light, edge)) - np.log(rate(dark, edge))
    return float(band + attn)


def _foreground(rec, rgb, points, geometry, source_lab):
    """Foreground and segmentation of one image (model input orientation) — None if there is no text line."""
    size = float(geometry.get('size', 0) or 0)
    if len(points) < 2 or geometry.get('samples', 0) < 2 or size <= 0:
        return None
    shape = rgb.shape[:2]
    lines = tr_lines(geometry)
    strip = strip_mask(lines, shape)
    gray = _normalized_gray(rgb, strip)
    dark, light = [_sauvola_positive(gray, size, h) for h in (False, True)]
    band, edge = _expanded_band(lines, size, shape)
    ink_high = bool(_band_ink_high(dark, light, band, edge, shape))
    mp = px_maps(list(rec['maps']), rec['gw'], rec['gh'], rec['wh0'], rec['rotated']) if rec['maps'].size else None
    if mp is not None and edge.any():
        ink_high = not (_logsum_polarity(dark, light, edge, mp) > 0)
    fg = light if ink_high else dark
    lab = np.asarray(source_lab)
    if rec['rotated']:
        fg = np.rot90(fg); lab = np.rot90(np.asarray(lab))
    lab = np.asarray(lab, np.int16)
    return dict(fg=fg, lab=lab, fg32=fg.astype(np.float32), lab32=lab.astype(np.float32),
                ink_high=ink_high, polarity='logsum', size=size)


# ---------------------------------------------------------------- font metrics

def _font_paths():
    paths = []
    for pattern in FONT_PATTERNS:
        path, family, style = subprocess.check_output(
            ['fc-match', '-f', '%{file}\n%{family}\n%{style}', pattern], text=True).splitlines()
        expected_family, expected_style = pattern.split(':style=')
        if expected_family not in family.split(',') or expected_style not in style.split(','):
            raise RuntimeError(f'Font substitution: {pattern} -> {family} / {style}')
        paths.append((path, hashlib.sha256(Path(path).read_bytes()).hexdigest()))
    if len({h for _, h in paths}) != len(paths):
        raise RuntimeError('Requested glyph fonts resolve to duplicate files')
    return [p for p, _ in paths]


_FONT = {}
FONT_TABLE = Path(__file__).resolve().parent / 'data' / 'font_metrics.json'


def font_metrics():
    """Mean over 94 characters × 10 fonts — in x-height units (ink width · top · bottom above the x-height middle (up +) ·
    horizontal centre offset).
    Reads `data/font_metrics.json`, a frozen table of values measured in the manuscript run environment (openocr · PIL ·
    font file sha256 recorded in the table) — so machines with different font · PIL versions use the same values.
    To re-measure the table, `measure_font_metrics()`."""
    if not _FONT:
        import json
        for c, v in json.loads(FONT_TABLE.read_text())['metrics'].items():
            _FONT[c] = np.asarray(v, dtype=np.float64)
    return _FONT


def measure_font_metrics():
    """Re-measure the font_metrics table with this machine's fonts (fc-match · 200 px rendering) — only when building a
    new table."""
    from PIL import Image, ImageDraw, ImageFont
    out = {}
    chars = [chr(c) for c in range(33, 127)]
    acc = {c: [] for c in chars}
    for path in _font_paths():
        f = ImageFont.truetype(path, 200)
        def ink(c):
            im = Image.new('L', (600, 600), 0); ImageDraw.Draw(im).text((150, 150), c, fill=255, font=f)
            a = np.asarray(im) > 127; ys, xs = np.nonzero(a)
            return (xs.min(), xs.max() + 1, ys.min(), ys.max() + 1) if len(xs) else None
        x0, x1, y0, y1 = ink('x'); xh = y1 - y0; mid = (y0 + y1) / 2
        for c in chars:
            b = ink(c)
            if b is None:
                continue
            adv = f.getlength(c)
            acc[c].append(((b[1] - b[0]) / xh, (mid - b[2]) / xh, (mid - b[3]) / xh, ((b[0] + b[1]) / 2 - 150 - adv / 2) / xh))
    for c, v in acc.items():
        out[c] = np.mean(np.asarray(v), axis=0)
    return out


# ---------------------------------------------------------------- position frame · search window

def _axes(a):
    t = np.stack([np.cos(a), np.sin(a)], -1)
    return t, np.stack([-t[..., 1], t[..., 0]], -1)


def _proj(r, i, pts):
    t, n = _axes(r['angle'][i])
    d = np.asarray(pts, float) - r['ptl'][i]
    return d @ t, d @ n


def _neighbor_gaps(r, i):
    """Distances along the tangent to the previous · next p_tl (positive). If one is missing, the other · if both are
    missing, the median spacing."""
    n = len(r['ptl'])
    prev = -_proj(r, i, r['ptl'][i - 1][None])[0][0] if i > 0 else None
    nxt = _proj(r, i, r['ptl'][i + 1][None])[0][0] if i + 1 < n else None
    prev = None if prev is not None and prev <= 1e-6 else prev
    nxt = None if nxt is not None and nxt <= 1e-6 else nxt
    if prev is None and nxt is None:
        prev = nxt = r['spacing']
    return (prev if prev is not None else nxt), (nxt if nxt is not None else prev)


def _geometric(r, i):
    """Geometric box — horizontally to the previous · next p_tl midpoint × 1.2 · vertically decision-map half-height ×
    0.55."""
    gp, gn = _neighbor_gaps(r, i)
    hv = GEO_KV * r['radii'][i][1]
    return np.array([-GEO_KU * gp / 2, GEO_KU * gn / 2, -hv, hv])


def _margin(b):
    """Add a margin to a tight box — horizontally each side height × 0.10 · vertically height × 0.05 · widened if the width
    is below height × 0.5."""
    u0, u1, v0, v1 = b
    H = v1 - v0
    u0, u1, v0, v1 = u0 - PAD_U * H, u1 + PAD_U * H, v0 - PAD_V * H, v1 + PAD_V * H
    if u1 - u0 < PAD_MIN_W * H:
        c = (u0 + u1) / 2; u0, u1 = c - PAD_MIN_W * H / 2, c + PAD_MIN_W * H / 2
    return np.asarray([u0, u1, v0, v1], float)


def _search_window(r, i, vreach=VREACH, a=VSHIFT, reach=1.):
    """Search window of position i — horizontally to the previous · next p_tl · vertically ±vreach·max(ry, spacing/2) ·
    centre shifted by font metric × a.
    Window pixel density is 48 pixels per p_tl spacing (at least 1×). The decision map is bilinear from grid to pixels."""
    gp, gn = _neighbor_gaps(r, i)
    hv = vreach * max(r['radii'][i][1], .5 * r['spacing'])
    cv = 0.
    if a:
        M = font_metrics(); ch = r['ar'][i] if i < len(r['ar']) else 'x'
        _, top, bot, _ = M.get(ch, M['x']); cv = -(top + bot) / 2 * r['spacing'] * a
    U0, U1, V0, V1 = -reach * gp, reach * gn, cv - hv, cv + hv
    px = max(1., 48. / max(r['spacing'], 1e-6))
    t, n = _axes(r['angle'][i]); c = r['ptl'][i]
    W, H = max(4, int(round((U1 - U0) * px))), max(4, int(round((V1 - V0) * px)))
    su, sv = (U1 - U0) / W, (V1 - V0) / H
    uu = U0 + (np.arange(W) + .5) * su; vv = V0 + (np.arange(H) + .5) * sv
    X = c[0] + uu[None, :] * t[0] + vv[:, None] * n[0]
    Y = c[1] + uu[None, :] * t[1] + vv[:, None] * n[1]
    w, h = r['size']
    ins = (X >= -.5) & (X <= w - .5) & (Y >= -.5) & (Y <= h - .5)
    gx, gy = X / r['cell'][0], Y / r['cell'][1]
    gw, gh = r['grid']
    g = np.asarray(r['decided'][i], float).reshape(gh, gw)
    dec = map_coordinates(g, [gy, gx], order=1, mode='nearest')
    return dict(U0=U0, V0=V0, su=su, sv=sv, uu=uu, vv=vv, ins=ins, gp=gp, gn=gn, X=X, Y=Y, dec=dec)


def _components(fg):
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), 8)
    return n, lab, st


def _pick(Wd, fg, tau=TAU):
    """Foreground components with attention mass ÷ total mass on the foreground ≥ tau (area 3 or more) — mask of the
    selected pixels or None."""
    n, lab, st = _components(fg)
    A = Wd['dec'] * Wd['ins']
    mass = np.bincount(lab.ravel(), weights=(A * fg).ravel(), minlength=n)
    tot = mass[1:].sum()
    if tot <= 0:
        return None
    share = mass / tot
    keep = [j for j in range(1, n) if st[j, cv2.CC_STAT_AREA] >= 3 and share[j] >= tau]
    if not keep:
        return None
    m = np.isin(lab, keep) & fg
    return m if m.any() else None


def _bbox(Wd, m, clip=NO_CLIP):
    ys, xs = np.nonzero(m)
    a, b = Wd['U0'] + xs.min() * Wd['su'], Wd['U0'] + (xs.max() + 1) * Wd['su']
    c, d = Wd['V0'] + ys.min() * Wd['sv'], Wd['V0'] + (ys.max() + 1) * Wd['sv']
    a, b = max(a, -clip * Wd['gp'] / 2), min(b, clip * Wd['gn'] / 2)
    if b - a < 1e-3 or d - c < 1e-3:
        return None
    return np.array([a, b, c, d])


def _tight_box(r, i, parts):
    """Tight box [u0, u1, v0, v1] of position i (position frame) — None if nothing is selected.
    If the AR character is i · j · ! · ? · ; · :, add detached dot components within the horizontal extent of the
    selected box and above it (below for ! ?)."""
    Wd = _search_window(r, i)
    fg = (map_coordinates(parts['fg32'], [Wd['Y'], Wd['X']], order=0, mode='constant', cval=0) > .5) & Wd['ins']
    Sw = (map_coordinates(parts['lab32'], [Wd['Y'], Wd['X']], order=0, mode='constant', cval=-1) == i) & Wd['ins']
    rad = max(1, int(round(DIL * r['spacing'] / Wd['su'])))
    Sw = cv2.dilate(Sw.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))).astype(bool) & Wd['ins']
    m = _pick(Wd, fg & Sw)
    b = None if m is None else _bbox(Wd, m)
    ch = r['ar'][i] if i < len(r['ar']) else ''
    if b is not None and ch in DOT_CHARS and parts.get('size'):
        nn, lab, st = _components(fg & Sw)
        uc = Wd['U0'] + (st[:, cv2.CC_STAT_LEFT] + st[:, cv2.CC_STAT_WIDTH] / 2) * Wd['su']
        vc = Wd['V0'] + (st[:, cv2.CC_STAT_TOP] + st[:, cv2.CC_STAT_HEIGHT] / 2) * Wd['sv']
        reach, below = DOT_REACH * parts['size'], ch in '!?'
        add = [q for q in range(1, nn) if st[q, cv2.CC_STAT_AREA] >= 2 and b[0] <= uc[q] <= b[1]
               and ((b[3] < vc[q] <= b[3] + reach) if below else (b[2] - reach <= vc[q] < b[2]))]
        if add:
            b2 = _bbox(Wd, m | np.isin(lab, add))
            b = b if b2 is None else b2
    return b


def _quantile(values, weights, q):
    """Weighted inverse CDF (cell mass as is · no interpolation)."""
    order = np.argsort(values, kind='stable')
    return float(values[order[min(int(np.searchsorted(np.cumsum(weights[order]), q, side='left')), len(order)-1)]])


def _decision_radii(mass, xy, wh, angle, cell_size, keep=RBOX_MASS):
    """Half-width · half-height of the quantile box of decision-map cells projected onto the tangent · normal
    ((1−keep)/4 per tail · half-cell padding)."""
    p = np.asarray(mass, float)
    if not len(p) or np.any(p < 0) or not np.isfinite(p).all() or p.sum() <= 0:
        raise ValueError('Invalid decision map')
    p = p / p.sum()
    points = np.clip(np.asarray(xy, float), [0., 0.], np.asarray(wh)-1.)
    t = np.array([np.cos(angle), np.sin(angle)])
    n = np.array([-t[1], t[0]])
    u, v = points @ t, points @ n
    tail = (1-keep)/4
    du = float(np.abs(t) @ cell_size) / 2
    dv = float(np.abs(n) @ cell_size) / 2
    a, b, c, d = [_quantile(u, p, tail)-du, _quantile(u, p, 1-tail)+du,
                  _quantile(v, p, tail)-dv, _quantile(v, p, 1-tail)+dv]
    return [(b-a)/2, (d-c)/2]


# ---------------------------------------------------------------- rbox · window region

def current_boxes(rgb, lab, points, geometry, rec):
    """Four rbox corners per character (original px) — (corner list, meta). ValueError if they cannot be built.

    rgb (H, W, 3) uint8 · lab segmentation (H, W) · points K character points p_tl · geometry tl_4 · rec attention record.
    """
    rgb, lab = np.asarray(rgb, np.uint8), np.asarray(lab)
    pts = np.asarray(points, float).reshape(-1, 2)
    if lab.shape != rgb.shape[:2] or len(pts) != len(rec['text']):
        raise ValueError('rbox RGB, segmentation and text/point shapes differ')
    rotated = bool(rec['rotated']); nw = rgb.shape[1]
    view = np.rot90(rgb) if rotated else rgb
    h, w = view.shape[:2]; gw, gh = rec['gw'], rec['gh']
    ptl = np.column_stack([pts[:, 1], nw-1-pts[:, 0]]) if rotated else pts.copy()
    angle = angles(geometry, points) - (math.pi/2 if rotated else 0.)
    clean, _ = common_min_maps(rec['maps'])
    fixed = corner_prior_maps(neighbor_competition_maps(clean), gw, gh, rec['wh0'])
    _, sels = grid_points(rec['maps'], gw, gh, text=rec['text'], wh0=rec['wh0'])
    mass = np.asarray([f*np.asarray(s) for f, s in zip(fixed, sels)])
    if not np.isfinite(mass).all() or np.any(mass.sum(1) <= 0):
        raise ValueError('rbox has an empty/nonfinite decision map')
    mass /= mass.sum(1, keepdims=True)
    yy, xx = np.indices((gh, gw)); cell = np.array([w/max(gw-1, 1), h/max(gh-1, 1)])
    xy = np.clip(np.column_stack([xx.ravel(), yy.ravel()])*cell, [0, 0], [w-1, h-1])
    radii = [_decision_radii(m, xy, (w, h), a, cell) for m, a in zip(mass, angle)]
    gaps = np.linalg.norm(np.diff(ptl, axis=0), axis=1); valid = gaps[gaps > 1e-6]
    if not len(valid):
        raise ValueError('rbox needs nondegenerate inter-character spacing')
    r = dict(size=(w, h), grid=(gw, gh), ar=rec['text'], ptl=ptl, angle=angle, radii=np.array(radii),
             decided=mass, cell=cell, spacing=float(np.median(valid)))
    parts = _foreground(rec, rgb, points, geometry, lab)
    if parts is None:
        raise ValueError('rbox foreground unavailable')
    boxes = []
    for i in range(len(ptl)):
        tight = _tight_box(r, i, parts)
        boxes.append((np.asarray(_geometric(r, i), float), 'fallback_geo') if tight is None
                     else (np.asarray(_margin(tight), float), 'si'))
    corners = []
    for p, a, (b, _) in zip(ptl, angle, boxes):
        t = np.array([np.cos(a), np.sin(a)]); n = np.array([-t[1], t[0]])
        q = np.array([p+u*t+v*n for u, v in [(b[0], b[2]), (b[1], b[2]), (b[1], b[3]), (b[0], b[3])]])
        corners.append(np.column_stack([nw-1-q[:, 1], q[:, 0]]) if rotated else q)
    return corners, dict(version=RBOX_VERSION, preparation=PREPARATION, padding=True,
                         ink_high=bool(parts['ink_high']), polarity=parts['polarity'],
                         boxes=[dict(bounds=b.tolist(), how=how) for b, how in boxes])


def trer_region(shape, geometry, boxes):
    """TRER = TR ∪ (rboxes that overlap TR) — each rbox is rasterized as a convex polygon at 1/256 px. None if boxes is
    empty or not a list."""
    if not isinstance(boxes, list) or not boxes:
        return None
    base = tr_strip(shape, geometry); region = base.copy(); hit = 0
    for box in boxes:
        mask = np.zeros(shape, np.uint8)
        cv2.fillConvexPoly(mask, np.rint(np.asarray(box) * 256).astype(np.int32), 1, shift=RASTER_SHIFT)
        if np.any(mask.astype(bool) & base):
            region |= mask.astype(bool); hit += 1
    return region, dict(region_mode='trer', trer_boxes=len(boxes), trer_included=hit, pixels=int(region.sum()))


def rbox_or_reason(rgb, lab, points, geometry, rec):
    """List of rbox corners (list of [[x, y]×4]) or a failure reason string — same shape as the precomputed pickle
    values."""
    if rec is None:
        return 'missing_attention_record'
    try:
        boxes, _ = current_boxes(rgb, lab, points, geometry, rec)
        return [np.asarray(b, float).tolist() for b in boxes]
    except ImportError:                                # missing module: stop, not a silent TRW1.5 per image
        raise
    except Exception as e:
        return f'{type(e).__name__}: {e}'


def window_region(rgb, lab, points, geometry, rec, *, boxes=None, fallback='trw15'):
    """W window region — (region bool (H, W), meta). TRER; images where no rbox could be built get TRW1.5
    (a one-character image gets the whole image).

    boxes has the same meaning as the old precomputed pickle values — None computes them on the spot with current_boxes
    (an exception means failure) · a non-empty list gives TRER from those rboxes · any other value (failure reason
    string · empty list) is an rbox failure and gives TRW1.5 without computing.
    """
    if fallback != 'trw15':
        raise ValueError("fallback must be 'trw15'")
    shape = np.asarray(rgb).shape[:2]
    if boxes is None:
        boxes = rbox_or_reason(rgb, lab, points, geometry, rec)
    got = trer_region(shape, geometry, boxes)
    if got is not None:
        return got
    region = wide_band(shape, geometry, half=TRW_HALF)
    return region, dict(region_mode='trer', trer_fallback='trw15', pixels=int(region.sum()))
