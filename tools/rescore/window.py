"""Character windows — build a W-AR window image for each character position from one AR attention record.

Flow (for each character position i · default = the new definition `DEFINITION`): character point p_tl · text-line curve
tl_4 (`position` · refined p_tl `REFINED_POLICY`)
→ position segment S (`segment.nearest` · nearest seed) → window region (`region` · TRW = text line ± 1.5 x-height band)
→ bounding rectangle of (own segment + neighbour segments) ∩ region (neighbours = segments touching in 4-connectivity
inside the base band TR · default one hop = W3)
→ resized to the AR input size (ratio 1–4 · no vertical rotation). The window is not rotated.
The character positions in a window are "the positions whose p_tl lies inside the window image" (the scoring side
`window_score` asks for the substring of that range).

Earlier window definition (`PAPER_V1`, not used in the paper): stored p_tl (old policy) · watershed segmentation (`segment.cut`) · region TRER (TRW1.5 when no rbox
can be built) · the window is rotated upright when the reading tangent tilts by 45° or more or points backwards —
`prepare(..., **PAPER_V1)` · `inputs(..., ptl='original', segmentation='watershed')`.

The window images and slot ranges of `PAPER_V1` are bit-identical to those of the research code that produced the
stored v1 scores (TRER region, TR neighbour band, no fill); the comparison lives in the research repository.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

from tools.rescore import position, region as regions, segment

MAX_LEN = 25            # maximum output length of the recognizer (svtrv2-ar) — longer transcripts get no windows
PAD_PX = 2              # margin at the window rectangle edge (px)
MIN_WIN_PX = 3          # minimum side (px) that keeps a 1-px mask that vanished after rotation
ROT_MIN_DEG = 45.0      # no rotation when the reading tangent tilts less than this
HOPS = 1                # neighbour hops — 0 = W1 (own segment only) · 1 = W3 (default) · 2 = W5 · 3 = W7 · 4 = W9
REGIONS = ('trer', 'trw15', 'none')   # window region — TRER (earlier window definition · TRW1.5 if rbox fails) · TRW (default · 1.5× band) · none
PTL = {'refined': position.REFINED_POLICY, 'original': None}   # p_tl policy — refined (default) · old (earlier window definition)
# Window definition — the default (refined p_tl · nearest seed · TRW band · no rotation) and v1
DEFINITION = dict(region='trw15', rotate=False, ptl='refined', segmentation='nearest')
PAPER_V1 = dict(region='trer', rotate=True, ptl='original', segmentation='watershed')
# AR input — public svtrv2-ar evaluation setting (ratio rounded to [1, 4] · stretched to that cell size without padding).
# W windows get no vertical rotation (RotateTall).
AR_BASE_SHAPE = ((64, 64), (96, 48), (112, 40), (128, 32))   # (width, height) for ratio 1 · 2 · 3 · 4
AR_MAX_RATIO = 4
AR_BASE_H = 32
BACKGROUND_POLICY = 'full_image_border_otsu_median_v1'

# Parts of the version name of the paper · stored scores (`stroke_glyph_train_c2w.METHOD_VERSION`) — for name matching only
# (not used in the computation)
_ASSIGNMENT_VERSION = 'w-ctc-greedy-v3-pos0.5-sim1-tau0-nofill'
_AGGREGATION_VERSION = 'all-w-mean-v1'
_R_POLICY = {'v1': 'w-r-inside-fill-v1', 'v2': 'w-r-inside-fill-v2'}


def method_version(region='trer', hops=HOPS, fallback='trw15', neighbour='tr', fill=False, r_policy='v2', *,
                   ptl='original', segmentation='watershed', rotate=True, boundary=True, aggregation='sum'):
    """The same string as the research-path version name — the `method_version` column of the paper tables
    (wconstruct.json etc.).
    region takes the research-path window region value as is ('trer' · 'r'/'tre' · 'band_wide15' · 'none' etc.).
    The keyword defaults are the earlier window definition, so the old names stay unchanged — the new definition adds a suffix for each of these
    columns (-ptlr · -near · -norot · -core · -cov):
    `method_version('band_wide15', ptl='refined', segmentation='nearest', rotate=False, boundary=False, aggregation='coverage')`."""
    region = 'r' if region == 'tre' else region
    out = _ASSIGNMENT_VERSION + '-pipeline-' + _AGGREGATION_VERSION + '-' + _R_POLICY[r_policy]
    if not fill:
        out += '-nofill'
    if region != 'r':
        out += '-' + region
        if region == 'trer' and fallback != 'tre':
            out += '-fb' + fallback
    if hops != 1:
        out += f'-hop{hops}'
    if neighbour == 'r0':
        out += '-nbr0'
    elif neighbour == 'tr':
        out += '-nbtr'
    out += ('-ptlr' if ptl == 'refined' else '') + ('-near' if segmentation == 'nearest' else '') + \
        ('' if rotate else '-norot') + ('' if boundary else '-core') + ('-cov' if aggregation == 'coverage' else '')
    return out


# ---------------------------------------------------------------- background colour · rotation · cropping

def background_rgb(rgb):
    """Background colour of one image — the RGB median of whichever of the two global-luminance Otsu clusters dominates
    the border (the global median when the luminance range < 12).
    Used for rotation margins · empty-window placeholders (not estimated separately per window)."""
    import cv2
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or not rgb.size:
        raise ValueError('background estimation requires a nonempty uint8 RGB image')
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    pixels = rgb.reshape(-1, 3)
    if int(gray.max()) - int(gray.min()) >= 12:
        threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
        bright = float((border > threshold).mean()) >= .5
        candidates = rgb[gray > threshold] if bright else rgb[gray <= threshold]
        if len(candidates):
            pixels = candidates
    return [int(x) for x in np.median(pixels, axis=0).astype('uint8')]


class _LazyBackground:
    """The background colour is needed only for rotation · empty windows — computed once on first use."""
    def __init__(self, rgb):
        self.rgb, self.value = rgb, None

    def get(self):
        if self.value is None:
            self.value = background_rgb(self.rgb)
        return self.value

    def __iter__(self):
        return iter(self.get())


def deploy_theta(theta):
    """Window rotation angle — when the reading direction points left (backwards), rotate upright as is; otherwise
    rotate only when |θ| ≥ 45°."""
    if math.cos(theta) < 0:
        return theta
    return theta if abs(math.degrees(theta)) >= ROT_MIN_DEG else 0.0


def mask_aabb(mask, pad=PAD_PX):
    """Bounding rectangle with margin [x0, y0, x1, y1] of a mask — None for an empty mask."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return [max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
            min(w, int(xs.max()) + 1 + pad), min(h, int(ys.max()) + 1 + pad)]


def rotate_crop(rgb, mask, theta, background, pad=PAD_PX):
    """Rotate mask back by the reading tangent θ and crop a horizontal window → (PIL image, the 6 flattened values of
    the 2×3 window → original coordinate matrix) · (None, None) for an empty mask."""
    import cv2
    from PIL import Image
    if abs(theta) < 1e-9:
        box = mask_aabb(mask, pad)
        if box is None:
            return None, None
        image = Image.fromarray(np.ascontiguousarray(rgb[box[1]:box[3], box[0]:box[2]]))
        return image, [1.0, 0.0, float(box[0]), 0.0, 1.0, float(box[1])]
    h, w = mask.shape
    center = (w / 2, h / 2)
    mat = cv2.getRotationMatrix2D(center, math.degrees(theta), 1.0)
    cosine, sine = abs(mat[0, 0]), abs(mat[0, 1])
    ow, oh = int(w * cosine + h * sine) + 2, int(w * sine + h * cosine) + 2
    mat[0, 2] += ow / 2 - center[0]
    mat[1, 2] += oh / 2 - center[1]
    out = cv2.warpAffine(np.asarray(rgb), mat, (ow, oh), flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=tuple(int(v) for v in background))
    rmask = cv2.warpAffine(mask.astype(np.uint8), mat, (ow, oh), flags=cv2.INTER_NEAREST) > 0
    box = mask_aabb(rmask, pad)
    if box is None:
        # The rotated mask is empty — INTER_NEAREST dropped a 1-px stroke. Move the centroid before
        # rotation with the same matrix and build a box of the minimum side.
        ys, xs = np.nonzero(mask)
        if not len(xs):
            return None, None
        rx, ry = mat @ np.asarray([float(xs.mean()), float(ys.mean()), 1.0])
        half = max(MIN_WIN_PX, 2 * pad + 1) / 2.0
        x0 = int(max(0, min(ow - 1, round(rx - half))))
        y0 = int(max(0, min(oh - 1, round(ry - half))))
        box = [x0, y0, int(max(x0 + 1, min(ow, round(rx + half)))), int(max(y0 + 1, min(oh, round(ry + half))))]
    image = Image.fromarray(np.ascontiguousarray(out[box[1]:box[3], box[0]:box[2]]))
    inv = cv2.invertAffineTransform(mat)                       # rotated canvas → original
    inv[:, 2] += inv[:, :2] @ np.asarray(box[:2], float)       # window → canvas origin
    return image, [round(float(v), 9) for v in inv.reshape(-1)]


def window_mask(lab, bg0, point, region, contact, hops=HOPS):
    """The segment the character point falls on (own segment) + neighbour segments up to `hops` hops in the contact
    graph, all ∩ region.
    Contact is checked only inside contact (base band TR). Empty mask when the own segment is not inside the region."""
    lab, region = np.asarray(lab), np.asarray(region, bool)
    if lab.shape != region.shape:
        raise ValueError('S labels and R must have the same shape')
    contact = region if contact is None else np.asarray(contact, bool)
    own, was_bg = segment.label_at(lab, bg0, *point)
    present = own is not None and bool(((lab == own) & region).any())
    clipped = segment.clipped_labels(lab, contact, int64=True)
    seen, frontier = {own}, {own}
    for _ in range(hops if present else 0):
        nxt = set()
        for seg in frontier:
            nxt |= {i for i in segment.touching(clipped, seg, False) if i >= 0 and not (bg0 and i == 0)}
        frontier = nxt - seen
        seen |= frontier
    neighbours = sorted(seen - {own}) if present else []
    ids = sorted({own, *neighbours}) if present else []
    mask = region & np.isin(lab, ids)
    return mask, dict(own=own, own_present=present, was_bg=bool(was_bg), neighbours=neighbours, pixels=int(mask.sum()))


def ar_input(window):
    """Window image → AR input array (H, W, 3) uint8 — clip the ratio round(w/h) to [1, 4] and BICUBIC to that cell
    size."""
    from PIL import Image
    ratio = min(max(round(window.width / max(1, window.height)), 1), AR_MAX_RATIO)
    width, height = AR_BASE_SHAPE[ratio - 1] if ratio <= len(AR_BASE_SHAPE) else (AR_BASE_H * ratio, AR_BASE_H)
    return np.asarray(window.resize((width, height), Image.Resampling.BICUBIC), dtype=np.uint8)


def _window_slots(points, size, affine):
    """Original slot order, with translation-only crops avoiding per-slot inverses.

    Subtract the crop origin before testing the half-open bounds, as the original
    affine path does. Testing against origin + size can round differently at an
    edge. Keep the original scalar matrix arithmetic for rotated windows.
    """
    width, height = size
    if (affine[0, 0] == 1.0 and affine[0, 1] == 0.0
            and affine[1, 0] == 0.0 and affine[1, 1] == 1.0
            and np.isfinite(points).all()):
        xy = points - affine[:, 2]
        inside = ((xy[:, 0] >= 0.0) & (xy[:, 0] < width)
                  & (xy[:, 1] >= 0.0) & (xy[:, 1] < height))
        return np.flatnonzero(inside).tolist()
    inverse = np.linalg.inv(affine[:, :2])
    slots = []
    for slot, source_xy in enumerate(points):
        x, y = inverse @ (np.asarray(source_xy, dtype=np.float64) - affine[:, 2])
        if 0.0 <= x < width and 0.0 <= y < height:
            slots.append(slot)
    return slots


class _WindowBounds:
    """Cache contact neighbours and each label's bounds inside the text region.

    For an unrotated window the pixels outside the selected masks are not filled.
    Its crop is therefore exactly the bounding box of their union. Build these
    boxes once instead of materializing and scanning a full-image mask per slot.
    Legacy background/negative-label inputs keep the existing mask implementation.
    """

    def __init__(self, lab, bg0, area, contact):
        from scipy.ndimage import find_objects
        lab = np.asarray(lab)
        area = np.asarray(area, bool)
        if lab.shape != area.shape:
            raise ValueError('S labels and R must have the same shape')
        self.lab, self.graph = lab, {}
        self.boxes = None
        if bg0 or not np.issubdtype(lab.dtype, np.integer) or lab.min() < 0:
            return
        count = int(lab.max()) + 1
        if count > 65536:
            return
        labelled = np.where(area, lab.astype(np.int32) + 1, 0)
        self.boxes = find_objects(labelled, max_label=count)
        # All labels are nonnegative here. Test the contact mask on both sides
        # directly; avoid a full-image int64 cast and a second clipped copy.
        contact = np.broadcast_to(np.asarray(contact, bool), lab.shape)
        for a, b, ma, mb in ((lab[:, :-1], lab[:, 1:], contact[:, :-1], contact[:, 1:]),
                             (lab[:-1, :], lab[1:, :], contact[:-1, :], contact[1:, :])):
            valid = (a != b) & ma & mb
            pairs = np.unique(a[valid].astype(np.int64) * count + b[valid])
            for code in pairs:
                i, j = divmod(int(code), count)
                self.graph.setdefault(i, set()).add(j)
                self.graph.setdefault(j, set()).add(i)

    def crop(self, rgb, point, hops, pad=PAD_PX, *, source_image=None):
        from PIL import Image
        h, w = self.lab.shape
        # This fast path has already excluded background and negative labels.
        # Keep label_at's round-then-clamp rule without two scalar numpy calls.
        x = min(max(round(point[0]), 0), w - 1)
        y = min(max(round(point[1]), 0), h - 1)
        own = int(self.lab[y, x])
        if own is None or self.boxes[own] is None:
            return None, None
        seen, frontier = {own}, {own}
        for _ in range(hops):
            nxt = set()
            for label in frontier:
                nxt |= self.graph.get(label, set())
            frontier = nxt - seen
            seen |= frontier
        boxes = [self.boxes[i] for i in seen if self.boxes[i] is not None]
        y0 = max(0, min(b[0].start for b in boxes) - pad)
        x0 = max(0, min(b[1].start for b in boxes) - pad)
        y1 = min(h, max(b[0].stop for b in boxes) + pad)
        x1 = min(w, max(b[1].stop for b in boxes) + pad)
        image = (source_image.crop((x0, y0, x1, y1)) if source_image is not None else
                 Image.fromarray(np.ascontiguousarray(rgb[y0:y1, x0:x1])))
        return image, [1.0, 0.0, float(x0), 0.0, 1.0, float(y0)]


# ---------------------------------------------------------------- one image

def inputs(rgb, rec, *, ptl='refined', segmentation='nearest'):
    """attention record → ((p_tl points float32 (K, 2), tl_4 curve) | (None, None), segment lab | None) — computed on
    the spot.
    ptl: 'refined' (default · `position.REFINED_POLICY`) | 'original' (earlier window definition) · segmentation: 'nearest' (default) |
    'watershed' (earlier window definition).
    Segmentation uses the per-position attention maps resized back to the original size and the p_tl points.
    The tl_4 spline coefficients can differ by about 1e-12 across numpy · BLAS builds, and that difference can move
    segmentation boundaries by a few px · the rbox
    (the stored v1 geometry mixes values computed under numpy 1.26.4 and 2.1.3 — reproduce it with `stored_inputs`)."""
    if ptl not in PTL or segmentation not in segment.SEGMENTATIONS:
        raise ValueError(f'ptl must be one of {tuple(PTL)} and segmentation one of {tuple(segment.SEGMENTATIONS)}')
    points, geometry = position.derive(rec, policy=PTL[ptl])
    ok = len(points) and geometry is not None and len(points) == geometry.get('slots', -1)
    lab = None
    if len(points) and np.asarray(rec['maps']).size:
        def maps():
            return position.px_maps(list(rec['maps']), rec['gw'], rec['gh'], rec['wh0'], rec['rotated'])
        cs = [(float(x), float(y)) for x, y in points]
        if segmentation == 'nearest':                 # discs do not overlap → attention not expanded (same labels)
            lab = segment.nearest_lazy(np.asarray(rgb), rec['maps'], maps, cs)
        else:
            lab = segment.SEGMENTATIONS[segmentation](np.asarray(rgb), maps(), cs)
    return (points, geometry) if ok else (None, None), lab


def prepare(rgb, rec, *, boxes=None, region='trw15', hops=HOPS, fallback='trw15', geometry_inputs=None, bg0=False,
            rotate=False, ptl='refined', segmentation='nearest'):
    """One image (original RGB) and its AR attention record (text · maps · gw · gh · wh0 · rotated) → list of windows.

    Returns dict(text, windows=[per position dict(image=AR input array | None, valid, slot_ids, target_slot, theta,
    source_affine, skip_reason)], region_meta) — an image whose windows cannot be built gives dict(skip=reason) (no record ·
    empty transcript · mismatch between points and character count · transcript longer than MAX_LEN · segmentation failure).
    boxes: precomputed list of rbox corners (None → computed on the spot · a value that is not a list counts as rbox
    failure, i.e. TRW1.5) ·
    geometry_inputs: precomputed ((points, curve), lab) — for paper reproduction (`stored_inputs`) · comparison ·
    bg0: whether the segmentation has a background segment 0 (marker of the stored segmentation).
    rotate: whether to rotate the windows upright along the reading tangent (earlier window definition · the default does not rotate and does
    not compute the tangent either) ·
    ptl · segmentation: passed to `inputs` when geometry_inputs is absent. The earlier window definition is `prepare(..., **PAPER_V1)`."""
    if region not in REGIONS:
        raise ValueError(f'region must be one of {REGIONS}')
    if rec is None:
        return dict(skip='missing_charout')
    rgb = np.asarray(rgb, np.uint8)
    pair, lab = inputs(rgb, rec, ptl=ptl, segmentation=segmentation) if geometry_inputs is None else geometry_inputs
    if pair[0] is None or lab is None:
        return dict(skip='missing_charout')
    points, geometry = pair
    text = rec['text']
    points = [(float(x), float(y)) for x, y in points]
    if not text or not points:
        return dict(skip='empty_prediction')
    if len(text) != len(points):
        return dict(skip='length_mismatch')
    if len(text) > MAX_LEN:
        return dict(skip='prediction_too_long')
    height, width = rgb.shape[:2]
    # p_tl is an estimate moved from the attention grid, so it can land just outside the canvas (−0.1 px etc.) — pull it
    # inside the canvas.
    points = [(min(max(x, 0.0), width - 1.0), min(max(y, 0.0), height - 1.0)) for x, y in points]
    point_array = np.asarray(points, dtype=np.float64)
    contact = None
    if region == 'trer':
        area, region_meta = regions.window_region(rgb, lab, points, geometry, rec, boxes=boxes, fallback=fallback)
    elif region == 'trw15':
        area, contact = regions.wide_and_tr_bands(rgb.shape[:2], geometry, half=1.5)
        region_meta = dict(region_mode='band_wide15', pixels=int(area.sum()))
    else:
        area = np.ones(rgb.shape[:2], bool)
        region_meta = dict(region_mode='none', pixels=int(area.sum()))
    if contact is None:
        contact = regions.tr_strip(rgb.shape[:2], geometry)
    angles = position.angles(geometry, points) if rotate else None    # without rotation the tangent is not used
    background = _LazyBackground(rgb)
    bounds = _WindowBounds(lab, bg0, area, contact) if not rotate else None
    source_image = None
    if bounds is not None and bounds.boxes is not None and len(points) > 1:
        # Convert RGB once rather than copying overlapping numpy crops into PIL
        # once per slot. PIL crop and the original BICUBIC resize stay unchanged.
        from PIL import Image
        source_image = Image.fromarray(rgb)
    windows = []
    for target, (cx, cy) in enumerate(points):
        theta = deploy_theta(float(angles[target])) if rotate else 0.0
        if bounds is not None and bounds.boxes is not None:
            window, image_map = bounds.crop(rgb, (cx, cy), hops, source_image=source_image)
        else:
            mask, _ = window_mask(lab, bg0, (cx, cy), area, contact, hops)
            window, image_map = rotate_crop(rgb, mask, theta, background)
        if window is None:
            windows.append(dict(image=None, valid=False, slot_ids=[], target_slot=target, theta=theta,
                                source_affine=None, skip_reason='empty_R_local_window'))
            continue
        affine = np.asarray(image_map, dtype=np.float64).reshape(2, 3)
        slot_ids = _window_slots(point_array, window.size, affine)
        if target not in slot_ids:
            windows.append(dict(image=None, valid=False, slot_ids=[], target_slot=target, theta=theta,
                                source_affine=affine.tolist(), skip_reason='target_outside_R_local_window'))
            continue
        windows.append(dict(image=ar_input(window), valid=True, slot_ids=slot_ids,
                            target_slot=target, theta=theta, source_affine=affine.tolist(), size=window.size, skip_reason=None))
    return dict(text=text, windows=windows, region_meta=region_meta, points=points)


# ---------------------------------------------------------------- reading stored attention records (for paper reproduction)

def charout_root(env=True):
    """Root of the stored attention records (<root>/<dataset>/svtrv2-ar.lmdb) — `STR_CHAROUT_ROOT` (when env) >
    `local_paths.json` > `data/charout` in the repository (`tools._io.local_path`). Read again on every call."""
    from tools._io import local_path
    return Path(local_path('charout_root', env=env))


CHAROUT_ROOT = charout_root()
READER = 'svtrv2-ar'
FORCED = 'svtrv2-ar_afix'       # images whose stored transcript differs from the beam greedy a — records made
#                               by teacher forcing a (when present, that image uses it)
_ENVS = {}


def _env(dataset, name, root):
    import lmdb
    key = (str(root), dataset, name, os.getpid())
    if key not in _ENVS:
        path = Path(root) / dataset / f'{name}.lmdb'
        _ENVS[key] = lmdb.open(str(path), readonly=True, lock=False, readahead=False, max_readers=512,
                               map_size=64 << 30) if path.exists() else None
    return _ENVS[key]


def unpack_record(raw, maps=True):
    """Stored format (npz bytes) → dict(text · conf · gw · gh · wh0 · rotated · maps)."""
    import io
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        rec = dict(text=str(z['text'].item()) if z['text'].shape == () else str(z['text']),
                   conf=z['conf'].astype(np.float32), gw=int(z['gw']), gh=int(z['gh']),
                   wh0=tuple(int(v) for v in z['wh0']), rotated=bool(z['rotated']))
        if maps:
            rec['maps'] = z['maps'].astype(np.float32)
    return rec


def _reader_env(dataset, root):
    """The attention store of the recognizer for a dataset — stops when the LMDB is missing (otherwise every image would be
    skipped as missing_charout). The teacher-forced store (`FORCED`) is optional."""
    env = _env(dataset, READER, root)
    if env is None:
        raise FileNotFoundError(f'attention store not found: {Path(root) / dataset / (READER + ".lmdb")} — set STR_CHAROUT_ROOT')
    return env


def stored_record(dataset, index, *, root=None, forced=True):
    """One stored svtrv2-ar attention record — with forced=True, an image that has a teacher-forced record
    (svtrv2-ar_afix) reads that one. None when absent.
    The dataset's attention store itself must exist (`_reader_env`)."""
    root = CHAROUT_ROOT if root is None else Path(root)
    _reader_env(dataset, root)
    key = b'%09d' % int(index)
    for name in ((FORCED, READER) if forced else (READER,)):
        env = _env(dataset, name, root)
        if env is None:
            continue
        with env.begin() as txn:
            raw = txn.get(key)
        if raw is not None:
            return dict(unpack_record(raw), source=name)
    return None


# Definition versions of the stored geometry · segmentation that the paper runs read — used only when the marker of the
# stored row equals this (the same check as the research-path charout.geometry · seg).
STORED_DEFINITION = dict(map='last-decoder-layer-head-mean-v1', ptl='ptl-44cc92b89f9756cf', tl4='tl4-92cc8a768f03c02b',
                         seg='seg-1c5bf82d4a2373d5')


def stored_inputs(dataset, index, *, root=None):
    """Stored p_tl · tl_4 · segmentation that the paper runs read (derived rows of the svtrv2-ar store) →
    ((points, curve) | None, (lab, bg0) | None).
    A field is None when the row marker differs from STORED_DEFINITION or is missing — the caller re-measures only that
    field with `inputs` (as the research path does)."""
    import io
    import json
    root = CHAROUT_ROOT if root is None else Path(root)
    env = _reader_env(dataset, root)
    k = b'%09d' % int(index)
    with env.begin() as txn:
        mark = txn.get(b'm' + k)
        p, line, pv, lv = txn.get(b'p' + k), txn.get(b'l' + k), txn.get(b'v' + k), txn.get(b'L' + k)
        seg, sv = txn.get(b's' + k), txn.get(b'z' + k)
    ok_map = mark is not None and mark.decode() == STORED_DEFINITION['map']
    pair = None
    if ok_map and pv is not None and lv is not None and pv.decode() == STORED_DEFINITION['ptl'] and lv.decode() == STORED_DEFINITION['tl4'] \
            and p is not None and line is not None:
        points = np.load(io.BytesIO(p), allow_pickle=False)
        curve = json.loads(line)
        if curve.get('format') == 'tl4-spline-v1' and len(points) == curve['slots']:
            pair = (points, curve)
    cut = None
    if ok_map and seg is not None and sv is not None and sv.decode() == STORED_DEFINITION['seg']:
        with np.load(io.BytesIO(seg), allow_pickle=False) as z:
            cut = (z['lab'].astype(np.int16), bool(z['bg0']))
    return pair, cut


def paper_inputs(rgb, rec, dataset, index, *, root=None):
    """The same inputs as the paper runs — ordinary records read the stored geometry · segmentation and compute only
    the missing fields on the spot; teacher-forced records compute everything on the spot.
    → (geometry_inputs, bg0) — pass them to `prepare(..., geometry_inputs=…, bg0=…)`."""
    pair = cut = None
    if rec is not None and rec.get('source') != FORCED:
        pair, cut = stored_inputs(dataset, index, root=root)
    if pair is None or cut is None:
        fresh_pair, fresh_lab = inputs(rgb, rec, ptl='original', segmentation='watershed')
        pair = pair if pair is not None else fresh_pair
        cut = cut if cut is not None else (fresh_lab, False)
    return (pair, cut[0]), cut[1]


# ---------------------------------------------------------------- building attention records (when no stored copy exists)

AR_ROTATE_TALL = 1.5    # RotateTall of the whole-image input — 90° counterclockwise when height > 1.5 × width
#                       (public svtrv2-ar evaluation setting)
ENCODER_DOWN_H = 8      # vertical downsampling factor of the encoder feature grid — grid height = input height / 8


def whole_input(rgb):
    """Whole image → (AR input array, input (width, height), original (width, height) · after rotation, whether
    rotated) — the same as the public svtrv2-ar evaluation input."""
    from PIL import Image
    img = Image.fromarray(np.asarray(rgb, np.uint8))
    rotated = False
    w, h = img.size
    if h > AR_ROTATE_TALL * w:
        img = img.transpose(Image.ROTATE_90)
        rotated = True
    w0, h0 = img.size
    r = int(np.clip(round(w0 / max(1, h0)), 1, len(AR_BASE_SHAPE)))
    width, height = AR_BASE_SHAPE[r - 1]
    arr = np.array(img.convert('RGB').resize((width, height), Image.BICUBIC), dtype=np.uint8)
    return arr, (width, height), (w0, h0), rotated


def attention_record(model, rgb, text, char_index, device):
    """One image + transcript text (usually greedy a) → attention record dict(text · maps · gw · gh · wh0 · rotated).

    Runs the recognizer's greedy decoding loop as is, but forces only the next token to text (+ ⟨eos⟩), and at each step
    averages over heads the last query of the last decoder layer's cross-attention (the same map definition as the stored
    dump). If text is the greedy output, the record equals the greedy forward record.
    Each map sums to 1 per character position (rounded to six decimals) · float32 passed through float16, as in the stored
    format. None for out-of-vocabulary characters · more than 25 characters."""
    import torch
    if len(text) > MAX_LEN or any(c not in char_index for c in text):
        return None
    arr, (width, height), wh0, rotated = whole_input(rgb)
    x = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).unsqueeze(0).to(device)
    x = (x.float() / 255.0 - 0.5) / 0.5
    dec = model.decoder
    cross = dec.decoder[-1].cross_attn
    target = [char_index[c] for c in text] + [dec.eos]
    with torch.no_grad():
        memory = model.encoder(x)
        n_tok = int(memory.shape[1])
        if dec.encoder is not None:
            memory = dec.positional_encoding(memory)
            for layer in dec.encoder:
                memory = layer(memory)
        seq = torch.full((1, dec.max_len + 1), dec.ignore_index, dtype=torch.int64, device=device)
        seq[:, 0] = dec.bos
        maps = []
        for t in range(len(target)):
            emb = dec.positional_encoding(dec.embedding(seq[:, :t + 1]))
            mask = dec.generate_square_subsequent_mask(emb.shape[1], device)
            h = emb
            for layer in dec.decoder:
                h = layer(h, memory, self_mask=mask)
            maps.append(cross.attn_map[:, :, -1, :].mean(dim=1)[0])
            if t + 1 < len(target):                       # the step after the last (⟨eos⟩) is not used — for a
                                                          # 25-character transcript it would run past the end of seq
                seq[:, t + 1] = target[t]
        grid = torch.stack(maps, 0).float().cpu().numpy()
    gh = max(1, height // ENCODER_DOWN_H)
    gw = max(1, n_tok // gh)
    if gw * gh != n_tok:
        return None
    raw = [np.asarray(m, dtype=np.float32) for m in grid[:len(target) - 1]]          # character steps only, no ⟨eos⟩ step
    raw = [np.round(a / s, 6) for a in raw if (s := float(np.asarray(a, dtype=np.float64).sum())) > 0]
    stored = np.asarray(raw, dtype=np.float16) if raw else np.zeros((0, gw * gh), np.float16)
    return dict(text=text, maps=stored.astype(np.float32).reshape(len(raw), -1), gw=int(gw), gh=int(gh),
                wh0=tuple(int(v) for v in wh0), rotated=bool(rotated), source='computed')
