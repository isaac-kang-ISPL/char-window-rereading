"""Midpoint windows (MW) — the character windows of the manuscript (Section III-C, MW3 = adjacent positions).

Window of position i (0-indexed here, k = n // 2 neighbours on each side; MW3: k = 1):

    W_i = { q in R : m_i^- <= <q, t_i> <= m_i^+ }
    m_i^- = (d_{i-k-1} + d_{i-k}) / 2   (-inf when i - k <= 0)
    m_i^+ = (d_{i+k} + d_{i+k+1}) / 2   (+inf when i + k >= L - 1)

t_i is the unit tangent of the text line at p_i (`position.angles`), d_j = <p_j, t_i> made nondecreasing by a running
maximum, and R is the band of 1.5 x-heights on both sides of the text line (`region.wide_band`, TRW). The recognizer
reads the bounding rectangle of W_i with a margin of `window.PAD_PX` pixels, not rotated, resized by `window.ar_input`;
the positions a window reads are those whose points lie inside the crop (`window._window_slots`). There is no character
segmentation.

R is rasterized once per image and each scanline is kept as its inside intervals, which are intersected with the two
half-planes; the crop boxes are those of the pixel masks (pixel centres, the same tie rule).

Same windows as `tools/experiments/window_studies/midpoint_windows.py` (research tree), which produced the stored MW
tables of the manuscript — `tests/test_rescore_core.py` compares the two when the research module is present.
"""
from __future__ import annotations

import math
import numpy as np
from PIL import Image

from tools.rescore import position as P, window as W


class BandBounds:
    """TRW boundary runs; per-window cost depends on runs, not image area."""

    def __init__(self, area):
        area = np.asarray(area, bool)
        self.height, self.width = area.shape
        edges = np.diff(area.astype(np.int8), axis=1, prepend=np.int8(0), append=np.int8(0))
        yy, xx = np.nonzero(edges)
        self.y = yy[::2]
        self.start = xx[::2]
        self.stop = xx[1::2]  # exclusive
        assert np.array_equal(self.y, yy[1::2])
        self.pixels = int(np.sum(self.stop - self.start))

    def _transition(self, estimate, ux, dy, threshold, greater, strict):
        """First true integer x for a monotone predicate, with exact float ties.

        Division gives the usual answer. Verify its two neighbouring pixels
        using the original multiply/add order; binary search only runs where
        floating-point rounding moved the transition. This also handles an
        almost vertical cut, where division by tiny cos(theta) is ill-conditioned.
        """
        x = np.clip(estimate, self.start, self.stop).astype(np.int64)

        def pred(xx, yy):
            q = (xx + .5) * ux + yy
            if greater:
                return q > threshold if strict else q >= threshold
            return q < threshold if strict else q <= threshold

        good = ((x == self.stop) | pred(x, dy)) & ((x == self.start) | ~pred(x - 1, dy))
        bad = ~good
        if np.any(bad):
            low, high, yy = self.start[bad].copy(), self.stop[bad].copy(), dy[bad]
            while np.any(low < high):
                mid = (low + high) // 2
                hit = pred(mid, yy)
                active = low < high
                high = np.where(active & hit, mid, high)
                low = np.where(active & ~hit, mid + 1, low)
            x[bad] = low
        return x

    def box(self, u, lo, hi, pad=W.PAD_PX):
        ux, uy = map(float, u)
        if not len(self.y) or lo > hi or math.isnan(lo) or math.isnan(hi) or not np.isfinite([ux, uy]).all():
            return None
        dy = (self.y + .5) * uy
        if ux == 0:
            a, b = self.start, self.stop
            good = (dy >= lo) & (dy <= hi)
        else:
            first, last = (lo, hi) if ux > 0 else (hi, lo)
            with np.errstate(over='ignore', invalid='ignore'):
                lower = np.ceil((first - dy) / ux - .5)
                upper = np.floor((last - dy) / ux - .5) + 1
            a = self._transition(lower, ux, dy, first, ux > 0, False)
            b = self._transition(upper, ux, dy, last, ux > 0, True)
            good = a < b
        if not np.any(good):
            return None
        return [max(0, int(a[good].min()) - pad), max(0, int(self.y[good].min()) - pad),
                min(self.width, int(b[good].max()) + pad), min(self.height, int(self.y[good].max()) + 1 + pad)]


def geometry_only(rgb, rec, ptl='refined', **_unused):
    """Compatibility pair for W preparation: derive points/curve, never labels."""
    points, geometry = P.derive(rec, policy=W.PTL[ptl])
    ok = len(points) and geometry is not None and len(points) == geometry.get('slots', -1)
    ok = ok and np.asarray(rec['maps']).size > 0
    # None in the labels position is intentional: midpoint windows do not use it.
    return ((points, geometry) if ok else (None, None)), None


def make_windows(rgb, rec, widths, *, geometry_inputs=None):
    widths = tuple(widths)
    if any(n not in (1, 3, 5, 7, 9) for n in widths):
        raise ValueError('MW width must be one of 1,3,5,7,9')

    def skip(reason):
        return {n: dict(skip=reason) for n in widths}

    if rec is None:
        return skip('missing_charout')
    pair, _ = geometry_only(rgb, rec) if geometry_inputs is None else geometry_inputs
    if pair[0] is None:
        return skip('missing_charout')
    pts, geom = pair
    text = rec['text']
    if not text or not len(pts):
        return skip('empty_prediction')
    if len(text) != len(pts):
        return skip('length_mismatch')
    if len(text) > W.MAX_LEN:
        return skip('prediction_too_long')
    raw_points = np.asarray(pts, float).reshape(-1, 2)
    height, width = rgb.shape[:2]
    points = np.clip(raw_points, [0., 0.], [width - 1., height - 1.])
    band = BandBounds(W.regions.wide_band(rgb.shape[:2], geom, half=1.5))
    angles = np.asarray(P.angles(geom, raw_points), float)
    out = {n: dict(text=text, windows=[], points=points.tolist(),
                  region_meta=dict(region_mode='band_wide15', pixels=band.pixels)) for n in widths}
    source = Image.fromarray(rgb)
    for target, point in enumerate(points):
        # Keep the old coincident/clipped-point lookup and projection arithmetic.
        i = int(np.argmin(np.hypot(raw_points[:, 0] - point[0], raw_points[:, 1] - point[1])))
        u = np.array([math.cos(angles[i]), math.sin(angles[i])])
        s = np.maximum.accumulate(raw_points @ u)
        for n in widths:
            k = n // 2
            lo = -float('inf') if i - k <= 0 else (s[i-k-1] + s[i-k]) / 2
            hi = float('inf') if i + k >= len(s) - 1 else (s[i+k] + s[i+k+1]) / 2
            box = band.box(u, lo, hi)
            win = dict(image=None, valid=False, slot_ids=[], target_slot=target,
                       theta=0., source_affine=None, skip_reason='empty_R_local_window')
            if box is not None:
                image = source.crop(box)
                affine = np.array([[1., 0., float(box[0])], [0., 1., float(box[1])]])
                ids = W._window_slots(points, image.size, affine)
                win.update(source_affine=affine.tolist(), skip_reason='target_outside_R_local_window')
                if target in ids:
                    win.update(image=W.ar_input(image), valid=True, slot_ids=ids, size=image.size, skip_reason=None)
            out[n]['windows'].append(win)
    return out


def prepare(rgb, rec, *, hops=1, geometry_inputs=None, **kw):
    """`window.prepare`-compatible entry: MW(2·hops + 1), hops = 1 is MW3."""
    if kw.get('rotate', False) or kw.get('region', 'trw15') != 'trw15' or kw.get('ptl', 'refined') != 'refined':
        raise ValueError('Optimized MW supports the current refined, TRW, no-rotation definition only')
    return make_windows(rgb, rec, [2 * hops + 1], geometry_inputs=geometry_inputs)[2 * hops + 1]
