"""attention maps → character points p_tl · text line curve tl_4.

From the cross-attention maps (K, gh·gw) that the SVTRv2-AR decoder emits for each character position (timestep),

1. Map cleanup — remove the min common to all positions → adjacent-position competition → corner prior
2. Initial text line tl_1 — fit a cubic curve to the max cell p_argmax of the cleaned maps; per-position
   reference point · tangent
3. Observation point p90 — axis_prior (distance weight along the text line) → top 90 % cumulative mass →
   blob_soft (blob distance weight) centroid
4. Refinement — fit tl_2 → per-character height correction along the tl_2 normal → refit tl_4 · nearest projection
   → smooth abrupt spacing changes → pull back toward the observation point by the reliability

and produces the character points p_tl in original crop px and the tl_4 curve coefficients (geometry).
Coordinate convention: grid (x, y) = (column, row),
px are original crop coordinates: the grid ends are stretched to (W, H) and the rotation (RotateTall) is undone.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / 'data'
GLYPH_OFFSET = DATA / 'glyph_offset.json'   # per-character offset table (synthetic measured · x-height units · up is +)

MASS_Q = 0.9                        # cumulative mass kept by the observation point p90
INITIAL_MASS_Q = 0.7                # tl_1 fit weight = selected mass of the top 70 %
NEIGHBOR_COMPETITION_LAM = 0.5      # adjacent-position competition (0 off · 1 full penalty by the non-owned share)
CORNER_PRIOR_SIGMA_RATIO = 0.25     # corner prior σ = short side × this value
CORNER_PRIOR_MIN_MASS = 1e-14       # if the mass left after the corner prior is ≤ this, revert to the uncorrected map
ONE_ALPHA = 0.2                     # α of axis_prior · blob_soft distance weight max(1 − α·d², 0) (d in character spacings)
PPL_HALF = 3.0                      # spread at which the reliability 1/(1 + ppl/PPL_HALF) is 0.5

# Explicit refinement recipe. The no-policy path retains the published cache
# definition; callers must use a separate geometry/crop key for this recipe.
REFINED_VERSION = 'centroid70-cubic4-local-projection-v1'
REFINED_POLICY = dict(initial='top70', initial_drop='cubic', align_drop='cubic', slot_projection=True)

#: Text line fit settings — shared by tl_1 · tl_2 · tl_4.
ALIGN = dict(
    lam0=0.003,        # curvature penalty λ (2nd difference of coefficients)
    gam=0.0,           # exponent that rescales the per-image λ by the spread (0 = off)
    gam1=0.0,          # log term of the same exponent (0 = off)
    ppl_ref=1.0,       # reference spread of that exponent
    deg=3,             # curve degree
    huber=0.1,         # IRLS Huber threshold (character widths)
    iters=1,           # IRLS iterations
    tau=1.0,           # influential-point threshold — removing a point moves the curve more than this (char widths)
    drops=2,           # max number of influential points removed
    lams=0.3,          # spacing smoothing penalty (2nd difference of arc length)
    d0=1.0,            # pull-back distance scale (character widths)
    end_guard=True,    # if the tangent flips in the end position, pull in the start of the tangent extension
)


# ── Map cleanup ─────────────────────────────────────────────────────────────

def normalize_attention_maps(rmaps):
    """Normalize each position's attention to a (K, cells) distribution that sums to 1. Uniform if the sum is 0."""
    if len(rmaps) == 0:
        return np.empty((0, 0), dtype=np.float64)
    maps = np.asarray(rmaps, dtype=np.float64).reshape(len(rmaps), -1)
    total = maps.sum(axis=1, keepdims=True)
    uniform = np.full_like(maps, 1.0 / max(maps.shape[1], 1))
    return np.divide(maps, np.maximum(total, 1e-18), out=uniform,
                     where=total > 1e-18)


def common_min_maps(rmaps):
    """Subtract the per-cell minimum common to all positions and renormalize -> (maps, common min).

    Positions that are all 0 after the removal revert to the original normalized map.
    """
    raw = normalize_attention_maps(rmaps)
    if len(raw) == 0:
        return raw, np.zeros(raw.shape[1] if raw.ndim == 2 else 0, dtype=np.float64)
    common = raw.min(axis=0)
    clean = np.maximum(raw - common[None, :], 0.0)
    total = clean.sum(axis=1, keepdims=True)
    clean = np.divide(clean, np.maximum(total, 1e-18), out=raw.copy(),
                      where=total > 1e-18)
    return clean, common


def neighbor_competition_maps(rmaps, lam=NEIGHBOR_COMPETITION_LAM):
    """Adjacent positions share the same cell once, simultaneously.

    owner_i(x) = a_i(x) / Σ_{j∈{i−1,i,i+1}} a_j(x) · out_i(x) ∝ a_i(x)·((1−lam) + lam·owner_i(x)).
    """
    maps = normalize_attention_maps(rmaps)
    if len(maps) <= 1 or lam <= 0:
        return maps
    lam = float(np.clip(lam, 0.0, 1.0))
    local_total = np.stack([
        maps[max(0, i - 1):min(len(maps), i + 2)].sum(axis=0)
        for i in range(len(maps))
    ])
    owner = np.divide(maps, np.maximum(local_total, 1e-18),
                      out=np.ones_like(maps), where=local_total > 1e-18)
    out = maps * ((1.0 - lam) + lam * owner)
    total = out.sum(axis=1, keepdims=True)
    return np.divide(out, np.maximum(total, 1e-18), out=maps.copy(),
                     where=total > 1e-18)


def corner_prior_weights(gw, gh, wh0=None, sigma_ratio=None):
    """g = 1 − exp(−d²/(2σ²)) from the distance d to the nearest image corner (per cell · flattened).

    Cell coordinates are x = column·W/(gw−1), y = row·H/(gh−1). No decay if one axis has a single cell.
    """
    ratio = CORNER_PRIOR_SIGMA_RATIO if sigma_ratio is None else float(sigma_ratio)
    if ratio < 0 or not np.isfinite(ratio):
        raise ValueError('corner prior sigma ratio must be finite and nonnegative')
    if ratio == 0 or gw <= 1 or gh <= 1:
        return np.ones(gw * gh, dtype=np.float64)
    W, H = wh0 if wh0 is not None else (gw - 1, gh - 1)
    if min(W, H) <= 0:
        raise ValueError('corner prior image dimensions must be positive')
    yy, xx = np.mgrid[0:gh, 0:gw]
    x, y = xx * W / (gw - 1), yy * H / (gh - 1)
    d2 = np.minimum(x, W-x)**2 + np.minimum(y, H-y)**2
    return (-np.expm1(-d2 / (2 * (ratio * min(W, H))**2))).ravel()


def corner_prior_maps(rmaps, gw, gh, wh0=None, sigma_ratio=None):
    """Multiply by the corner prior once and normalize each position to sum 1.

    Positions whose mass vanished revert to the map before correction."""
    maps = normalize_attention_maps(rmaps)
    if not len(maps):
        return maps
    if maps.shape[1] != gw * gh:
        raise ValueError('attention cells do not match grid dimensions')
    weights = corner_prior_weights(gw, gh, wh0, sigma_ratio)
    weighted = maps * weights
    mass = weighted.sum(axis=1, keepdims=True)
    return np.divide(weighted, np.maximum(mass, CORNER_PRIOR_MIN_MASS), out=maps.copy(),
                     where=mass > CORNER_PRIOR_MIN_MASS)


_GRID_CACHE: dict = {}


def _grid(gw, gh):
    """Grid cell coordinates (same size for every image, so built once · read-only).

    -> (cell index ij, x, y, (x, y) column stack, mgrid rows · columns)."""
    key = (int(gw), int(gh))
    hit = _GRID_CACHE.get(key)
    if hit is None:
        ij = np.arange(gw * gh)
        xy = np.column_stack((np.arange(gw * gh, dtype=np.float64) % gw, np.arange(gw * gh, dtype=np.float64) // gw))
        rr, cc = np.mgrid[0:gh, 0:gw]
        hit = dict(ij=ij, gx=ij % gw, gy=ij // gw, xy=xy, rr=rr, cc=cc)
        for v in hit.values():
            v.setflags(write=False)
        if len(_GRID_CACHE) > 64:
            _GRID_CACHE.clear()
        _GRID_CACHE[key] = hit
    return hit


def top_q_observation(pr, gw, gh, q):
    """Mask of the top-q cumulative mass -> (probability-weighted center (x, y) inside it, mask, retained mass)."""
    values = normalize_attention_maps(np.asarray(pr, dtype=np.float64).reshape(1, -1))[0]
    order = np.argsort(-values, kind='stable')
    count = int(np.searchsorted(np.cumsum(values[order]), q)) + 1
    selected = np.zeros(values.size, dtype=bool)
    selected[order[:count]] = True
    mass = values * selected
    retained = float(mass.sum())
    xy = _grid(gw, gh)['xy']
    if retained <= 1e-18:
        point = np.asarray(((gw - 1.0) / 2.0, (gh - 1.0) / 2.0))
    else:
        point = (mass[:, None] * xy).sum(axis=0) / retained
    return point, selected, retained


def perplexity(pr, gw, gh, K):
    """Attention spread of one position — exp(entropy) ÷ cells per position (gw·gh/K)."""
    p = np.asarray(pr, np.float64)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum())) / max(gw * gh / K, 1e-9)


def reliability(ppl, half=PPL_HALF):
    """Spread -> reliability 1/(1 + ppl/half)."""
    return 1.0 / (1.0 + np.asarray(ppl, float) / half)


def clip_points_to_image(pts, gw, gh, wh0=None):
    """Clip grid character points into the image bounds [0, W−1]×[0, H−1] mapped to the grid (new array).

    wh0 = (W, H) in the STR input orientation. Without it, the grid bounds [0, gw−1]×[0, gh−1].
    """
    points = np.asarray(pts, dtype=float).reshape(-1, 2)
    upper = np.maximum(np.asarray([gw, gh], dtype=float) - 1.0, 0.0)
    if wh0 is not None:
        size = np.asarray(wh0, dtype=float)
        upper *= np.maximum(size - 1.0, 0.0) / size
    return np.clip(points, 0.0, upper)


# ── Curve fitting (P-spline) ────────────────────────────────────────────────

def _spacing(P):
    """Median spacing between neighboring points (x-height estimate). None if there is one point or all spacings are 0."""
    q = np.asarray(P, float)
    if len(q) < 2:
        return None
    d = np.linalg.norm(np.diff(q, axis=0), axis=1)
    d = d[d > 1e-6]
    return float(np.median(d)) if len(d) else None


def _irls(P, w, solve, huber, iters):
    """IRLS — multiply the confidence weight w by the residual Huber weight min(1, δ/|r|) and refit iters times.

    -> (fitted values, last weights)."""
    ww = w.copy()
    G = solve(ww)
    if huber is None:
        return G, ww
    for _ in range(iters):
        res = np.linalg.norm(P - G, axis=1)
        ww = w * np.minimum(1.0, huber / np.maximum(res, 1e-9))
        G = solve(ww)
    return G, ww


def design(n, deg):
    """Number of positions n -> (knots t, basis B (n×m), 2nd difference D, actual degree).

    No interior knots (the whole curve is one polynomial)."""
    t, B, D, degree = _design_cached(n, deg)
    return t.copy(), B.copy(), D.copy(), degree


@lru_cache(maxsize=128)
def _design_cached(n, deg):
    from scipy.interpolate import BSpline
    deg = int(min(deg, max(1, n - 1)))
    inner = np.zeros(0)
    t = np.r_[[0.0] * (deg + 1), inner, [n - 1.0] * (deg + 1)]
    B = BSpline.design_matrix(np.arange(n, dtype=float), t, deg).toarray()
    m = B.shape[1]
    D = np.diff(np.eye(m), n=2, axis=0) if m >= 3 else np.zeros((0, m))
    return t, B, D, deg


def _end_guard_domain(sp, n, radius):
    """If the tangent reverses (dot product ≤ 0) within the last position at an end, start the tangent
    extension just before the curvature radius becomes smaller than radius."""
    lo, hi = 0.0, float(n - 1)
    if sp.k < 3 or radius <= 0:
        return lo, hi
    for left, u in ((True, np.linspace(1.0, 0.0, 129)),
                    (False, np.linspace(n - 2.0, n - 1.0, 129))):
        if np.dot(sp(u[0], 1), sp(u[-1], 1)) > 0:
            continue
        velocity, acceleration = sp(u, 1), sp(u, 2)
        speed = np.linalg.norm(velocity, axis=1)
        cross = np.abs(velocity[:, 0] * acceleration[:, 1]
                       - velocity[:, 1] * acceleration[:, 0])
        folded = np.flatnonzero(radius * cross > speed ** 3)
        if len(folded):
            cut = float(u[max(int(folded[0]) - 1, 0)])
            if left:
                lo = cut
            else:
                hi = cut
    # if the valid interval vanishes, use the original interval
    return (lo, hi) if lo < hi else (0.0, float(n - 1))


def pspline(pts, w, lam, cw, deg=3, huber=None, iters=3, asp=1.0, end_guard=False):
    """Weighted smoothing B-spline — x(k) · y(k) with the position index k as the parameter.

    Coordinates are in character-width cw units; the penalty is the 2nd difference of coefficients:
    (BᵀWB + λ·n/rows·DᵀD) c = BᵀW p.
    Beyond the ends, straight extension along the end tangent. -> (k -> (x, y) in original units, last weights).
    """
    from scipy.interpolate import BSpline
    P = np.asarray(pts, float) / cw
    P[:, 1] *= asp
    n = len(P)
    w = np.asarray(w, float)
    w = w / max(w.mean(), 1e-9)
    if n == 1:
        return (lambda k: (float(P[0, 0] * cw), float(P[0, 1] * cw / asp))), w
    nz = int((w > 0).sum())          # zero-weight points are not data — lower the degree accordingly
    t, B, D, deg = design(n, min(deg, max(1, nz - 1)))
    m = B.shape[1]

    lam_eff = lam * n / max(len(D), 1)
    penalty = lam_eff * D.T @ D
    ridge = 1e-9 * np.eye(m)

    def solve(ww):
        A = B.T @ (ww[:, None] * B) + penalty + ridge
        c = np.linalg.solve(A, B.T @ (ww[:, None] * P))
        solve.c = c
        return B @ c
    G, ww = _irls(P, w, solve, huber, iters)
    sp = BSpline(t, solve.c, deg, extrapolate=False)
    lo, hi = (_end_guard_domain(sp, n, 0.5 * (_spacing(P) or 0.0)) if end_guard
              else (0.0, float(n - 1)))
    d0, d1 = sp([lo, hi], 1)
    g0, g1 = sp([lo, hi])

    def f_(k):
        ka = np.asarray(k, dtype=float)
        out = np.asarray(sp(np.clip(ka, lo, hi)), dtype=float)
        out = np.where((ka <= lo)[..., None], g0 + d0 * (ka - lo)[..., None], out)
        out = np.where((ka >= hi)[..., None],
                       g1 + d1 * (ka - hi)[..., None], out)
        return out

    def scaled(k):
        ka = np.asarray(k)
        out = np.array(f_(k), dtype=float, copy=True)
        out[..., 0] *= cw
        out[..., 1] *= cw / asp
        if ka.ndim == 0:
            return float(out[0]), float(out[1])
        return out

    # for storage — coefficients of the actual solve (original units)
    scaled.spline = dict(knots=sp.t.tolist(),
                         coefficients=(sp.c * [cw, cw / asp]).tolist(),
                         degree=int(sp.k))
    if lo != 0.0 or hi != n - 1.0:
        scaled.spline['spline_domain'] = [lo, hi]
    return scaled, ww


def _influence(P, w, lam, cw, deg, huber, iters, asp, *, isolated=False):
    """The point whose removal moves the curve the most -> (index, that distance [character widths]).

    Solves the n leave-one-out fits (pspline's weight normalization, degree reduction and IRLS unchanged) at once with
    numpy; the chosen index and distance equal a loop that calls pspline once per point."""
    n = len(P)
    if n == 1:
        return -1, 0.0
    P = np.asarray(P, float)
    w = np.asarray(w, float)
    alive = np.flatnonzero(w > 0)
    if not len(alive):
        return -1, 0.0
    Q = P / cw
    Q[:, 1] *= asp
    W = np.repeat(w[None, :], len(alive) + 1, 0)           # row 0 = reference (none removed) · row j+1 = alive[j] removed
    W[np.arange(1, len(alive) + 1), alive] = 0.0
    W = W / np.maximum(W.mean(1, keepdims=True), 1e-9)
    dg_all = np.minimum(deg, np.maximum(1, (W > 0).sum(1) - 1))
    G_all = np.zeros((len(W), n, 2))
    for dg in np.unique(dg_all):                           # rows whose degree drops (few points left) are solved apart
        sel = np.flatnonzero(dg_all == dg)
        _t, B, D, _deg = design(n, int(dg))
        penalty = lam * n / max(len(D), 1) * D.T @ D
        ridge = 1e-9 * np.eye(B.shape[1])

        def solve(ww):
            A = np.einsum('ki,sk,kj->sij', B, ww, B) + penalty + ridge
            rhs = np.einsum('ki,sk,kd->sid', B, ww, Q)
            return np.einsum('ki,sid->skd', B, np.linalg.solve(A, rhs))
        Ws = W[sel]
        G = solve(Ws)
        if huber is not None:
            for _ in range(iters):
                res = np.linalg.norm(Q[None] - G, axis=2)
                G = solve(Ws * np.minimum(1.0, huber / np.maximum(res, 1e-9)))
        G_all[sel] = G
    diff = G_all[1:] - G_all[:1]
    diff[..., 1] /= asp
    d = np.linalg.norm(diff * cw, axis=2).max(1)
    if isolated:                                           # interior points with live neighbors on both sides only · capped by their interpolation deviation
        for j, i in enumerate(alive):
            left = np.flatnonzero(w[:i] > 0)
            right = np.flatnonzero(w[i + 1:] > 0) + i + 1
            if not len(left) or not len(right):
                d[j] = 0.0
                continue
            lo, hi = int(left[-1]), int(right[0])
            expected = P[lo] + (P[hi] - P[lo]) * ((i - lo) / (hi - lo))
            d[j] = min(float(d[j]), float(np.linalg.norm(P[i] - expected)))
    j = int(np.argmax(d))                                  # first maximum, as in the old loop · none if ≤ 0
    if d[j] <= 0:
        return -1, 0.0
    return int(alive[j]), float(d[j]) / max(cw, 1e-9)


def fit_textline(pts, weights, cw, ppl=None, *, drop_policy='influence'):
    """Shared fit for tl_1 · tl_2 · tl_4 — robust cubic P-spline after influential-point removal (at most drops points).

    Returns the curve in the coordinates of pts as given -> (curve, last weights).
    cubic keeps degree+1 support points. none removes nothing and uses only the existing Huber weights.
    isolated is a comparison option that excludes only interior points that also deviate by at least one
    character width from the interpolation of their two neighbors.
    """
    o = ALIGN
    Pfit = np.asarray(pts, float)
    K = len(Pfit)
    if K == 0:
        raise ValueError('textline fitting needs at least one point')
    wf = np.asarray(weights, float).copy()
    lam = o['lam0']
    if ppl is not None:
        pbar = float(np.exp(np.log(np.maximum(np.asarray(ppl, float), 1e-9)).mean()))
        rel = pbar / max(float(o['ppl_ref']), 1e-9)
        gam = o['gam'] + float(o['gam1']) * np.log(max(rel, 1e-9))
        lam *= rel ** gam
    alive = int((wf > 0).sum())
    if drop_policy not in ('influence', 'none', 'isolated', 'cubic'):
        raise ValueError(f'Unknown point-exclusion policy: {drop_policy}')
    for _ in range(int(o['drops']) if drop_policy != 'none' else 0):
        if alive <= (max(3, int(o['deg']) + 1) if drop_policy == 'cubic' else 3):
            break
        i, v = _influence(Pfit, wf, lam, cw, o['deg'], o['huber'], o['iters'], 1.0,
                          isolated=drop_policy == 'isolated')
        if i < 0 or v <= o['tau']:
            break
        wf[i] = 0.0
        alive -= 1
    return pspline(Pfit, wf, lam, cw, deg=o['deg'], huber=o['huber'],
                   iters=o['iters'], asp=1.0, end_guard=o['end_guard'])


def nearest_sample_indices(curve, points, max_pairs=65536):
    """For each point, the index of the curve sample with the smallest squared distance (earlier index on ties)."""
    c = np.asarray(curve)
    p = np.asarray(points)
    if c.ndim != 2 or c.shape[1] != 2:
        raise ValueError('curve must have shape (samples, 2)')
    if p.size == 0:
        return np.empty(0, dtype=np.intp)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError('points must have shape (points, 2)')
    if len(c) == 0:
        raise ValueError('cannot project points onto an empty sampled curve')
    if max_pairs < 1:
        raise ValueError('max_pairs must be positive')
    result = np.empty(len(p), dtype=np.intp)
    block = max(1, int(max_pairs) // len(c))
    for first in range(0, len(p), block):
        part = p[first:first + block]
        dx = c[None, :, 0] - part[:, None, 0]
        dy = c[None, :, 1] - part[:, None, 1]
        np.square(dx, out=dx)
        np.square(dy, out=dy)
        np.add(dx, dy, out=dx)
        result[first:first + len(part)] = np.argmin(dx, axis=1)
    return result


# ── Observation point p90 ────────────────────────────────────────────────────

def _initial_geometry(prs, gw, gh, wh0, *, initial='argmax', drop_policy='influence'):
    """Cleaned maps -> tl_1 (per-position reference point p_initial, unit tangent, median spacing of fitted points | None).

    Fit points = max cell (argmax), overall centroid (centroid), or centroid of the top 70 % mass (top70).
    For every option, weight = selected mass of the top 70 %, and the points are clipped into the image.
    With one position, that point itself is the reference point, without a curve.
    """
    prs = normalize_attention_maps(prs)
    K = len(prs)
    ij = np.arange(gw * gh, dtype=np.float64)
    xy = np.column_stack((ij % gw, ij // gw))
    p_argmax = np.asarray([xy[int(np.argmax(pr))] for pr in prs], dtype=np.float64).reshape(-1, 2)
    observations = [top_q_observation(pr, gw, gh, INITIAL_MASS_Q) for pr in prs]   # once only — for both the mass and the top70 points
    p70_mass = np.asarray([o[2] for o in observations], dtype=np.float64)
    p_argmax = clip_points_to_image(p_argmax, gw, gh, wh0)
    if initial == 'argmax':
        p_fit = p_argmax.copy()
    elif initial == 'centroid':
        p_fit = prs @ xy
    elif initial == 'top70':
        p_fit = np.asarray([o[0] for o in observations]).reshape(-1, 2)
    else:
        raise ValueError(f'Unknown initial observation: {initial}')
    p_fit = clip_points_to_image(p_fit, gw, gh, wh0)
    if K <= 1:
        return p_fit, np.empty((0, 2)), None

    raw_step = np.linalg.norm(np.diff(p_fit, axis=0), axis=1)
    positive = raw_step[raw_step > 1e-9]
    cw = float(np.median(positive)) if len(positive) else max(gw / K, 1.0)
    ppl = [perplexity(pr, gw, gh, K) for pr in prs]
    curve, _ = fit_textline(p_fit, np.maximum(p70_mass, 1e-6), cw, ppl=ppl, drop_policy=drop_policy)
    slots = np.arange(K, dtype=np.float64)
    p_initial = clip_points_to_image(curve(slots), gw, gh, wh0)
    eps = 1e-3
    tangent = np.asarray(curve(slots + eps), dtype=np.float64) - \
        np.asarray(curve(slots - eps), dtype=np.float64)
    tangent_norm = np.linalg.norm(tangent, axis=1)
    valid = tangent_norm > 1e-9
    tangents = np.zeros_like(tangent)
    tangents[valid] = tangent[valid] / tangent_norm[valid, None]
    # a degenerate tangent takes the nearest valid tangent
    if valid.any():
        valid_idx = np.flatnonzero(valid)
        for k in np.flatnonzero(~valid):
            tangents[k] = tangents[valid_idx[np.argmin(np.abs(valid_idx - k))]]

    fitted_step = np.linalg.norm(np.diff(p_initial, axis=0), axis=1)
    fitted_positive = fitted_step[fitted_step > 1e-9]
    spacing = float(np.median(fitted_positive)) if len(fitted_positive) else None
    return p_initial, tangents, spacing


def _blobs(sel, pr, gw, gh):
    """Selection mask -> (8-neighbor blob labels (gh, gw), number of blobs, grid probabilities (gh, gw))."""
    from scipy import ndimage
    lab, nc = ndimage.label(sel.reshape(gh, gw), structure=np.ones((3, 3)))
    return lab, nc, pr.reshape(gh, gw)


def _mirror(p, gw, gh):
    """Point reflection about the grid center."""
    return (float(gw - 1) - p[0], float(gh - 1) - p[1])


def _last_est(ctx, gw, gh):
    """Extrapolated position of the last position p_{K−2} + (p_{K−2} − p_{K−3}).

    For a 2-character image, the point reflection of the previous point. None if not the last position."""
    prev = ctx.get('prev') or []
    if not ctx['last'] or ctx['first'] or len(prev) < 1:
        return None
    if len(prev) == 1:
        return _mirror(prev[-1], gw, gh)
    (x1, y1), (x2, y2) = prev[-2], prev[-1]
    return (2 * x2 - x1, 2 * y2 - y1)


def _blob_soft(sel, pr, gw, gh, ctx, *, tail_extrapolate=True):
    """For each 8-neighbor blob b of the selected cells, give a per-cell distance weight f_b = max(1 − α·d_b², 0).

    d_b = distance from the reference point to the blob centroid ÷ character spacing. The reference point is the
    centroid of the whole selection,
    for the last position the extrapolation of the two previous final points (_last_est), or ctx['ref'] if given.
    With one blob, unchanged; if all weights are 0, only the nearest blob (the heavier one on ties).
    """
    lab, nc, P = _blobs(sel, pr, gw, gh)
    if nc <= 1:
        return sel
    m0 = sel.reshape(gh, gw)
    grid = _grid(gw, gh)
    rr, cc = grid['rr'], grid['cc']
    w0 = P * m0
    tot = w0.sum()
    if tot <= 0:
        return sel
    px, py = (w0 * cc).sum() / tot, (w0 * rr).sum() / tot
    est = _last_est(ctx, gw, gh) if tail_extrapolate else None
    if est is not None:
        px, py = est
    if ctx.get('ref') is not None:
        px, py = ctx['ref']
    unit = ctx['cw'] if ctx['cw'] else float(gw)
    alpha = ctx['alpha']
    out = np.zeros((gh, gw), float)
    best, bk = 1, None
    for c in range(1, nc + 1):
        m = lab == c
        w = P[m]
        mass = float(w.sum())
        bx, by = float((w * cc[m]).sum() / mass), float((w * rr[m]).sum() / mass)
        d = float(np.hypot(bx - px, by - py)) / max(unit, 1e-9)
        out[m] = max(1.0 - alpha * d * d, 0.0)
        key = (d, -mass)
        if bk is None or key < bk:
            best, bk = c, key
    if out.sum() <= 0:
        return (lab == best).ravel()
    return out.ravel()


def top_mass(pr, gw, gh, first, last, cw, prev, ref, axis, spacing, *, prior_blend=False, tail_extrapolate=True):
    """Cleaned map of one position (gw·gh,) -> per-cell selection (boolean) or weight (float).

    axis_prior: multiply by max(1 − α·d², 0), with d the distance along the text line measured from reference
    point ref along tangent axis, then select the top MASS_Q.
    blob_soft: blob distance weight of the selected cells. If axis_prior was applied, its weight is multiplied
    in again per cell.
    """
    ctx = {'first': first, 'last': last, 'cw': cw, 'alpha': ONE_ALPHA,
           'prev': prev or [], 'ref': ref}
    pre = None
    if ref is not None and axis is not None and spacing is not None and spacing > 1e-12:
        grid = _grid(gw, gh)
        gx, gy = grid['gx'], grid['gy']
        along = ((gx - ref[0]) * axis[0] + (gy - ref[1]) * axis[1]) / spacing
        candidate = np.maximum(1.0 - ctx['alpha'] * along * along, 0.0)
        if prior_blend:
            # A poorly supported initial curve must not erase the observation.
            # Retained attention mass is the confidence in this geometric prior.
            mass = float(np.clip(np.sum(pr * candidate) / max(float(pr.sum()), 1e-18), 0., 1.))
            candidate = mass * candidate + (1. - mass)
        pre = candidate if float((pr * candidate).sum()) > 0 else np.ones_like(pr)

    ranked = pr if pre is None else pr * pre
    ranked = ranked / ranked.sum()
    o = np.argsort(-ranked)
    n = int(np.searchsorted(np.cumsum(ranked[o]), MASS_Q)) + 1
    sel = np.zeros(pr.size, bool)
    sel[o[:n]] = True
    cond_pr = ranked if pre is not None else pr
    # with a tangent · spacing, ref is only for axis_prior — the blob_soft reference point follows the original rule
    cond_ctx = dict(ctx, ref=None) if (axis is not None and spacing is not None) else ctx
    sel = _blob_soft(sel, cond_pr, gw, gh, cond_ctx, tail_extrapolate=tail_extrapolate)
    if pre is not None:
        sel = np.asarray(sel, dtype=float) * pre
    return sel


def grid_points(rmaps, gw, gh, text=None, wh0=None, *, initial='argmax', initial_drop='influence', prior_blend=False, tail_extrapolate=True):
    """Attention maps of one image -> (per-position grid observation points p90 [(x, y)…],
    per-position selection/weights [arrays…]).

    text is unused (kept for compatibility with the old call signature).
    """
    K = len(rmaps)
    ij = np.arange(gw * gh)
    gx, gy = (ij % gw).astype(float), (ij // gw).astype(float)
    clean, _common = common_min_maps(rmaps)
    prs = corner_prior_maps(neighbor_competition_maps(clean), gw, gh, wh0)
    refs, tangents, spacing = _initial_geometry(prs, gw, gh, wh0, initial=initial, drop_policy=initial_drop)
    cw = spacing if spacing is not None else gw / max(K, 1)
    pts, sels = [], []
    for k, pr in enumerate(prs):
        axis = tangents[k] if k < len(tangents) else None
        ref = refs[k] if axis is not None and k < len(refs) else None
        sel = top_mass(pr, gw, gh, first=(k == 0), last=(k == K - 1), cw=cw, prev=pts,
                       ref=ref, axis=axis, spacing=spacing, prior_blend=prior_blend, tail_extrapolate=tail_extrapolate)
        w = pr * sel
        w = w / w.sum()
        point = clip_points_to_image([[(w * gx).sum(), (w * gy).sum()]], gw, gh, wh0)[0]
        pts.append(tuple(map(float, point)))
        sels.append(sel)
    return pts, sels


# ── Refinement: p90 -> p_tl · tl_4 ──────────────────────────────────────────

_GLYPH: dict = {}


def glyph_offset() -> dict:
    """{character: (offset, spread)} — read once."""
    global _GLYPH
    if not _GLYPH:
        j = json.loads(GLYPH_OFFSET.read_text(encoding='utf-8'))
        _GLYPH = {r['ch']: (float(r['d']), float(r['spread'])) for r in j['rows']}
    return _GLYPH


def precorrect(P, text, tangent=None):
    """Subtract the per-character offset from the character points along the text line normal
    -> (moved points, weight factor, spacing).

    Shift = median spacing × offset (up is +), weight ∝ 1/(0.15 + spread) (mean 1).
    Without tangent, the neighbor-point gradient is used as the tangent. Without a transcription or spacing,
    (P, None, None).
    """
    P = np.asarray(P, float)
    if not text:
        return P, None, None
    s = _spacing(P)
    if not s:
        return P, None, None
    tab = glyph_offset()
    K = len(P)
    chs = list(str(text))[:K] + ['?'] * max(0, K - len(str(text)))
    d = np.array([(tab.get(c) or tab.get(c.lower()) or tab.get(c.upper()) or (0.0, 0.5))[0]
                  for c in chs], float)
    sp = np.array([(tab.get(c) or tab.get(c.lower()) or tab.get(c.upper()) or (0.0, 0.5))[1]
                   for c in chs], float)
    if tangent is None:
        tg = np.gradient(P, axis=0) if len(P) >= 2 else np.array([[1.0, 0.0]])
    else:
        tg = np.asarray(tangent, dtype=float).copy()
        if tg.shape != P.shape:
            raise ValueError(f'tangent shape {tg.shape} != point shape {P.shape}')
    nrm = np.linalg.norm(tg, axis=1, keepdims=True)
    dead = (nrm[:, 0] < 1e-6)
    if dead.any():
        g = P[-1] - P[0]
        gn = float(np.linalg.norm(g))
        tg[dead] = (g / gn) if gn > 1e-9 else np.array([1.0, 0.0])
        nrm[dead] = 1.0
    tg = tg / np.maximum(nrm, 1e-9)
    nv = np.stack([-tg[:, 1], tg[:, 0]], 1)
    Q = P + nv * (s * d)[:, None]      # d is + upward and nv points toward the bottom of the character
    wf = 1.0 / (0.15 + sp)
    return Q, wf / wf.mean(), s


def align(pts, ppl, gw, gh, wh0, rotated, text, *, drop_policy='influence', slot_projection=False):
    """Grid observation points p90 -> (grid character points p_tl [(x, y)…], tl_4 geometry in original px).

    Coordinates · distances are in character-width (gw/K) units, and y is measured after multiplying by the
    pixel aspect ratio (sy/sx).
    At every step the points are clipped inside the image bounds. With fewer than 2 positions, the input points
    without a curve.
    slot_projection limits the projection range of the i-th point to curve parameters i±0.5.
    It prevents projections past the midpoint to a neighboring position; it is not a rule that forces final
    backward steps to 0.
    """
    o = ALIGN
    P0 = clip_points_to_image(pts, gw, gh, wh0)
    K = len(P0)
    if K < 2:
        out0 = [tuple(map(float, p)) for p in P0]
        return out0, dict(format='tl4-spline-v1', knots=[], coefficients=[],
                          degree=0, slots=K, parameter_range=[0.0, 0.0], samples=0, size=0.0)
    w0i, h0i = wh0
    sx, sy = w0i / max(gw - 1, 1), h0i / max(gh - 1, 1)
    asp = sy / sx
    cw = gw / max(K, 1)
    ppl = np.asarray(ppl, float)
    P = P0.copy()
    P[:, 1] *= asp                                   # coordinates scaled to the pixel aspect ratio
    upper = clip_points_to_image([[gw, gh]], gw, gh, wh0)[0] * [1.0, asp]

    def bound(Pscaled):
        return np.clip(Pscaled, 0.0, upper)
    _unused, wg, _s = precorrect(P, text)            # only the weights are used
    w = 1.0 / np.maximum(ppl, 1e-6)
    if wg is not None:
        w = w * wg
    box = (gw - 1.0, (gh - 1.0) * asp)

    def sample_curve(f_):
        """Sample the curve out to both crop ends

        -> (samples in character-width coordinates, cumulative arc length, arc length -> point)."""
        n1, e = K - 1, 1e-3
        p0e, p1e, before, after = np.asarray(f_(np.array([0.0, float(n1), -e, n1 + e])))
        d0v = (p0e - before) / e
        d1v = (after - p1e) / e

        def exit_box(pp, dv):
            cand = []
            for c_, dc, hi in ((pp[0], dv[0], box[0]), (pp[1], dv[1], box[1])):
                if dc > 1e-9:
                    cand.append(max((hi - c_) / dc, 0.0))
                elif dc < -1e-9:
                    cand.append(max((0.0 - c_) / dc, 0.0))
            return min(cand) if cand else 0.0

        t0 = min(-0.5, -min(4.0 * K, exit_box(p0e, d0v)))
        t1 = max(n1 + 0.5, n1 + min(4.0 * K, exit_box(p1e, d1v)))
        ts = np.linspace(t0, t1, max(4000, 200 * K))
        f_.sampling = (float(t0), float(t1), len(ts))
        C_ = np.asarray(f_(ts)) / cw
        L_ = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(C_, axis=0), axis=1))]

        def at_(v):
            return np.stack([np.interp(v, L_, C_[:, 0]), np.interp(v, L_, C_[:, 1])], axis=-1)
        return C_, L_, at_

    # tl_2 — observation points before correction
    f2, _wh2 = fit_textline(P, w, cw, ppl=ppl, drop_policy=drop_policy)

    # per-character height correction — along the normal of the per-position tangent of tl_2
    ks = np.arange(K, dtype=float)
    eps = 1e-3
    tangent2 = np.asarray(f2(ks + eps)) - np.asarray(f2(ks - eps))
    Pc, _wg2, _s2 = precorrect(P, text, tangent=tangent2)
    Pc = bound(Pc)

    # tl_4 — refit to the corrected points and project onto the nearest sample
    f4, wh = fit_textline(Pc, w, cw, ppl=ppl, drop_policy=drop_policy)
    C, L, at = sample_curve(f4)
    def project(q, slots):
        if not slot_projection:
            return nearest_sample_indices(C, q)
        parameters = np.linspace(*f4.sampling)
        result = []
        for point, slot in zip(q, slots):
            allowed = np.flatnonzero(np.abs(parameters - slot) <= .5)
            result.append(int(allowed[np.argmin(np.sum((C[allowed] - point) ** 2, axis=1))]))
        return np.asarray(result, dtype=np.intp)
    nearest = project(Pc / cw, np.arange(K))
    s = L[nearest].copy()
    Pn = C[nearest].copy()
    projected_outside = ((Pn * cw < 0.0) | (Pn * cw > upper)).any(axis=1)
    Pn = bound(Pn * cw) / cw
    # points clamped to the bounds have their arc length read again
    outside = np.flatnonzero(projected_outside)
    if len(outside):
        s[outside] = L[project(Pn[outside], outside)]

    # spacing smoothing — min Σ w_i (s_i − ŝ_i)² + λs Σ (ŝ_{k+1} − 2ŝ_k + ŝ_{k−1})²
    wv = wh if (wh > 0).sum() >= 2 else np.ones(K)
    D2 = np.diff(np.eye(K), n=2, axis=0)
    if len(D2):
        A = np.diag(wv) + (o['lams'] * K / len(D2)) * (D2.T @ D2)
        sh = np.linalg.solve(A, wv * s)
    else:
        sh = s
    Pn[:] = at(sh)
    Pn = bound(Pn * cw) / cw

    # pull-back — toward the observation point before correction by α = reliability / (1 + distance/d0)
    P2 = P / cw
    dd = np.linalg.norm(Pn - P2, axis=1)
    a = reliability(ppl) / (1.0 + dd / max(o['d0'], 1e-6))
    Pn = Pn + (P2 - Pn) * a[:, None]

    G = bound(np.asarray(Pn * cw, float))
    G[:, 1] /= asp
    out = [tuple(map(float, p)) for p in G]
    # f4 is in aspect-corrected grid coordinates — multiplying both axes by sx gives STR input px;
    # undo the rotation if rotated
    coeff = np.asarray(f4.spline['coefficients'], float) * sx
    if rotated:
        coeff = np.column_stack([h0i - 1 - coeff[:, 1], coeff[:, 0]])
    t0, t1, samples = f4.sampling
    geometry = dict(f4.spline, format='tl4-spline-v1', coefficients=coeff.tolist(),
                    slots=K, parameter_range=[t0, t1], samples=samples,
                    size=float((_spacing(P) or (1.0 / sx)) * sx))
    return out, geometry


def grid_points_px(points, gw, gh, wh_orig, rotated):
    """Grid points -> original crop px (float64 before rounding). Stretch the grid ends to (W, H) and undo the rotation."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if not len(pts):
        return pts
    w0, h0 = wh_orig
    out = pts * [w0 / max(gw - 1, 1), h0 / max(gh - 1, 1)]
    if rotated:
        out = np.column_stack([h0 - 1 - out[:, 1], out[:, 0]])
    return out


def derive(rec, *, policy=None):
    """One attention record -> (original px character points p_tl float32 (K, 2) · two decimal places, tl_4 geometry).

    rec keys: maps (K, gh·gw) · gw · gh · wh0 (W, H in the STR input orientation) · rotated · text.
    geometry: knots · coefficients (original px) · degree · [spline_domain] · format · slots ·
    parameter_range · samples · size (median spacing of observation points, px).
    policy=None reproduces the existing computation unchanged. The validated improved combination is policy=REFINED_POLICY.
    Geometry · segmentation · window caches with a different policy must not be reused under the same definition.
    """
    maps, gw, gh = rec['maps'], rec['gw'], rec['gh']
    if not len(maps):
        return (np.asarray(np.empty((0, 2)), np.float32).reshape(-1, 2),
                dict(format='tl4-spline-v1', knots=[], coefficients=[],
                     degree=0, slots=0, parameter_range=[0, 0], samples=0, size=0.0))
    policy = dict(policy or {})
    align_drop = policy.pop('align_drop', 'influence')
    slot_projection = policy.pop('slot_projection', False)
    raw, _ = grid_points(maps, gw, gh, text=rec['text'], wh0=rec['wh0'], **policy)
    ppl = [perplexity(np.asarray(m, float) / max(float(np.sum(m)), 1e-18), gw, gh, len(maps))
           for m in maps]
    points, geometry = align(raw, ppl, gw, gh, rec['wh0'], rec['rotated'], rec['text'],
                             drop_policy=align_drop, slot_projection=slot_projection)
    px = grid_points_px(points, gw, gh, rec['wh0'], rec['rotated'])
    return np.asarray(np.round(px, 2), np.float32).reshape(-1, 2), geometry


# ── Reading a stored tl_4 ────────────────────────────────────────────────────

def evaluate(geometry, parameters, derivative=False):
    """Evaluate tl_4 at the parameters (first derivative if derivative=True).

    Outside the fitted interval, straight extension along the end tangent."""
    from scipy.interpolate import BSpline
    u = np.asarray(parameters, dtype=float)
    if not geometry['samples']:
        return np.empty((0, 2), dtype=float)
    spline = BSpline(geometry['knots'], geometry['coefficients'], geometry['degree'],
                     extrapolate=False)
    start, end = geometry.get('spline_domain', (0.0, geometry['slots'] - 1.0))
    clipped = np.clip(u, start, end)
    if derivative:
        return np.asarray(spline(clipped, nu=1), dtype=float)
    points = np.asarray(spline(clipped), dtype=float)
    for edge, mask in ((start, u < start), (end, u > end)):
        tail = spline(edge) + (u - edge)[..., None] * spline(edge, nu=1)
        points = np.where(mask[..., None], tail, points)
    return points


def line(geometry):
    """Original px curve samples of tl_4 (same range · count as at fit time)."""
    return evaluate(geometry, np.linspace(*geometry['parameter_range'], geometry['samples']))


def angles(geometry, points):
    """Reading-direction tangent angle at the nearest tl_4 sample for each character point
    (radian · direction of increasing position index).

    If the tangent is degenerate, the tangent of the nearest valid sample; if all are degenerate, (1, 0).
    0 if there is no curve.
    """
    if not geometry['samples'] or not len(points):
        return np.zeros(len(points))
    ts = np.linspace(*geometry['parameter_range'], geometry['samples'])
    curve = evaluate(geometry, ts)
    nearest = nearest_sample_indices(curve, points)
    tangent = evaluate(geometry, ts[nearest], derivative=True)
    dead = np.linalg.norm(tangent, axis=1) <= 1e-9
    if dead.any():
        all_tangents = evaluate(geometry, ts, derivative=True)
        valid = np.flatnonzero(np.linalg.norm(all_tangents, axis=1) > 1e-9)
        for i in np.flatnonzero(dead):
            tangent[i] = (all_tangents[valid[np.argmin(np.abs(valid - nearest[i]))]]
                          if len(valid) else [1.0, 0.0])
    return np.arctan2(tangent[:, 1], tangent[:, 0])


# ── Original px maps ─────────────────────────────────────────────────────────

def px_maps(rmaps, gw, gh, wh0, rotated):
    """Grid maps -> per-position maps at original px size (K, h, w) float32.

    Linear interpolation to the input size, then the rotation is undone."""
    import cv2
    w0, h0 = wh0
    out = []
    for a in rmaps:
        g = np.asarray(a, np.float32).reshape(gh, gw)
        m = cv2.resize(g, (w0, h0), interpolation=cv2.INTER_LINEAR)
        out.append(np.rot90(m, -1) if rotated else m)
    return np.stack(out)
