"""attention 지도 → 글자 위치 p_tl · 글줄 곡선 tl_4.

SVTRv2-AR 디코더가 자리(timestep)마다 낸 cross-attention 지도 (K, gh·gw) 에서

1. 지도 손질 — 자리 공통 min 제거 → 인접 자리 competition → 구석 prior
2. 초기 글줄 tl_1 — 손질한 지도의 최대 칸 p_argmax 에 3차 곡선을 맞춰 자리별 기준점 · 접선
3. 관측점 p90 — axis_prior(글줄 방향 거리 무게) → 누적 질량 상위 90 % → blob_soft(덩어리 거리 무게) 무게중심
4. 정돈 — tl_2 맞춤 → tl_2 법선으로 글자별 높이 보정 → tl_4 재맞춤 · 최근접 투영 → 간격 급변 완화 → 신뢰도만큼 관측점 쪽으로 되돌림

을 거쳐 원본 crop px 의 자리점 p_tl 과 tl_4 곡선 계수(geometry)를 낸다. 좌표 규약: 격자 (x, y) = (열, 행),
px 는 격자 끝을 (W, H) 로 늘린 뒤 회전(RotateTall)을 되돌린 원본 crop 좌표다.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / 'data'
GLYPH_OFFSET = DATA / 'glyph_offset.json'   # 글자별 치우침 표 (합성 실측 · x-height 단위 · 위가 +)

MASS_Q = 0.9                        # 관측점 p90 이 남기는 누적 질량
INITIAL_MASS_Q = 0.7                # tl_1 맞춤 무게 = 상위 70 % 선택 질량
NEIGHBOR_COMPETITION_LAM = 0.5      # 인접 자리 competition 세기 (0 끔 · 1 소유 몫만큼 전부 감점)
CORNER_PRIOR_SIGMA_RATIO = 0.25     # 구석 prior σ = 짧은 변 × 이 값
CORNER_PRIOR_MIN_MASS = 1e-14       # 구석 prior 뒤 남은 질량이 이 이하면 보정 전 지도로 되돌림
ONE_ALPHA = 0.2                     # axis_prior · blob_soft 거리 무게 max(1 − α·d², 0) 의 α (d 는 글자 간격 단위)
PPL_HALF = 3.0                      # 신뢰도 1/(1 + ppl/PPL_HALF) 가 0.5 가 되는 퍼짐

# Explicit refinement recipe. The no-policy path retains the published cache
# definition; callers must use a separate geometry/crop key for this recipe.
REFINED_VERSION = 'centroid70-cubic4-local-projection-v1'
REFINED_POLICY = dict(initial='top70', initial_drop='cubic', align_drop='cubic', slot_projection=True)

#: 글줄 맞춤 설정 — tl_1 · tl_2 · tl_4 공용.
ALIGN = dict(
    lam0=0.003,        # 휨 벌점 λ (계수 2차 차분)
    gam=0.0,           # 장별 λ 를 퍼짐으로 바꾸는 지수 (0 = 끔)
    gam1=0.0,          # 같은 지수의 로그 항 (0 = 끔)
    ppl_ref=1.0,       # 그 지수의 기준 퍼짐
    deg=3,             # 곡선 차수
    huber=0.1,         # IRLS Huber 문턱 (글자폭)
    iters=1,           # IRLS 반복
    tau=1.0,           # 영향점 제거 문턱 — 한 점을 빼면 곡선이 이만큼(글자폭) 넘게 움직일 때
    drops=2,           # 영향점 제거 최대 개수
    lams=0.3,          # 간격 완화 벌점 (호 길이 2차 차분)
    d0=1.0,            # 되돌림 거리 척도 (글자폭)
    end_guard=True,    # 끝 한 자리에서 접선이 뒤집히면 접선 연장 시작을 당긴다
)


# ── 지도 손질 ──────────────────────────────────────────────────────────────

def normalize_attention_maps(rmaps):
    """자리별 attention 을 각각 합 1 인 (K, cells) 분포로. 합이 0 이면 균등."""
    if len(rmaps) == 0:
        return np.empty((0, 0), dtype=np.float64)
    maps = np.asarray(rmaps, dtype=np.float64).reshape(len(rmaps), -1)
    total = maps.sum(axis=1, keepdims=True)
    uniform = np.full_like(maps, 1.0 / max(maps.shape[1], 1))
    return np.divide(maps, np.maximum(total, 1e-18), out=uniform,
                     where=total > 1e-18)


def common_min_maps(rmaps):
    """모든 자리에 공통인 칸별 최솟값을 빼고 다시 정규화 -> (지도, 공통 min).

    제거 뒤 전부 0 인 자리는 원래 정규화 지도로 되돌린다.
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
    """인접 자리끼리 같은 칸을 동시에 한 번 나눠 갖는다.

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
    """가장 가까운 이미지 구석과의 거리 d 로 g = 1 − exp(−d²/(2σ²)) (칸별 · 평탄화).

    칸 좌표는 x = 열·W/(gw−1), y = 행·H/(gh−1). 한 축이 1칸이면 감쇠하지 않는다.
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
    """구석 prior 를 한 번 곱하고 자리별 합 1. 질량이 사라진 자리는 보정 전으로."""
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
    """격자 칸 좌표(장마다 같은 크기라 한 번 만들어 둔다 · 읽기 전용) — (칸 번호 ij, x, y, (x, y) 열 묶음, mgrid 행 · 열)."""
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
    """누적 질량 상위 q 마스크 -> (그 안의 확률가중 중심 (x, y), 마스크, 남은 질량)."""
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
    """자리 하나의 attention 퍼짐 — exp(엔트로피) ÷ 자리당 칸 수(gw·gh/K)."""
    p = np.asarray(pr, np.float64)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum())) / max(gw * gh / K, 1e-9)


def reliability(ppl, half=PPL_HALF):
    """퍼짐 -> 신뢰도 1/(1 + ppl/half)."""
    return 1.0 / (1.0 + np.asarray(ppl, float) / half)


def clip_points_to_image(pts, gw, gh, wh0=None):
    """격자 자리점을 이미지 경계 [0, W−1]×[0, H−1] 를 격자로 옮긴 사각형 안으로 (새 배열).

    wh0 = STR 입력 방향의 (W, H). 없으면 격자 경계 [0, gw−1]×[0, gh−1].
    """
    points = np.asarray(pts, dtype=float).reshape(-1, 2)
    upper = np.maximum(np.asarray([gw, gh], dtype=float) - 1.0, 0.0)
    if wh0 is not None:
        size = np.asarray(wh0, dtype=float)
        upper *= np.maximum(size - 1.0, 0.0) / size
    return np.clip(points, 0.0, upper)


# ── 곡선 맞춤 (P-spline) ───────────────────────────────────────────────────

def _spacing(P):
    """이웃 점 간격의 중앙값 (x-height 어림값). 점이 하나이거나 간격이 모두 0 이면 None."""
    q = np.asarray(P, float)
    if len(q) < 2:
        return None
    d = np.linalg.norm(np.diff(q, axis=0), axis=1)
    d = d[d > 1e-6]
    return float(np.median(d)) if len(d) else None


def _irls(P, w, solve, huber, iters):
    """IRLS — 신뢰 무게 w 에 잔차 Huber 무게 min(1, δ/|r|) 를 곱해 iters 번 다시 맞춘다 -> (마디값, 마지막 무게)."""
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
    """자리 수 n -> (마디 t, 기저 B (n×m), 2차 차분 D, 실제 차수). 내부 마디 없음(곡선 전체가 다항식 하나)."""
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
    """끝 한 자리에서 접선이 역방향(내적 ≤ 0)이면, 곡률 반경이 radius 보다 작아지기 직전에서 접선 연장을 시작한다."""
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
    # 유효 구간이 사라지면 원래 구간
    return (lo, hi) if lo < hi else (0.0, float(n - 1))


def pspline(pts, w, lam, cw, deg=3, huber=None, iters=3, asp=1.0, end_guard=False):
    """가중 smoothing B-spline — 자리 번호 k 를 매개변수로 x(k) · y(k).

    좌표는 글자폭 cw 단위, 벌점은 계수 2차 차분: (BᵀWB + λ·n/행수·DᵀD) c = BᵀW p.
    끝 밖은 끝 접선으로 직선 연장. -> (k -> (x, y) 원래 단위, 마지막 무게).
    """
    from scipy.interpolate import BSpline
    P = np.asarray(pts, float) / cw
    P[:, 1] *= asp
    n = len(P)
    w = np.asarray(w, float)
    w = w / max(w.mean(), 1e-9)
    if n == 1:
        return (lambda k: (float(P[0, 0] * cw), float(P[0, 1] * cw / asp))), w
    nz = int((w > 0).sum())          # 무게 0 인 점은 자료가 아니다 — 차수도 그만큼 내린다
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

    # 저장용 — 실제 풀이의 계수 (원래 단위)
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
    W = np.repeat(w[None, :], len(alive) + 1, 0)           # 줄 0 = 기준(아무것도 안 뺌) · 줄 j+1 = alive[j] 를 뺀 무게
    W[np.arange(1, len(alive) + 1), alive] = 0.0
    W = W / np.maximum(W.mean(1, keepdims=True), 1e-9)
    dg_all = np.minimum(deg, np.maximum(1, (W > 0).sum(1) - 1))
    G_all = np.zeros((len(W), n, 2))
    for dg in np.unique(dg_all):                           # 남은 점이 적어 차수가 내려가는 줄은 따로 푼다
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
    if isolated:                                           # 양옆에 살아 있는 점이 있는 내부점만 · 양옆 보간과의 어긋남으로 줄임
        for j, i in enumerate(alive):
            left = np.flatnonzero(w[:i] > 0)
            right = np.flatnonzero(w[i + 1:] > 0) + i + 1
            if not len(left) or not len(right):
                d[j] = 0.0
                continue
            lo, hi = int(left[-1]), int(right[0])
            expected = P[lo] + (P[hi] - P[lo]) * ((i - lo) / (hi - lo))
            d[j] = min(float(d[j]), float(np.linalg.norm(P[i] - expected)))
    j = int(np.argmax(d))                                  # 옛 반복과 같게 첫 최대 · 0 이하면 없음
    if d[j] <= 0:
        return -1, 0.0
    return int(alive[j]), float(d[j]) / max(cw, 1e-9)


def fit_textline(pts, weights, cw, ppl=None, *, drop_policy='influence'):
    """tl_1 · tl_2 · tl_4 공용 맞춤 — 영향점 제거(최대 drops 개) 뒤 robust 3차 P-spline.

    pts 좌표 그대로 곡선을 돌려준다 -> (곡선, 마지막 무게).
    cubic 은 차수+1 개의 지지점을 보존한다. none 은 삭제 없이 기존 Huber만 쓴다.
    isolated 는 양옆 점 보간과도 한 글자폭 이상 어긋난 내부점만 제외하는 비교 옵션이다.
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
    """곡선 표본 중 각 점에서 제곱 거리가 가장 작은 표본 번호 (같으면 앞 번호)."""
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


# ── 관측점 p90 ─────────────────────────────────────────────────────────────

def _initial_geometry(prs, gw, gh, wh0, *, initial='argmax', drop_policy='influence'):
    """손질한 지도 -> tl_1 의 (자리별 기준점 p_initial, 단위 접선, 맞춘 점 간격 중앙값 | None).

    맞춤 점 = 최대 칸(argmax), 전체 무게중심(centroid), 상위 70 % 질량의 무게중심(top70).
    어느 옵션이든 무게 = 상위 70 % 선택 질량, 점은 이미지 안으로 제한한다.
    한 자리면 곡선 없이 그 점 자체가 기준점이다.
    """
    prs = normalize_attention_maps(prs)
    K = len(prs)
    ij = np.arange(gw * gh, dtype=np.float64)
    xy = np.column_stack((ij % gw, ij // gw))
    p_argmax = np.asarray([xy[int(np.argmax(pr))] for pr in prs], dtype=np.float64).reshape(-1, 2)
    observations = [top_q_observation(pr, gw, gh, INITIAL_MASS_Q) for pr in prs]   # 한 번만 — 질량과 top70 점에 같이 쓴다
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
    # 퇴화한 접선은 가장 가까운 유효 접선으로
    if valid.any():
        valid_idx = np.flatnonzero(valid)
        for k in np.flatnonzero(~valid):
            tangents[k] = tangents[valid_idx[np.argmin(np.abs(valid_idx - k))]]

    fitted_step = np.linalg.norm(np.diff(p_initial, axis=0), axis=1)
    fitted_positive = fitted_step[fitted_step > 1e-9]
    spacing = float(np.median(fitted_positive)) if len(fitted_positive) else None
    return p_initial, tangents, spacing


def _blobs(sel, pr, gw, gh):
    """선택 마스크 -> (8-이웃 덩어리 라벨 (gh, gw), 덩어리 수, 격자 확률 (gh, gw))."""
    from scipy import ndimage
    lab, nc = ndimage.label(sel.reshape(gh, gw), structure=np.ones((3, 3)))
    return lab, nc, pr.reshape(gh, gw)


def _mirror(p, gw, gh):
    """격자 중심에 대한 점대칭 위치."""
    return (float(gw - 1) - p[0], float(gh - 1) - p[1])


def _last_est(ctx, gw, gh):
    """끝자리의 외삽 위치 p_{K−2} + (p_{K−2} − p_{K−3}). 2글자 장은 앞 점의 점대칭. 끝자리가 아니면 None."""
    prev = ctx.get('prev') or []
    if not ctx['last'] or ctx['first'] or len(prev) < 1:
        return None
    if len(prev) == 1:
        return _mirror(prev[-1], gw, gh)
    (x1, y1), (x2, y2) = prev[-2], prev[-1]
    return (2 * x2 - x1, 2 * y2 - y1)


def _blob_soft(sel, pr, gw, gh, ctx, *, tail_extrapolate=True):
    """선택 칸의 8-이웃 덩어리 b 마다 거리 무게 f_b = max(1 − α·d_b², 0) 를 칸별로 준다.

    d_b = 기준점 → 덩어리 무게중심 거리 ÷ 글자 간격. 기준점은 선택 전체의 무게중심,
    끝자리는 앞 두 최종 점의 외삽(_last_est), ctx['ref'] 가 있으면 그것.
    덩어리가 하나면 그대로, 무게가 전부 0 이면 가장 가까운(같으면 무거운) 덩어리 하나만.
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
    """한 자리의 손질한 지도 (gw·gh,) -> 칸별 선택(불리언) 또는 무게(실수).

    axis_prior: 기준점 ref 와 접선 axis 로 잰 글줄 방향 거리 d 로 max(1 − α·d², 0) 를 곱한 뒤 상위 MASS_Q 선택.
    blob_soft: 선택 칸의 덩어리 거리 무게. axis_prior 를 걸었으면 그 무게를 칸별로 다시 곱한다.
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
    # 접선 · 간격이 있으면 ref 는 axis_prior 전용 — blob_soft 기준점은 원래 규칙
    cond_ctx = dict(ctx, ref=None) if (axis is not None and spacing is not None) else ctx
    sel = _blob_soft(sel, cond_pr, gw, gh, cond_ctx, tail_extrapolate=tail_extrapolate)
    if pre is not None:
        sel = np.asarray(sel, dtype=float) * pre
    return sel


def grid_points(rmaps, gw, gh, text=None, wh0=None, *, initial='argmax', initial_drop='influence', prior_blend=False, tail_extrapolate=True):
    """한 장의 attention 지도들 -> (자리별 격자 관측점 p90 [(x, y)…], 자리별 선택/무게 [배열…]).

    text 는 쓰지 않는다(옛 호출 꼴 호환).
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


# ── 정돈: p90 -> p_tl · tl_4 ──────────────────────────────────────────────

_GLYPH: dict = {}


def glyph_offset() -> dict:
    """{글자: (치우침, 흩어짐)} — 한 번 읽어 둔다."""
    global _GLYPH
    if not _GLYPH:
        j = json.loads(GLYPH_OFFSET.read_text(encoding='utf-8'))
        _GLYPH = {r['ch']: (float(r['d']), float(r['spread'])) for r in j['rows']}
    return _GLYPH


def precorrect(P, text, tangent=None):
    """자리점에서 글자별 치우침을 글줄 법선 방향으로 뺀다 -> (옮긴 점, 무게 곱, 간격).

    이동 = 간격 중앙값 × 치우침(위가 +), 무게 ∝ 1/(0.15 + 흩어짐) (평균 1).
    tangent 가 없으면 이웃 점 gradient 를 접선으로 쓴다. 전사나 간격이 없으면 (P, None, None).
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
    Q = P + nv * (s * d)[:, None]      # d 는 위가 + 이고 nv 는 글자의 아래쪽
    wf = 1.0 / (0.15 + sp)
    return Q, wf / wf.mean(), s


def align(pts, ppl, gw, gh, wh0, rotated, text, *, drop_policy='influence', slot_projection=False):
    """격자 관측점 p90 -> (격자 자리점 p_tl [(x, y)…], 원본 px 의 tl_4 geometry).

    좌표 · 거리는 글자폭(gw/K) 단위이고 y 는 픽셀 종횡비(sy/sx)를 곱해 잰다.
    단계마다 점을 이미지 경계 안으로 제한한다. 자리가 2 미만이면 곡선 없이 입력점.
    slot_projection 은 i번째 점의 투영 범위를 곡선 매개변수 i±0.5로 제한한다.
    이웃 자리의 중간을 넘는 투영을 막으며, 최종 역행을 0으로 강제하는 규칙은 아니다.
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
    P[:, 1] *= asp                                   # 픽셀 비율로 맞춘 좌표
    upper = clip_points_to_image([[gw, gh]], gw, gh, wh0)[0] * [1.0, asp]

    def bound(Pscaled):
        return np.clip(Pscaled, 0.0, upper)
    _unused, wg, _s = precorrect(P, text)            # 무게만 쓴다
    w = 1.0 / np.maximum(ppl, 1e-6)
    if wg is not None:
        w = w * wg
    box = (gw - 1.0, (gh - 1.0) * asp)

    def sample_curve(f_):
        """곡선을 crop 양끝까지 표본화 -> (글자폭 좌표 표본, 누적 호 길이, 호 길이 -> 점)."""
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

    # tl_2 — 보정 전 관측점
    f2, _wh2 = fit_textline(P, w, cw, ppl=ppl, drop_policy=drop_policy)

    # 글자별 높이 보정 — tl_2 의 자리별 접선의 법선 방향
    ks = np.arange(K, dtype=float)
    eps = 1e-3
    tangent2 = np.asarray(f2(ks + eps)) - np.asarray(f2(ks - eps))
    Pc, _wg2, _s2 = precorrect(P, text, tangent=tangent2)
    Pc = bound(Pc)

    # tl_4 — 보정점에 다시 맞추고 최근접 표본으로 투영
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
    # 경계로 물린 점은 호 길이를 다시 읽는다
    outside = np.flatnonzero(projected_outside)
    if len(outside):
        s[outside] = L[project(Pn[outside], outside)]

    # 간격 완화 — min Σ w_i (s_i − ŝ_i)² + λs Σ (ŝ_{k+1} − 2ŝ_k + ŝ_{k−1})²
    wv = wh if (wh > 0).sum() >= 2 else np.ones(K)
    D2 = np.diff(np.eye(K), n=2, axis=0)
    if len(D2):
        A = np.diag(wv) + (o['lams'] * K / len(D2)) * (D2.T @ D2)
        sh = np.linalg.solve(A, wv * s)
    else:
        sh = s
    Pn[:] = at(sh)
    Pn = bound(Pn * cw) / cw

    # 되돌림 — α = 신뢰도 / (1 + 거리/d0) 만큼 보정 전 관측점 쪽으로
    P2 = P / cw
    dd = np.linalg.norm(Pn - P2, axis=1)
    a = reliability(ppl) / (1.0 + dd / max(o['d0'], 1e-6))
    Pn = Pn + (P2 - Pn) * a[:, None]

    G = bound(np.asarray(Pn * cw, float))
    G[:, 1] /= asp
    out = [tuple(map(float, p)) for p in G]
    # f4 는 종횡비를 맞춘 격자 좌표 — 두 축 모두 sx 배 하면 STR 입력 px, 회전이면 되돌린다
    coeff = np.asarray(f4.spline['coefficients'], float) * sx
    if rotated:
        coeff = np.column_stack([h0i - 1 - coeff[:, 1], coeff[:, 0]])
    t0, t1, samples = f4.sampling
    geometry = dict(f4.spline, format='tl4-spline-v1', coefficients=coeff.tolist(),
                    slots=K, parameter_range=[t0, t1], samples=samples,
                    size=float((_spacing(P) or (1.0 / sx)) * sx))
    return out, geometry


def grid_points_px(points, gw, gh, wh_orig, rotated):
    """격자 점 -> 원본 crop px (반올림 전 float64). 격자 끝을 (W, H) 로 늘리고 회전을 되돌린다."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if not len(pts):
        return pts
    w0, h0 = wh_orig
    out = pts * [w0 / max(gw - 1, 1), h0 / max(gh - 1, 1)]
    if rotated:
        out = np.column_stack([h0 - 1 - out[:, 1], out[:, 0]])
    return out


def derive(rec, *, policy=None):
    """attention 레코드 하나 -> (원본 px 자리점 p_tl float32 (K, 2) · 소수 둘째 자리, tl_4 geometry).

    rec 열쇠: maps (K, gh·gw) · gw · gh · wh0 (STR 입력 방향 W, H) · rotated · text.
    geometry: knots · coefficients(원본 px) · degree · [spline_domain] · format · slots ·
    parameter_range · samples · size(관측점 간격 중앙값 px).
    policy=None 은 기존 계산을 그대로 재현한다. 검증한 개선 조합은 policy=REFINED_POLICY.
    정책이 다른 기하·분할·창 캐시는 같은 정의로 재사용하면 안 된다.
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


# ── 저장된 tl_4 읽기 ───────────────────────────────────────────────────────

def evaluate(geometry, parameters, derivative=False):
    """tl_4 를 매개변수에서 평가 (derivative=True 면 1차 미분). 맞춘 구간 밖은 끝 접선 직선 연장."""
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
    """tl_4 의 원본 px 곡선 표본 (맞출 때와 같은 범위 · 개수)."""
    return evaluate(geometry, np.linspace(*geometry['parameter_range'], geometry['samples']))


def angles(geometry, points):
    """자리점마다 가장 가까운 tl_4 표본에서의 읽기 방향 접선각 (radian · 자리 번호가 느는 방향).

    접선이 퇴화하면 가장 가까운 유효 표본의 접선, 전부 퇴화하면 (1, 0). 곡선이 없으면 0.
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


# ── 원본 px 지도 ───────────────────────────────────────────────────────────

def px_maps(rmaps, gw, gh, wh0, rotated):
    """격자 지도 -> 원본 px 크기의 자리별 지도 (K, h, w) float32 — 입력 크기로 선형 보간 뒤 회전을 되돌린다."""
    import cv2
    w0, h0 = wh0
    out = []
    for a in rmaps:
        g = np.asarray(a, np.float32).reshape(gh, gw)
        m = cv2.resize(g, (w0, h0), interpolation=cv2.INTER_LINEAR)
        out.append(np.rot90(m, -1) if rotated else m)
    return np.stack(out)
