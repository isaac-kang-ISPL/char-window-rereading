"""글줄 띠 TR · rbox · W 창 영역(TRER · rbox 를 못 만들면 TRW1.5).

입력은 원본 RGB · 조각 분할 lab(`segment.cut`) · 글자 위치 p_tl · 글줄 곡선 tl_4(`position.derive`) · attention 기록 rec
(키 text · maps · gw · gh · wh0 · rotated). 좌표는 모두 원본 그림의 px 이다.

    TR     = 글줄 곡선에서 cap-height ~ 기준선 사이 띠(잉크 확장 전 기본 띠). 이웃 조각 맞닿음 판정에 쓴다.
    rbox   = 글자마다 p_tl 중심 · tl_4 접선 방향 사각형. 전경(Sauvola k 0.2 · 극성 = 띠 테두리 log 비 + attention log 비)
             ∩ 넓힌 조각 S_i 에서 attention 몫 ≥ 0.2 인 덩어리의 외곽 → 떨어진 점 붙이기 → 여유.
    TRER   = TR ∪ (TR 과 겹치는 rbox). W 창의 픽셀 범위다.
    TRW1.5 = 글줄 중심 ± 1.5 x-height 띠. rbox 를 못 만든 장의 대체(한 글자 장은 글줄이 없어 그림 전체).

공개 함수: tr_lines · strip_mask · tr_strip · wide_band · current_boxes · trer_region · window_region.

글줄 곡선점 솎기(`_spaced_indices`)는 같은 산술의 C++ 반복(`curve_spacing.cpp`)으로 돈다 — 처음 부를 때 segment 의 커널과 같은
곳(`LEGIT_RESCORE_KERNEL_DIR`)에 g++ 로 빌드하고, 빌드나 적재가 안 되면 Python 반복으로 대신한다(같은 결과 · 느림 · `spacing_backend()`).
`LEGIT_REGION_BACKEND=python` 이면 처음부터 Python, `=cpp` 면 커널이 없을 때 멈춘다.
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

# 옛 경로가 이 계산을 불러 올 때 OpenCV 를 단일 스레드로 두었다(작은 연산이 많아 그쪽이 빠르다) — 같게 둔다.
cv2.setNumThreads(1)

# ---------------------------------------------------------------- 상수

CAP_OVER_XH = 1.333                 # cap-height ÷ x-height (합성 글꼴 여섯의 중앙값)
BORDER_UP, BORDER_DN = 2.5, 1.5     # 글자 영역 위 · 아래 경계 — 기준선에서 x-height 배수(위 +2.5 · 아래 −1.5)
EXPANDED_PAD = 0.5                  # 극성 판정용 확장 띠 — TR 을 법선 방향으로 x-height × 0.5 씩 넓힘
SAUVOLA_K, SAUVOLA_R = .2, 127.5    # Sauvola 문턱(창 = x-height 1배 홀수)
POL_FLOOR = 1e-3                    # 극성 점수의 전경률 하한
VREACH = .8                         # 찾는 창의 세로 반폭 = VREACH × max(결정 지도 반높이, 간격/2)
VSHIFT = .5                         # 찾는 창의 세로 중심을 글꼴 계량 (윗변+아랫변)/2 × 간격 × VSHIFT 만큼 옮김
TAU = .2                            # 덩어리 고르기 문턱 — attention 질량 ÷ 전경 위 질량 합
DIL = .05                           # 조각 S_i 를 넓히는 반지름 — p_tl 간격 × DIL
DOT_CHARS, DOT_REACH = 'ij!?;:', .5  # 떨어진 점을 붙일 글자 · 상자 위(! ? 는 아래) x-height × DOT_REACH 안
NO_CLIP = 99.                       # 가로 자르기 없음 — 찾는 창이 앞 · 뒤 p_tl 까지라 상자는 그 안에 머문다
GEO_KU, GEO_KV = 1.2, .55           # 덩어리를 못 고른 자리의 기하 상자 — 이웃 p_tl 중점 × 1.2 · 결정 지도 반높이 × 0.55
PAD_U, PAD_V, PAD_MIN_W = .10, .05, .5   # 상자 여유 — 가로 양옆 높이 × 0.10 · 세로 높이 × 0.05 · 폭 최소 높이 × 0.5
RBOX_MASS = .9                      # 결정 지도 분위 상자가 담는 질량(반높이 추정용)
TRW_HALF = 1.5                      # rbox 실패 장의 대체 띠 — 글줄 중심 ± 1.5 x-height
RASTER_SHIFT = 8                    # rbox 다각형 래스터 정밀도 1/256 px

RBOX_VERSION = 'rbox-si-k0.2-dot-logsumpol-v6'
PREPARATION = 'current-rbox-prepared-v1'

# 글꼴 계량(글자별 윗변 · 아랫변)을 재는 글꼴 10개 — fc-match 로 찾는다.
FONT_PATTERNS = ('DejaVu Sans:style=Book', 'DejaVu Sans:style=Bold',
                 'DejaVu Serif:style=Book', 'DejaVu Serif:style=Bold', 'DejaVu Sans Mono:style=Book',
                 'Liberation Sans:style=Regular', 'Liberation Sans:style=Bold',
                 'Liberation Serif:style=Regular', 'Liberation Serif:style=Bold',
                 'Liberation Sans Narrow:style=Regular')


# ---------------------------------------------------------------- 글줄 선 · 띠

SPACING_SRC = Path(__file__).with_name('curve_spacing.cpp')
_SPACING = None
_SPACING_BACKEND = None


def _spacing_kernel():
    """C++ 솎기 함수(처음 한 번 빌드 · 적재) — 못 쓰면 None 이고 그 까닭을 `_SPACING_BACKEND` 와 stderr 에 남긴다."""
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
    except Exception as e:                                         # 컴파일러 없음 · 빌드 실패 · 적재 실패
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
    """곡선점 솎기에 쓰는 구현 — {'name': 'cpp' | 'python', 'reason', 'library'}."""
    _spacing_kernel()
    return dict(_SPACING_BACKEND)


def _spaced_indices(points, threshold):
    """점열에서 앞서 남긴 점과의 거리가 threshold 이상인 점만 남긴 번호(첫 점 포함) — C++ 반복, 못 쓰면 `_spaced_indices_py`.
    경계 근처(|거리 − threshold| ≤ 1e-12·max(1, threshold))와 무한 거리는 numpy norm 으로 정하는 Python 규칙을 그대로 따른다."""
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
    """글줄 곡선 점열과 x-height s 에서 여섯 선(기준선 · 글줄 · x-height · cap-height · 위 · 아래 경계).

    미는 양은 기준선 기준 x-height 배수(위가 +)이고 글줄 법선 방향으로 민다. 법선은 읽기 방향을 따른다.
    """
    return {k: v.tolist() for k, v in _line_arrays(f_pts, s, up, down).items()}


def _line_arrays(f_pts, s, up=BORDER_UP, down=BORDER_DN, keys=None):
    """`_derive_lines` 의 선들을 float 배열 (n, 2) 로 — 값은 같다(소수 셋째 자리 반올림 · 짧은 점열은 반올림 없음).
    keys 를 주면 그 선만 계산한다(띠 둘은 위 · 아래 경계와 cap-height · 기준선만 쓴다)."""
    up_ = {'baseline': 0.0, 'textline': 0.5, 'xheight': 1.0,
           'capheight': CAP_OVER_XH,
           'bordup': up, 'borddn': -down}
    off = {k: 0.5 - v for k, v in up_.items() if keys is None or k in keys}   # 글줄 기준 · 화면 좌표
    P = np.asarray(f_pts, float)
    if len(P) < 2:
        return {k: np.asarray([[float(x), float(y + v * s)] for x, y in f_pts], float).reshape(-1, 2) for k, v in off.items()}
    # 점열을 max(s/4, 0.5px) 간격으로 솎은 뒤 접선을 잰다(촘촘한 점열의 반올림 지터 방지).
    step_ = max(float(s) * 0.25, 0.5)
    keep = _spaced_indices(P, step_)
    if keep[-1] != len(P) - 1:
        keep.append(len(P) - 1)
    P = P[keep]
    if len(P) < 2:
        P = np.asarray(f_pts, float)[[0, -1]]
    tg = np.gradient(P, axis=0)
    nrm = np.linalg.norm(tg, axis=1, keepdims=True)
    # 겹친 점(기울기 0)은 전체 방향으로 메운다.
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
    """tl_4 곡선 표본에서 여섯 선 — 크기는 geometry['size'](x-height)."""
    size = float(geometry.get('size', 0) or 0)
    return _derive_lines(line(geometry), size)


def strip_mask(lines, shape):
    """cap-height 선과 기준선 사이 띠를 이웃 삼각형 합집합으로 래스터(1/256 px) — 그림 밖은 잘림."""
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
    """기본 띠 TR(cap-height ~ 기준선) — 글줄 곡선이 없으면 그림 전체."""
    size = float(geometry.get('size', 0))
    if geometry.get('samples', 0) < 2 or size <= 0:
        return np.ones(shape, bool)
    return strip_mask(_line_arrays(line(geometry), size, keys=('capheight', 'baseline')), shape)


def wide_band(shape, geometry, half=TRW_HALF):
    """글줄 중심(기준선 +0.5) ± half x-height 띠 — 글줄 곡선이 없으면 그림 전체."""
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


# ---------------------------------------------------------------- 전경 · 극성

def _normalized_gray(rgb, reference_mask):
    """TR 안 명도 최소 · 최대로 그림 전체를 min-max 정규화한 회색조(띠가 비거나 평평하면 그대로)."""
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
    """Sauvola(창 = x-height 1배 홀수 · k 0.2) 전경 — ink_high 면 밝은 글자(255 − 명도)."""
    size = max(3, int(round(float(xheight) * 1)) | 1)
    source = 255 - gray if ink_high else gray
    threshold = threshold_sauvola(source, window_size=size, k=SAUVOLA_K, r=SAUVOLA_R)
    return source <= threshold


def _full_image_ink_high(dark, light, shape):
    """그림 테두리 전경이 적은 쪽이 글자(같으면 전체 전경이 적은 쪽 · 그다음 어두운 글자)."""
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
    """극성 판정용 확장 띠와 그 테두리(두 곡선 · 띠 안 그림 테두리)."""
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
    """확장 띠 테두리 전경이 적은 쪽이 글자(같으면 띠 전경 · 그다음 어두운 글자). 테두리가 없으면 그림 테두리 판정."""
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
    """극성 점수 band + attn(> 0 이면 어두운 글자).

    attention 마스크 = 자리별 지도를 자기 최대로 나눈 뒤 화소별 최대 → 0–255 Otsu.
    attn = ln(dark 안/밖) − ln(light 안/밖) · band = ln(light 띠 테두리 전경률) − ln(dark 띠 테두리 전경률).
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
    """장 하나의 전경과 조각 분할(모델 입력 방향) — 글줄이 없으면 None."""
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


# ---------------------------------------------------------------- 글꼴 계량

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
    """94자 × 글꼴 10개 평균 — x-height 단위로 (잉크 폭 · x-height 가운데 위 윗변 · 아랫변(위 +) · 가로 중심 어긋남).
    원고 회차 환경(openocr · PIL · 글꼴 파일 sha256 은 표에 기록)에서 잰 값을 굳힌 표 `data/font_metrics.json` 을 읽는다 —
    글꼴 · PIL 판이 다른 기계에서도 같은 값을 쓰게. 표를 다시 재려면 `measure_font_metrics()`."""
    if not _FONT:
        import json
        for c, v in json.loads(FONT_TABLE.read_text())['metrics'].items():
            _FONT[c] = np.asarray(v, dtype=np.float64)
    return _FONT


def measure_font_metrics():
    """font_metrics 표를 이 기계의 글꼴로 다시 잰다(fc-match · 200 px 렌더링) — 표를 새로 만들 때만."""
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


# ---------------------------------------------------------------- 자리 틀 · 찾는 창

def _axes(a):
    t = np.stack([np.cos(a), np.sin(a)], -1)
    return t, np.stack([-t[..., 1], t[..., 0]], -1)


def _proj(r, i, pts):
    t, n = _axes(r['angle'][i])
    d = np.asarray(pts, float) - r['ptl'][i]
    return d @ t, d @ n


def _neighbor_gaps(r, i):
    """접선 위 앞 · 뒤 p_tl 까지 거리(양수). 없으면 다른 쪽 · 둘 다 없으면 간격 중앙값."""
    n = len(r['ptl'])
    prev = -_proj(r, i, r['ptl'][i - 1][None])[0][0] if i > 0 else None
    nxt = _proj(r, i, r['ptl'][i + 1][None])[0][0] if i + 1 < n else None
    prev = None if prev is not None and prev <= 1e-6 else prev
    nxt = None if nxt is not None and nxt <= 1e-6 else nxt
    if prev is None and nxt is None:
        prev = nxt = r['spacing']
    return (prev if prev is not None else nxt), (nxt if nxt is not None else prev)


def _geometric(r, i):
    """기하 상자 — 가로 앞 · 뒤 p_tl 중점까지 × 1.2 · 세로 결정 지도 반높이 × 0.55."""
    gp, gn = _neighbor_gaps(r, i)
    hv = GEO_KV * r['radii'][i][1]
    return np.array([-GEO_KU * gp / 2, GEO_KU * gn / 2, -hv, hv])


def _margin(b):
    """빈틈없는 상자에 여유 — 가로 양옆 높이 × 0.10 · 세로 높이 × 0.05 · 폭이 높이 × 0.5 보다 좁으면 넓힘."""
    u0, u1, v0, v1 = b
    H = v1 - v0
    u0, u1, v0, v1 = u0 - PAD_U * H, u1 + PAD_U * H, v0 - PAD_V * H, v1 + PAD_V * H
    if u1 - u0 < PAD_MIN_W * H:
        c = (u0 + u1) / 2; u0, u1 = c - PAD_MIN_W * H / 2, c + PAD_MIN_W * H / 2
    return np.asarray([u0, u1, v0, v1], float)


def _search_window(r, i, vreach=VREACH, a=VSHIFT, reach=1.):
    """자리 i 의 찾는 창 — 가로 앞 · 뒤 p_tl 까지 · 세로 ±vreach·max(ry, 간격/2) · 중심은 글꼴 계량 × a 만큼 옮김.
    창 화소는 p_tl 간격 48 화소 밀도(최소 1 배). 결정 지도는 격자 → 화소 쌍선형."""
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
    """전경 덩어리 중 attention 질량 ÷ 전경 위 질량 합 ≥ tau(넓이 3 이상) — 고른 화소 mask 또는 None."""
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
    """자리 i 의 빈틈없는 상자 [u0, u1, v0, v1](자리 틀) — 못 고르면 None.
    AR 글자가 i · j · ! · ? · ; · : 이면 고른 상자의 가로 범위 안 · 상자 위(! ? 는 아래) 떨어진 점 덩어리를 더한다."""
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
    """가중 역누적분포(칸 질량 그대로 · 보간 없음)."""
    order = np.argsort(values, kind='stable')
    return float(values[order[min(int(np.searchsorted(np.cumsum(weights[order]), q, side='left')), len(order)-1)]])


def _decision_radii(mass, xy, wh, angle, cell_size, keep=RBOX_MASS):
    """결정 지도 칸을 접선 · 법선에 투영한 분위 상자(꼬리마다 (1−keep)/4 · 반쪽 칸 덧댐)의 반폭 · 반높이."""
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


# ---------------------------------------------------------------- rbox · 창 영역

def current_boxes(rgb, lab, points, geometry, rec):
    """글자마다 rbox 네 모서리(원본 px) — (모서리 목록, meta). 못 만들면 ValueError.

    rgb (H, W, 3) uint8 · lab 조각 분할(H, W) · points 글자 위치 p_tl K 개 · geometry tl_4 · rec attention 기록.
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
    """TRER = TR ∪ (TR 과 겹치는 rbox) — rbox 는 볼록 다각형 1/256 px 래스터. boxes 가 비었거나 목록이 아니면 None."""
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
    """rbox 모서리 목록(list of [[x, y]×4]) 또는 실패 사유 문자열 — 미리 계산 pickle 값과 같은 모양."""
    if rec is None:
        return 'missing_attention_record'
    try:
        boxes, _ = current_boxes(rgb, lab, points, geometry, rec)
        return [np.asarray(b, float).tolist() for b in boxes]
    except ImportError:                                # 모듈이 없으면 장마다 조용히 TRW1.5 로 떨어지지 않게 멈춘다
        raise
    except Exception as e:
        return f'{type(e).__name__}: {e}'


def window_region(rgb, lab, points, geometry, rec, *, boxes=None, fallback='trw15'):
    """W 창 영역 — (영역 bool (H, W), meta). TRER, rbox 를 못 만든 장은 TRW1.5(한 글자 장은 그림 전체).

    boxes 는 옛 미리 계산 pickle 값과 같은 뜻이다 — None 이면 current_boxes 로 그 자리에서 계산(예외면 실패) ·
    비어 있지 않은 list 면 그 rbox 로 TRER · 그 밖의 값(실패 사유 문자열 · 빈 list)은 계산 없이 rbox 실패로 TRW1.5.
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
