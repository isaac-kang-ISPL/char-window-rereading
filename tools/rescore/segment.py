"""자리 조각 분할 S_i — 자리별 attention 지도와 자리점으로 그림을 K 조각으로 가른다.

조합:
    지도   자리마다 최대 1 로 편 attention 지도(원본 해상도로 선형 보간)
    씨앗   자리점 원판(반지름 SEED_R · 자리 간격) · 겹친 화소는 지도 값이 큰 자리 · 자리점 화소는 제 번호로 못 박음
    가르기 `nearest`(기본) — 모든 화소를 유클리드 거리로 가장 가까운 씨앗 화소의 자리에 붙인다
           `cut`(원고 v1) — 지형 -(max_k H_k) 위 씨앗 제어 watershed + 거리 항(compactness COMPACT)
배경은 없다 — 모든 화소가 어느 자리엔가 속한다(라벨 0..K-1).

물채움은 scikit-image 0.26 의 compact watershed 를 그대로 옮긴 C++ 커널(`compact_watershed.cpp`)로 돈다 —
처음 부를 때 g++ 로 빌드해 `LEGIT_RESCORE_KERNEL_DIR`(기본 ~/.cache/legit_rescore/kernels)에 둔다. 빌드나 적재가
안 되면 scikit-image 의 `watershed` 로 대신하고(같은 라벨 · 느림) 그 사실을 stderr 에 한 번 남긴다 — 어느 쪽이
돌았는지는 `backend()`. `LEGIT_SEGMENT_BACKEND=skimage` 면 처음부터 scikit-image, `=cpp` 면 커널이 없을 때 멈춘다.

창 이웃 판정용 보조 함수: `label_at`(점이 떨어진 조각) · `touching`(4-이웃으로 맞닿은 조각) ·
`clipped_labels`(마스크 밖 화소를 -1 로 지운 라벨).

    lab = nearest(rgb, maps_px, points)      # (H,W) int16 · 값 0..K-1 · 씨앗이 없으면 None (기본)
    lab = cut(rgb, maps_px, points)          # 같은 모양 · watershed (원고 v1)
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

#: 원판 씨앗 반지름 — 자리 간격(`unit_w`)에 대한 비율.
SEED_R = 0.25
#: 물채움의 거리 항 무게(skimage watershed 의 compactness).
COMPACT = 1e-4
#: 잉크(Sobel 크기)의 가우시안 흐림 σ. 0 이면 안 흐린다.
INK_BLUR = 1.0

#: C++ 물채움 원본 · 빌드한 공유 라이브러리를 두는 곳 · 컴파일 옵션(-ffp-contract=off: 곱셈-덧셈을 합치지 않아 값이 원본과 같다).
KERNEL_SRC = Path(__file__).with_name('compact_watershed.cpp')
KERNEL_DIR = Path(os.environ.get('LEGIT_RESCORE_KERNEL_DIR', str(Path.home() / '.cache' / 'legit_rescore' / 'kernels')))
CXX_FLAGS = ('-O3', '-std=c++17', '-fPIC', '-shared', '-ffp-contract=off')
#: 씨앗을 큐에 넣는 값 — 'terrain'(scikit-image 0.26 · 그 자리의 지형값) | '-inf'(scikit-image 0.25).
SEED_COST = os.environ.get('LEGIT_SEGMENT_SEED', 'terrain')
_FLOOD = None
_BACKEND = None


def unit_w(cs, shape):
    """자리 간격(px) — 이웃 자리점 거리의 중앙값(하한 1). 점이 하나면 그림 긴 변 ÷ 자리 수."""
    d = [float(np.hypot(cs[i + 1][0] - cs[i][0], cs[i + 1][1] - cs[i][1]))
         for i in range(len(cs) - 1)]
    return max(float(np.median(d)) if d else max(shape[:2]) / max(len(cs), 1), 1.0)


def ink(rgb: np.ndarray) -> np.ndarray:
    """(H,W) 0~1 획 증거 — 회색조 Sobel 크기 · σ=INK_BLUR 흐림 · 99 백분위로 나눔."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    m = np.hypot(gx, gy)
    if INK_BLUR > 0:
        m = cv2.GaussianBlur(m, (0, 0), INK_BLUR)
    hi = float(np.percentile(m, 99.0))
    return np.clip(m / hi, 0, 1) if hi > 0 else np.zeros_like(m)


def _up(maps: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """지도 (K,h,w) -> (K,H,W) float32 · 크기가 다르면 선형 보간."""
    h, w = hw
    if maps.shape[1:] == (h, w):
        return maps.astype(np.float32)
    return np.stack([cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
                     for m in maps.astype(np.float32)])


def norm(maps: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """지도 (K,h,w) -> (K,H,W), 자리마다 최대 1 로 나누고 [0,1] 로 자른다(최대가 0 이하면 그대로)."""
    m = _up(np.asarray(maps, np.float32), hw)
    if not len(m):
        return m
    mx = m.reshape(len(m), -1).max(1)
    mx[mx <= 0] = 1.0
    return np.clip(m / mx[:, None, None], 0.0, 1.0)


def _disc_argmax_markers(m, cs, disc):
    """비정상 입력(지도에 NaN · 음수, 반지름 NaN)용 원판 씨앗 — 원판 전체를 그린 뒤 argmax 로 임자를 정한다."""
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
    """원판 씨앗 -> (H,W) int32 `0`=없음 · `1..K`.

    `m` 정규화 지도 (K,H,W) · `cs` 자리점 K 개(px) · `r` 반지름(px). 겹친 화소는 지도 값이
    큰 자리(같으면 앞 자리)가 갖고, 마지막에 자리점 화소를 제 번호로 못 박는다.
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
        take = claim & (values > score)  # 엄격 비교 — 값이 같으면 앞 자리
        score[take], dst[take] = values[take], k + 1
    for k, (x, y) in enumerate(cs):
        out[int(min(max(round(y), 0), H-1)), int(min(max(round(x), 0), W-1))] = k + 1
    return out


def relief(m: np.ndarray, ink_map: np.ndarray, beta: float = 0.0) -> np.ndarray:
    """watershed 지형 `-(max_k H_k + beta*ink)` — 글자 안이 골짜기, 글자 사이가 능선."""
    return -(m.max(0) + beta * np.clip(ink_map, 0, 1)) if len(m) else -ink_map * 0


def build(force: bool = False) -> Path:
    """C++ 커널을 빌드하고 공유 라이브러리 경로를 돌려준다 — 이름에 원본 sha256 이 들어가 원본이 바뀌면 새로 빌드한다.
    임시 이름으로 만든 뒤 옮기므로 여러 프로세스가 함께 빌드해도 반쯤 쓴 파일을 읽지 않는다. 컴파일러는 `CXX`(기본 g++)."""
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
    """커널이 옮긴 판(0.26)인가 — 0.25 는 씨앗을 -inf 로 넣어 대신 경로의 라벨이 다를 수 있다."""
    return _skimage_version().split('.')[:2] == ['0', '26']


def _skimage_seed_cost():
    """설치된 scikit-image 의 씨앗 규칙 — 0.26 'terrain' · 0.25 '-inf' · 그 밖은 확인 안 함(None)."""
    return {('0', '26'): 'terrain', ('0', '25'): '-inf'}.get(tuple(_skimage_version().split('.')[:2]))


def _kernel():
    """C++ 물채움 함수(처음 한 번 빌드 · 적재) — 못 쓰면 None 이고 그 까닭을 `_BACKEND` 와 stderr 에 남긴다."""
    global _FLOOD, _BACKEND
    if _BACKEND is not None:
        return _FLOOD
    want = os.environ.get('LEGIT_SEGMENT_BACKEND', 'auto')
    if want == 'skimage':
        _BACKEND = dict(name='skimage', reason='LEGIT_SEGMENT_BACKEND=skimage', library=None)
        return None
    try:
        lib = build()
        fn = ctypes.CDLL(str(lib)).compact_flood
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64,
                       ctypes.c_void_p, ctypes.c_int64, ctypes.c_double, ctypes.c_int, ctypes.c_void_p]
        fn.restype = ctypes.c_int
    except Exception as e:                                         # 컴파일러 없음 · 빌드 실패 · 적재 실패
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
    """물채움에 쓰는 구현 — {'name': 'cpp' | 'skimage', 'reason', 'library'}."""
    _kernel()
    return dict(_BACKEND)


def _flood(terrain: np.ndarray, mk: np.ndarray) -> np.ndarray:
    """`skimage.segmentation.watershed(terrain, markers=mk, connectivity=2, compactness=COMPACT)` 와 같은 라벨(int32).

    입력 변환 · 패딩 · 이웃 순서는 scikit-image 의 함수를 그대로 쓰고, 힙 물채움만 C++ 로 돈다. 거리는 원본처럼 패딩한
    지형의 strides 로 잰다(F 순서 지형이면 1차원 거리 — 회전 attention 지도에서 생기는 원본의 동작을 그대로 둔다).
    씨앗이 C 순서가 아니면 원본이 씨앗을 복사본에 채워 결과가 달라지므로 scikit-image 를 그대로 부른다."""
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
    """씨앗 `mk` 로 물채움+거리 watershed -> (H,W) int16 라벨 0..K-1. `maps` 는 여기서 한 번 더 편다."""
    m = norm(maps, ink_map.shape)
    if not len(m):
        return np.zeros(ink_map.shape, np.int16)
    lab = _flood(relief(m, ink_map, 0.0), mk)
    return (lab.astype(np.int16) - 1)


def cut(rgb, maps_px, points):
    """자리 조각 분할 -> `lab` (H,W) int16, 값이 자리 번호 0..K-1 · 씨앗이 없으면 None.

    `rgb` 원본 (H,W,3) uint8 · `maps_px` 자리별 attention 지도 (K,h0,w0)(원본 좌표계 · `position.px_maps`) ·
    `points` 자리점 K 개 [(x, y), ...](원본 px · p_tl). 여기서 Python float 로 바꾼다 — float32 배열을
    그대로 쓰면 자리 간격이 float32 로 계산돼 원판 반지름이 달라진다.
    """
    cs = [(float(x), float(y)) for x, y in points]
    hw = np.asarray(rgb).shape[:2]
    m = norm(maps_px, hw)
    mk = disc_markers(m, cs, SEED_R * unit_w(cs, m.shape[1:]))
    if mk.max() == 0:
        return None
    # `watershed(m, ink(rgb), mk)` 와 같은 값을 덜 계산한다 — 잉크는 무게 0 이라 모양만 쓰이고(0·잉크 = 0),
    # 편 지도를 다시 펴도 그대로다. 0 배열을 더하는 것은 남긴다: 지형의 배열 순서(C/F)가 거리 계산을 정한다.
    return (_flood(-(m.max(0) + np.zeros(hw, np.float32)), mk).astype(np.int16) - 1)


def nearest(rgb, maps_px, points):
    """자리 조각 분할(기본) -> `lab` (H,W) int16, 값이 자리 번호 0..K-1 · 씨앗이 없으면 None.

    씨앗은 `cut` 과 같은 원판(`disc_markers`)이고, 씨앗이 아닌 화소는 유클리드 거리 변환으로 가장 가까운 씨앗 화소의
    자리를 받는다(거리가 같으면 거리 변환이 고른 화소). attention 지도는 씨앗을 정할 때만 쓰인다 — 원판이 겹치지 않으면
    지도를 펴지 않는 같은 결과의 길은 `nearest_lazy`.
    """
    cs = [(float(x), float(y)) for x, y in points]
    hw = np.asarray(rgb).shape[:2]
    m = norm(maps_px, hw)
    return nearest_labels(disc_markers(m, cs, SEED_R * unit_w(cs, m.shape[1:])))


def nearest_labels(mk):
    """씨앗 (H,W) int32(0 = 없음 · 1..K) -> 가장 가까운 씨앗 화소의 자리 번호 (H,W) int16 · 씨앗이 없으면 None."""
    from scipy.ndimage import distance_transform_edt
    if not np.any(mk > 0):
        return None
    index = distance_transform_edt(mk == 0, return_distances=False, return_indices=True)
    return mk[tuple(index)].astype(np.int16) - 1


def disjoint_disc_markers(hw, cs, raw_maps, r):
    """`disc_markers` 와 같은 씨앗을 기하만으로 — 어느 두 원판(자리점 화소 포함)도 같은 화소를 갖지 않을 때만, 아니면 None.
    attention 은 겹친 화소의 임자를 정할 때만 쓰이므로(편 지도는 [0, 1] 이라 원판 화소를 모두 얻는다) 겹침이 없으면 지도 없이
    같은 씨앗이다. 지도에 NaN · 음수가 있거나 반지름이 유한하지 않으면 None(지금 계산으로)."""
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
    """`nearest(rgb, maps_px(), points)` 와 같은 라벨 — 자리 원판이 겹치지 않으면 attention 지도를 펴지 않는다.
    raw_maps: 격자 지도(기록의 maps) · maps_px: 원본 px 지도를 만드는 인자 없는 함수(겹칠 때만 부른다)."""
    cs = [(float(x), float(y)) for x, y in points]
    hw = np.asarray(rgb).shape[:2]
    r = SEED_R * unit_w(cs, hw)
    mk = disjoint_disc_markers(hw, cs, raw_maps, r)
    if mk is None:
        mk = disc_markers(norm(maps_px(), hw), cs, r)
    return nearest_labels(mk)


SEGMENTATIONS = {'nearest': nearest, 'watershed': cut}


def label_at(lab, bg0, cx, cy):
    """점 (cx, cy) 가 떨어진 조각 -> (라벨, 배경이었나).

    `bg0` 이 참이고 그 화소가 배경(0)이면 가장 가까운 0 아닌 조각으로 대신한다 — 없으면 (None, True).
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
    """조각 `lid` 에 4-이웃으로 직접 맞닿은 조각 라벨 집합(자기 제외 · `bg0` 이면 0 도 뺀다)."""
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
    """`mask` 밖 화소를 -1 로 지운 라벨(`int64` 면 int64 로 올린 뒤)."""
    return np.where(mask, lab.astype(np.int64) if int64 else lab, -1)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Build and check the C++ watershed kernel')
    ap.add_argument('--build', action='store_true', help='rebuild even when the source is unchanged')
    a = ap.parse_args()
    if a.build:
        build(force=True)
    print(backend())
