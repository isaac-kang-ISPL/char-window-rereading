"""글자 창 — AR 의 attention 기록 한 장에서 자리마다 W-AR 창 그림을 만든다.

흐름(자리 i 마다 · 기본 = 새 정의 `DEFINITION`): 글자 위치 p_tl · 글줄 곡선 tl_4(`position` · 개선 p_tl `REFINED_POLICY`)
→ 자리 조각 S(`segment.nearest` · 가장 가까운 씨앗) → 창 영역(`region` · TRW = 글줄 ± 1.5 x-height 띠)
→ 자기 조각 + 이웃 조각(이웃은 기본 띠 TR 안에서 4-이웃으로 맞닿은 조각 · 기본 한 단계 = W3) ∩ 영역의 외곽 사각형
→ AR 입력 크기(비율 1–4 · 세로 회전 없음)로 늘림. 창은 돌리지 않는다.
창 안 자리 범위는 「창 그림 안에 p_tl 이 든 자리」다(점수 쪽 `window_score` 가 그 범위의 부분 문자열을 묻는다).

원고 v1(`PAPER_V1`): 저장 p_tl(옛 정책) · watershed 분할(`segment.cut`) · 영역 TRER(rbox 를 못 만들면 TRW1.5) ·
읽기 접선이 45° 이상 기울었거나 거꾸로면 창을 바로 세움 — `prepare(..., **PAPER_V1)` · `inputs(..., ptl='original', segmentation='watershed')`.

The window images and slot ranges of `PAPER_V1` are bit-identical to those of the research code that produced the
stored v1 scores (TRER region, TR neighbour band, no fill); the comparison lives in the research repository.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

from tools.rescore import position, region as regions, segment

MAX_LEN = 25            # 인식기(svtrv2-ar)의 최대 출력 길이 — 더 긴 전사는 창을 만들지 않는다
PAD_PX = 2              # 창 사각형 가장자리 여유(px)
MIN_WIN_PX = 3          # 회전 뒤 1px mask 가 사라졌을 때 살리는 최소 변(px)
ROT_MIN_DEG = 45.0      # 읽기 접선이 이보다 덜 기울면 돌리지 않는다
HOPS = 1                # 이웃 단계 — 0 = W1(자기 조각만) · 1 = W3(기본) · 2 = W5 · 3 = W7 · 4 = W9
REGIONS = ('trer', 'trw15', 'none')   # 창 영역 — TRER(원고 v1 · rbox 실패 시 TRW1.5) · TRW(기본 · 1.5배 띠만) · 영역 없음
PTL = {'refined': position.REFINED_POLICY, 'original': None}   # p_tl 정책 — 개선(기본) · 옛 계산(원고 v1)
# Window definition — the default (refined p_tl · nearest seed · TRW band · no rotation) and v1
DEFINITION = dict(region='trw15', rotate=False, ptl='refined', segmentation='nearest')
PAPER_V1 = dict(region='trer', rotate=True, ptl='original', segmentation='watershed')
# AR 입력 — 공개 svtrv2-ar 평가 설정(비율을 반올림해 [1, 4] · 그 칸 크기로 패딩 없이 늘림). W 창에는 세로 회전(RotateTall)을 걸지 않는다.
AR_BASE_SHAPE = ((64, 64), (96, 48), (112, 40), (128, 32))   # 비율 1 · 2 · 3 · 4 의 (폭, 높이)
AR_MAX_RATIO = 4
AR_BASE_H = 32
BACKGROUND_POLICY = 'full_image_border_otsu_median_v1'

# 원고 · 저장 점수의 판 이름(`stroke_glyph_train_c2w.METHOD_VERSION`)을 이루는 조각 — 이름 대조용(계산에는 안 쓰인다)
_ASSIGNMENT_VERSION = 'w-ctc-greedy-v3-pos0.5-sim1-tau0-nofill'
_AGGREGATION_VERSION = 'all-w-mean-v1'
_R_POLICY = {'v1': 'w-r-inside-fill-v1', 'v2': 'w-r-inside-fill-v2'}


def method_version(region='trer', hops=HOPS, fallback='trw15', neighbour='tr', fill=False, r_policy='v2', *,
                   ptl='original', segmentation='watershed', rotate=True, boundary=True, aggregation='sum'):
    """연구 경로 판 이름과 같은 문자열 — 원고 표(wconstruct.json 등)의 `method_version` 칸.
    region 은 연구 경로의 LEGIT_W_REGION 값('trer' · 'r'/'tre' · 'band_wide15' · 'none' 등)을 그대로 받는다.
    키워드 인자 기본값은 원고 v1 이라 옛 이름은 그대로다 — 새 정의는 그 칸마다 꼬리(-ptlr · -near · -norot · -core · -cov)가 붙는다:
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


# ---------------------------------------------------------------- 배경색 · 회전 · 자르기

def background_rgb(rgb):
    """그림 한 장의 배경색 — 전체 명도 Otsu 의 두 무리 중 테두리에 많은 쪽의 RGB 중앙값(명도 범위 < 12 면 전체 중앙값).
    회전 여백 · 빈 창 자리표시에 쓴다(창마다 따로 추정하지 않는다)."""
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
    """배경색은 회전 · 빈 창에서만 필요하다 — 처음 쓸 때 한 번 계산."""
    def __init__(self, rgb):
        self.rgb, self.value = rgb, None

    def get(self):
        if self.value is None:
            self.value = background_rgb(self.rgb)
        return self.value

    def __iter__(self):
        return iter(self.get())


def deploy_theta(theta):
    """창 회전각 — 읽기 방향이 왼쪽을 향하면(역방향) 그대로 돌려 세우고, 아니면 |θ| ≥ 45° 일 때만 돌린다."""
    if math.cos(theta) < 0:
        return theta
    return theta if abs(math.degrees(theta)) >= ROT_MIN_DEG else 0.0


def mask_aabb(mask, pad=PAD_PX):
    """마스크의 여유 포함 외곽 사각형 [x0, y0, x1, y1] — 빈 마스크면 None."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return [max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
            min(w, int(xs.max()) + 1 + pad), min(h, int(ys.max()) + 1 + pad)]


def rotate_crop(rgb, mask, theta, background, pad=PAD_PX):
    """mask 를 읽기 접선 θ 만큼 반대로 돌려 수평 창으로 자른다 → (PIL 그림, 창 좌표 → 원본 좌표 2×3 행렬 펼친 6개) · 빈 마스크면 (None, None)."""
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
        # 회전한 mask 가 비었다 — INTER_NEAREST 가 1px 획을 떨어뜨린 경우. 회전 전 무게중심을 같은 행렬로 옮겨 최소 변 상자를 만든다.
        ys, xs = np.nonzero(mask)
        if not len(xs):
            return None, None
        rx, ry = mat @ np.asarray([float(xs.mean()), float(ys.mean()), 1.0])
        half = max(MIN_WIN_PX, 2 * pad + 1) / 2.0
        x0 = int(max(0, min(ow - 1, round(rx - half))))
        y0 = int(max(0, min(oh - 1, round(ry - half))))
        box = [x0, y0, int(max(x0 + 1, min(ow, round(rx + half)))), int(max(y0 + 1, min(oh, round(ry + half))))]
    image = Image.fromarray(np.ascontiguousarray(out[box[1]:box[3], box[0]:box[2]]))
    inv = cv2.invertAffineTransform(mat)                       # 회전 화폭 → 원본
    inv[:, 2] += inv[:, :2] @ np.asarray(box[:2], float)       # 창 → 화폭 원점
    return image, [round(float(v), 9) for v in inv.reshape(-1)]


def window_mask(lab, bg0, point, region, contact, hops=HOPS):
    """자리 점이 떨어진 조각(자기 조각) + 맞닿음 그래프에서 hops 단계까지의 이웃 조각, 모두 ∩ 영역.
    맞닿음은 contact(기본 띠 TR) 안에서만 본다. 자기 조각이 영역 안에 없으면 빈 마스크."""
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
    """창 그림 → AR 입력 배열 (H, W, 3) uint8 — 비율 round(w/h) 를 [1, 4] 로 자르고 그 칸 크기로 BICUBIC."""
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


# ---------------------------------------------------------------- 장 하나

def inputs(rgb, rec, *, ptl='refined', segmentation='nearest'):
    """attention 기록 → ((p_tl 점 float32 (K, 2), tl_4 곡선) | (None, None), 조각 lab | None) — 그 자리에서 계산.
    ptl: 'refined'(기본 · `position.REFINED_POLICY`) | 'original'(원고 v1) · segmentation: 'nearest'(기본) | 'watershed'(원고 v1).
    조각 분할은 원본 크기로 되돌린 자리별 attention 지도와 p_tl 점에서 한다.
    tl_4 스플라인 계수는 numpy · BLAS 판에 따라 1e-12 쯤 달라질 수 있고, 그 차이가 분할 경계 몇 px · rbox 를 움직일 수 있다
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
        if segmentation == 'nearest':                 # 원판이 겹치지 않으면 attention 을 펴지 않는다(같은 라벨)
            lab = segment.nearest_lazy(np.asarray(rgb), rec['maps'], maps, cs)
        else:
            lab = segment.SEGMENTATIONS[segmentation](np.asarray(rgb), maps(), cs)
    return (points, geometry) if ok else (None, None), lab


def prepare(rgb, rec, *, boxes=None, region='trw15', hops=HOPS, fallback='trw15', geometry_inputs=None, bg0=False,
            rotate=False, ptl='refined', segmentation='nearest'):
    """그림 한 장(원본 RGB)과 그 장의 AR attention 기록(text · maps · gw · gh · wh0 · rotated) → 창 목록.

    반환 dict(text, windows=[자리마다 dict(image=AR 입력 배열 | None, valid, slot_ids, target_slot, theta, source_affine, skip_reason)],
    region_meta) — 창을 못 만드는 장은 dict(skip=이유)(기록 없음 · 빈 전사 · 점과 글자 수 불일치 · 전사가 MAX_LEN 보다 김 · 분할 실패).
    boxes: 미리 계산한 rbox 모서리 목록(None 이면 그 자리에서 계산 · 목록이 아닌 값은 rbox 실패로 보아 TRW1.5) ·
    geometry_inputs: 미리 계산한 ((점, 곡선), lab) — 원고 재현(`stored_inputs`) · 대조용 · bg0: 분할에 배경 조각 0 이 있나(저장 분할의 표식).
    rotate: 읽기 접선대로 창을 바로 세우나(원고 v1 · 기본은 돌리지 않으며 접선도 계산하지 않는다) ·
    ptl · segmentation: geometry_inputs 가 없을 때 `inputs` 에 넘긴다. 원고 v1 은 `prepare(..., **PAPER_V1)`."""
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
    # p_tl 은 attention 격자를 옮긴 추정값이라 화폭 바로 밖(−0.1 px 등)에 찍힐 수 있다 — 화폭 안으로 들인다.
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
    angles = position.angles(geometry, points) if rotate else None    # 회전하지 않으면 접선이 쓰이지 않는다
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


# ---------------------------------------------------------------- 저장된 attention 기록 읽기(원고 재현용)

def charout_root(env=True):
    """Root of the stored attention records (<root>/<dataset>/svtrv2-ar.lmdb) — `STR_CHAROUT_ROOT` (when env) >
    `local_paths.json` > `data/charout` in the repository (`tools._io.local_path`). Read again on every call."""
    from tools._io import local_path
    return Path(local_path('charout_root', env=env))


CHAROUT_ROOT = charout_root()
READER = 'svtrv2-ar'
FORCED = 'svtrv2-ar_afix'       # 저장 전사가 빔 greedy a 와 다른 장 — a 로 교사 강제해 만든 기록(있으면 그 장은 이것)
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
    """저장 형식(npz bytes) → dict(text · conf · gw · gh · wh0 · rotated · maps)."""
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
    """저장된 svtrv2-ar attention 기록 한 장 — forced=True 면 교사 강제 기록(svtrv2-ar_afix)이 있는 장은 그것을 읽는다. 없으면 None.
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


# 원고 회차가 읽은 저장 기하 · 분할의 정의 판 — 저장 행의 표식이 이것과 같을 때만 쓴다(연구 경로 charout.geometry · seg 와 같은 검사).
STORED_DEFINITION = dict(map='last-decoder-layer-head-mean-v1', ptl='ptl-44cc92b89f9756cf', tl4='tl4-92cc8a768f03c02b',
                         seg='seg-1c5bf82d4a2373d5')


def stored_inputs(dataset, index, *, root=None):
    """원고 회차가 읽은 저장 p_tl · tl_4 · 조각 분할(svtrv2-ar 저장소의 파생 행) → ((점, 곡선) | None, (lab, bg0) | None).
    행 표식이 STORED_DEFINITION 과 다르거나 없으면 그 칸 None — 부르는 쪽이 `inputs` 로 그 칸만 다시 잰다(연구 경로와 같음)."""
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
    """원고 회차와 같은 입력 — 보통 기록은 저장 기하 · 분할을 읽고 없는 칸만 그 자리에서, 교사 강제 기록은 모두 그 자리에서.
    → (geometry_inputs, bg0) — `prepare(..., geometry_inputs=…, bg0=…)` 에 넘긴다."""
    pair = cut = None
    if rec is not None and rec.get('source') != FORCED:
        pair, cut = stored_inputs(dataset, index, root=root)
    if pair is None or cut is None:
        fresh_pair, fresh_lab = inputs(rgb, rec, ptl='original', segmentation='watershed')
        pair = pair if pair is not None else fresh_pair
        cut = cut if cut is not None else (fresh_lab, False)
    return (pair, cut[0]), cut[1]


# ---------------------------------------------------------------- attention 기록 만들기(저장본이 없을 때)

AR_ROTATE_TALL = 1.5    # 통짜 그림 입력의 RotateTall — 세로 > 1.5 × 가로면 반시계 90°(공개 svtrv2-ar 평가 설정)
ENCODER_DOWN_H = 8      # 인코더 특징 격자의 세로 축소 배율 — 격자 높이 = 입력 높이 / 8


def whole_input(rgb):
    """통짜 그림 → (AR 입력 배열, 입력 (폭, 높이), 원본 (폭, 높이) · 돌린 뒤, 돌렸나) — 공개 svtrv2-ar 평가 입력과 같다."""
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
    """그림 한 장 + 전사 text(보통 greedy a) → attention 기록 dict(text · maps · gw · gh · wh0 · rotated).

    인식기의 greedy 디코딩 루프를 그대로 돌되 다음 토큰만 text(+ ⟨eos⟩)로 강제하고, 단계마다 마지막 디코더 층 cross-attention 의
    마지막 질의를 head 평균한다(저장 덤프와 같은 지도 정의). text 가 greedy 출력이면 greedy 순전파 기록과 같다.
    지도는 자리마다 합 1(소수 여섯째 자리 반올림) · 저장 형식처럼 float16 을 거친 float32. 사전 밖 글자 · 25자 초과면 None."""
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
            if t + 1 < len(target):                       # 마지막(⟨eos⟩) 단계 뒤는 쓰지 않는다 — 25자 전사면 seq 끝을 넘는다
                seq[:, t + 1] = target[t]
        grid = torch.stack(maps, 0).float().cpu().numpy()
    gh = max(1, height // ENCODER_DOWN_H)
    gw = max(1, n_tok // gh)
    if gw * gh != n_tok:
        return None
    raw = [np.asarray(m, dtype=np.float32) for m in grid[:len(target) - 1]]          # ⟨eos⟩ 단계는 빼고 글자 단계만
    raw = [np.round(a / s, 6) for a in raw if (s := float(np.asarray(a, dtype=np.float64).sum())) > 0]
    stored = np.asarray(raw, dtype=np.float16) if raw else np.zeros((0, gw * gh), np.float16)
    return dict(text=text, maps=stored.astype(np.float32).reshape(len(raw), -1), gw=int(gw), gh=int(gh),
                wh0=tuple(int(v) for v in wh0), rotated=bool(rotated), source='computed')
