"""tools.rescore — the array decision rule equals the loop over candidates; normalize_36_paper; reading clusters;
the C++ segmentation flood equals scikit-image."""
from __future__ import annotations

import math
import random
import string

import numpy as np

from tools._io import cluster_key, normalize_paper
from tools.rescore.decide import Batch, merge_readings
from tools.rescore.metrics import accuracy, pair, sign_p


def _loop_decide(items, pll, mu, beta, tau, omega=0., cm=2):
    """The rule as a loop — the reference the arrays must match. ΔPLL = 0 when a or y is shorter than cm (cm ≤ 1: no
    exception, a without a PLL value is never changed when α > 0)."""
    pred = []
    for k, base, g, cands in items:
        p0 = pll.get(base)
        best, by = -math.inf, None
        for y, dlp, dl, dw in cands:
            p = pll.get(y)
            active = cm <= 1 or (len(y) >= cm and len(base) >= cm)
            if mu and active and (p is None or p0 is None):
                continue
            if omega and dw is None:
                continue
            s = dlp + (mu * (p - p0 if active else 0.) if mu else 0.) + beta * dl + (omega * dw if omega else 0.)
            if s > best:
                best, by = s, y
        pred.append(by if by is not None and best > tau else base)
    return pred


def _items(n=400, seed=0):
    rnd = random.Random(seed)
    words = ['CAT', 'cat', 'Cat!', 'COAT', 'CART', '', '!!', 'DOG', 'D0G', 'BOOK', 'B00K', 'book.']
    items, pll = [], {}
    for i in range(n):
        a = rnd.choice(words)
        cands = []
        for y in rnd.sample(words, rnd.randint(0, 6)):
            if y == a:
                continue
            dw = None if rnd.random() < .2 else rnd.uniform(-3, 3)
            cands.append((y, rnd.uniform(-8, 0), len(y) - len(a), dw))
        items.append((f"{rnd.choice(['test/SVT', 'u14m/curve', 'u14m/general'])}#{i}", a, rnd.choice(words), cands))
    for w in words:
        if rnd.random() < .85:
            pll[w] = rnd.uniform(-20, -2)
    return items, pll


def test_decide_matches_loop():
    items, pll = _items()
    b = Batch(items)
    for cm in (1, 2):
        for mu in (0., .05, .4):
            for beta in (0., .5):
                for tau in (0., .25):
                    for om in (0., .1):
                        got = list(b.decide(pll, mu, beta, tau, om, context_min=cm))
                        assert got == _loop_decide(items, pll, mu, beta, tau, om, cm)
    assert list(b.decide(pll, .4, .5, 0., .1)) == _loop_decide(items, pll, .4, .5, 0., .1, 2)   # default: two characters


def test_short_strings_have_no_context_term():
    """a or y of one character: ΔPLL = 0 even without ByT5 values; two characters on both sides: the term counts."""
    items = [('test/SVT#1', 'A', 'AB', [('AB', -1., 1, None)]),          # a short: y wins on Δrec alone
             ('test/SVT#2', 'AB', 'A', [('A', -1., -1, None)]),          # y short, no PLL value: still a candidate
             ('test/SVT#3', 'AB', 'AC', [('AC', -1., 0, None)])]         # both long: PLL decides
    pll = {'AB': -1., 'AC': -30.}
    b = Batch(items)
    assert list(b.decide(pll, 1., 0., -5.)) == ['AB', 'A', 'AB']
    assert list(b.decide(pll, 1., 0., -5., context_min=1)) == ['A', 'AB', 'AB']


def test_merged_batch_matches_merged_items():
    items, pll = _items(seed=1)
    m1 = Batch(items).merged()
    m2 = Batch(merge_readings(items))
    for mu, beta, tau, om in ((0., 0., 0., 0.), (.05, .5, 0., .1), (.2, 1., .25, .4)):
        assert list(m1.decide(pll, mu, beta, tau, om)) == list(m2.decide(pll, mu, beta, tau, om))
        assert list(m1.decide(pll, mu, beta, tau, om)) == _loop_decide(merge_readings(items), pll, mu, beta, tau, om)


def test_empty_normalized_form_is_its_own_cluster():
    assert cluster_key('!!') != cluster_key('..')
    items = [('test/SVT#1', '!!', 'x', [('..', -1., 0, None), ('!!!', -2., 1, None)])]
    (_, _, _, c), = merge_readings(items)
    assert [y for y, *_ in c] == ['..', '!!!']                      # nothing merged with a or with each other


def test_normalize_paper_keeps_ascii_alnum_only():
    def ref(t):
        keep = string.digits + string.ascii_letters
        return ''.join(ch for ch in t if ch in keep).lower()
    for code in list(range(0, 0x3000)) + [0x1D7CE, 0xFF21, 0x0660]:
        ch = chr(code)
        assert normalize_paper(ch) == ref(ch)
    assert normalize_paper('Télég-ABC 12!') == 'tlgabc12'


def test_accuracy_and_pair():
    items, pll = _items(seed=2)
    b = Batch(items)
    d = b.decide(pll, .05, .5, 0., .1)
    a = accuracy(b, d)
    ok = np.array([normalize_paper(y) == normalize_paper(g) for y, (_, _, g, _) in zip(d, items)])
    assert a['correct'] == int(ok.sum()) and a['n'] == len(items)
    r = pair(b, b.base(), d)
    assert r['net'] == r['fixed'] - r['broken'] and r['p'] == sign_p(r['fixed'], r['broken'])
    assert sign_p(0, 0) == 1. and abs(sign_p(10, 0) - 2 / 2 ** 10) < 1e-15


def _old_cut(S, rgb, maps, points):
    """segment.cut before the C++ flood — ink, maps normalised twice, scikit-image watershed."""
    cs = [(float(x), float(y)) for x, y in points]
    ik = S.ink(rgb)
    m = S.norm(maps, ik.shape)
    mk = S.disc_markers(m, cs, S.SEED_R * S.unit_w(cs, m.shape[1:]))
    if mk.max() == 0:
        return None
    return S._ws(S.relief(S.norm(m, ik.shape), ik, 0.0), markers=mk, connectivity=2, compactness=S.COMPACT).astype(np.int16) - 1


def _segment_native(monkeypatch):
    """segment with the seed rule of the installed scikit-image (0.26 'terrain', 0.25 '-inf') — the fallback must match."""
    import pytest
    from tools.rescore import segment as S
    rule = S._skimage_seed_cost()
    if rule is None:
        pytest.skip(f'the kernel ports scikit-image 0.25 / 0.26 (installed {S._skimage_version()})')
    monkeypatch.setattr(S, 'SEED_COST', rule)
    return S


def test_cpp_flood_equals_skimage(monkeypatch):
    """C/F-order terrain, plateaus, seed blobs, thin images, five compactness values — label for label."""
    S = _segment_native(monkeypatch)
    assert S.backend()['name'] == 'cpp'
    rng = np.random.default_rng(7)
    for case in range(60):
        h, w = (int(v) for v in rng.integers(1, 40, size=2))
        terrain = (rng.random((h, w)) if case % 3 == 0 else rng.integers(0, 4, (h, w)) / 3 if case % 3 == 1
                   else np.zeros((h, w)))
        mk = np.zeros((h, w), np.int32)
        for k in range(int(rng.integers(1, 12))):
            y, x = int(rng.integers(h)), int(rng.integers(w))
            mk[max(0, y - case % 3):y + 1 + case % 3, max(0, x - case % 3):x + 1 + case % 3] = k + 1
        if case % 4 == 0:
            terrain, mk = terrain.T, np.ascontiguousarray(mk.T)
        for c in (1e-8, 1e-4, .1, 1., 10.):
            monkeypatch.setattr(S, 'COMPACT', c)
            assert np.array_equal(S._flood(terrain, mk), S._ws(terrain, markers=mk, connectivity=2, compactness=c)), (case, c)


def test_cut_equals_old_cut_and_fallback(monkeypatch):
    """cut (C++) = cut before the change = cut with the scikit-image fallback, on random maps incl. F-order (rotated) maps."""
    S = _segment_native(monkeypatch)
    rng = np.random.default_rng(11)
    for case in range(24):
        h, w = (int(v) for v in rng.integers(8, 48, size=2))
        k = int(rng.integers(1, 7))
        maps = rng.random((k, max(2, h // 3), max(2, w // 3))).astype(np.float32)
        if case % 3 == 0:                  # maps of rotated images at image size arrive F-ordered: the terrain stays F-ordered
            maps = np.ascontiguousarray(rng.random((k, w, h)).astype(np.float32)).transpose(0, 2, 1)
        rgb = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        pts = [(float(rng.uniform(0, w)), float(rng.uniform(0, h))) for _ in range(k)]
        want = _old_cut(S, rgb, maps, pts)
        assert np.array_equal(S.cut(rgb, maps, pts), want)
        keep = S._FLOOD, S._BACKEND
        S._FLOOD, S._BACKEND = None, dict(name='skimage', reason='test', library=None)
        try:
            assert np.array_equal(S.cut(rgb, maps, pts), want)
        finally:
            S._FLOOD, S._BACKEND = keep



def test_midpoint_box_equals_pixel_mask():
    """MW crop box from the scanline intervals = bounding box (+ margin) of the pixel mask R ∩ {lo ≤ <q, t> ≤ hi},
    with pixel centres q, on random regions, directions (incl. axis-parallel and almost vertical) and limits."""
    from tools.rescore import midpoint as M
    rng = np.random.default_rng(3)
    for case in range(300):
        h, w = (int(v) for v in rng.integers(1, 40, size=2))
        area = rng.random((h, w)) < (.2 if case % 2 else .9)
        if case % 5 == 0:
            area[:] = False
        theta = (0., math.pi / 2, math.pi / 2 - 1e-9, math.pi)[case % 4] if case % 3 == 0 else float(rng.uniform(-math.pi, math.pi))
        u = np.array([math.cos(theta), math.sin(theta)])
        yy, xx = np.mgrid[0:h, 0:w]
        proj = (xx + .5) * u[0] + (yy + .5) * u[1]
        lo, hi = sorted(rng.uniform(proj.min() - 3, proj.max() + 3, size=2))
        if case % 7 == 0:
            lo = -math.inf
        if case % 11 == 0:
            hi = math.inf
        mask = area & (proj >= lo) & (proj <= hi)
        got = M.BandBounds(area).box(u, lo, hi, pad=2)
        if not mask.any():
            assert got is None, case
            continue
        ys, xs = np.nonzero(mask)
        want = [max(0, xs.min() - 2), max(0, ys.min() - 2), min(w, xs.max() + 1 + 2), min(h, ys.max() + 1 + 2)]
        assert got == [int(v) for v in want], (case, got, want)


def _synthetic_record(rng, n, width, height, slope):
    """An attention record of a recognizer input of `width` × `height` (no rotation) with one Gaussian map per character
    along a straight line of the given slope (grid = input / 4, as SVTRv2-AR)."""
    gw, gh = width // 4, height // 4
    yy, xx = np.mgrid[0:gh, 0:gw]
    maps = []
    for i in range(n):
        cx = (i + .5) * gw / n + rng.normal(0, .3)
        cy = gh / 2 + slope * (cx - gw / 2) + rng.normal(0, .2)
        m = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * .8 ** 2))
        maps.append((m / m.sum()).ravel().astype(np.float32))
    text = ''.join(rng.choice(list(string.ascii_uppercase), size=n))
    return dict(maps=np.asarray(maps), gw=gw, gh=gh, wh0=(width, height), rotated=False, text=text)


def _same_windows(core, want):
    assert core.keys() == want.keys()
    if 'windows' not in want:
        assert core == want
        return 0
    assert core['points'] == want['points'] and core['region_meta'] == want['region_meta']
    n = 0
    for a, b in zip(core['windows'], want['windows'], strict=True):
        assert {k: v for k, v in a.items() if k != 'image'} == {k: v for k, v in b.items() if k != 'image'}
        assert (a['image'] is None) == (b['image'] is None)
        if a['image'] is not None:
            assert np.array_equal(a['image'], b['image'])
            n += 1
    return n


def test_midpoint_windows_match_research_module():
    """Core MW windows (`pipeline.prepare_windows`, default MW3) = the research module that produced the stored MW tables
    (skipped in the release tree, which does not ship it)."""
    import pytest
    research = pytest.importorskip('tools.experiments.window_studies.midpoint_windows')
    from tools.rescore import midpoint as M
    from tools.rescore.pipeline import WINDOW, prepare_windows
    assert WINDOW == 'mw3'
    rng = np.random.default_rng(5)
    compared = 0
    for case in range(40):
        n = int(rng.integers(1, 9))
        width, height = (128, 32) if case % 2 else (96, 48)
        rec = _synthetic_record(rng, n, width, height, float(rng.uniform(-.3, .3)))
        rgb = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        for kind, nn in (('mw1', 1), ('mw3', 3), ('mw5', 5)):
            compared += _same_windows(prepare_windows(rgb, rec, kind), research.make_windows(rgb, rec, [nn])[nn])
        _same_windows(prepare_windows(rgb, rec), research.prepare(rgb, rec))
        _same_windows(M.prepare(rgb, rec), research.prepare(rgb, rec))
    assert compared > 50
