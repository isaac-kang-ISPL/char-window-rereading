"""Accuracy and paired comparisons — word accuracy with normalize_36_paper (ASCII letters and digits, lower case).

`accuracy` gives the number of correct images and the macro averages over the 6 CB sets and the 7 UB sets (General
included) — the averages the manuscript reports. `pair` counts the images one output list fixes and breaks against
another, with a two-sided sign test; `groups` splits the same counts into CB, the six harder UB sets, and General.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

from tools.rescore.decide import Batch, Decision, norm
from tools.rescore.store import CB, UB, UB_HARD

SIG = '분명'                     # verdict strings of the stored results: p < .05 / not
NOT_SIG = '구분 안 됨'
GROUPS = (('CB 6셋', lambda d: d.startswith('test/')), ('어려운 UB 6셋', lambda d: d in UB_HARD),
          ('general', lambda d: d == 'u14m/general'))


class _Info:
    __slots__ = ('n', 'pg', 'dnames', 'did', 'dn')

    def __init__(self, items):
        if isinstance(items, Batch):
            c = items.core
            self.n, self.pg, self.dnames, self.did, self.dn = c.n, c.pg, c.dnames, c.did, c.dn
            return
        self.n = len(items)
        self.pg = [norm(g) for _, _, g, _ in items]
        ds = [k.rsplit('#', 1)[0] for k, *_ in items]
        self.dnames = list(dict.fromkeys(ds))
        di = {d: i for i, d in enumerate(self.dnames)}
        self.did = np.fromiter((di[d] for d in ds), np.int64, self.n)
        self.dn = np.bincount(self.did, minlength=len(self.dnames))


_INFO = OrderedDict()


def info(items):
    hit = _INFO.get(id(items))
    if hit is not None and hit[0] is items:
        return hit[1]
    v = _Info(items)
    _INFO[id(items)] = (items, v)
    while len(_INFO) > 32:
        _INFO.popitem(last=False)
    return v


def correct(items, pred):
    """Boolean array: output i reads like label i."""
    inf = info(items)
    if isinstance(pred, Decision) and len(pred) == inf.n and (pred.core.pg is inf.pg or pred.core.pg == inf.pg):
        return pred.ok
    return np.fromiter((norm(x) == q for x, q in zip(pred, inf.pg)), bool, inf.n)


def accuracy(items, pred):
    """dict(correct, n, cb, ub, per) — per-set accuracy in %, cb / ub = mean over the sets present (None if none)."""
    inf = info(items)
    ok = correct(items, pred)
    c = np.bincount(inf.did, weights=ok, minlength=len(inf.dnames)) if inf.n else np.zeros(0)
    cnt = [int(x) for x in c.tolist()]
    tot = [int(x) for x in inf.dn.tolist()]
    a = {d: 100 * cnt[i] / tot[i] for i, d in enumerate(inf.dnames)}
    cb = [a[k] for k in CB if k in a]
    ub = [a[k] for k in UB if k in a]
    return dict(correct=sum(cnt), n=sum(tot), cb=(sum(cb) / len(cb) if cb else None), ub=(sum(ub) / len(ub) if ub else None), per=a)


def _sign_sum(n, k):
    c = s = 1
    for i in range(1, k + 1):
        c = c * (n - i + 1) // i
        s += c
    return s


_SIGN = {}


def sign_p(a, b):
    """Two-sided sign test: min(1, 2·Σ_{i ≤ min(a, b)} C(a+b, i) / 2^(a+b))."""
    key = (a, b)
    if key in _SIGN:
        return _SIGN[key]
    n = a + b
    if not n:
        p = 1.
    else:
        p = min(1., 2 * _sign_sum(n, min(a, b)) / 2 ** n)
    _SIGN[key] = p
    return p


def verdict(p):
    return SIG if p < .05 else NOT_SIG


def pair(items, p0, p1):
    """p1 against p0 on the same images — fixed (p1 right, p0 wrong), broken, net, sign-test p, verdict."""
    a0, a1 = correct(items, p0), correct(items, p1)
    a = int(np.count_nonzero(a1 & ~a0))
    b = int(np.count_nonzero(a0 & ~a1))
    p = sign_p(a, b)
    return dict(fixed=a, broken=b, net=a - b, p=p, verdict=verdict(p))


def _dmask(inf, sel):
    dm = np.fromiter((bool(sel(d)) for d in inf.dnames), bool, len(inf.dnames))
    return dm[inf.did] if inf.n else np.zeros(0, bool)


def groups(items, p1, p0):
    """pair() split into CB 6 sets, the six harder UB sets, and General — {group: dict(win, lose, verdict)}."""
    inf = info(items)
    a0, a1 = correct(items, p0), correct(items, p1)
    out = {}
    for nm, sel in GROUPS:
        m = _dmask(inf, sel)
        w = int(np.count_nonzero(a1 & ~a0 & m))
        l = int(np.count_nonzero(a0 & ~a1 & m))
        out[nm] = dict(win=w, lose=l, verdict=verdict(sign_p(w, l)))
    return out


def set_rows(items, pred, sets):
    """{set: dict(n, correct, acc)} plus '전체' (all images) — the tables of the new benchmarks."""
    ok = correct(items, pred)
    ds = np.array([k.rsplit('#', 1)[0] for k, *_ in items])
    out = {d: dict(n=int((ds == d).sum()), correct=int(ok[ds == d].sum()), acc=float(ok[ds == d].mean() * 100)) for d in sets}
    out['전체'] = dict(n=len(items), correct=int(ok.sum()), acc=float(ok.mean() * 100))
    return out
