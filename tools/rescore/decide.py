"""The decision rule — S(y) over the candidates of each image, change a to the best y when max S > τ.

    S(y) = Δb(y) + α·ΔV(y) + β·Δlen(y) + γ·ΔPLL(y)              (`Batch.decide`, the rule of the paper)
    S(y) = Δb(y) − δ·ΔILM(y) + γ·ΔPLL(y) + β·Δlen(y)            (`Batch.decide_ilm`, comparison rows, not the method)

Δb is the recognizer's log P (`dlp`), ΔV the window score (`dw`; the window tables call it W). In code the
coefficients are named after the stored results: omega = α (window), beta = β (length), mu = γ (ByT5), tau = τ,
lam = δ (ILM). A term with coefficient 0 is left out of the sum. A candidate without a value for a term whose
coefficient is not 0 is not chosen. Ties go to the first candidate in beam order. Scores are summed term by term in
the order of `Batch.decide` (log P, ByT5, length, window) with IEEE doubles, so the arrays here give the same
decisions as a loop over the candidates.

Context term: ΔPLL(y) = 0 when a or y has fewer than two characters (`CONTEXT_MIN`) — a one-character string has no
visible context once its character is masked, so its ByT5 value is close to a character frequency. Such a candidate
needs no ByT5 value. `context_min=1` gives the rule without this exception (the stored values of earlier runs).

Items are kept as flat candidate arrays (`Batch`); score tables are attached once per batch and reused across the
coefficient grid. Reading clusters are not part of the method; `Batch.merged()` keeps them for the
comparison rows of earlier results: Δrec(y) becomes
    log Σ_{y' ≡ y} P(y') − log Σ_{y' ≡ a} P(y')        (≡ : same `cluster_key`, i.e. normalize_36_paper; a included)
and candidates in the cluster of a are dropped. A hypothesis whose normalized form is empty is its own cluster.
"""
from __future__ import annotations

import functools

import numpy as np

from tools._io import cluster_key, normalize_paper

NEG = -np.inf
CONTEXT_MIN = 2         # ΔPLL = 0 when a or y is shorter than this (1 = no exception)
norm = functools.lru_cache(maxsize=None)(normalize_paper)
ckey = functools.lru_cache(maxsize=None)(cluster_key)


def _merge(items):
    """merge_readings() and, for each kept candidate, its flat index in `items` (candidate order)."""
    red = np.logaddexp.reduce
    out, keep, off = [], [], 0
    for k, b, g, cands in items:
        kb = ckey(b)
        ks = [ckey(c[0]) for c in cands]
        mass = {kb: [0.]}
        for q, c in zip(ks, cands):
            v = mass.get(q)
            if v is None:
                mass[q] = [c[1]]
            else:
                v.append(c[1])
        M = {q: (float(v[0]) if len(v) == 1 else float(red(v))) for q, v in mass.items()}
        mb = M[kb]
        row = []
        for j, (q, c) in enumerate(zip(ks, cands)):
            if q != kb:
                row.append((c[0], M[q] - mb, c[2], c[3]))
                keep.append(off + j)
        out.append((k, b, g, row))
        off += len(cands)
    return out, np.array(keep, np.int64)


def merge_readings(items):
    """Reading clusters — see the module doc. Item order and the ΔW of each kept candidate are unchanged."""
    return _merge(items)[0]


def window_delta(V, items):
    """Absolute window scores {key: {y: W}} → {key: {y: W(y) − W(a)}} for y ≠ a (images whose a has a score)."""
    out = {}
    for k, b, _, _ in items:
        v = V.get(k)
        if v and b in v:
            out[k] = {y: x - v[b] for y, x in v.items() if y != b}
    return out


def with_window(items, Wd):
    """Items with ΔV from {key: {y: ΔV}} — a candidate without a value gets None (not chosen when α > 0)."""
    out = []
    for k, b, g, cands in items:
        w = (Wd or {}).get(k, {})
        out.append((k, b, g, [(y, dlp, dl, w.get(y)) for y, dlp, dl, _ in cands]))
    return out


def _vals(vs, m):
    """Values (None = missing) → (present, values with NaN for missing). A NaN value counts as missing too: in the
    loop form a NaN score never wins `s > best`, and a NaN for a makes every score of the image NaN."""
    v = np.array(vs, dtype=np.float64) if m else np.zeros(0)
    assert len(v) == m
    return ~np.isnan(v), v


class _Core:
    """Candidate arrays that do not depend on score tables or windows."""

    def __init__(self, items, parent=None, keep=None):
        n = len(items)
        lens = np.fromiter((len(it[3]) for it in items), np.int64, n)
        off = np.zeros(n + 1, np.int64)
        np.cumsum(lens, out=off[1:])
        m = int(off[-1])
        self.n, self.m = n, m
        self.seg = np.repeat(np.arange(n), lens)
        self.ne = np.flatnonzero(lens > 0)
        self.starts = off[self.ne]
        self.arange = np.arange(m)
        self.dlp = np.fromiter((x[1] for it in items for x in it[3]), np.float64, m)
        if parent is not None:                                         # a subset of the parent's candidates (reading clusters)
            P = parent
            self.ys_o = P.ys_o[keep]
            self.ys = self.ys_o.tolist()
            self.base, self.base_o, self.pg, self.okb = P.base, P.base_o, P.pg, P.okb
            self.dl, self.okc = P.dl[keep], P.okc[keep]
            self.lc, self.lb = P.lc[keep], P.lb
            self.dnames, self.did, self.dn = P.dnames, P.did, P.dn
            return
        self.ys = [x[0] for it in items for x in it[3]]
        self.base = [it[1] for it in items]
        self.ys_o = np.empty(m, object)
        self.ys_o[:] = self.ys
        self.base_o = np.empty(n, object)
        self.base_o[:] = self.base
        self.dl = np.fromiter((x[2] for it in items for x in it[3]), np.float64, m)
        self.lc = np.fromiter((len(y) for y in self.ys), np.int64, m)
        self.lb = np.fromiter((len(b) for b in self.base), np.int64, n)
        pg = [norm(g) for _, _, g, _ in items]
        self.pg = pg
        self.okb = np.fromiter((norm(b) == q for b, q in zip(self.base, pg)), bool, n)
        self.okc = np.fromiter((norm(y) == pg[s] for y, s in zip(self.ys, self.seg.tolist())), bool, m)
        ds = [k.rsplit('#', 1)[0] for k, *_ in items]
        self.dnames = list(dict.fromkeys(ds))                       # first-appearance order
        di = {d: i for i, d in enumerate(self.dnames)}
        self.did = np.fromiter((di[d] for d in ds), np.int64, n)
        self.dn = np.bincount(self.did, minlength=len(self.dnames))

    def context(self, pp, cm):
        """(ΔPLL with 0 where a or y is shorter than cm, valid) — valid: the candidate and a have ByT5 values, or the term is 0."""
        diff = pp.c - pp.b[self.seg]
        both = pp.c_ok & pp.b_ok[self.seg]
        if cm <= 1:
            return diff, both
        act = (self.lc >= cm) & (self.lb[self.seg] >= cm)
        return np.where(act, diff, 0.), ~act | both


class _Table:
    def __init__(self, core, tab):
        tab = tab or {}
        self.c_ok, self.c = _vals([tab.get(y) for y in core.ys], core.m)
        self.b_ok, self.b = _vals([tab.get(b) for b in core.base], core.n)


class Decision(list):
    """Outputs of one decision (strings in item order) with their correctness against the labels."""
    __slots__ = ('ok', 'core')


class Batch:
    """A list of items in candidate arrays — build once, decide many times.

    Three kinds share arrays: a plain batch (from items), a windowed batch (`with_window`: same candidates, ΔW from a
    table) and a cluster batch (`merged`: a subset of the parent's candidates with cluster Δrec). The item list of a
    windowed batch is built only when read."""

    def __init__(self, items, core=None, dw=None, root=None, W=None, parent=None, keep=None):
        self._items = items
        self.core = core if core is not None else _Core(items)
        if dw is None:
            dw = _vals([x[3] for it in items for x in it[3]], self.core.m)
        self.dw_ok, self.dw = dw
        self._root, self._W = root, W                                 # windowed: the batch it was made from, its ΔW table
        self._parent, self._keep = parent, keep                       # cluster batch: the plain batch and the kept candidates
        self._tabs = {}
        self._merged = None
        self._windowed = {}

    @property
    def items(self):
        if self._items is None:
            self._items = with_window(self._root.items, self._W)
        return self._items

    def __len__(self):
        return self.core.n

    def with_window(self, Wd):
        """Same candidates with ΔW from {key: {y: ΔW}} (replacing any ΔW of the items)."""
        root = self._root or self
        hit = root._windowed.get(id(Wd))
        if hit is not None and hit[0] is Wd:
            return hit[1]
        if root._parent is not None:                                  # cluster batch: the parent's ΔW of the kept candidates
            pw = root._parent.with_window(Wd)
            dw = (pw.dw_ok[root._keep], pw.dw[root._keep])
        else:
            c = root.core
            vals = []
            for k, b, g, cands in root.items:
                w = (Wd or {}).get(k, {})
                vals.extend(w.get(x[0]) for x in cands)
            dw = _vals(vals, c.m)
        b = Batch(None, core=root.core, dw=dw, root=root, W=Wd)
        root._windowed[id(Wd)] = (Wd, b)
        if len(root._windowed) > 8:
            root._windowed.pop(next(iter(root._windowed)))
        return b

    def merged(self):
        """Reading clusters (`merge_readings`); a windowed batch keeps its ΔW table."""
        if self._root is not None:
            return self._root.merged().with_window(self._W)
        if self._merged is None:
            its, keep = _merge(self.items)
            core = _Core(its, parent=self.core, keep=keep)
            self._merged = Batch(its, core=core, dw=(self.dw_ok[keep], self.dw[keep]), parent=self, keep=keep)
        return self._merged

    def table(self, tab):
        """Score table values of the candidates and of a — attached once per table."""
        if self._root is not None:
            return self._root.table(tab)
        hit = self._tabs.get(id(tab))
        if hit is None or hit[0] is not tab:
            if self._parent is not None:
                pt = self._parent.table(tab)
                t = _Table.__new__(_Table)
                t.c_ok, t.c, t.b_ok, t.b = pt.c_ok[self._keep], pt.c[self._keep], pt.b_ok, pt.b
            else:
                t = _Table(self.core, tab)
            hit = (tab, t)
            self._tabs[id(tab)] = hit
            if len(self._tabs) > 8:
                self._tabs.pop(next(iter(self._tabs)))
        return hit[1]

    def _pick(self, s, valid, tau):
        c = self.core
        bad = np.isnan(s)
        if valid is not None:
            bad |= ~valid
        s = np.where(bad, NEG, s)
        mx = np.full(c.n, NEG)
        idx = np.full(c.n, -1, np.int64)
        if len(c.ne):
            mx[c.ne] = np.maximum.reduceat(s, c.starts)
            pos = np.where(s == mx[c.seg], c.arange, c.m)
            idx[c.ne] = np.minimum.reduceat(pos, c.starts)
        choose = (mx > tau) & (mx > NEG)
        return self._decision(choose, idx)

    def _decision(self, choose, idx):
        c = self.core
        d = Decision(list(c.base) if not c.m else np.where(choose, c.ys_o[np.maximum(idx, 0)], c.base_o).tolist())
        d.ok = c.okb.copy() if not c.m else np.where(choose, c.okc[np.maximum(idx, 0)], c.okb)
        d.core = c
        return d

    def decide(self, lm, mu, beta, tau, omega=0., context_min=None):
        """S = Δb + (γ·ΔPLL if γ) + β·Δlen + (α·ΔV if α) — mu = γ, omega = α; ΔPLL with the short-string rule
        (`CONTEXT_MIN`)."""
        c = self.core
        valid = None
        if mu:
            ctx, valid = c.context(self.table(lm), CONTEXT_MIN if context_min is None else context_min)
            s = c.dlp + mu * ctx
        else:
            s = c.dlp + 0.
        s = s + beta * c.dl
        if omega:
            valid = self.dw_ok if valid is None else (valid & self.dw_ok)
            s = s + omega * self.dw
        return self._pick(s, valid, tau)

    def decide_ilm(self, lm, ilm, lam, mu, beta, tau, context_min=None):
        """S = Δb − δ·ΔILM + (γ·ΔPLL if γ) + β·Δlen (no window term; comparison rows). δ = 0 is `decide` without the
        window."""
        if not lam:
            return self.decide(lm, mu, beta, tau, context_min=context_min)
        c = self.core
        ip = self.table(ilm)
        valid = ip.c_ok & ip.b_ok[c.seg]
        s = c.dlp - lam * (ip.c - ip.b[c.seg])
        if mu:
            ctx, ok = c.context(self.table(lm), CONTEXT_MIN if context_min is None else context_min)
            valid = valid & ok
            s = s + mu * ctx
        else:
            s = s + 0.
        s = s + beta * c.dl
        return self._pick(s, valid, tau)

    def base(self):
        """The greedy transcripts — the row without re-reading."""
        return self._decision(np.zeros(self.core.n, bool), np.full(self.core.n, -1, np.int64))


_BATCH = {}


def batch(items):
    """Batch of an item list — the same list object gives the same Batch (a few kept)."""
    if isinstance(items, Batch):
        return items
    hit = _BATCH.get(id(items))
    if hit is not None and hit[0] is items:
        return hit[1]
    b = Batch(items)
    _BATCH[id(items)] = (items, b)
    if len(_BATCH) > 8:
        _BATCH.pop(next(iter(_BATCH)))
    return b
