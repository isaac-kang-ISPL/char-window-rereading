"""The candidate rule — compare the greedy transcript a with the top K hypotheses of the beam.

An item is one image: (key, a, label, [(y, Δrec, Δlen, ΔW)]) where
    Δrec = log P_rec(y) − log P_rec(a)   (recognizer score of the beam, a's score from the greedy record)
    Δlen = len(y) − len(a)
    ΔW   = window score of y − window score of a, or None until a window table is attached (`decide.with_window`).
Candidates are the hypotheses of beam rank < K other than a; hypotheses that read like a (same normalize_36_paper form)
and the empty hypothesis stay. An image whose a has no recognizer score stops the run (`no_baseline`), except where the
recognizer cannot score a (a longer than its maximum output length, 25).
"""
from __future__ import annotations

import os
import sys

from tools.rescore import store

K = int(os.environ.get('PAPER_K', '10'))
_SRC = __import__('hashlib').sha1(open(__file__, 'rb').read()).hexdigest()[:12]
MAX_LEN = 25
READERS = ('svtrv2-ar', 'parseq', 'trba', 'svtrv2', 'mdiff4str')     # order of the recognizer tables
REFINE = ('parseq', 'mdiff4str')                                     # recognizers with refinement (cloze) scores


def no_baseline(name, keys, allowed=()):
    bad = sorted(set(keys) - set(allowed))
    if bad:
        raise RuntimeError(f'{name}: {len(bad):,} images without a score of the greedy transcript (e.g. {bad[:3]})')
    if allowed:
        print(f'{name}: {len(set(allowed)):,} images whose usual output is longer than {MAX_LEN} keep it', flush=True)


def greedy_score(c):
    """log P of a in the SVTRv2-AR beam record — 'ar_lp' when the record carries it, else the entry of a in the list."""
    if 'ar_lp' in c:
        return c['ar_lp']
    x = next((x for x in c['cands'] if x['y'] == c['ar']), None)
    return None if x is None else x['ar']


def ar_items(C, gt, k=None):
    """SVTRv2-AR — a = greedy transcript, candidates = beam rank < K other than a."""
    k = K if k is None else k
    out, miss = [], []
    for key, c in C.items():
        if key not in gt:
            continue
        a, s0 = c['ar'], greedy_score(c)
        cands = []
        if s0 is None:
            miss.append(key)
        else:
            for x in c['cands']:
                if x['nb'] >= k or x['y'] == a:
                    continue
                cands.append((x['y'], x['ar'] - s0, len(x['y']) - len(a), None))
        out.append((key, a, gt[key], cands))
    no_baseline('svtrv2-ar', miss)
    return out


def ctc_items(B, gt, k=None):
    """SVTRv2 (CTC) — a = greedy path, candidates = prefix beam top K other than a, Δrec = prefix log P difference."""
    k = K if k is None else k
    out, miss = [], []
    for key, b in B.items():
        if key not in gt:
            continue
        g = b['greedy']
        lp = {t: s for t, s in b['beam']}
        s0 = b['greedy_lp'] if 'greedy_lp' in b else lp.get(g)
        cands = []
        if s0 is None:
            miss.append(key)
        else:
            for t, s in b['beam'][:k]:
                if t != g:
                    cands.append((t, s - s0, len(t) - len(g), None))
        out.append((key, g, gt[key], cands))
    no_baseline('svtrv2', miss)
    return out


def model_items(D, gt, m, lam=0., k=None):
    """PARSeq, TRBA, MDiff4STR — a = usual output. With refinement scores (PARSeq, MDiff4STR) the recognizer score is
    (1 − λ)·left-to-right log P + λ·cloze log P; otherwise the teacher-forced log P of a and the beam scores."""
    k = K if k is None else k
    out, miss, long_ = [], [], []
    for key, v in D.items():
        if key not in gt:
            continue
        u, beam = v['usual'], v['beam']
        if 'cands' in v:
            sc = {t: None if (a is None or c is None) else (1 - lam) * a + lam * c for t, a, c, _, _ in v['cands']}
            s0 = sc.get(u)
            if s0 is None:
                (long_ if len(u) > MAX_LEN else miss).append(key)
            cands = [] if s0 is None else [(t, sc[t] - s0, len(t) - len(u), None) for t, a, c, src, rk in v['cands']
                                            if t != u and 0 <= rk < k and sc[t] is not None]
            out.append((key, u, gt[key], cands))
            continue
        s0 = v['usual_lp']
        if s0 is None:
            (long_ if len(u) > MAX_LEN else miss).append(key)
        cands = [] if s0 is None else [(t, s - s0, len(t) - len(u), None) for t, s in beam[:k] if t != u]
        out.append((key, u, gt[key], cands))
    no_baseline(m, miss + long_, allowed=long_)
    return out


def items(m, split, k=None):
    """Items of recognizer m on a split ('natdev', 'full', 'newsets') — kept in the disk cache by their input files."""
    k = K if k is None else k
    env = [os.environ.get('CTC_GREEDY_FIX', '1')]
    if split == 'natdev' and store.natdev_v2():               # redefined validation set: built from the merged lists, not cached
        return _build(m, split, k)
    return store.cached(f'items_{m}_{split}_k{k}', store.beam_files(m, split), [m, split, k, env, _SRC],
                        lambda: _build(m, split, k))


def _build(m, split, k):
    gt = store.labels(split)
    if m == 'svtrv2-ar':
        return ar_items(store.ar_beams(split), gt, k)
    if m == 'svtrv2':
        return ctc_items(store.ctc_beams(split), gt, k)
    lam = store.cand_lambda().get(m, 0.) if m in REFINE else 0.
    return model_items(store.other_beams(m, split), gt, m, lam, k)


_ITEMS = {}


def get(m, split, k=None):
    """items() kept for the process — the experiments read the same lists many times."""
    key = (m, split, K if k is None else k)
    if key not in _ITEMS:
        _ITEMS[key] = items(m, split, k)
    return _ITEMS[key]


def strings(item_lists):
    """Every a and candidate of the lists — the strings a score table must cover."""
    s = set()
    for its in item_lists:
        for _, a, _, c in its:
            s.add(a)
            s.update(y for y, *_ in c)
    return s


if __name__ == '__main__':
    for m in READERS:
        for sp in ('natdev', 'full'):
            it = get(m, sp)
            print(m, sp, len(it), sum(len(c) for *_, c in it), file=sys.stderr)
