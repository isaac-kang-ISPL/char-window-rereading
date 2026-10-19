"""Shared pieces of the manuscript experiments — output root, result writing, the inputs every table reads."""
from __future__ import annotations

import json
import os
from pathlib import Path

from tools.rescore import hypotheses as H
from tools.rescore import store
from tools.rescore.decide import batch

def out_root():
    """Output root — read at call time, so a driver can set LEGIT_RESCORE_OUT after the modules are imported."""
    return Path(os.environ.get('LEGIT_RESCORE_OUT', str(store.DATA))).expanduser()


def out(rel):
    """Output path of a result file (public relative path; `store.research_path` gives its folder on disk)."""
    return out_root() / store.research_path(rel)


def result(rel):
    """A result an experiment reads from an earlier experiment — the output root first, else the data root (not in a
    full `reproduce` run, which must read its own results: LEGIT_RESCORE_STRICT=1)."""
    p = out_root() / store.research_path(rel)
    if p.exists() or os.environ.get('LEGIT_RESCORE_STRICT') == '1':
        return p
    return store.at(rel)


def rel_out(p):
    """Path as stored inside result files — relative to the output root (or the data root)."""
    p = Path(p)
    for root in (out_root(), store.DATA):
        try:
            return str(p.relative_to(root))
        except ValueError:
            pass
    return str(p)


def dump(rel, obj, trailing=True, default=float):
    """Write a result the way the experiments always did: UTF-8 JSON, indent 1, a final newline."""
    p = out(rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, indent=1, default=default) + ('\n' if trailing else '')
    tmp = p.with_suffix(p.suffix + f'.tmp{os.getpid()}')
    tmp.write_text(text)
    os.replace(tmp, p)
    return p


def load(rel):
    return json.loads(result(rel).read_text())


_PLL = {}


def pll(model='byt5-xl', field='ci', strings_file='strings.json'):
    """ByT5 PLL table (kept for the process)."""
    key = (model, field, strings_file)
    if key not in _PLL:
        tab = store.pll_table(model, field, strings_file)
        if tab is None:
            raise FileNotFoundError(f'{model} PLL table: {store.N_SHARDS} shards and {strings_file} needed in '
                                    f'{store.rel(store.PLL / model)} (data root {store.DATA})')
        _PLL[key] = tab
    return _PLL[key]


def ar(split):
    """SVTRv2-AR items as a Batch (validation 'natdev' or evaluation 'full')."""
    return batch(H.get('svtrv2-ar', split))


def reader(m, split):
    return batch(H.get(m, split))


_WIN = {}


def window_delta(tag, b):
    """ΔW table of a SVTRv2-AR window tag against the greedy transcripts of batch b."""
    from tools.rescore.decide import window_delta as wd
    key = (tag, id(b))
    hit = _WIN.get(key)
    if hit is None or hit[0] is not b:
        hit = (b, wd(store.window_table(tag), b.items))
        _WIN[key] = hit
    return hit[1]


def workers():
    """Worker processes a stage may use (LEGIT_STAGE_WORKERS · reproduce sets cores / parallel stages)."""
    return max(1, int(os.environ.get('LEGIT_STAGE_WORKERS', '1')))


def pmap(fn, args, n=None, chunksize=1):
    """[fn(a) for a in args] in forked worker processes (order kept) — fn and its data are inherited by fork, so they need
    not be picklable; only the arguments and results travel. One worker (or one item) runs in this process."""
    args = list(args)
    n = min(n or workers(), len(args))
    if n <= 1:
        return [fn(a) for a in args]
    import multiprocessing as mp
    global _PMAP_FN
    _PMAP_FN = fn
    with mp.get_context('fork').Pool(n) as pool:
        return pool.map(_pmap_call, args, chunksize=chunksize)


_PMAP_FN = None


def _pmap_call(a):
    return _PMAP_FN(a)
