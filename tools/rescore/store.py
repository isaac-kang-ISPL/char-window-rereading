"""Stored inputs of the re-reading experiments — beam lists of each recognizer and the score tables.

Every input lives under one data root (`CWR_DATA`, default the repository root) at a public relative path
(`beams/…`, `windows/…`, `pll/…`, `benchmarks/…`, `reference/…`) — the layout of the published data archive as unpacked.
`research_path` maps a public path to the folder that holds it on disk: the same path in a release, the research
folder name in the research tree (prefix pairs in `research_layout.json` at the repository root, which a release does
not ship). The GPU stages of `cli` write the same layout, so a table computed here from a fresh run and one computed
from the stored run read alike.

Tables that take seconds to assemble (ByT5 PLL over 16 shards, window scores over hundreds of chunk files) are kept
in a content-keyed disk cache (`cached`): the key covers every file read (path, size, mtime, inode, ctime), the
arguments, the Python and numpy versions, and the source of this module, so a changed input or rule never returns an
old table. `CWR_RESCORE_CACHE=0` turns the cache off.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import pickle
import platform
import re
import sys
from pathlib import Path

import numpy as np

from tools._io import cache_dir, env

REPO = Path(__file__).resolve().parents[2]
DATA = Path(env('CWR_DATA', str(REPO))).expanduser()
API = DATA / 'materials' / 'api'
TMP = DATA / 'materials' / 'tmp'
CACHE = Path(env('CWR_RESCORE_CACHE_DIR', cache_dir()))
USE_CACHE = env('CWR_RESCORE_CACHE', '1') != '0'

# benchmark sets — order of the macro averages
CB = ('test/IIIT5k', 'test/SVT', 'test/IC13_857', 'test/IC15_1811', 'test/SVTP', 'test/CUTE80')
UB = ('u14m/curve', 'u14m/multi_oriented', 'u14m/artistic', 'u14m/contextless', 'u14m/salient', 'u14m/multi_words',
      'u14m/general')
UB_HARD = ('u14m/artistic', 'u14m/contextless', 'u14m/curve', 'u14m/multi_oriented', 'u14m/multi_words', 'u14m/salient')
NEW_SETS = ('wordart_test', 'OST/weak', 'OST/heavy')



def _layout():
    """[(public prefix, research prefix)] of research_layout.json — empty in a release (the file is not shipped).
    A pair whose public side ends in '/' maps a folder and everything under it, any other pair one file."""
    f = REPO / 'research_layout.json'
    return [tuple(pair) for pair in json.loads(f.read_text())['prefixes']] if f.is_file() else []


LAYOUT = _layout()


def research_path(public):
    """Relative path on disk of a public relative path (the first matching pair wins; unmapped paths stay as given)."""
    s = str(public)
    for pub, res in LAYOUT:
        if pub.endswith('/'):
            if s == pub[:-1]:
                return res[:-1]
            if s.startswith(pub):
                return res + s[len(pub):]
        elif s == pub:
            return res
    return s


def at(public):
    """Data-root path of a public relative path."""
    return DATA / research_path(public)


def public_path(stored):
    """Public relative path of a relative path on disk — the inverse of `research_path` (unmapped paths stay as given)."""
    s = str(stored)
    for pub, res in LAYOUT:
        if pub.endswith('/'):
            if s == res[:-1]:
                return pub[:-1]
            if s.startswith(res):
                return pub + s[len(res):]
        elif s == res:
            return pub
    return s


# where each stored input is (public path under the data root) — one place, so a release can list exactly what it reads
BEAMS = at('beams/svtrv2-ar')                                   # SVTRv2-AR beam lists
FULL_ROWS = at('benchmarks/cb-ub/rows.json.gz')                 # CB + UB images and their labels
CTC = at('beams/svtrv2-ctc')                                    # SVTRv2 (CTC) prefix beam lists
CTC_FIX = at('beams/svtrv2-ctc-greedy')                         # CTC lists recounted where the greedy path left the beam
OTHER = at('beams/parseq-trba')                                 # PARSeq, TRBA beam lists
MDIFF = at('beams/mdiff4str')                                   # MDiff4STR beam lists
CAND_LAMBDA = at('beams/candidate-lambda.json')                 # weight of the refinement log P (PARSeq, MDiff4STR)
PLL = at('pll/byt5')                                            # ByT5 PLL per string (16 shards, one run per model)
ILM = at('ilm/svtrv2-ar')                                       # internal LM log P per string
WINDOWS = at('windows/svtrv2-ar')                               # SVTRv2-AR window scores per image and hypothesis
OWN_WINDOWS = at('windows/other-recognizers')                   # windows of the other recognizers, read by themselves
VENN = at('benchmarks/test-sample/beams.json')                  # stratified 10,000-image test sample of CB + UB
NEWSETS = at('beams/new-benchmarks')                            # WordArt, OST weak / heavy
N_SHARDS = 16
READER_DIR = {'parseq': 'parseq', 'trba': 'trba', 'mdiff4str': 'mdiff', 'svtrv2': 'svtrv2'}


def rel(p):
    """Path relative to the data root (the form stored inside result files)."""
    p = Path(p)
    try:
        return str(p.relative_to(DATA))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------- content-keyed cache

def _sig(paths):
    out = []
    for p in sorted({str(p) for p in paths}):
        try:
            s = os.stat(p)
            out.append((p, s.st_size, s.st_mtime_ns, s.st_ino, s.st_dev, s.st_ctime_ns))
        except FileNotFoundError:
            out.append((p, None))
    return out


def _source(fn):
    import inspect
    try:
        return hashlib.sha1(inspect.getsource(fn).encode()).hexdigest()[:12]
    except (OSError, TypeError):
        return None


def cached(name, files, params, build, source=None):
    """build() once per content of `files` and `params` — pickled under CACHE. The key includes the source of the loader
    that builds the value (`source`, default the caller), so editing a loader rebuilds only its own tables."""
    if not USE_CACHE:
        return build()
    if source is None:                                          # the calling loader (a module-level function)
        caller = sys._getframe(1)
        source = caller.f_globals.get(caller.f_code.co_name)
    key = hashlib.sha1(json.dumps([name, _sig(files), params, sys.version, np.__version__, platform.machine(),
                                   _source(source) if source is not None else None, _source(_string_table)],
                                  default=str).encode()).hexdigest()[:20]
    f = CACHE / f'{name}_{key}.pkl'
    value = _load(f)
    if value is not None:
        return value[0]
    import fcntl
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(CACHE / f'{name}_{key}.lock', 'w') as lk:          # parallel stages: one builds, the others wait and read
        fcntl.flock(lk, fcntl.LOCK_EX)
        value = _load(f)
        if value is not None:
            return value[0]
        value = build()
        tmp = f.with_suffix(f'.tmp{os.getpid()}')
        with open(tmp, 'wb') as fh:
            pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, f)
    _prune(name)
    return value


def _load(f):
    """(value,) from a cache file, or None when it is missing or broken."""
    if not f.exists():
        return None
    try:
        with open(f, 'rb') as fh:
            return (pickle.load(fh),)
    except Exception:                                           # broken cache file — rebuild
        return None


KEEP = 2                                                        # versions kept per table (e.g. two environments)


def _prune(name):
    """Drop older versions of a table — same name, another key (changed inputs or loader). The newest KEEP stay."""
    olds = sorted(CACHE.glob(f'{name}_' + '?' * 20 + '.pkl'), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in olds[KEEP:]:
        for q in (p, p.with_suffix('.lock')):
            try:
                q.unlink()
            except OSError:
                pass


def _json(path):
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as fh:
            return json.load(fh)
    return json.loads(path.read_text())


def _need(files, what, where):
    """Stop when a stored input is missing — an empty table would silently turn every decision into a (no window score,
    no beam list)."""
    if not files:
        raise FileNotFoundError(f'{what}: no input files in {where} (data root {DATA}; set CWR_DATA)')
    return files


def _merge_json_dir(d, glob='*.json', skip='_limit'):
    out = {}
    for f in sorted(Path(d).glob(glob)):
        if skip and skip in f.name:
            continue
        out.update(json.loads(f.read_text()))
    return out


def _complete(d):
    """A score folder is complete when its end marks _done_s<i>of<n> cover every shard i of some n."""
    d = Path(d)
    marks = [re.match(r'_done_s(\d+)of(\d+)$', f.name) for f in d.glob('_done_s*')] if d.exists() else []
    ns = {int(x.group(2)) for x in marks if x}
    return any(len({int(x.group(1)) for x in marks if x and int(x.group(2)) == n}) == n for n in ns)


# ---------------------------------------------------------------- the validation set (natdev10k_v2 switch)

NATDEV_V2_DEFAULT = Path(__file__).resolve().parents[2] / 'materials/api/natdev_replace27/natdev10k_v2_keys.json'
NATDEV_V2_PUBLIC = 'benchmarks/validation-set.json'             # the same key list in the published data archive


def natdev_v2():
    """The validation set 'natdev' (default natdev10k_v2, user decision 2026-10-22: natdev10k without the 27 images identical
    to UB General plus ranks 50,001-50,027 of the same order), or None for the original natdev10k. CWR_NATDEV_V2 = a JSON
    file with 'keys' (the validation images, in order) and 'extra_split' (the split name under which the stores hold the images
    that natdev10k lacks, in the natdev layout); CWR_NATDEV_V2=0 selects the original natdev10k. Without the research tree's
    key list (a release) the key list of the data root is read (NATDEV_V2_PUBLIC), and without either the original split."""
    default = NATDEV_V2_DEFAULT if NATDEV_V2_DEFAULT.exists() else at(NATDEV_V2_PUBLIC)
    f = env('CWR_NATDEV_V2', str(default) if default.exists() else '0')
    if f in ('', '0', 'natdev10k'):
        return None
    d = json.loads(Path(f).read_text())
    return dict(keys=tuple(d['keys']), added=frozenset(d.get('added', ())), extra=d['extra_split'],
                name=d.get('name', Path(f).stem))


def _v2(split, base, extra, sparse=False):
    """natdev under CWR_NATDEV_V2: natdev and the extra split merged, restricted to the listed keys in their order.
    A listed image without input stops the run — a selection never runs on fewer images than the definition. sparse: a store
    that lacks some natdev10k images by construction (own windows: 2 images of TRBA and of SVTRv2-CTC); there only an added
    image without input stops the run, and a natdev10k image keeps the absence it had."""
    v = natdev_v2() if split == 'natdev' else None
    if v is None:
        return base
    d = dict(base)
    d.update(extra(v['extra']))
    miss = [k for k in v['keys'] if k not in d and (not sparse or k in v['added'])]
    if miss:
        raise SystemExit(f"validation set {v['name']}: {len(miss)} validation images without input in this store "
                         f"(split {v['extra']} missing or incomplete), e.g. {miss[:3]}; CWR_NATDEV_V2=0 reads the original natdev10k")
    return {k: d[k] for k in v['keys'] if k in d}


# ---------------------------------------------------------------- images and labels

def labels(split):
    """{key: label} — 'full' = CB + UB (414,482 images) · 'natdev' = the validation set · 'newsets' = WordArt, OST."""
    if split == 'full':
        return {r['key']: r['gt'] for r in _json(FULL_ROWS)}
    if split == 'natdev':
        return _v2(split, {k: v['gt'] for k, v in _json(BEAMS / 'natdev_cands.json').items()},
                   lambda x: {k: v['gt'] for k, v in _json(BEAMS / f'{x}_cands.json').items()})
    if split == 'newsets':
        return {k: c['gt'] for k, c in _json(NEWSETS / 'cands.json').items()}
    raise ValueError(split)


def venn_keys():
    """Keys of the stratified 10,000-image test sample (venn10k)."""
    return {r['key'] for r in _json(VENN)['rows']}


# ---------------------------------------------------------------- beam lists

def ar_beams(split):
    """SVTRv2-AR beam lists — {key: {'ar': greedy a, 'gt', 'ar_lp'?: log P of a, 'cands': [{'y', 'ar': log P, 'nb': rank}]}}."""
    f = {'natdev': BEAMS / 'natdev_cands.json', 'full': BEAMS / 'arnb_cands_full.json.gz',
         'newsets': NEWSETS / 'cands.json'}[split]
    return _v2(split, cached(f'ar_beams_{split}', [f], [], lambda: _json(f)), lambda x: _json(BEAMS / f'{x}_cands.json'))


def ctc_beams(split, fix=True):
    """SVTRv2 (CTC) prefix-beam lists — {key: {'greedy', 'beam': [[t, log P], …], 'greedy_lp'?}}. Images whose greedy path
    left the beam top 10 are read from CTC_FIX, where a and the top 10 were recounted with the exact CTC forward."""
    d = CTC / ('beams' if split == 'full' else f'beams_{split}')
    fx = CTC_FIX / f'ctc_{split}.json'
    use_fix = fix and fx.exists() and os.environ.get('CTC_GREEDY_FIX', '1') != '0'
    files = _need(sorted(d.glob('*.json')), f'SVTRv2-CTC beam lists ({split})', d) + ([fx] if use_fix else [])

    def build():
        B = _merge_json_dir(d, skip=None)
        if use_fix:
            for k, v in _json(fx).items():
                if k in B:
                    B[k] = dict(greedy=v['greedy'], beam=v['beam'])
        return B
    return _v2(split, cached(f'ctc_beams_{split}', files, [use_fix], build), lambda x: ctc_beams(x, fix))


def other_beams(m, split):
    """PARSeq / TRBA / MDiff4STR beam lists (width 20, 20 expansions) — {key: {'usual', 'usual_lp'?, 'beam', 'cands'?}}."""
    if m == 'mdiff4str':
        d = MDIFF / split
    else:
        d = OTHER / ((m if split == 'full' else f'{m}_{split}') + '_b20t20')
    files = _need(sorted(d.glob('*.json')), f'{m} beam lists ({split})', d)
    return _v2(split, cached(f'beams_{m}_{split}', files, [], lambda: _merge_json_dir(d, skip=None)), lambda x: other_beams(m, x))


def beam_files(m, split):
    """Files a recognizer's beam lists and labels are read from — the key of the item cache."""
    lab = {'full': [FULL_ROWS], 'natdev': [BEAMS / 'natdev_cands.json'], 'newsets': [NEWSETS / 'cands.json']}[split]
    if m == 'svtrv2-ar':
        return lab + [{'natdev': BEAMS / 'natdev_cands.json', 'full': BEAMS / 'arnb_cands_full.json.gz',
                       'newsets': NEWSETS / 'cands.json'}[split]]
    if m == 'svtrv2':
        d = CTC / ('beams' if split == 'full' else f'beams_{split}')
        return lab + sorted(d.glob('*.json')) + [CTC_FIX / f'ctc_{split}.json']
    d = MDIFF / split if m == 'mdiff4str' else OTHER / ((m if split == 'full' else f'{m}_{split}') + '_b20t20')
    return lab + sorted(d.glob('*.json')) + [CAND_LAMBDA]


def cand_lambda():
    """{reader: λ} — weight of the refinement (cloze) log P in the recognizer score of PARSeq and MDiff4STR."""
    return {m: v.get('lambda', 0.) for m, v in _json(CAND_LAMBDA).items()}


# ---------------------------------------------------------------- score tables

def _string_table(npz_files, strings, field):
    for f in npz_files:
        z = np.load(f)
        off = np.cumsum(np.r_[0, z['lengths']])
        v = z[field].astype(np.float64)
        for n, ix in enumerate(z['index']):
            a = v[off[n]:off[n + 1]]
            if np.all(np.isfinite(a)):
                yield strings[ix], float(a.sum())


def pll_table(model='byt5-xl', field='ci', strings_file='strings.json'):
    """{string: Σ per-position log P} of ByT5 (field 'ci' = case-insensitive position terms). None if a shard is missing.
    Strings added later (`<model>_more`) fill only strings the main run did not score."""
    d = PLL / model
    fs = sorted(d.glob(f'*_pll_shard*of{N_SHARDS}.npz'))
    if len(fs) < N_SHARDS:
        return None
    more = PLL / f'{model}_more'
    fm = sorted((more / model).glob(f'*_pll_shard*of{N_SHARDS}.npz')) if (more / 'strings.json').exists() else []
    fm = fm if len(fm) == N_SHARDS else []
    files = fs + fm + [PLL / strings_file] + ([more / 'strings.json'] if fm else [])

    def build():
        tab = dict(_string_table(fs, json.loads((PLL / strings_file).read_text()), field))
        if fm:
            for t, v in _string_table(fm, json.loads((more / 'strings.json').read_text()), field):
                tab.setdefault(t, v)
        return tab
    tab = cached(f'pll_{model}_{field}', files, [model, field, strings_file], build)
    v = natdev_v2()
    add = PLL / f"{model}_{v['extra']}_{field}.json" if v else None
    if add is not None and add.exists():
        tab = dict(tab)
        for t, x in json.loads(add.read_text()).items():
            tab.setdefault(t, x)
    return tab


def pll_folder(d, field='ci'):
    """PLL table of one work folder (strings.json + 16 shards, in d/byt5-xl or d) — the new-benchmark runs."""
    d = Path(d)
    fs = sorted((d / 'byt5-xl').glob(f'byt5-xl_pll_shard*of{N_SHARDS}.npz')) or sorted(d.glob(f'*_pll_shard*of{N_SHARDS}.npz'))
    if len(fs) < N_SHARDS:
        raise SystemExit(f'PLL shards {len(fs)}/{N_SHARDS} ({rel(d)})')
    return cached(f'pllf_{hashlib.sha1(str(d).encode()).hexdigest()[:8]}', fs + [d / 'strings.json'], [field],
                  lambda: dict(_string_table(fs, json.loads((d / 'strings.json').read_text()), field)))


def ilm_table(m):
    """{string: internal-LM log P} of recognizer m (the decoder with the visual input removed)."""
    files = []
    for tag in ('', '_b20t20'):
        sf, vf = ILM / f'strings_{m}{tag}.json', ILM / f'ilm_{m}{tag}.npz'
        if sf.exists() and vf.exists():
            files += [sf, vf]
    _need(files, f'internal-LM table of {m}', ILM)

    def build():
        out = {}
        for k in range(0, len(files), 2):
            v = np.load(files[k + 1])['logp']
            out.update({t: float(x) for t, x in zip(json.loads(files[k].read_text()), v) if np.isfinite(x)})
        return out
    return cached(f'ilm_{m}', files, [m], build)


def natdev_merge(split, load):
    """load(split) of a per-image store; under CWR_NATDEV_V2 natdev also reads load(<extra split>) and keeps the listed keys
    (the hook of the scripts whose stores are not read through this module)."""
    return _v2(split, load(split), load)


def window_table(tag):
    """window_table_ of the tag; under CWR_NATDEV_V2 a tag with the segment 'natdev' (trwnn_natdev, wf2_mwargmax_natdev_b10t10tf32,
    ...) also reads the same tag with the extra split in that place."""
    parts = tag.split('_')
    if 'natdev' in parts and natdev_v2():
        alt = lambda x: window_table_('_'.join(x if s == 'natdev' else s for s in parts))
        return _v2('natdev', window_table_(tag), alt)
    return window_table_(tag)


def window_table_(tag):
    """{key: {hypothesis: window score}} of SVTRv2-AR — absolute scores, the greedy transcript a included.
    `<tag>` first, then `<tag>_gfix` (hypotheses added when the evaluation candidates took the validation rule; fills
    only missing hypotheses), then `<tag>_afix` (images whose windows were rebuilt from a; replaces the image)."""
    main, gfix, afix = WINDOWS / tag, WINDOWS / f'{tag}_gfix', WINDOWS / f'{tag}_afix'
    use_afix = env('CWR_AFIX', '1') != '0'
    files = [f for d in (main, gfix) + ((afix,) if use_afix else ()) for f in sorted(d.glob('*.json')) if '_limit' not in f.name]
    _need([f for f in files if f.parent == main], f'window table {tag}', main)

    def build():
        V = _merge_json_dir(main)
        for k, d in _merge_files(sorted(gfix.glob('*.json'))):
            w = V.setdefault(k, {})
            for y, v in d.items():
                w.setdefault(y, v)
        if use_afix:
            V.update(_merge_json_dir(afix))
        return V
    return cached(f'win_{tag}', files, [tag, use_afix], build)


def _merge_files(files):
    for f in files:
        if '_limit' in f.name:
            continue
        yield from json.loads(f.read_text()).items()


def own_window_table(m, split, tag=None):
    """Windows of recognizer m built from its own attention and read by itself — {key: {hypothesis: ΔW against a}}.
    `<split>_<tag>` (+ `_same`: hypotheses that read like a; + `_usual`: images whose windows were rebuilt from the usual
    output, replacing the image; + `_refine` · `_refine_usual`: hypotheses outside the beam strings, the refinement outputs of
    PARSeq and MDiff4STR, added to the image's table). None until every shard of the main run is done."""
    tag = tag or os.environ.get('OWN_TAG', 'owntrwnn')
    base = OWN_WINDOWS / READER_DIR[m]
    main, same, usual = base / f'{split}_{tag}', base / f'{split}_{tag}_same', base / f'{split}_{tag}_usual'
    refine, refine_usual = base / f'{split}_{tag}_refine', base / f'{split}_{tag}_refine_usual'
    if not _complete(main):
        return None
    use_same = _complete(same)
    use_usual = env('CWR_OWN_USUAL', '1') != '0' and usual.exists()
    use_refine = env('CWR_OWN_REFINE', '1') != '0' and _complete(refine)
    use_refine_usual = use_refine and use_usual and refine_usual.exists()
    files = [f for d, on in ((main, True), (same, use_same), (usual, use_usual), (refine, use_refine),
                             (refine_usual, use_refine_usual)) if on
             for f in sorted(d.glob('*.json')) if '_limit' not in f.name]

    def build():
        W = {}
        for k, d in _merge_files(sorted(main.glob('*.json'))):
            W.setdefault(k, {}).update(d)
        if use_same:
            for k, d in _merge_files(sorted(same.glob('*.json'))):
                for y, v in d.items():
                    W.setdefault(k, {}).setdefault(y, v)
        if use_usual:
            for k, d in _merge_files(sorted(usual.glob('*.json'))):
                W[k] = dict(d)
        for d_ in (refine, refine_usual) if use_refine else ():
            if d_ is refine_usual and not use_refine_usual:
                continue
            for k, d in _merge_files(sorted(d_.glob('*.json'))):
                w = W.setdefault(k, {})
                for y, v in d.items():
                    if w.get(y) is None:                          # absent, or stored without a value by the beam-string run
                        w[y] = v
        return W
    key = [m, split, tag, use_same, use_usual] + ([use_refine, use_refine_usual] if use_refine else [])
    out = cached(f'own_{m}_{split}_{tag}', files, key, build)
    return out if out is None else _v2(split, out, lambda x: own_window_table(m, x, tag) or {}, sparse=True)


def window_files(tag):
    """Files `window_table(tag)` reads — for the cache key of a table built from it."""
    use_afix = env('CWR_AFIX', '1') != '0'
    dirs = (WINDOWS / tag, WINDOWS / f'{tag}_gfix') + ((WINDOWS / f'{tag}_afix',) if use_afix else ())
    return [f for d in dirs for f in sorted(d.glob('*.json')) if '_limit' not in f.name]


def pll_files(model='byt5-xl', strings_file='strings.json'):
    """Files `pll_table(model, …)` reads."""
    more = PLL / f'{model}_more'
    fm = sorted((more / model).glob(f'*_pll_shard*of{N_SHARDS}.npz')) if (more / 'strings.json').exists() else []
    return (sorted((PLL / model).glob(f'*_pll_shard*of{N_SHARDS}.npz')) + [PLL / strings_file] + fm
            + ([more / 'strings.json'] if fm else []))


def window_done(tag):
    """Both runs of a window tag finished (validation: 1 shard · evaluation: 16 shards)."""
    return (WINDOWS / f'{tag}_natdev' / '_done_s0of1').exists() and len(list((WINDOWS / f'{tag}_full').glob('_done_s*of16'))) == 16
