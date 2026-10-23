#!/usr/bin/env python3
"""Recompute every result file of the manuscript tables from the stored inputs, then compare with a reference tree.

Stages run in dependency order. With --jobs > 1 the stored inputs shared by most stages (SVTRv2-AR validation and
evaluation lists, ByT5 PLL, default window scores) are loaded once and the stages run in forked workers.

    python -m tools.experiments.rescore.reproduce --out /tmp/legit_repro [--jobs 8] [--ref <tree>] [--only a,b]
    python -m tools.experiments.rescore.reproduce --list
Results land under --out at the paths the manuscript generator reads; --ref is compared byte for byte, and the run exits
non-zero when a file differs. The default reference is `<data root>/reference` — the stored results of the manuscript run
whose inputs the data root holds (in the research tree a link: the MW3 root → materials/mw3, the 10 · 10 TF32 root →
materials/b10t10tf32) — or the data root itself when it has none.
"""
from __future__ import annotations

import argparse
import filecmp
import importlib
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

# (name, module:function, environment, depends on, result file) — every experiment module lists its own stages in
# `REPRODUCE` (same tuple, function name without the module); this driver collects them, so adding an experiment does not
# touch this file.
def _collect():
    import pkgutil
    import tools.experiments.rescore as pkg
    out = []
    for info in sorted(pkgutil.iter_modules(pkg.__path__), key=lambda i: i.name):
        if info.name in ('common', 'reproduce'):
            continue
        mod = importlib.import_module(f'tools.experiments.rescore.{info.name}')
        for name, fn, env, deps, res in getattr(mod, 'REPRODUCE', ()):
            out.append((name, fn if ':' in fn else f'{info.name}:{fn}', env, tuple(deps), res))
        RUN_RECORD.update(getattr(mod, 'RUN_RECORD', {}))
    return tuple(out)


# Fields that record where a stage read its input (commit and path of the producing run), not results: compared after
# removing them, and reported as such ({result path: fields}, from the RUN_RECORD of each experiment module).
RUN_RECORD = {}
STAGES = _collect()
BY = {s[0]: s for s in STAGES}


# numpy chooses exp / log / reduction kernels by the CPU, OpenBLAS its dot kernels: an AVX-512 machine and an AVX2 machine
# then differ in the last digit of a few results (log-sum-exp of cluster masses, a correlation over millions of pairs).
# With these settings both take the AVX2 paths and write the same bytes.
PORTABLE = dict(NPY_DISABLE_CPU_FEATURES='AVX512F AVX512CD AVX512VPOPCNTDQ AVX512VL AVX512BW AVX512DQ AVX512VNNI AVX512IFMA '
                                         'AVX512VBMI AVX512VBMI2 AVX512BITALG AVX512FP16 AVX512_SKX AVX512_CLX AVX512_CNL '
                                         'AVX512_ICL AVX512_SPR',
                OPENBLAS_CORETYPE='Haswell')


def _child(name, tx):
    """Run one stage in a forked process and send (ok, seconds | traceback) back."""
    import traceback
    try:
        tx.send((True, run_stage(name)[1]))
    except BaseException:                                        # noqa: BLE001 — report every failure to the driver
        tx.send((False, traceback.format_exc()))
    finally:
        tx.close()


def _ordered(names):
    """Dependency order (stable): a stage after the stages it reads."""
    left, out = list(names), []
    while left:
        for n in left:
            if all(d in out or d not in names for d in BY[n][3]):
                out.append(n)
                left.remove(n)
                break
        else:
            raise ValueError(f'dependency cycle among {left}')
    return out


_OPENED = set()


def _audit(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
        _OPENED.add(os.fsdecode(args[0]))


def run_stage(name):
    _, target, env, _, _ = BY[name]
    os.environ.update(env)
    rec = os.environ.get('LEGIT_RECORD_INPUTS')
    if rec:                                                       # files the stage opens (an audit hook cannot be removed —
        sys.addaudithook(_audit)                                  # the worker process ends with the stage)
    mod, fn = target.split(':')
    t = time.time()
    getattr(importlib.import_module(f'tools.experiments.rescore.{mod}'), fn)()
    if rec:
        Path(rec).mkdir(parents=True, exist_ok=True)
        (Path(rec) / f'{name}.txt').write_text('\n'.join(sorted(_OPENED)) + '\n')
    return name, time.time() - t


def write_inputs(rec, out):
    """Union of the files the stages opened under the data root → <out>/data_inputs.json (path, size, SHA-256): the data a
    release needs. Run with a fresh cache (LEGIT_RESCORE_CACHE_DIR), or cached tables hide their source files."""
    import hashlib
    import json
    from tools.rescore import store
    # A data root may be a tree of links (the MW3 and TF32 roots): paths are kept as opened, not resolved, so that a file is
    # listed at its place under the root; sizes and hashes are those of the file the link points to.
    root = Path(os.path.abspath(store.DATA))
    paths = set()
    for f in Path(rec).glob('*.txt'):
        for line in f.read_text().split('\n'):
            if not line:
                continue
            p = Path(os.path.abspath(line))
            if p.is_file() and root in p.parents:
                paths.add(p)
    for p in list(paths):                                        # end marks of the score folders — checked, not opened
        paths.update(q for q in p.parent.glob('_done_*') if q.is_file())
    rows = []
    for p in sorted(paths):
        rel = p.relative_to(root).as_posix()
        if rel.startswith(('tools/', 'scripts/', 'configs/', 'openrec/', 'strtrain/', 'tests/')):
            continue                                              # code, not data
        rows.append(dict(path=store.public_path(rel), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    (Path(out) / 'data_inputs.json').write_text(json.dumps(dict(root='LEGIT_DATA', files=rows), indent=1) + '\n')
    print(f'data inputs: {len(rows)} files · {sum(r["bytes"] for r in rows) / 1e9:.2f} GB → {Path(out) / "data_inputs.json"}', flush=True)


def preload():
    """Inputs most stages share — loaded before forking so workers inherit them."""
    from tools.experiments.rescore import common as C
    dev, full = C.ar('natdev'), C.ar('full')
    C.pll()
    C.window_delta('trwnn_natdev', dev)
    C.window_delta('trwnn_full', full)


# Stages whose stored inputs are in the published data archive (the SVTRv2-AR accuracy values of the main table, of the
# comparison with published methods and of the image-conditioned table: `paper_tables`; the recorded files). A release — no
# research_layout.json, so the data root is the unpacked archive — runs and lists only these; the other stages need
# inputs of the research tree and are listed as "data not included".
ARCHIVE_STAGES = ('table_main', 'table_condition', 'recorded')


def _available():
    from tools.rescore import store
    return [s for s in STAGES if store.LAYOUT or s[0] in ARCHIVE_STAGES]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, help='output root (LEGIT_RESCORE_OUT)')
    ap.add_argument('--ref', type=Path, help='reference tree to compare with (default: the data root)')
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--only', default='', help='comma-separated stage names (their dependencies must exist in --out or the data root)')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--no-compare', action='store_true')
    ap.add_argument('--record-inputs', action='store_true', help='list the data files the stages open → <out>/data_inputs.json')
    ap.add_argument('--portable', action='store_true',
                    help='same bytes on AVX2 and AVX-512 machines: numpy without AVX-512 dispatch, OpenBLAS Haswell kernels')
    a = ap.parse_args()
    if a.portable and os.environ.get('NPY_DISABLE_CPU_FEATURES') != PORTABLE['NPY_DISABLE_CPU_FEATURES']:
        os.execve(sys.executable, [sys.executable, '-m', 'tools.experiments.rescore.reproduce'] + sys.argv[1:],
                  {**os.environ, **PORTABLE})                     # numpy fixes its SIMD paths at import — restart with them set
    if a.list:
        avail = {s[0] for s in _available()}
        for name, target, env, deps, res in STAGES:
            what = res if isinstance(res, str) else (f"{len(res)} files" if res else "(part)")
            print(f'{name:14s} {target:30s} {env} deps={list(deps)} → {what}' if name in avail else
                  f'{name:14s} (data not included in the archive)')
        return 0
    if a.out:
        os.environ['LEGIT_RESCORE_OUT'] = str(a.out.resolve())
    if a.record_inputs:
        os.environ['LEGIT_RECORD_INPUTS'] = str(((a.out or Path('.')).resolve()) / '_opened')
        # files opened in a stage's worker processes (common.pmap) are not recorded, and a table cached there would hide its
        # sources — so no workers
        os.environ['LEGIT_STAGE_WORKERS'] = '1'
    # OpenMP threads per stage (scikit-learn's boosting reranker) — unlimited, parallel stages oversubscribe a busy machine
    os.environ.setdefault('OMP_NUM_THREADS', str(max(1, min(8, (os.cpu_count() or 8) // max(1, a.jobs)))))
    # worker processes inside a stage (common.pmap) — the machine's cores shared among the parallel stages
    os.environ.setdefault('LEGIT_STAGE_WORKERS', str(max(1, (os.cpu_count() or 8) // max(1, a.jobs))))
    from tools.rescore import store
    unknown = sorted(set(filter(None, a.only.split(','))) - set(BY))
    if unknown:
        raise SystemExit(f'unknown stages {unknown} (see --list)')
    avail = [s[0] for s in _available()]
    missing = sorted(set(filter(None, a.only.split(','))) - set(avail))
    if missing:
        raise SystemExit(f'stages {missing}: their inputs are not in the data archive (see --list)')
    if not a.out and not a.no_compare:
        raise SystemExit('give --out: the results would be written into the data root (and compared with themselves)')
    todo = _ordered([n for n in avail if not a.only or n in a.only.split(',')])
    if not a.only:
        os.environ['LEGIT_RESCORE_STRICT'] = '1'        # a full run reads earlier results only from its own output root
    print(f'data root {store.DATA} · output root {os.environ.get("LEGIT_RESCORE_OUT", store.DATA)} · Python {sys.version.split()[0]} · '
          f'stages {len(todo)} · jobs {a.jobs}', flush=True)
    t0 = time.time()
    if a.record_inputs:                                           # the parent too: it preloads the shared tables before forking
        sys.addaudithook(_audit)
    if a.jobs <= 1:
        for name in todo:
            print(f'{name}: {run_stage(name)[1]:.1f}s', flush=True)
    else:
        preload()
        print(f'preload {time.time() - t0:.1f}s', flush=True)
        if a.record_inputs:
            Path(os.environ['LEGIT_RECORD_INPUTS']).mkdir(parents=True, exist_ok=True)
            (Path(os.environ['LEGIT_RECORD_INPUTS']) / '_preload.txt').write_text('\n'.join(sorted(_OPENED)) + '\n')
        ctx = mp.get_context('fork')
        done, running, failed = set(), {}, []
        pending = list(todo)
        while pending or running:                                 # one fresh (non-daemon) process per stage, so a stage may
            for name in list(pending):                            # run its own worker pool (common.pmap)
                if len(running) >= a.jobs:
                    break
                if all(d in done or d not in todo for d in BY[name][3]):
                    rx, tx = ctx.Pipe(duplex=False)
                    p = ctx.Process(target=_child, args=(name, tx))
                    p.start()
                    tx.close()
                    running[name] = (p, rx)
                    pending.remove(name)
            for name, (p, rx) in list(running.items()):
                if rx.poll():
                    ok, sec = rx.recv()
                    p.join()
                    if not ok:
                        failed.append(name)
                        print(f'{name}: FAILED\n{sec}', flush=True)
                    else:
                        print(f'{name}: {sec:.1f}s', flush=True)
                    done.add(name)
                    del running[name]
                elif not p.is_alive():
                    p.join()
                    failed.append(name)
                    print(f'{name}: FAILED (exit {p.exitcode})', flush=True)
                    done.add(name)
                    del running[name]
            time.sleep(.2)
        if failed:
            print(f'failed stages: {failed}', flush=True)
    print(f'all stages {time.time() - t0:.1f}s', flush=True)
    if os.environ.get('LEGIT_RECORD_INPUTS'):
        from tools.experiments.rescore.common import out_root
        write_inputs(os.environ['LEGIT_RECORD_INPUTS'], out_root())
    if a.no_compare:
        return 0
    from tools.experiments.rescore.common import out_root
    OUT = out_root()
    # a release keeps the reference results under reference/ of the data root (part of their public path); the research
    # tree links <data root>/reference to the manuscript's result tree, laid out like the data root
    ref = Path(a.ref) if a.ref else (store.DATA / 'reference' if store.LAYOUT and (store.DATA / 'reference').is_dir()
                                     else store.DATA)
    if OUT.resolve() == Path(ref).resolve():
        print('output root = reference tree — nothing to compare (give --out)', flush=True)
        return 1
    bad = 0
    files = [r for name in todo for r in ((BY[name][4],) if isinstance(BY[name][4], str) else (BY[name][4] or ()))]   # None: a part
    for res in files:
        disk = store.research_path(res)
        same = (OUT / disk).exists() and (ref / disk).exists() and filecmp.cmp(OUT / disk, ref / disk, shallow=False)
        note = ''
        if not same and res in RUN_RECORD and (OUT / disk).exists() and (ref / disk).exists():
            import json
            a, b = (json.loads((t / disk).read_text()) for t in (OUT, ref))
            for k in RUN_RECORD[res]:
                a.pop(k, None), b.pop(k, None)
            same, note = a == b, f'  (values; run record {list(RUN_RECORD[res])} differs)'
        bad += not same
        print(f'{"same" if same else "DIFFERENT":9s} {res}{note}', flush=True)
    print(f'{len(files) - bad}/{len(files)} result files byte-identical with {ref}', flush=True)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
