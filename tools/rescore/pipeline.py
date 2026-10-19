"""Bounded, resumable CPU preparation for the main rescore entry point.

Input bundles contain original RGB, the *same* greedy pass's attention, and
candidate strings. They can be copied in batches to a CPU server and prepared
there without loading an STR or language model. Prepared bundles are keyed by
the input bytes and the current window implementation; older versions survive.
Pickles are for trusted, local/project-produced artifacts only.

Windows: `WINDOW` = MW3, the midpoint windows of the manuscript (`midpoint`); MW1 · MW5 · MW7 · MW9 widen or narrow
them, and W3 is the segment window of `window.prepare` (the supplementary tables before the midpoint windows). The
choice is part of protocol.json and of the preparation identity; a stored protocol without it is a W3 run.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import pickle
import sqlite3
import tempfile
import time
from pathlib import Path

FORMAT = 'rescore-pipeline-v1'
WINDOWS = ('mw1', 'mw3', 'mw5', 'mw7', 'mw9', 'w3')
WINDOW = 'mw3'


def prepare_windows(rgb, rec, kind=WINDOW):
    """Windows of one image in the `window.prepare` record format, for a kind of `WINDOWS`."""
    if kind not in WINDOWS:
        raise ValueError(f'unknown window kind {kind!r}; one of {WINDOWS}')
    if kind == 'w3':
        from tools.rescore import window
        return window.prepare(rgb, rec)
    from tools.rescore import midpoint
    n = int(kind[2:])
    return midpoint.make_windows(rgb, rec, [n])[n]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_bundle(path, kind, identity, payload):
    body = pickle.dumps(payload, protocol=5)
    envelope = dict(format=FORMAT, kind=kind, identity=identity, sha256=digest(body), body=body)
    atomic_bytes(path, pickle.dumps(envelope, protocol=5))


def read_bundle(path, kind, identity=None):
    with open(path, 'rb') as fh:
        obj = pickle.load(fh)
    if (obj.get('format') != FORMAT or obj.get('kind') != kind
            or (identity is not None and obj.get('identity') != identity)
            or digest(obj['body']) != obj.get('sha256')
            or (kind == 'input' and obj.get('identity') != obj.get('sha256'))):
        raise ValueError(f'incompatible or damaged {kind} bundle: {path}')
    return obj['identity'], pickle.loads(obj['body'])


def preparation_id(window=WINDOW):
    """Hash rules + window kind + numerical libraries, not server names or absolute paths."""
    root = Path(__file__).parent
    sources = {p.name: file_digest(p) for p in sorted(root.glob('*.py'))
               if p.stem in ('window', 'midpoint', 'position', 'region', 'segment', 'window_score', 'pipeline')}
    sources.update({p.name: file_digest(p) for p in sorted(root.glob('*.cpp'))})
    sources.update({str(p.relative_to(root)): file_digest(p) for p in sorted((root / 'data').glob('*.json'))})
    versions = {}
    for name in ('numpy', 'scipy', 'Pillow', 'opencv-python', 'opencv-python-headless', 'scikit-image'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return digest(json.dumps([FORMAT, window, sources, versions], sort_keys=True).encode())


def prepared_path(cache, input_id, definition):
    key = digest(f'{input_id}:{definition}'.encode())
    return Path(cache) / definition / (key + '.pkl'), key


def init_worker():
    """Limit CPU work per worker; spawn avoids inheriting initialized CUDA."""
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    import cv2
    cv2.setNumThreads(1)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except ImportError:
        pass


def prepare_bundle(path, cache, definition, cache_only=False, window=WINDOW):
    """Reuse an exact prepared bundle, otherwise prepare only this missing chunk."""
    from tools.rescore import window_score
    if definition != preparation_id(window):
        raise ValueError('CPU preparation code/libraries differ from the requested definition')
    input_id, payload = read_bundle(path, 'input')
    target, identity = prepared_path(cache, input_id, definition)
    if target.exists():
        _, records = read_bundle(target, 'windows', identity)
        return records, True
    if cache_only:
        raise FileNotFoundError(f'CPU input not prepared: {target}; run cli prepare on a CPU server first')
    records = []
    for row in payload['rows']:
        rec = row['attention']
        got = prepare_windows(row['rgb'], rec, window) if rec is not None else dict(skip='no_record')
        record = dict(key=row['key'], **got)
        if 'windows' in got:
            record['plan'] = window_score.plan_record(record, row['hyps'], payload['char_index'])
        records.append(record)
    write_bundle(target, 'windows', identity, records)
    return records, False


class ScoreCache:
    """Persistent exact-string PLL results, isolated by explicit run protocol.

No lowercasing, prefix reuse or changing of the PLL definition. Only completed
finite scores are committed; missing strings are returned in caller order.
"""
    def __init__(self, path, protocol):
        self.protocol = digest(json.dumps(protocol, sort_keys=True).encode())
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=60)
        self.db.execute('CREATE TABLE IF NOT EXISTS pll (protocol TEXT, text TEXT, score REAL, '
                        'PRIMARY KEY(protocol,text))')

    def get(self, texts):
        out = {}
        texts = list(dict.fromkeys(texts))
        for start in range(0, len(texts), 400):
            part = texts[start:start + 400]
            marks = ','.join('?' for _ in part)
            for text, score in self.db.execute(f'SELECT text,score FROM pll WHERE protocol=? AND text IN ({marks})',
                                               [self.protocol] + part):
                out[text] = score
        return out

    def put(self, scores):
        import math
        if any(not math.isfinite(float(v)) for v in scores.values()):
            raise ValueError('nonfinite PLL scores cannot be cached')
        with self.db:
            self.db.executemany('INSERT OR IGNORE INTO pll VALUES (?,?,?)',
                                [(self.protocol, k, float(v)) for k, v in scores.items()])

    def close(self):
        self.db.close()


def batches(items, size):
    from itertools import islice
    it = iter(items)
    while part := list(islice(it, size)):
        yield part


def lm_revision():
    """Local HF snapshot identity, without downloading or initializing a model."""
    try:
        from huggingface_hub import try_to_load_from_cache
        config = try_to_load_from_cache('google/byt5-xl', 'config.json')
        if isinstance(config, str):
            return dict(config_sha256=file_digest(config), snapshot=Path(config).parent.name)
    except ImportError:
        pass
    return None


def protocol(dataset, keys, limit, batch, checkpoint, pll_images, legacy_attention=False, no_pll_cache=False, *,
             recognizer=None, window=WINDOW):
    """Run definition. recognizer = `nbest.setting(..., device=)` (beam width · expansions · KV cache · TF32 requested and
    used) — None: the defaults on cuda:0. precision.matmul_tf32 is the precision the recognizer actually runs in."""
    import torch
    from tools.rescore import nbest, window_score, hypotheses as H, lm
    recognizer = recognizer or nbest.setting(device='cuda:0' if torch.cuda.is_available() else 'cpu')
    root = Path(__file__).parent
    repo = root.parents[1]
    versions = {}
    for name in ('torch', 'transformers'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    reader_sources = [window_score.CONFIG, repo / 'strtrain/utils/EN_symbol_dict.txt']
    reader_sources += sorted((repo / 'openrec/modeling').rglob('*.py'))
    reader_sources += sorted((repo / 'openrec/postprocess').rglob('*.py'))
    return dict(format=FORMAT, dataset=dataset, limit=limit, batch=batch, pll_images=pll_images, candidate_k=H.K,
                window=window, preparation_id=preparation_id(window),
                keys_sha256=digest(json.dumps(keys).encode()),
                checkpoint_sha256=file_digest(checkpoint or window_score.CHECKPOINT),
                sources={p.name: file_digest(p) for p in sorted(root.glob('*.py'))},
                reader_sources={str(p.relative_to(repo)): file_digest(p) for p in reader_sources},
                versions=versions,
                recognizer=recognizer,
                precision=dict(matmul_tf32=recognizer['matmul_tf32'],
                               cudnn_tf32=torch.backends.cudnn.allow_tf32, cudnn_benchmark=torch.backends.cudnn.benchmark),
                attention='legacy-per-image' if legacy_attention else 'same-greedy-pass-head-mean-round6-f16',
                pll=dict(model='google/byt5-xl', dtype='bf16', field='ci', cache=not no_pll_cache,
                         encoder_policy=lm.CONTEXT_REUSE_VERSION if lm.context_reuse_enabled() else 'legacy'))


def check_protocol(out, definition):
    path = out / 'protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != definition:
            raise ValueError('Run inputs/code/settings changed; use a new --out directory; old results are preserved')
    elif (out / 'scores.jsonl').exists() or any((out / 'inputs').glob('*.pkl')):
        raise ValueError('Existing run has no protocol; copy protocol.json too or use a new --out')
    else:
        atomic_bytes(path, (json.dumps(definition, sort_keys=True, indent=2) + '\n').encode())


def check_source(out, dataset):
    """Guard source changes when extending a run; finish consumes frozen bundles only.

    A cheap LMDB file identity guard, not a claim of a full-dataset content hash.
    Exported bundles independently hash the actual image/attention/label bytes.
    """
    from tools._io import DATA_ROOT
    source = (Path(DATA_ROOT) / dataset / 'data.mdb').resolve()
    if not source.exists():
        return  # _images reports missing LMDB; also permits synthetic/custom input readers
    st = source.stat()
    identity = dict(path=str(source), bytes=st.st_size, mtime_ns=st.st_mtime_ns,
                    ctime_ns=st.st_ctime_ns, inode=st.st_ino, device=st.st_dev)
    target = out / 'input_source.json'
    if target.exists() and json.loads(target.read_text()) != identity:
        raise ValueError('LMDB source changed; do not mix exported inputs with a different dataset')
    if not target.exists():
        atomic_bytes(target, json.dumps(identity, sort_keys=True).encode())


def decode_input(part, model, char_index, device, legacy_attention=False, recognizer=None):
    """Reuse first greedy attention/encoder; no per-image second inference. recognizer: `nbest.setting` (None → defaults)."""
    import numpy as np
    from tools.rescore import nbest, window, hypotheses as H
    rec_setting = recognizer or nbest.setting()
    vocab = sorted(char_index, key=char_index.get)
    arrays = [nbest.prepare(img) for _, img, _ in part]
    beams = {}
    for shape in dict.fromkeys(a.shape for a in arrays):
        sel = [j for j, a in enumerate(arrays) if a.shape == shape]
        g, b, maps = nbest.decode(model, vocab, [arrays[j] for j in sel], device, rec_setting['beam_width'],
                                  rec_setting['token_topk'], return_attention=True, kv=rec_setting['kv_cache'],
                                  tf32=rec_setting['tf32_requested'])
        for j, anchor, cands, attn in zip(sel, g, b, maps):
            beams[j] = anchor, cands, attn
    rows = [dict(key=key, gt=label or '', greedy=beams[j][0]['text'], nbest=beams[j][1])
            for j, (key, _, label) in enumerate(part)]
    C = nbest.candidates(rows)
    result = []
    for j, (key, img, _) in enumerate(part):
        text, maps = rows[j]['greedy'], beams[j][2]
        width, height = img.size
        rotated = height > window.AR_ROTATE_TALL * width
        wh0 = (height, width) if rotated else (width, height)
        gh = arrays[j].shape[0] // window.ENCODER_DOWN_H
        rec = None
        if text and len(text) <= window.MAX_LEN and maps.shape[1] % gh == 0:
            rec = dict(text=text, maps=maps, gh=gh, gw=maps.shape[1] // gh, wh0=wh0,
                       rotated=rotated, source='computed')
            if legacy_attention:
                with nbest.precision(device, rec_setting['tf32_requested']):
                    rec = window.attention_record(model, np.asarray(img), text, char_index, device)
        c = C[key]
        result.append(dict(key=key, candidate=c, rgb=np.asarray(img), attention=rec,
                           hyps=[x['y'] for x in c['cands'] if x['nb'] < H.K]))
    return dict(char_index=char_index, rows=result)


def prepare_run(run, prepared_cache=None, workers=8):
    """CPU-only portable bundles; no torch/model initialization on this path."""
    import contextlib
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    if workers < 0:
        raise ValueError('workers must be nonnegative')
    run = Path(run)
    cache = Path(prepared_cache or run / 'prepared')
    stored = run / 'protocol.json'
    window = json.loads(stored.read_text()).get('window', 'w3') if stored.exists() else WINDOW
    definition = preparation_id(window)
    paths = sorted((run / 'inputs').glob('*.pkl'))
    if not paths:
        raise FileNotFoundError('No input bundles; run score --stage beam first')
    hits = count = 0
    with contextlib.ExitStack() as stack:
        ex = stack.enter_context(ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn'),
                                                     initializer=init_worker)) if workers > 0 else None
        for group in batches(paths, max(1, workers * 2)):
            tasks = [ex.submit(prepare_bundle, p, cache, definition, False, window) for p in group] if ex else None
            for i, p in enumerate(group):
                records, hit = tasks[i].result() if tasks else prepare_bundle(p, cache, definition, False, window)
                hits += int(hit); count += len(records)
    print(f'CPU prepared {count:,} images; reused {hits}/{len(paths)} bundles; {cache}', flush=True)
    return dict(images=count, bundles=len(paths), cache_hits=hits, definition=definition, window=window)


def materialize(out):
    """Streaming atomic JSONL: complete chunks survive an interrupted write."""
    out = Path(out)
    fd, name = tempfile.mkstemp(prefix='scores.', suffix='.tmp', dir=out)
    try:
        seen = set()
        with os.fdopen(fd, 'w') as fh:
            for p in sorted((out / 'scored').glob('*.json')):
                for r in json.loads(p.read_text()):
                    if r['key'] in seen:
                        raise ValueError(f'duplicate scored image: {r["key"]}')
                    seen.add(r['key'])
                    fh.write(json.dumps(r, ensure_ascii=False) + '\n')
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(name, out / 'scores.jsonl')
    finally:
        if os.path.exists(name):
            os.unlink(name)


def score(dataset, out, device='cuda:0', keys=None, limit=0, batch=48, checkpoint=None, workers=8, *,
          stage='all', prepared_cache=None, cache_only=False, pll_images=256, legacy_attention=False, no_pll_cache=False,
          beam_width=None, token_topk=None, kv=None, tf32=None, window=WINDOW):
    """Bounded streaming main path; portable CPU bundles and persistent PLL.

    all: overlap CPU preparation with later GPU beam/W/PLL work.
    beam: export inputs only; finish: consume exported/prepared bundles.
    Changed BF16 grouping is explicit in protocol, not bit-parity with old runs.
    beam_width · token_topk · kv · tf32: the recognizer (None → `nbest` defaults: 10 · 10 · cached · TF32 on Ampere or
    newer, the setting of the manuscript) — resolved once and stored in protocol.json. window: a kind of `WINDOWS`
    (default MW3); a stored protocol without it is a W3 run.
    """
    import contextlib
    import multiprocessing as mp
    from collections import deque
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    from tools.rescore import lm, nbest, window_score
    from tools.rescore.cli import _images
    if stage not in ('all', 'beam', 'finish') or batch < 1 or pll_images < 1 or workers < 0:
        raise ValueError('invalid stage, batch, pll_images or workers')
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if stage == 'finish' and not (out / 'protocol.json').exists():
        raise FileNotFoundError('finish needs the exported protocol.json as well as inputs/')
    recognizer = nbest.setting(beam_width, token_topk, kv, tf32, device)
    definition = protocol(dataset, keys, limit, batch, checkpoint, pll_images, legacy_attention, no_pll_cache,
                          recognizer=recognizer, window=window)
    window = definition.get('window', 'w3')     # a stored protocol (replayed by a caller) without the field is W3
    check_protocol(out, definition)
    if stage != 'finish':
        check_source(out, dataset)
    run_id = digest(json.dumps(definition, sort_keys=True).encode())
    cache = Path(prepared_cache or out / 'prepared')
    prep_id = preparation_id(window)
    chunks = out / 'scored'
    chunks.mkdir(exist_ok=True)
    done = set()
    for p in sorted(chunks.glob('*.json')):
        done.update(r['key'] for r in json.loads(p.read_text()))
    reader = char_index = pll = None
    stats = dict(status='running', input_bundles=0, prepared_hits=0, prepared_misses=0,
                 pll_hits=0, pll_misses=0, images=0, beam_seconds=0., prep_wait_seconds=0.,
                 window_score_seconds=0., pll_seconds=0.)

    def load_reader():
        nonlocal reader, char_index
        if reader is None:
            reader, char_index = window_score.load_reader(device, checkpoint)

    def inputs():
        existing = sorted((out / 'inputs').glob('*.pkl'))
        known = set()
        for p in existing:
            _, payload = read_bundle(p, 'input')
            if payload.get('run_id') != run_id:
                raise ValueError(f'input bundle belongs to another run: {p}')
            row_keys = {r['key'] for r in payload['rows']}
            if known & row_keys:
                raise ValueError('duplicate image across input bundles')
            known.update(row_keys)
            if not row_keys <= done:
                if row_keys & done:
                    raise ValueError('partial scored input bundle')
                yield p, payload
        if stage == 'finish':
            if not existing:
                raise FileNotFoundError('No input bundles; run score --stage beam first')
            return
        marker = out / 'inputs_complete.json'
        meta = json.loads(marker.read_text()) if marker.exists() else {}
        if (meta.get('run_id') == run_id and set(meta.get('bundles', [])) == {p.name for p in existing}
                and meta.get('images') == len(known)):
            return  # Fully exported, frozen inputs: do not decode the LMDB again on resume.
        todo = (r for r in _images(dataset, keys, limit) if r[0] not in known and r[0] not in done)
        for part in batches(todo, batch):
            load_reader()
            t0 = time.perf_counter()
            payload = decode_input(part, reader, char_index, device, legacy_attention=legacy_attention, recognizer=recognizer)
            stats['beam_seconds'] += time.perf_counter() - t0
            payload['run_id'] = run_id
            identity = digest(pickle.dumps(payload, protocol=5))
            p = out / 'inputs' / f'{len(known):09d}_{identity}.pkl'
            write_bundle(p, 'input', identity, payload)
            known.update(r['key'] for r in payload['rows'])
            yield p, payload
        check_source(out, dataset)
        atomic_bytes(out / 'inputs_complete.json', json.dumps(dict(run_id=run_id, images=len(known),
                     bundles=[p.name for p in sorted((out / 'inputs').glob('*.pkl'))])).encode())

    try:
        if stage == 'beam':
            for _, payload in inputs():
                stats['input_bundles'] += 1
                stats['images'] += len(payload['rows'])
            stats['status'] = 'beam_complete'
            return stats
        with contextlib.ExitStack() as stack:
            if cache_only and workers > 0:
                ex = stack.enter_context(ThreadPoolExecutor(workers))  # I/O only; no CPU geometry/model initialization
            else:
                ex = stack.enter_context(ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn'),
                                                             initializer=init_worker)) if workers > 0 else None
            revision = lm_revision()
            manifest = out / 'pll_model.json'
            if manifest.exists() and json.loads(manifest.read_text()) != revision:
                raise ValueError('ByT5 snapshot changed or unavailable; preserve this run and use a new --out')
            pll_cache = ScoreCache(out / 'pll.sqlite', dict(run=definition, revision=revision))
            stack.callback(pll_cache.close)
            waiting = deque()
            ready = []

            def flush():
                nonlocal pll
                if not ready:
                    return
                texts = sorted({y for _, payload, _ in ready for r in payload['rows']
                                for y in [r['candidate']['ar']] + r['hyps']})
                ok = [t for t in texts if t and all(c in char_index for c in t)]
                values = {} if no_pll_cache else pll_cache.get(ok)
                missing = [t for t in ok if t not in values]
                stats['pll_hits'] += len(values); stats['pll_misses'] += len(missing)
                if missing:
                    if pll is None:
                        pll = lm.PLL('byt5-xl', device)
                        # The first load may populate a previously absent local HF snapshot.
                        actual_revision = lm_revision()
                        if manifest.exists() and json.loads(manifest.read_text()) != actual_revision:
                            raise ValueError('ByT5 snapshot changed during loading')
                        atomic_bytes(manifest, json.dumps(actual_revision, sort_keys=True).encode())
                        pll_cache.protocol = digest(json.dumps(dict(run=definition, revision=actual_revision), sort_keys=True).encode())
                    t0 = time.perf_counter()
                    got = pll.table(missing)
                    stats['pll_seconds'] += time.perf_counter() - t0
                    if set(got) != set(missing):
                        raise ValueError('PLL did not return every eligible string')
                    if not no_pll_cache:
                        pll_cache.put(got)
                    values.update(got)
                for path, payload, W in ready:
                    result = []
                    for r in payload['rows']:
                        c = r['candidate']; ys = [c['ar']] + r['hyps']
                        result.append(dict(key=r['key'], gt=c['gt'], ar=c['ar'], ar_lp=c['ar_lp'],
                                           cands=c['cands'], window=W.get(r['key'], {}),
                                           pll={y: values[y] for y in ys if y in values}))
                    atomic_bytes(chunks / (path.stem + '.json'), json.dumps(result, ensure_ascii=False).encode())
                    stats['images'] += len(result)
                ready.clear()
                print(f'{dataset}: +{stats["images"]:,} scored; PLL new={stats["pll_misses"]:,}, '
                      f'reused={stats["pll_hits"]:,}', flush=True)

            def consume():
                path, payload, future = waiting.popleft()
                t0 = time.perf_counter()
                records, hit = future.result() if future else prepare_bundle(path, cache, prep_id, cache_only, window)
                stats['prep_wait_seconds'] += time.perf_counter() - t0
                stats['prepared_hits' if hit else 'prepared_misses'] += 1
                load_reader()
                if payload['char_index'] != char_index:
                    raise ValueError('prepared input vocabulary differs from loaded recognizer')
                hyps = {r['key']: r['hyps'] for r in payload['rows']}
                valid = [r for r in records if 'windows' in r]
                t0 = time.perf_counter()
                W = (window_score.score_windows(valid, hyps, reader, char_index, device, tf32=recognizer['tf32_requested'])
                     if valid else {})
                stats['window_score_seconds'] += time.perf_counter() - t0
                compact = dict(rows=[{k: r[k] for k in ('key', 'candidate', 'hyps')} for r in payload['rows']])
                ready.append((path, compact, W))
                if sum(len(p['rows']) for _, p, _ in ready) >= pll_images:
                    flush()

            for path, payload in inputs():
                stats['input_bundles'] += 1
                task = ex.submit(prepare_bundle, path, cache, prep_id, cache_only, window) if ex else None
                waiting.append((path, payload, task))
                if len(waiting) >= max(2, workers * 2):
                    consume()
            while waiting:
                consume()
            flush()
            marker = out / 'inputs_complete.json'
            meta = json.loads(marker.read_text()) if marker.exists() else {}
            actual = {p.name for p in (out / 'inputs').glob('*.pkl')}
            scored_keys = {r['key'] for p in chunks.glob('*.json') for r in json.loads(p.read_text())}
            complete = (meta.get('run_id') == run_id and set(meta.get('bundles', [])) == actual
                        and meta.get('images') == len(scored_keys))
            stats['status'] = 'complete' if complete else 'partial_inputs'
    except BaseException as exc:
        stats['status'] = 'interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed'
        stats['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        # Committed chunks, not a partial append, are the resume authority.
        if stage != 'beam':
            materialize(out)
        atomic_bytes(out / 'pipeline_stats.json', json.dumps(stats, indent=2).encode())
    return stats
