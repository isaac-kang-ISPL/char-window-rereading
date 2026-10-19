"""Window term V(y) (code: W) — re-read hypothesis y with the same recognizer (svtrv2-ar) over every character window of the
image and aggregate.

For window w (character position range lo..hi · the slot_ids of `window.prepare`), edit-align y onto a, take the substring
c_w(y) of y that corresponds to that range, and use as window score s_w(y) the log-likelihood of teacher forcing
c_w(y) + ⟨eos⟩ with the window image as input. The default is the overlap-weighted sum:

    s_w(y) = log P_AR(c_w(y) ⟨eos⟩ | window w)          V(y) = Σ_w ω_w s_w(y)          ω_w = mean_{i ∈ slot_ids(w)} 1 / n_i

n_i = number of valid windows covering character position i (the same weight for all hypotheses of an image). The earlier window definition
(`SCORE_PAPER_V1`, not used in the paper) uses as window score the logsumexp over the core ± one boundary character on each side (up to four
variants) and sums without weights:
s_w(y) = logsumexp_{v ∈ variants_w(y)} log P_AR(v ⟨eos⟩ | window w) · V(y) = Σ_w s_w(y).

The result is {hypothesis: V(y)} per image — greedy a is included too (Δ is formed by the decision side). Rules:
hypotheses with an empty string · out-of-vocabulary characters are dropped · an image with no valid window gives 0.0 for
all hypotheses · when a substring cannot be measured in a window, that window term is −1e30
(overlap weights add the measured terms in window order, then −1e30 × the number of unmeasured windows · the sum of the earlier window definition
adds −1e30 in place, in window order — as in the stored scores) ·
an image whose windows cannot be built (`window.prepare` skips it) has no key in the result.
The recognizer's float32 matmuls default to TF32 (Ampere or newer · `nbest.precision`). The paper's stored scores are
fp32 — `tf32=False` · `--fp32` · LEGIT_AR_TF32=0 (`PAPER_V1_RUN` includes fp32).

`SCORE_PAPER_V1` gives the stored scores of the earlier window definition bit for bit on the same machine and batches as the research code that
produced them; the comparison lives in the research repository.

    PYTHONPATH=. python -m tools.rescore.window_score --pool <TSV(dataset, sample_idx)> --hyps <JSON {key: [y, …]}> --out <dir> \
        --device cuda:0 [--checkpoint <UnionST-U14MFilter.pth>]
    # earlier window definition (not used in the paper): … --paper-v1
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).resolve().with_name('data') / 'svtrv2-ar.yml'   # inference configuration of the recognizer
MAX_LEN = 25            # maximum length of a teacher-forced substring (recognizer output length limit)
CHUNK = 5000            # storage chunk — the sorted image indices of each dataset, 5,000 images each
WINDOW_BATCH = 256      # windows fed to the encoder at once (same input size only)
CANDIDATE_BATCH = 2048  # (window, substring) pairs fed to the decoder at once
AGGREGATIONS = ('coverage', 'sum')
SCORE_DEFINITION = dict(boundary=False, aggregation='coverage')   # default — one core substring · overlap weights
SCORE_PAPER_V1 = dict(boundary=True, aggregation='sum')           # earlier window definition — logsumexp of boundary variants · sum


# ---------------------------------------------------------------- hypothesis ↔ window correspondence

def edit_ops(a, g):
    """Edit alignment a → g (Levenshtein DP · backtracking from the end · order equal/substitute > a only > g only)
    → [(op, i, j)] · op '=' · 'S' · 'I' (a only) · 'D' (g only)."""
    n, m = len(a), len(g); D = np.zeros((n + 1, m + 1), int); D[:, 0] = range(n + 1); D[0, :] = range(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (a[i - 1] != g[j - 1]))
    i, j, out = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (a[i - 1] != g[j - 1]):
            out.append(('=' if a[i - 1] == g[j - 1] else 'S', i - 1, j - 1)); i -= 1; j -= 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            out.append(('I', i - 1, None)); i -= 1
        else:
            out.append(('D', None, j - 1)); j -= 1
    return out[::-1]


def coords(a, y):
    """Coordinate on a for each character of y — equal · substitute get that position i; a character only in y gets the
    previous matched position + 0.5."""
    out = []; last = -1
    for op, i, j in edit_ops(a, y):
        if op in ('=', 'S'):
            out.append((float(i), y[j])); last = i
        elif op == 'D':
            out.append((last + .5, y[j]))
    return out


def variants(lo, hi, L, units):
    """Substrings of y corresponding to the positions lo..hi of a covered by the window — the core ± one boundary
    character on each side (up to four).
    The first window adds to the core the characters prepended to y, the last window those appended to y. If the core is
    empty (y dropped all those positions), only the boundary characters."""
    core = [k for k, (c, _) in enumerate(units) if lo <= c <= hi or (c < 0 and lo == 0) or (c > L - 1 and hi == L - 1)]
    if core:
        k0, k1 = core[0], core[-1]
    else:
        before = [k for k, (c, _) in enumerate(units) if c < lo]
        k0 = (before[-1] + 1) if before else 0; k1 = k0 - 1
    mid = ''.join(ch for _, ch in units[k0:k1 + 1])
    left = units[k0 - 1][1] if k0 - 1 >= 0 else None
    right = units[k1 + 1][1] if k1 + 1 < len(units) else None
    return [l + mid + r for l in ([''] + ([left] if left else [])) for r in ([''] + ([right] if right else []))]


# ---------------------------------------------------------------- teacher-forcing log-likelihood

def encode_text(text, char_index, max_len):
    if len(text) > max_len or any(char not in char_index for char in text):
        return None
    return [char_index[char] for char in text]


def forced_scores(decoder, memory, requests, char_index, candidate_batch, max_len):
    """[(window index, string)] → {(window index, string): (log P(string ⟨eos⟩ | window), mean per token)} —
    teacher-forces string + ⟨eos⟩ after ⟨bos⟩ in one pass.
    Strings of similar length are batched together to reduce padding. Strings with out-of-vocabulary characters · longer
    than max_len are absent from the result."""
    import torch
    encoded = []
    for image_index, text in requests:
        tokens = encode_text(text, char_index, max_len)
        if tokens is not None:
            encoded.append((image_index, text, tokens))
    encoded.sort(key=lambda item: len(item[2]))
    output = {}
    for start in range(0, len(encoded), candidate_batch):
        part = encoded[start:start + candidate_batch]
        width = max(len(tokens) for _image, _text, tokens in part) + 2
        ids_np = np.full((len(part), width), decoder.ignore_index, dtype=np.int64)
        ids_np[:, 0] = decoder.bos
        lengths = np.empty(len(part), dtype=np.int64)
        for row, (_image, _text, tokens) in enumerate(part):
            n = len(tokens)
            if n:
                ids_np[row, 1:1 + n] = tokens
            ids_np[row, 1 + n] = decoder.eos
            lengths[row] = n + 1
        ids = torch.from_numpy(ids_np).to(memory.device, non_blocking=True)
        image_index = torch.as_tensor([item[0] for item in part], dtype=torch.long, device=memory.device)
        with torch.inference_mode():
            logits = decoder.forward_train(memory.index_select(0, image_index), ids)
            logp = logits.float().log_softmax(-1)
            targets = ids[:, 1:]
            gathered = logp.gather(-1, targets.clamp_max(logp.shape[-1] - 1).unsqueeze(-1)).squeeze(-1)
            valid = torch.arange(gathered.shape[1], device=gathered.device)[None, :] < torch.from_numpy(lengths).to(gathered.device)[:, None]
            raw = torch.where(valid, gathered, torch.zeros_like(gathered)).sum(1).cpu().numpy()
        for row, (image_i, text, _tokens) in enumerate(part):
            value = float(raw[row])
            output[(image_i, text)] = (value, value / int(lengths[row]))
    return output


def checkpoint_path(checkpoint=None):
    """The SVTRv2-AR checkpoint — `checkpoint`, else LEGIT_AR_CHECKPOINT > `local_paths.json` (`tools._io.local_path`;
    a release has no default). Stops with the setting to use when the file is not there."""
    from tools._io import local_path
    path = checkpoint or local_path('ar_checkpoint')
    if not path or not Path(path).is_file():
        raise FileNotFoundError('SVTRv2-AR checkpoint (UnionST-U14MFilter.pth) ' + (f'not found: {path}' if path else 'not set')
                                + ' — set LEGIT_AR_CHECKPOINT or pass --checkpoint')
    return Path(path)


def __getattr__(name):
    """`CHECKPOINT` (LEGIT_AR_CHECKPOINT > `local_paths.json`) is resolved when it is read, so a release without a
    checkpoint imports cleanly and stops only where the checkpoint is needed; `load_reader` also checks that it exists."""
    if name == 'CHECKPOINT':
        from tools._io import local_path
        path = local_path('ar_checkpoint')
        return Path(path) if path else checkpoint_path()
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def reader_config():
    """`CONFIG` as a dict, with the dictionary path taken from the repository root instead of the working directory."""
    from strtrain.engine.config import Config
    cfg = Config(str(CONFIG)).cfg
    for part in (cfg['Global'], cfg['PostProcess']):
        path = part.get('character_dict_path')
        if path and not os.path.isabs(os.path.expanduser(path)):
            part['character_dict_path'] = os.path.normpath(os.path.join(REPO, path))
    return cfg


def load_reader(device, checkpoint=None):
    """svtrv2-ar (public weights UnionST-U14MFilter) → (model, vocabulary {character: index}) — the last two decoder
    outputs (special tokens) are excluded from the vocabulary."""
    import torch
    from openrec.modeling import build_model
    from openrec.postprocess import build_post_process
    cfg = reader_config()
    post = build_post_process(cfg['PostProcess'], cfg['Global'])
    cfg['Architecture']['Decoder']['out_channels'] = post.get_character_num()
    model = build_model(cfg['Architecture'])
    state = torch.load(checkpoint_path(checkpoint), map_location='cpu', weights_only=False)
    model.load_state_dict(state.get('state_dict', state.get('model', state)), strict=True)
    model = model.eval().to(device)
    vocab = list(post.character)[:model.decoder.out_channels - 2]
    return model, {c: k for k, c in enumerate(vocab)}


def coverage_weights(slots, length):
    """ω_w (code: u) = mean_{i ∈ slots_w} 1 / (number of valid windows covering position i) — one weight per window (independent
    of the hypothesis)."""
    count = np.zeros(length, dtype=np.int64)
    for ids in slots:
        count[ids] += 1
    return [float(np.mean(1. / count[ids])) for ids in slots]


def aggregate(raw, aggregation='coverage'):
    """Per-window raw scores → {key: {y: V(y)}} — exactly the aggregation `score_windows` uses (GPU runs and CPU
    re-aggregation are byte-identical).
    raw: the second value of `score_windows(detail=True)`, or the object read from a <chunk>.raw.pkl.gz written by `run`
    (dict(definition, rows)).
    'sum' (earlier window definition): add in window order, inserting −1e30 for unmeasured windows · 'coverage': add the measured terms
    ω_w · s_w in window order, then −1e30 × the number of unmeasured windows."""
    if aggregation not in AGGREGATIONS:
        raise ValueError(f'aggregation must be one of {AGGREGATIONS}')
    rows = raw['rows'] if 'rows' in raw and 'definition' in raw else raw
    out = {}
    for key, r in rows.items():
        weights = [w['u'] for w in r['windows']]
        res = {}
        for y, each in r['s'].items():
            tot = 0.
            if aggregation == 'sum':
                for value in each:
                    tot += value if value is not None else -1e30
            else:
                for value, u_w in zip(each, weights):
                    if value is not None:
                        tot += u_w * value
                tot = tot - 1e30 * float(sum(value is None for value in each))
            res[y] = tot
        out[key] = res
    return out


def hypotheses_of(text, hyps_y, char_index):
    """Hypotheses to score — a first; duplicates · empty strings · hypotheses with out-of-vocabulary characters are
    dropped."""
    return [y for y in dict.fromkeys([text] + list(hyps_y)) if y and all(c in char_index for c in y)]


def plan_record(record, hyps_y, char_index, boundary=False):
    """Scoring plan of one image — the Python computation before the GPU (edit alignment · coords · substrings · window
    weights), done per image.
    `run` builds it in advance in the CPU workers (`_prepare`) and attaches it to the record's 'plan'; `score_windows` uses
    an attached plan as is when it matches the current arguments, and builds it on the spot with this function when it is
    absent or does not match — either way the values are the same.
    → dict(text, ys, boundary, n_windows, local=[r['windows'] indices of the valid windows], meta=[dict(slot_ids, target, u)],
           per=[(y, [(valid window ordinal, [substrings…])…])], need=[substrings to ask per valid window (sorted · no
           duplicates)])."""
    a = record['text']; L = len(a)
    ys = hypotheses_of(a, hyps_y, char_index)
    local, spans, meta = [], [], []
    for k, w in enumerate(record['windows']):
        if not w['valid'] or not w['slot_ids']:
            continue
        local.append(k); spans.append((min(w['slot_ids']), max(w['slot_ids'])))
        meta.append(dict(slot_ids=[int(i) for i in w['slot_ids']], target=int(w.get('target_slot', -1))))
    weights = coverage_weights([m['slot_ids'] for m in meta], L)
    for m, u_w in zip(meta, weights):
        m['u'] = u_w
    per, sets = [], [set() for _ in local]
    for y in ys:
        u = coords(a, y)
        row = [(n, variants(lo, hi, L, u) if boundary else variants(lo, hi, L, u)[:1]) for n, (lo, hi) in enumerate(spans)]
        for n, vs in row:
            sets[n].update(vs)
        per.append((y, row))
    return dict(text=a, ys=ys, boundary=bool(boundary), n_windows=len(record['windows']), local=local, meta=meta, per=per,
                need=[sorted(s) for s in sets])


def _plan_for(record, hyps, char_index, boundary):
    """The plan attached to the record if it matches this call (transcript · hypotheses · boundary variants · number of
    windows), else a newly built plan."""
    p = record.get('plan')
    ys = hypotheses_of(record['text'], hyps.get(record['key'], []), char_index)
    if (p is not None and p.get('text') == record['text'] and p.get('ys') == ys and p.get('boundary') == bool(boundary)
            and p.get('n_windows') == len(record['windows'])):
        return p
    return plan_record(record, hyps.get(record['key'], []), char_index, boundary)


def score_windows(records, hyps, model, char_index, device, cand_batch=CANDIDATE_BATCH, *, boundary=False,
                  aggregation='coverage', detail=False, tf32=None):
    """records = [dict(key, text=a, windows=window list of `window.prepare`[, plan=`plan_record`])] ·
    hyps = {key: [y, …]} → {key: {y: V(y)}} (a first).
    boundary: window score = logsumexp of the boundary variants (up to four) (earlier window definition) · the default is one core substring ·
    aggregation: 'coverage' (default · overlap weights) | 'sum' (earlier window definition) · detail: if true, (W, per-window raw scores) —
    raw scores = {key: dict(windows=[dict(slot_ids, target, u)], s={y: [s_w(y) per window | None]})} (None = not measured
    in that window).
    When the record has a prebuilt plan (`plan_record`) that matches this call, it is used (less serial Python work in the
    GPU process · the same values).
    tf32: matmul precision of the recognizer (`nbest.precision` · None → LEGIT_AR_TF32 · default TF32)."""
    import torch
    from tools.rescore import nbest
    if aggregation not in AGGREGATIONS:
        raise ValueError(f'aggregation must be one of {AGGREGATIONS}')
    dec = model.decoder
    wins, plan = [], []                                   # windows (image index · AR input) · per image
                                                          # (plan, global index of its first window)
    for ri, r in enumerate(records):
        p = _plan_for(r, hyps, char_index, boundary)
        off = len(wins)
        wins.extend((ri, r['windows'][k]['image']) for k in p['local'])
        plan.append((p, off))
    by_win = {off + n: vs for p, off in plan for n, vs in enumerate(p['need']) if vs}
    scores, groups = {}, {}
    for wi, (_, arr) in enumerate(wins):
        if wi in by_win:
            groups.setdefault(arr.shape, []).append(wi)
    with nbest.precision(device, tf32):
        for shape, ids in groups.items():
            for s in range(0, len(ids), WINDOW_BATCH):
                part = ids[s:s + WINDOW_BATCH]
                x = torch.from_numpy(np.stack([wins[i][1] for i in part])).permute(0, 3, 1, 2).to(device)
                with torch.inference_mode():
                    mem = model.encoder((x.float() / 255 - .5) / .5)
                req = [(j, v) for j, wi in enumerate(part) for v in by_win[wi]]
                got = forced_scores(dec, mem, req, char_index, cand_batch, MAX_LEN)
                for (j, v), (raw, _) in got.items():
                    scores[(part[j], v)] = raw
    raw = {}
    for r, (p, off) in zip(records, plan):
        s_all = {}
        for y, ws in p['per']:
            each = []
            for n, vs in ws:
                wi = off + n
                vals = [scores[(wi, v)] for v in vs if (wi, v) in scores]
                each.append(float(np.logaddexp.reduce(vals)) if vals else None)
            s_all[y] = each
        # build the window info with constant keys — keys of dicts unpickled from workers are distinct
        # string objects per image, so the raw-score pickle bytes would differ
        raw[r['key']] = dict(windows=[dict(slot_ids=list(m['slot_ids']), target=m['target'], u=m['u']) for m in p['meta']], s=s_all)
    out = aggregate(raw, aggregation)
    return (out, raw) if detail else out


# ---------------------------------------------------------------- batch runs (stored attention records · original LMDB)

_W = {}


def _init(dataset, data_root, charout_root, boxes_path, kwargs, stored_geometry=True, planning=None):
    """Worker initialization — dataset is the default dataset (if None, each task passes it via
    `_prepare(..., dataset=)` · each dataset's original LMDB is opened on first use) ·
    with planning = dict(char_index, boundary), `_prepare` also builds the scoring plan (`plan_record`) per image."""
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    import cv2
    cv2.setNumThreads(1)
    _W.update(dataset=dataset, data_root=data_root, charout_root=charout_root, kwargs=kwargs, stored_geometry=stored_geometry,
              planning=planning, sources={})
    if dataset is not None:
        _source(dataset)
    if boxes_path:
        import pickle
        with open(boxes_path, 'rb') as fh:
            _W['boxes'] = pickle.load(fh)


def read_image(env, index):
    """One image of the original LMDB (image-%09d) → RGB uint8 array | None."""
    import io
    from PIL import Image
    with env.begin() as txn:
        raw = txn.get(b'image-%09d' % int(index))
    return None if raw is None else np.asarray(Image.open(io.BytesIO(raw)).convert('RGB'))


def _source(dataset):
    """The original LMDB inside a worker — opened once per dataset."""
    env = _W['sources'].get(dataset)
    if env is None:
        import lmdb
        env = _W['sources'][dataset] = lmdb.open(str(Path(_W['data_root']) / dataset), readonly=True, lock=False, readahead=False,
                                                 meminit=False)
    return env


def _prepare(indices, hyps=None, dataset=None):
    """Windows per image (`window.prepare`) — with hyps ({key: [y, …]}) and the planning of `_init`, the scoring plan
    (`plan_record`) is attached too.
    Without dataset, the default dataset of `_init`."""
    from tools.rescore import window
    dataset = dataset or _W['dataset']
    out = []
    for index in indices:
        try:
            rgb = read_image(_source(dataset), index)
            if rgb is None:
                out.append(dict(index=index, skip='missing_image')); continue
            rec = window.stored_record(dataset, index, root=_W['charout_root'])
            # The precomputed rbox was built from the geometry of the stored record (old transcript) — not used for
            # images that read a teacher-forced record; computed on the spot instead.
            # If the pickle value is not a list (an error string), it counts as rbox failure (TRW1.5).
            boxes = None if rec is None or rec.get('source') == window.FORCED else _W.get('boxes', {}).get(f"{dataset}#{index}")
            gi, bg0 = (window.paper_inputs(rgb, rec, dataset, index, root=_W['charout_root'])
                       if rec is not None and _W['stored_geometry'] else (None, False))
            got = window.prepare(rgb, rec, boxes=boxes, geometry_inputs=gi, bg0=bg0, **_W['kwargs'])
            got['index'] = index
            planning = _W.get('planning')
            if hyps is not None and planning and 'text' in got and 'windows' in got:
                key = f"{dataset}#{index}"
                try:                                                 # planning failed → no plan attached; the GPU side
                                                                     # rebuilds it and raises the same error
                    got['plan'] = plan_record(dict(got, key=key), hyps.get(key, []), planning['char_index'], planning['boundary'])
                except Exception:
                    pass
            out.append(got)
        except Exception as exc:                                     # one image's error does not stop the batch — the
                                                                     # image has no key in the result
            out.append(dict(index=index, error=f'{type(exc).__name__}: {exc}'))
    return out


def _batches(pool, indices, batch_size, worker_chunk, hyps=None, dataset=None):
    """batch_size images at a time, in input order — the next batch is submitted one ahead. With hyps · dataset, each
    task gets only the hypotheses of its images."""
    def task(part):
        if hyps is None:
            return pool.submit(_prepare, part)
        return pool.submit(_prepare, part, {k: hyps[k] for k in (f'{dataset}#{i}' for i in part) if k in hyps})

    def submit(start):
        stop = min(len(indices), start + batch_size)
        return [task(indices[i:min(i + worker_chunk, stop)]) for i in range(start, stop, worker_chunk)]
    pending = submit(0)
    try:
        for start in range(0, len(indices), batch_size):
            batch = [row for job in pending for row in job.result()]
            pending = submit(start + batch_size)
            yield batch
    finally:
        for job in pending:
            job.cancel()


def _stream(pool, jobs, batch_size, worker_chunk, hyps=None):
    """jobs = [(dataset, image indices, chunk index)] → (job ordinal, batch, whether it is the job's last batch) —
    batches are cut batch_size images at a time only within a job (the batch composition per job is the same as
    `_batches`), and the next batch is submitted one ahead even when it is the first batch of the next job (one pool for
    the whole run)."""
    order = [(j, s) for j, (_d, idx, _c) in enumerate(jobs) for s in range(0, len(idx), batch_size)]

    def task(dataset, part):
        sub = None if hyps is None else {k: hyps[k] for k in (f'{dataset}#{i}' for i in part) if k in hyps}
        return pool.submit(_prepare, part, sub, dataset)

    def submit(k):
        if k >= len(order):
            return []
        j, start = order[k]
        dataset, idx, _c = jobs[j]
        stop = min(len(idx), start + batch_size)
        return [task(dataset, idx[i:min(i + worker_chunk, stop)]) for i in range(start, stop, worker_chunk)]
    pending = submit(0)
    try:
        for k, (j, _start) in enumerate(order):
            batch = [row for job in pending for row in job.result()]
            pending = submit(k + 1)
            yield j, batch, k + 1 == len(order) or order[k + 1][0] != j
    finally:
        for job in pending:
            job.cancel()


def run(pool_rows, hyps, out, device, *, data_root=None, charout_root=None, boxes_path=None, workers=14, cpu_batch=256,
        shard=(0, 1), region='trw15', hops=1, fallback='trw15', stored_geometry=False, rotate=False, ptl='refined',
        segmentation='nearest', boundary=False, aggregation='coverage', raw=True, tf32=None, checkpoint=None):
    """Image list [(dataset, index)] → out/<dataset>__c<chunk>.json ({key: {y: V(y)}}) · existing chunks are skipped ·
    out/_done_s<i>of<n> at the end.
    The default is the new definition (refined p_tl · nearest seed · TRW · no rotation · core substring · overlap
    weights) · the earlier window definition is `**PAPER_V1_RUN`.
    stored_geometry=True first reads the p_tl · tl_4 · segmentation rows of the charout store, as the paper runs did
    (`window.paper_inputs` · earlier window definition) ·
    boxes_path is the rbox pickle of the paper runs (not used for images with a teacher-forced record) ·
    raw=True also writes the per-window raw scores (`score_windows(detail=True)`) to a <chunk>.raw.pkl.gz of the same
    name — to redo the aggregation · coefficients without a GPU.
    The CPU workers build the scoring plan (`plan_record` · edit alignment · substrings) along with the windows, so the GPU
    process runs only the encoder · decoder and the aggregation.
    There is one worker pool for the whole run, and the window preparation of the next batch overlaps the GPU scoring even
    when the dataset · 5,000-image chunk changes (`_stream`).
    tf32: matmul precision of the recognizer (None → LEGIT_AR_TF32 · default TF32) — the value actually used is recorded
    in the definition of the raw-score copy ·
    checkpoint: recognizer weights (None → `checkpoint_path`)."""
    from tools._io import local_path
    from tools.rescore import nbest, window
    data_root = Path(os.path.expanduser(data_root)) if data_root else Path(local_path('data_root'))
    charout_root = Path(os.path.expanduser(charout_root)) if charout_root else window.charout_root()
    by = OrderedDict()
    for dataset, index in pool_rows:
        by.setdefault(dataset, []).append(int(index))
    jobs = [(d, sorted(v)[c:c + CHUNK], c // CHUNK) for d, v in by.items() for c in range(0, len(v), CHUNK)]
    jobs = [j for n, j in enumerate(jobs) if n % shard[1] == shard[0]]
    model, char_index = load_reader(device, checkpoint)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if stored_geometry and (ptl != 'original' or segmentation != 'watershed'):
        raise ValueError('stored_geometry reads the geometry of the earlier window definition (original p_tl · watershed) — pass its ptl/segmentation')
    kwargs = dict(region=region, hops=hops, fallback=fallback, rotate=rotate, ptl=ptl, segmentation=segmentation)
    t0 = time.time()
    todo = []
    for dataset, idx, c in jobs:
        f = out / f"{dataset.replace('/', '__')}__c{c:03d}.json"
        if f.exists():
            print('SKIP', f.name, flush=True)
        else:
            todo.append((dataset, idx, c))

    def finish(dataset, c, res, detail, skipped):
        f = out / f"{dataset.replace('/', '__')}__c{c:03d}.json"
        if raw:                                                       # raw-score copy first — a chunk is finished when its
                                                                      # chunk JSON exists
            import gzip
            import pickle
            rf = f.with_name(f.stem + '.raw.pkl.gz'); rt = rf.with_name(rf.name + '.tmp')
            with gzip.open(rt, 'wb', compresslevel=1) as fh:
                pickle.dump(dict(definition=dict(kwargs, boundary=boundary, aggregation=aggregation, stored_geometry=stored_geometry,
                                                 matmul_tf32=nbest.matmul_tf32(device, tf32)),
                                 rows=detail), fh, protocol=5)
            rt.replace(rf)
        tmp = f.with_suffix('.tmp'); tmp.write_text(json.dumps(res, ensure_ascii=False)); tmp.replace(f)
        print(f'{dataset}__c{c:03d}: {len(res):,} images · skipped {skipped} · {time.time() - t0:.0f}s so far', flush=True)
    errors = []
    if todo:
        planning = dict(char_index=char_index, boundary=boundary)    # scoring plan also in the CPU workers
        with ProcessPoolExecutor(workers, initializer=_init,
                                 initargs=(None, data_root, charout_root, boxes_path, kwargs, stored_geometry, planning)) as ex:
            res, detail, skipped = {}, {}, 0
            for j, batch, last in _stream(ex, todo, cpu_batch, 2, hyps=hyps):
                dataset, _idx, c = todo[j]
                good = [r for r in batch if 'text' in r and 'windows' in r]
                skipped += len(batch) - len(good)
                errors += [dict(key=f"{dataset}#{r['index']}", error=r['error']) for r in batch if 'error' in r]
                for r in good:
                    r['key'] = f"{dataset}#{r['index']}"
                if good:
                    got, d = score_windows(good, hyps, model, char_index, device, boundary=boundary, aggregation=aggregation,
                                           detail=True, tf32=tf32)
                    res.update(got); detail.update(d)
                if last:                                              # last batch of this chunk — the workers build the next
                                                                      # batch while it is written
                    finish(dataset, c, res, detail, skipped)
                    res, detail, skipped = {}, {}, 0
    efile = out / f'_errors_s{shard[0]}of{shard[1]}.json'
    if errors:                    # an exception is not a skip reason: the chunks are kept, the run is not marked complete
        efile.write_text(json.dumps(errors, ensure_ascii=False, indent=1) + '\n')
    if efile.exists():            # also on a resume that skipped the finished chunks: delete their files and this list to redo them
        raise RuntimeError(f'images raised while their windows were built (e.g. {json.loads(efile.read_text())[0]}); '
                           f'not marked done — see {efile}')
    (out / f'_done_s{shard[0]}of{shard[1]}').write_text(time.strftime('%F %T') + '\n')


PAPER_V1_RUN = dict(region='trer', rotate=True, ptl='original', segmentation='watershed', boundary=True, aggregation='sum',
                    stored_geometry=True, tf32=False)   # `run(..., **PAPER_V1_RUN)` — the stored scores of the earlier window definition (fp32)


def pack_record(rec):
    """attention record → stored-format bytes (npz · maps float16) — the shape `window.unpack_record` reads."""
    import io
    buf = io.BytesIO()
    maps = np.asarray(rec['maps'], dtype=np.float16)
    np.savez(buf, text=str(rec['text']), conf=np.asarray(rec.get('conf', []), dtype=np.float32), maps=maps,
             gw=int(rec['gw']), gh=int(rec['gh']), wh0=np.asarray(rec['wh0'], dtype=np.int32), rotated=bool(rec['rotated']))
    return buf.getvalue()


def dump_records(pool_rows, texts, root, device, *, data_root=None, tf32=None, checkpoint=None):
    """For starting without a stored attention dump — builds `window.attention_record` per image from the image + a
    transcript (usually the beam's greedy a) and writes it in the stored format to root/<dataset>/svtrv2-ar.lmdb.
    Then `run(..., charout_root=root, stored_geometry=False)`.
    texts = {key: a} · images already present are skipped · tf32: recognizer matmul precision (`nbest.precision`) ·
    checkpoint: recognizer weights."""
    import lmdb
    from tools._io import local_path
    from tools.rescore import nbest, window
    data_root = Path(os.path.expanduser(data_root)) if data_root else Path(local_path('data_root'))
    model, char_index = load_reader(device, checkpoint)
    by = OrderedDict()
    for dataset, index in pool_rows:
        by.setdefault(dataset, []).append(int(index))
    for dataset, idx in by.items():
        dest = Path(root) / dataset / f'{window.READER}.lmdb'
        dest.parent.mkdir(parents=True, exist_ok=True)
        src = lmdb.open(str(Path(data_root) / dataset), readonly=True, lock=False, readahead=False, meminit=False)
        env = lmdb.open(str(dest), map_size=64 << 30, subdir=True, readahead=False)
        with env.begin() as txn:
            have = {int(k) for k in txn.cursor().iternext(values=False) if len(k) == 9 and k.isdigit()}
        rows, t0 = [], time.time()
        for n, index in enumerate(sorted(i for i in idx if i not in have)):
            key = f'{dataset}#{index}'
            rgb = read_image(src, index)
            text = texts.get(key)
            if rgb is None or not text:
                continue
            with nbest.precision(device, tf32):
                rec = window.attention_record(model, rgb, text, char_index, device)
            if rec is not None:
                rows.append((b'%09d' % index, pack_record(rec)))
            if len(rows) >= 500:
                with env.begin(write=True) as txn:
                    for k, v in rows:
                        txn.put(k, v)
                rows = []
        with env.begin(write=True) as txn:
            for k, v in rows:
                txn.put(k, v)
        env.sync(); env.close(); src.close()
        print(f'{dataset}: records for {len(idx):,} images · {time.time() - t0:.0f}s', flush=True)


def main(argv=None):
    from tools._io import read_tsv
    ap = argparse.ArgumentParser(description='Window scores V(y) of beam hypotheses, in chunks of 5,000 images per dataset')
    ap.add_argument('--pool', type=Path, required=True, help='TSV with the columns dataset and sample_idx')
    ap.add_argument('--hyps', type=Path, required=True, help='JSON {key: [y, …]} — key = "<dataset>#<sample_idx>"')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--checkpoint', type=Path, default=None, help='SVTRv2-AR checkpoint (default: LEGIT_AR_CHECKPOINT)')
    ap.add_argument('--workers', type=int, default=14)
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--boxes', type=Path, default=None,
                    help='precomputed rbox pickle {key: [corners…]} — computed on the spot when absent')
    ap.add_argument('--region', default='trw15', choices=('trer', 'trw15', 'none'))
    ap.add_argument('--hops', type=int, default=1)
    ap.add_argument('--rotate', action='store_true', help='rotate each window upright along the reading tangent (v1)')
    ap.add_argument('--ptl', default='refined', choices=('refined', 'original'))
    ap.add_argument('--segmentation', default='nearest', choices=('nearest', 'watershed'))
    ap.add_argument('--boundary', action='store_true',
                    help='window score = logsumexp over the boundary variants (up to four) (v1)')
    ap.add_argument('--aggregation', default='coverage', choices=AGGREGATIONS)
    ap.add_argument('--stored-geometry', action='store_true',
                    help='read the stored p_tl, tl_4 and segmentation of the attention store first (v1 definition)')
    ap.add_argument('--fresh-geometry', action='store_true', help='(old flag; now the default) do not read stored geometry')
    ap.add_argument('--paper-v1', action='store_true',
                    help='the whole v1 definition: TRER, rotation, original p_tl, watershed, boundary variants, sum, stored geometry')
    ap.add_argument('--no-raw', action='store_true', help='do not write the per-window raw scores (<chunk>.raw.pkl.gz)')
    ap.add_argument('--fp32', action='store_true', help='fp32 matmuls in the recognizer (the stored v1 scores; default TF32 on Ampere or newer)')
    ap.add_argument('--records', type=Path, default=None,
                    help='root of the attention records (<root>/<dataset>/svtrv2-ar.lmdb) — images without one are recorded first '
                         'from the image and the first hypothesis (greedy a); the geometry is then measured afresh')
    a = ap.parse_args(argv)
    rows = [(r['dataset'], int(r['sample_idx'])) for r in read_tsv(a.pool)]
    hyps = json.loads(a.hyps.read_text())
    definition = dict(PAPER_V1_RUN) if a.paper_v1 else dict(
        region=a.region, rotate=a.rotate, ptl=a.ptl, segmentation=a.segmentation, boundary=a.boundary, aggregation=a.aggregation,
        stored_geometry=a.stored_geometry, tf32=False if a.fp32 else None)
    if a.fp32:
        definition['tf32'] = False
    if a.records is not None:
        dump_records(rows, {k: v[0] for k, v in hyps.items() if v}, a.records, a.device, tf32=definition['tf32'],
                     checkpoint=a.checkpoint)
    if a.records is not None or a.fresh_geometry:
        definition['stored_geometry'] = False
    run(rows, hyps, a.out, a.device, charout_root=a.records, boxes_path=a.boxes, workers=a.workers,
        shard=tuple(map(int, a.shard.split('/'))), hops=a.hops, raw=not a.no_raw, checkpoint=a.checkpoint, **definition)


if __name__ == '__main__':
    main()
