"""End-to-end entry point — images → beam → windows → ByT5 PLL → decision; coefficient selection on a validation run.

    python -m tools.rescore.cli score  --dataset Union14M-L-val --keys keys.txt --out run_dev  --device cuda:0
    python -m tools.rescore.cli score  --dataset test/CUTE80 --out run_cute --device cuda:0
    python -m tools.rescore.cli select --run run_dev  --out coef.json          # grid on the validation run (CPU)
    python -m tools.rescore.cli apply  --run run_cute --coef coef.json        # outputs and accuracy (CPU)
    python -m tools.rescore.cli read   word1.png word2.jpg [--condition]        # demo on image files (GPU)

Image-conditioned extension (manuscript Section III-F, `condition`): `judge --run <run> --out judge.jsonl` asks
Qwen3.5-27B-GPTQ-Int4 (vLLM 0.28) about the images whose output can change with the condition coefficients, and
`apply --run <run> --condition judge.jsonl` decides with them; `read --condition` does both for image files.

The defaults are the setting of the manuscript: SVTRv2-AR beam 10 · 10 (width 10, 10 characters expanded per
hypothesis) with cached keys / values and TF32 matmuls on Ampere or newer GPUs, MW3 midpoint windows (`--window`;
mw1 · mw5 · mw7 · mw9 change the width, w3 is the segment window of the supplementary tables), ByT5-XL PLL in bfloat16,
and the coefficients `FINAL` (window α = omega 0.4, length β = beta 1, ByT5 γ = mu 0.1, τ = tau 0). `--v1-recognizer` is
the recognizer of the first manuscript version (beam 20 · 20, no cache, fp32), and --beam-width / --token-topk / --no-kv /
--fp32 change one part. The setting is stored in the run protocol, so runs of different settings never mix.

`score` writes one JSON line per image to <out>/scores.jsonl: the greedy transcript a with its log P, the beam top 10,
the window score of a and of every hypothesis, and the ByT5-XL PLL of every string — everything the decision needs,
so `select` and `apply` run without a GPU. Without --coef, `apply` uses the coefficients of the manuscript (`FINAL`,
chosen on the 10,000-image validation set). Reading clusters are off; a coefficient file with "clusters": true (earlier
results) applies them.

Main score path captures attention during the original greedy pass, overlaps a bounded CPU queue with GPU work,
and batches PLL across at least --pll-images images (default 256; whole AR batches retained). Exact strings already
scored in this run are read from PLL cache; ByT5 loads only for misses. These batch changes may change floating-point
scores, so the protocol is versioned and old runs are never silently resumed under the new definition.

CPU-heavy preparation can be done on a separate server without any STR/ByT5 model:
    score --dataset ... --out run --stage beam             # GPU: export inputs/ + protocol.json + inputs_complete.json
    prepare --run run --workers 8                          # CPU server: prepare portable inputs, write prepared/
    score --dataset ... --out run --stage finish --cache-only  # GPU: consume prepared/, no geometry on this host
Copy completed bundles together with their protocol.json. Partial inputs may be consumed
but are reported as partial_inputs until the export manifest is fully satisfied. Missing caches are computed only when
--cache-only is absent. --legacy-attention and --pll-images 48 --no-pll-cache retain diagnostic reference paths.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tools.rescore import hypotheses as H
from tools.rescore.decide import Batch, window_delta, with_window
from tools.rescore.metrics import accuracy
from tools.rescore.pipeline import WINDOW, WINDOWS

# coefficients of the manuscript (code name = paper symbol: omega = α window, beta = β length, mu = γ ByT5-XL, tau = τ)
FINAL = dict(mu=0.1, beta=1.0, tau=0.0, omega=0.4, clusters=False)


def _images(dataset, keys=None, limit=0):
    from tools._io import open_image, open_lmdb, read_sample
    env = open_lmdb(dataset)
    with env.begin() as t:
        n = int(t.get(b'num-samples').decode())
    idx = [int(k.rsplit('#', 1)[1]) for k in keys] if keys else range(1, n + 1)
    for c, i in enumerate(idx):
        if limit and c >= limit:
            break
        raw, label = read_sample(env, i)
        if raw is not None:
            yield f'{dataset}#{i}', open_image(raw).convert('RGB'), label


def score(dataset, out, device='cuda:0', keys=None, limit=0, batch=48, checkpoint=None, workers=8, **kwargs):
    """Main production path: shared attention, portable CPU cache, bounded overlap, cached/batched PLL."""
    from tools.rescore.pipeline import score as run
    return run(dataset, out, device, keys, limit, batch, checkpoint, workers, **kwargs)


def load_run(run):
    """scores.jsonl → (items with ΔW, PLL table)."""
    protocol_path = Path(run) / 'protocol.json'
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        if protocol.get('candidate_k', H.K) != H.K:
            raise ValueError('PAPER_K differs from the scored run; additional candidates have not been scored')
    rows = [json.loads(line) for line in open(Path(run) / 'scores.jsonl')]
    C = {r['key']: r for r in rows}
    gt = {r['key']: r['gt'] for r in rows}
    items = H.ar_items(C, gt)
    V = {r['key']: r['window'] for r in rows}
    pll = {y: v for r in rows for y, v in r['pll'].items()}
    return with_window(items, window_delta(V, items)), pll


def select(run, out):
    from tools.rescore.select import select_counted
    items, pll = load_run(run)
    coef, n = select_counted(Batch(items), pll)
    res = dict(coef, clusters=False, validation_correct=n, images=len(items))
    Path(out).write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps(res), flush=True)


def apply(run, coef=None, out=None, condition=None):
    """Outputs of a scored run. condition: judge answers {key: (p_O, p_H, p_S)} (`judge`) — the image-conditioned
    extension with the coefficients of the manuscript instead of `coef`."""
    items, pll = load_run(run)
    B = Batch(items)
    routes = None
    if condition is not None:
        from tools.rescore import condition as CD
        pred, routes = CD.decide(B, pll, condition)
        missing = routes.count('missing')
        if missing:
            print(f'{missing:,} images need the judge but have no answer — they keep the fixed output', flush=True)
    else:
        coef = coef or FINAL
        X = B.merged() if coef.get('clusters', False) else B
        pred = X.decide(pll, coef['mu'], coef['beta'], coef['tau'], coef['omega'])
    a0, a1 = accuracy(B, B.base()), accuracy(B, pred)
    print(f"greedy {a0['correct']:,}/{a0['n']:,} · re-read {a1['correct']:,}/{a1['n']:,} · per set {a1['per']}", flush=True)
    if out:
        with open(out, 'w') as fh:
            for i, ((k, a, g, _), y) in enumerate(zip(items, pred)):
                row = dict(key=k, greedy=a, output=y, gt=g)
                if routes is not None:
                    row['route'] = routes[i]
                fh.write(json.dumps(row, ensure_ascii=False) + '\n')
    return pred


def _read_key(key):
    """RGB array of a `<dataset>#<index>` key from the LMDB benchmarks (STR_DATA_ROOT)."""
    import numpy as np
    from tools._io import open_image, open_lmdb, read_sample
    dataset, index = key.rsplit('#', 1)
    raw, _ = read_sample(open_lmdb(dataset), int(index))
    return np.asarray(open_image(raw).convert('RGB'))


def judge(run, out, model=None, gpu_memory_utilization=.8):
    """Image-conditioned extension, GPU: ask the judge about the images of a scored run that need it (`condition.needs_judge`)
    and write {key, p: [p_O, p_H, p_S]} lines to out (resumes by key)."""
    from tools.rescore import condition as CD
    items, pll = load_run(run)
    need = CD.needs_judge(Batch(items), pll)
    keys = [k for (k, *_), n in zip(items, need) if n]
    done = {json.loads(line)['key'] for line in open(out)} if Path(out).exists() else set()
    todo = [k for k in keys if k not in done]
    print(f'{len(keys):,} of {len(items):,} images need the judge; {len(todo):,} to ask', flush=True)
    if not todo:
        return
    J = CD.Judge(model or CD.MODEL, gpu_memory_utilization=gpu_memory_utilization)
    with open(out, 'a') as fh:
        for b in range(0, len(todo), 96):
            part = todo[b:b + 96]
            for k, p in J.ask({k: _read_key(k) for k in part}).items():
                fh.write(json.dumps(dict(key=k, p=None if p is None else list(p))) + '\n')
            fh.flush()


def load_judge(path):
    return {r['key']: None if r['p'] is None else tuple(r['p']) for r in map(json.loads, open(path))}


def read(paths, device='cuda:0', coef=None, condition=False, judge_model=None, window=WINDOW, checkpoint=None):
    """Demo on image files: greedy transcript, beam, windows, ByT5-XL PLL and the decision for each image, in memory
    (no run directory). condition: also the image-conditioned extension (loads Qwen3.5-27B-GPTQ-Int4 when an image needs
    it). Returns one dict per image."""
    from PIL import Image
    from tools.rescore import lm, nbest, window_score
    from tools.rescore.pipeline import decode_input, prepare_windows
    recognizer = nbest.setting(device=device)
    model, char_index = window_score.load_reader(device, checkpoint)
    part = [(str(p), Image.open(p).convert('RGB'), '') for p in paths]
    rows = decode_input(part, model, char_index, device, recognizer=recognizer)['rows']
    records = []
    for r in rows:
        got = prepare_windows(r['rgb'], r['attention'], window) if r['attention'] is not None else dict(skip='no_record')
        rec = dict(key=r['key'], **got)
        if 'windows' in got:
            rec['plan'] = window_score.plan_record(rec, r['hyps'], char_index)
            records.append(rec)
    V = window_score.score_windows(records, {r['key']: r['hyps'] for r in rows}, model, char_index, device,
                                   tf32=recognizer['tf32_requested']) if records else {}
    texts = sorted({y for r in rows for y in [r['candidate']['ar']] + r['hyps'] if y and all(c in char_index for c in y)})
    scorer = lm.PLL('byt5-xl', device)
    pll = scorer.table(texts)
    items = H.ar_items({r['key']: r['candidate'] for r in rows}, {r['key']: '' for r in rows})
    B = Batch(with_window(items, window_delta(V, items)))
    coef = coef or FINAL
    out = list(B.decide(pll, coef['mu'], coef['beta'], coef['tau'], coef['omega']))
    routes = [None] * len(out)
    if condition:
        from tools.rescore import condition as CD
        need = CD.needs_judge(B, pll)
        probs = {}
        if need.any():
            import gc
            import torch
            del scorer, model
            gc.collect(); torch.cuda.empty_cache()
            probs = CD.Judge(judge_model or CD.MODEL).ask({r['key']: r['rgb'] for r, n in zip(rows, need) if n})
        out, routes = CD.decide(B, pll, probs)
    res = []
    for (k, a, _, cands), y, route in zip(B.items, out, routes):
        res.append(dict(image=k, greedy=a, output=y, hypotheses=[c[0] for c in cands], **({'route': route} if condition else {})))
        print(json.dumps(res[-1], ensure_ascii=False), flush=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('score')
    s.add_argument('--dataset', required=True)
    s.add_argument('--keys', type=Path, help='file with one key <dataset>#<index> per line')
    s.add_argument('--limit', type=int, default=0)
    s.add_argument('--out', type=Path, required=True)
    s.add_argument('--device', default='cuda:0')
    s.add_argument('--checkpoint', type=Path)
    s.add_argument('--workers', type=int, default=8, help='processes for the window construction (CPU)')
    s.add_argument('--batch', type=int, default=48, help='AR image batch; stored in the run protocol')
    s.add_argument('--pll-images', type=int, default=256, help='minimum images grouped per PLL call; whole AR batches retained')
    s.add_argument('--stage', choices=('all', 'beam', 'finish'), default='all',
                   help='beam exports RGB/attention/candidates; finish consumes them; all streams both')
    s.add_argument('--prepared-cache', type=Path, help='prepared bundles copied from a CPU server; matching entries reused')
    s.add_argument('--cache-only', action='store_true', help='fail on missing CPU cache instead of computing on GPU host')
    s.add_argument('--legacy-attention', action='store_true', help='diagnostic: repeat old per-image attention inference')
    s.add_argument('--no-pll-cache', action='store_true', help='diagnostic: recompute all PLL strings within each group')
    s.add_argument('--beam-width', type=int, help='AR beam width (default 10)')
    s.add_argument('--token-topk', type=int, help='characters expanded per live hypothesis (default 10)')
    s.add_argument('--no-kv', action='store_true', help='re-run the whole prefix at every decoding step (default: cached)')
    s.add_argument('--fp32', action='store_true', help='fp32 matmuls in the recognizer (default: TF32 on Ampere or newer)')
    s.add_argument('--v1-recognizer', '--paper-recognizer', dest='v1_recognizer', action='store_true',
                   help='the first manuscript version: beam 20 · 20, no cache, fp32 (single options above still apply)')
    s.add_argument('--window', choices=WINDOWS, default=WINDOW,
                   help='character windows: MW3 midpoint windows of the manuscript (default), other widths, or W3 segments')
    sp = sub.add_parser('prepare', help='CPU-only window preparation; no GPU or model loading')
    sp.add_argument('--run', type=Path, required=True, help='run directory containing inputs/ from score --stage beam')
    sp.add_argument('--prepared-cache', type=Path)
    sp.add_argument('--workers', type=int, default=8)
    s2 = sub.add_parser('select')
    s2.add_argument('--run', type=Path, required=True)
    s2.add_argument('--out', type=Path, required=True)
    s3 = sub.add_parser('apply')
    s3.add_argument('--run', type=Path, required=True)
    s3.add_argument('--coef', type=Path)
    s3.add_argument('--condition', type=Path, help='judge answers of `judge` — image-conditioned extension (instead of --coef)')
    s3.add_argument('--out', type=Path)
    s4 = sub.add_parser('judge', help='image-conditioned extension: Qwen3.5-27B-GPTQ-Int4 answers for the images that need them (GPU)')
    s4.add_argument('--run', type=Path, required=True)
    s4.add_argument('--out', type=Path, required=True)
    s4.add_argument('--model', help='local path or name of Qwen3.5-27B-GPTQ-Int4 (default: the Hugging Face name)')
    s4.add_argument('--gpu-memory-utilization', type=float, default=.8)
    s5 = sub.add_parser('read', help='demo: read image files end to end and print the outputs (GPU)')
    s5.add_argument('images', type=Path, nargs='+')
    s5.add_argument('--device', default='cuda:0')
    s5.add_argument('--checkpoint', type=Path)
    s5.add_argument('--window', choices=WINDOWS, default=WINDOW)
    s5.add_argument('--condition', action='store_true', help='also the image-conditioned extension (loads the judge if needed)')
    s5.add_argument('--judge-model', help='local path or name of Qwen3.5-27B-GPTQ-Int4')
    a = ap.parse_args(argv)
    if a.cmd == 'score':
        from tools.rescore.nbest import PAPER
        keys = a.keys.read_text().split() if a.keys else None
        paper = PAPER if a.v1_recognizer else {}
        score(a.dataset, a.out, a.device, keys, a.limit, batch=a.batch, checkpoint=a.checkpoint, workers=a.workers,
              stage=a.stage, prepared_cache=a.prepared_cache, cache_only=a.cache_only, pll_images=a.pll_images,
              legacy_attention=a.legacy_attention, no_pll_cache=a.no_pll_cache,
              beam_width=a.beam_width or paper.get('width'), token_topk=a.token_topk or paper.get('token_topk'),
              kv=False if a.no_kv else paper.get('kv'), tf32=False if a.fp32 else paper.get('tf32'), window=a.window)
    elif a.cmd == 'prepare':
        from tools.rescore.pipeline import prepare_run
        prepare_run(a.run, a.prepared_cache, a.workers)
    elif a.cmd == 'select':
        select(a.run, a.out)
    elif a.cmd == 'judge':
        judge(a.run, a.out, a.model, a.gpu_memory_utilization)
    elif a.cmd == 'read':
        read(a.images, a.device, None, a.condition, a.judge_model, a.window, a.checkpoint)
    else:
        apply(a.run, json.loads(a.coef.read_text()) if a.coef else None, a.out,
              load_judge(a.condition) if a.condition else None)


if __name__ == '__main__':
    sys.exit(main())
