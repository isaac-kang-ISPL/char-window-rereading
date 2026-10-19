"""Image-conditioned rescoring of SVTRv2-AR (manuscript Section III-F, Table condition) from the stored inputs and the stored
judge answers, with `tools.rescore.condition`.

Candidates and scores (data root `LEGIT_DATA`, MW3 midpoint windows, beam 10 · 10):
  CB + UB         the evaluation lists of Table main, window table `trwnn_full`, the ByT5-XL PLL table
  OST weak/heavy  the new-benchmark run (`newsets` lists, window table `newsets_trwnn`, its PLL folder)
  ORT, GNHK, CHAOS  the extra-set runs: `beams/extra-sets/<set>/scores.jsonl` (beam lists with their PLL) and the MW3
                  window scores `windows/extra-sets/<set>/c*.json`; one batch per image
Judge answers: `judge/answers.jsonl.gz` — the Qwen3.5-27B-GPTQ-Int4 joint answers (p_O, p_H, p_S; null
for an invalid answer) of every image whose output can change with the condition coefficients (the other images keep the
fixed output whatever the answers are). Rules: all three branches (H -> O -> S, thresholds .75 / .10 / .65) and each branch
alone (.75 / .10 / .50), as in `condition`. Accuracy is normalize_36_paper; CB and UB are macro averages over their sets.

    python -m tools.experiments.rescore.reproduce --out <dir> --only condition
"""
from __future__ import annotations

import gzip
import json

import numpy as np

from tools._io import normalize_paper as NP
from tools.experiments.rescore import common as C_
from tools.rescore import condition as C, hypotheses as H, store
from tools.rescore.decide import Batch, batch, window_delta, with_window

# public paths under the data root (`store.at` maps them to the research folders in the research tree)
RESULT = 'reference/condition_table/results.json'
ANSWERS = 'judge/answers.jsonl.gz'
EXTRA_BEAMS = 'beams/extra-sets'                                 # beam lists of the extra sets, with their PLL
EXTRA_WINDOWS = 'windows/extra-sets'                             # MW3 window scores of the extra sets
SPECIAL = (('WOST', 'OST/weak'), ('HOST', 'OST/heavy'), ('ORT', 'ort_val_occluded'), ('GNHK', 'gnhk_test'),
           ('CHAOS', 'chaos_bench_words'))
RULES = {'all_three': C.ALL_THREE, 'handwriting_only': C.SINGLE['handwriting'], 'occlusion_only': C.SINGLE['occlusion'],
         'spelling_only': C.SINGLE['spelling']}


def answers():
    with gzip.open(store.at(ANSWERS), 'rt') as f:
        return {r['key']: tuple(r['p']) for r in map(json.loads, f)}


def _cbub():
    items = H.get('svtrv2-ar', 'full')
    items = with_window(items, window_delta(store.window_table('trwnn_full'), items))
    return [batch(items)], [store.pll_table()]


def _ost(ds):
    cands = store.ar_beams('newsets')
    gt = {k: c['gt'] for k, c in cands.items() if k.startswith(ds + '#')}
    items = H.ar_items({k: v for k, v in cands.items() if k in gt}, gt)
    items = with_window(items, window_delta(store.window_table('newsets_trwnn'), items))
    return [batch(items)], [store.pll_folder(store.NEWSETS / 'pll')]


def _extra(ds):
    safe = ds.replace('/', '__')
    W = {}
    for p in sorted(store.at(f'{EXTRA_WINDOWS}/{safe}').glob('c*.json')):
        W.update(json.loads(p.read_text()))
    batches, plls = [], []
    with open(store.at(f'{EXTRA_BEAMS}/{safe}/scores.jsonl')) as f:
        for line in f:
            r = json.loads(line)
            if not NP(r['gt']):
                continue
            a, s0 = r['ar'], r['ar_lp']
            it = [(r['key'], a, r['gt'], [(c['y'], c['ar'] - s0, len(c['y']) - len(a), None) for c in r['cands']
                                          if c['nb'] < H.K and c['y'] != a])]
            batches.append(Batch(with_window(it, window_delta(W, it))))
            plls.append(r['pll'])
    return batches, plls


def table():
    probs = answers()
    sets = {'CB+UB': _cbub()}
    for label, ds in SPECIAL:
        sets[label] = _ost(ds) if ds.startswith('OST/') else _extra(ds)
    rows = {}
    for name, rule in RULES.items():
        row = {}
        for label, (batches, plls) in sets.items():
            final, routes, gts, ks = [], [], [], []
            for B, pl in zip(batches, plls):
                f, r = C.decide(B, pl, probs, rule)
                final += f; routes += r
                gts += [g for _, _, g, _ in B.items]; ks += [k for k, *_ in B.items]
            ok = np.array([NP(y) == NP(g) for y, g in zip(final, gts)])
            route = {x: routes.count(x) for x in sorted(set(routes))}
            if label == 'CB+UB':
                ds = np.array([k.rsplit('#', 1)[0] for k in ks])
                for part, names in (('CB', store.CB), ('UB', store.UB)):
                    row[part] = dict(macro=float(np.mean([ok[ds == d].mean() * 100 for d in names])),
                                     images=int(sum((ds == d).sum() for d in names)))
                row['routes CB+UB'] = route
            else:
                row[label] = dict(correct=int(ok.sum()), images=len(ok), accuracy=float(ok.mean() * 100), routes=route)
        rows[name] = row
    res = dict(rule=__doc__.split('\n\n')[0], judge_answers=len(probs), answers_file=ANSWERS,
               rules={n: [list(x) for x in r] for n, r in RULES.items()}, rows=rows)
    C_.dump(RESULT, res)
    return res


REPRODUCE = (
    ('condition', 'table', {}, (), RESULT),
)

if __name__ == '__main__':
    table()
