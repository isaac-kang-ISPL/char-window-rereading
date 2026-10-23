#!/usr/bin/env python3
"""Accuracy values of the SVTRv2-AR tables of the manuscript, recomputed from the stored inputs.

Inputs (data root `LEGIT_DATA`): SVTRv2-AR beam lists (beam 10 · 10: the greedy transcript a and the top 10 hypotheses),
the ByT5-XL PLL tables, the window scores of the MW3 windows, and for the image-conditioned table the stored answers of the
judge (Qwen3.5-27B-GPTQ-Int4). Word accuracy follows normalize_36_paper (ASCII letters and digits, case-insensitive); CB
and UB are macro averages over their 6 and 7 sets. Coefficients: α window, β length, γ ByT5-XL, τ margin of the decision
rule S(y) > τ.

  table_main       the main table and the rows of our evaluation in the comparison with published methods: α, β, γ, τ
                   selected once on the 10,000 validation images (Union14M-L-val, natdev10k_v2: images identical to an
                   evaluation image left out), then five rows on CB + UB (414,482 images) — the greedy transcript a, Beam
                   (beam top 1), Beam + re-read (γ = 0), Beam + ByT5 (α = 0) and Beam + re-read + ByT5 — with per-set
                   accuracy, CB / UB averages, and the images each rescored row corrects / breaks against Beam
  table_condition  the image-conditioned table on CB, UB and the five specialized sets (WOST, HOST: weak / heavy OST;
                   ORT val; GNHK test; CHAOS): fixed coefficients (the rows of the main table, and the single tuple
                   selected on all development data of the extension), each condition branch alone, all three branches
                   with the stored judge answers (`tools.rescore.condition`), and routing by the dataset name

    python -m tools.experiments.rescore.reproduce --out <dir> --only table_main,table_condition
"""
from __future__ import annotations

import gzip
import hashlib
import json

import numpy as np

from tools._io import normalize_paper as NP
from tools.experiments.rescore import common as C_
from tools.rescore import condition as CO, hypotheses as H, store
from tools.rescore import select as S
from tools.rescore.decide import Batch, batch, window_delta, with_window
from tools.rescore.metrics import accuracy, correct

# public paths under the data root (`store.at` maps them to the research folders in the research tree)
MAIN = 'reference/paper_tables/main.json'
CONDITION = 'reference/paper_tables/condition.json'
ANSWERS = 'judge/answers.jsonl.gz'
EXTRA_BEAMS = 'beams/extra-sets'                                 # beam lists of the extra sets, with their PLL
EXTRA_WINDOWS = 'windows/extra-sets'                             # MW3 window scores of the extra sets
WTAG = 'trwnn'                                                   # window tables of the MW3 windows (validation, CB + UB)
SPECIAL = (('WOST', 'OST/weak'), ('HOST', 'OST/heavy'), ('ORT', 'ort_val_occluded'), ('GNHK', 'gnhk_test'),
           ('CHAOS', 'chaos_bench_words'))
SET_NAMES = {'test/IIIT5k': 'IIIT5K', 'test/SVT': 'SVT', 'test/IC13_857': 'IC13', 'test/IC15_1811': 'IC15',
             'test/SVTP': 'SVTP', 'test/CUTE80': 'CUTE80', 'u14m/curve': 'Curve', 'u14m/multi_oriented': 'Multi-Oriented',
             'u14m/artistic': 'Artistic', 'u14m/contextless': 'Contextless', 'u14m/salient': 'Salient',
             'u14m/multi_words': 'Multi-Words', 'u14m/general': 'General'}
# the single (α, β, γ) selected on all development data of the image-conditioned extension (natural validation images,
# synthetic occlusion, GNHK train, synthetic spelling — not in the archive) with the objective of its branch thresholds
GLOBAL = (.25, 1., .1)
# routing by the dataset name, which the extension does not see: the condition whose coefficients each set takes
BY_NAME = {'WOST': 'occlusion', 'HOST': 'occlusion', 'ORT': 'occlusion', 'GNHK': 'handwriting', 'CHAOS': 'spelling'}
# what each result file holds (written into it)
MAIN_RULE = ('alpha (window), beta (length), gamma (ByT5-XL) and tau (margin) selected once on the validation images by '
             'the number of correct images (ties: smaller alpha + gamma, then grid order) and applied to CB + UB; '
             'Beam + re-read and Beam + ByT5 keep the selected coefficients with the other term at 0; Beam is the beam '
             'top 1; vs_beam counts the images a row corrects (Beam wrong, row right) and breaks (Beam right, row wrong)')
CONDITION_RULE = ('fixed rows: one coefficient tuple for every image (tau 0); branch rows: tools.rescore.condition with '
                  'the stored judge answers (an image whose output no condition can change keeps the fixed output, '
                  'route skipped); routing by dataset name: the condition coefficients by the name of the set, the fixed '
                  'combination on CB and UB')


def coefficients(alpha, beta, gamma, tau=0.):
    return dict(alpha=alpha, beta=beta, gamma=gamma, tau=tau)


def _decide(B, pll, c):
    return B.decide(pll, c['gamma'], c['beta'], c['tau'], c['alpha'])      # Batch.decide(pll, mu=γ, beta=β, tau=τ, omega=α)


def _parts(items):
    ds = np.array([k.rsplit('#', 1)[0] for k, *_ in items])
    return {'CB': np.isin(ds, store.CB), 'UB': np.isin(ds, store.UB)}


def table_main():
    """Coefficients on the validation set, then the five rows of the main table on CB + UB."""
    pll = C_.pll()
    dev, full = C_.ar('natdev'), C_.ar('full')
    D = dev.with_window(C_.window_delta(f'{WTAG}_natdev', dev))
    T = full.with_window(C_.window_delta(f'{WTAG}_full', full))
    sel, n_dev = S.select_counted(D, pll)
    F = coefficients(sel['omega'], sel['beta'], sel['mu'], sel['tau'])
    spec = (('Beam', coefficients(0., 0., 0.)), ('Beam + re-read', dict(F, gamma=0.)), ('Beam + ByT5', dict(F, alpha=0.)),
            ('Beam + re-read + ByT5', F))
    part = _parts(full.items)
    rows, ok = {}, {}
    for name, c in (('Greedy', None),) + spec:
        pred = full.base() if c is None else _decide(T, pll, c)
        a = accuracy(full, pred)
        ok[name] = correct(full, pred)
        rows[name] = dict(coefficients=c, CB=a['cb'], UB=a['ub'], correct=a['correct'],
                          sets={d: a['per'][d] for d in store.CB + store.UB})
        if c is not None and name != 'Beam':
            rows[name]['vs_beam'] = {g: dict(corrected=int((ok[name] & ~ok['Beam'] & m).sum()),
                                             broken=int((~ok[name] & ok['Beam'] & m).sum())) for g, m in part.items()}
        print(name, c, round(a['cb'], 2), round(a['ub'], 2), a['correct'], flush=True)
    keys = ''.join(f'{k}\n' for k, *_ in dev.items)                  # the bytes of benchmarks/validation-keys.txt
    res = dict(rule=MAIN_RULE,
               validation=dict(images=len(dev), keys_sha256=hashlib.sha256(keys.encode()).hexdigest(), correct_selected=n_dev,
                               grid=dict(alpha=list(S.GRID['omega']), beta=list(S.GRID['beta']), gamma=list(S.GRID['mu']),
                                         tau=list(S.GRID['tau']))),
               evaluation=dict(CB=list(store.CB), UB=list(store.UB), images=len(full), set_names=SET_NAMES),
               coefficients=F, rows=rows)
    C_.dump(MAIN, res)
    return res


# ---------------------------------------------------------------- the image-conditioned table

def answers():
    with gzip.open(store.at(ANSWERS), 'rt') as f:
        return {r['key']: tuple(r['p']) for r in map(json.loads, f)}


def _cbub():
    items = H.get('svtrv2-ar', 'full')
    items = with_window(items, window_delta(store.window_table(f'{WTAG}_full'), items))
    return [batch(items)], [store.pll_table()]


def _ost(ds):
    cands = store.ar_beams('newsets')
    gt = {k: c['gt'] for k, c in cands.items() if k.startswith(ds + '#')}
    items = H.ar_items({k: v for k, v in cands.items() if k in gt}, gt)
    items = with_window(items, window_delta(store.window_table(f'newsets_{WTAG}'), items))
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


def condition_sets():
    """{label: (batches, PLL tables)} — CB + UB, then the five specialized sets."""
    sets = {'CB+UB': _cbub()}
    for label, ds in SPECIAL:
        sets[label] = _ost(ds) if ds.startswith('OST/') else _extra(ds)
    return sets


def _score(label, outs, items):
    """Accuracy of one set: CB and UB macro averages for 'CB+UB', else correct / images / accuracy."""
    ok = np.array([NP(y) == NP(g) for y, (_, _, g, _) in zip(outs, items)])
    if label != 'CB+UB':
        return {label: dict(correct=int(ok.sum()), images=len(ok), accuracy=float(ok.mean() * 100))}
    ds = np.array([k.rsplit('#', 1)[0] for k, *_ in items])
    return {part: float(np.mean([ok[ds == d].mean() * 100 for d in names])) for part, names in (('CB', store.CB), ('UB', store.UB))}


def table_condition():
    """Rows of the image-conditioned table: fixed coefficients, judge-routed branches, routing by the dataset name."""
    probs = answers()
    sets = condition_sets()
    alpha, beta, gamma = CO.FIXED
    fixed = {'Beam': coefficients(0., 0., 0.), 'Beam + ByT5': coefficients(0., beta, gamma),
             'Beam + re-read': coefficients(alpha, beta, 0.), 'Fixed combination': coefficients(*CO.FIXED),
             'Fixed, all development data': coefficients(*GLOBAL)}
    branch = {name: coefficients(*c) for name, _, c in CO.ALL_THREE}
    rules = {'Handwriting branch only': CO.SINGLE['handwriting'], 'Occlusion branch only': CO.SINGLE['occlusion'],
             'Spelling branch only': CO.SINGLE['spelling'], 'All three branches': CO.ALL_THREE}
    order = ['Greedy', 'Beam', 'Beam + ByT5', 'Beam + re-read', 'Fixed combination', 'Fixed, all development data', *rules,
             'Routing by dataset name']
    rows = {name: {} for name in order}
    images = {}
    for label, (batches, plls) in sets.items():
        items = [it for B in batches for it in B.items]
        if label == 'CB+UB':
            part = _parts(items)
            images.update(CB=int(part['CB'].sum()), UB=int(part['UB'].sum()))
        else:
            images[label] = len(items)
        out = {k: [] for k in ('Greedy', *fixed, *branch)}
        final = {k: [] for k in rules}
        routes = {k: [] for k in rules}
        for B, pl in zip(batches, plls):
            out['Greedy'] += [a for _, a, _, _ in B.items]
            for name, c in {**fixed, **branch}.items():
                out[name] += list(_decide(B, pl, c))
            for name, rule in rules.items():
                f, r = CO.decide(B, pl, probs, rule, out={'fixed': out['Fixed combination'][-len(B):],
                                                          **{n: out[n][-len(B):] for n, _, _ in rule}})
                final[name] += f
                routes[name] += r
        by_name = out['Fixed combination'] if label == 'CB+UB' else out[BY_NAME[label]]
        for name in order:
            outs = final[name] if name in rules else (by_name if name == 'Routing by dataset name' else out[name])
            rows[name].update(_score(label, outs, items))
            if name in rules:
                rows[name].setdefault('routes', {})[label] = {x: routes[name].count(x) for x in sorted(set(routes[name]))}
    for name in order:
        c = (None if name == 'Greedy' else fixed.get(name) or
             ({'rule': [dict(condition=n, threshold=t, coefficients=coefficients(*cf)) for n, t, cf in rules[name]]}
              if name in rules else dict(CB_UB=coefficients(*CO.FIXED), **{s: branch[b] for s, b in BY_NAME.items()})))
        rows[name] = dict(coefficients=c, **rows[name])
        print(name, {k: round(v, 2) if isinstance(v, float) else round(v['accuracy'], 2) for k, v in rows[name].items()
                     if k not in ('coefficients', 'routes')}, flush=True)
    res = dict(rule=CONDITION_RULE, judge_answers=len(probs),
               answers_file=ANSWERS, sets=dict(SPECIAL), images=images, rows=rows)
    C_.dump(CONDITION, res)
    return res


REPRODUCE = (
    ('table_main', 'table_main', {}, (), MAIN),
    ('table_condition', 'table_condition', {}, (), CONDITION),
)

if __name__ == '__main__':
    table_main()
    table_condition()
