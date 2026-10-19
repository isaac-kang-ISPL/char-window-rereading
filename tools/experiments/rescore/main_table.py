#!/usr/bin/env python3
"""SVTRv2-AR main table — configurations, rows with terms dropped, neighbor range, window construction, LM swap.

Inputs: SVTRv2-AR beam lists (width 20, 20 expansions, top 10), ByT5-XL PLL (bf16, one run, case-insensitive
position terms; difference 0 when a or y has fewer than two characters), window scores of the default windows
(`tools.rescore.window` defaults: refined p_tl, nearest-seed segments, band of 1.5 times the x-height, neighbors
judged in the TR band, no rotation; one core substring per window; coverage-weighted sum) under the tag WTAG
(default trwnn). No reading clusters. Coefficients on natdev, applied once to CB + UB (414,482 images).

    python -m tools.experiments.rescore.main_table [main|ablation|wsize|wconstruct|lmswap]
    WTAG=trwnn_w1 python -m tools.experiments.rescore.main_table ablation      (other window tags)
Results: reference/ar_trer_eval/{results, ablation[_<tag>], wsize, wconstruct, lmswap}.json
"""
from __future__ import annotations

import os
import sys

from tools.experiments.rescore import common as C
from tools.rescore import ablation as AB
from tools.rescore import select as S
from tools.rescore import store
from tools.rescore.decide import CONTEXT_MIN
from tools.rescore.metrics import accuracy, groups, pair

RES = 'reference/ar_trer_eval'
ARMS = ('빔 1위', '+ ByT5', '+ W 창', '+ ByT5 + W 창')
DEFAULT = 'trwnn'
WINDOW = '개선 p_tl · 가장 가까운 씨앗 분할 · TRW 띠(x-height 1.5배) · 이웃 TR · 회전 없음 · 핵심 부분 문자열 · 겹침 가중'
PLL_NAME = 'pll_variants V1 · ByT5-XL bf16 · 한 회차 · ci'


def _inputs(wtag):
    pll = C.pll()
    dev, full = C.ar('natdev'), C.ar('full')
    Wd = C.window_delta(f'{wtag}_natdev', dev)
    Wt = C.window_delta(f'{wtag}_full', full)
    return pll, dev, full, Wd, Wt


def main():
    """Six configurations, each with its own coefficients on natdev, and the ILM rows chosen on natdev."""
    wtag = os.environ.get('WTAG', DEFAULT)
    pll, dev, full, Wd, Wt = _inputs(wtag)
    venn = store.venn_keys()
    vi = [k for k, it in enumerate(full.items) if it[0] in venn]
    test = [full.items[k] for k in vi]
    D, T = dev.with_window(Wd), full.with_window(Wt)
    base = full.base()
    a0 = accuracy(full, base)
    cover = sum(x is not None for *_, c in D.items for *_, x in c) / max(1, sum(len(c) for *_, c in D.items))
    res = dict(settings=dict(pll=PLL_NAME, grid=S.GRID, grid_name=os.environ.get('LEGIT_COEF_GRID', '안 B · 위로 넓힘(α ≤ 1.6 · γ ≤ 3.2)'),
                             same_answer='모든 규칙에서 남김', window=WINDOW, clusters=False,
                             context_min=f'{CONTEXT_MIN} 글자 미만이면 ΔPLL 0',
                             w_tags=[f'{wtag}_natdev', f'{wtag}_full'], n_dev=len(dev), n_test=len(full), n_venn=len(test),
                             w_cover=dict(dev=cover)),
               base=dict(cb=a0['cb'], ub=a0['ub'], correct=a0['correct']), rows={})
    preds = {}
    for arm in ARMS:
        Dx, Tx = D, T
        prm = S.select_arm(Dx, pll, arm)
        pr = Tx.decide(pll, prm['mu'], prm['beta'], prm['tau'], prm['omega'])
        a = accuracy(full, pr)
        preds[arm] = pr
        venn_correct = accuracy(test, [pr[k] for k in vi])['correct']
        res['rows'][arm] = dict(params=prm, cb=a['cb'], ub=a['ub'], correct=a['correct'], venn10k_correct=venn_correct,
                                vs_base=pair(full, base, pr), groups=groups(full, pr, base))
        print(arm, prm, round(a['cb'], 2), round(a['ub'], 2), a['correct'], venn_correct, flush=True)
    ilm = store.ilm_table('svtrv2-ar')
    for name, mus in (('− ILM', (0.,)), ('− ILM + ByT5', S.GRID['mu'])):
        best = None
        for lam in S.LAM:
            for mu in mus:
                for beta in S.GRID['beta']:
                    for tau in S.GRID['tau']:
                        c = int(dev.decide_ilm(pll, ilm, lam, mu, beta, tau).ok.sum())
                        if best is None or (c, -lam - mu) > best[0]:
                            best = ((c, -lam - mu), dict(lam=lam, mu=mu, beta=beta, tau=tau), c)
        prm = best[1]
        pr = full.decide_ilm(pll, ilm, prm['lam'], prm['mu'], prm['beta'], prm['tau'])
        a = accuracy(full, pr)
        preds[name] = pr
        res['rows'][name] = dict(params=prm, dev=best[2], cb=a['cb'], ub=a['ub'], correct=a['correct'],
                                 venn10k_correct=accuracy(test, [pr[k] for k in vi])['correct'], vs_base=pair(full, base, pr),
                                 groups=groups(full, pr, base))
        print(name, prm, round(a['cb'], 2), round(a['ub'], 2), a['correct'], flush=True)
    res['vs_byt5'] = {k: dict(pair(full, preds['+ ByT5'], preds[k]), groups=groups(full, preds[k], preds['+ ByT5']))
                      for k in ('− ILM + ByT5', '+ ByT5 + W 창')}
    C.dump(f"{RES}/{'results.json' if wtag == DEFAULT else f'results_{wtag}.json'}", res)


def ablation():
    """Main-table rows: the final combination on natdev, other rows with terms dropped (`tools.rescore.ablation`)."""
    wtag = os.environ.get('WTAG', DEFAULT)
    pll, dev, full, Wd, Wt = _inputs(wtag)
    F = S.final_params(dev, Wd, pll)
    res, _ = AB.ablation(full, Wt, pll, F, dev, store.ilm_table('svtrv2-ar'))
    a0 = accuracy(full, full.base())
    res.update(reader='svtrv2-ar', w_tags=[f'{wtag}_natdev', f'{wtag}_full'], grid=S.GRID,
               greedy=dict(cb=a0['cb'], ub=a0['ub'], correct=a0['correct']))
    for k, v in res['rows'].items():
        print(k, v['params'], round(v['cb'], 2), round(v['ub'], 2), v['correct'], flush=True)
    C.dump(f"{RES}/{'ablation.json' if wtag == DEFAULT else f'ablation_{wtag}.json'}", res)


SIZES = (('W3', DEFAULT), ('W1', f'{DEFAULT}_w1'), ('W5', f'{DEFAULT}_w5'), ('W7', f'{DEFAULT}_w7'), ('W9', f'{DEFAULT}_w9'))
REREAD_ROWS = ('+ re-read', '+ ByT5 + re-read')


def wsize():
    """Neighbor range of the windows (W1 … W9). rows: each W with its own final combination (check); rows_refit: the
    rule of the manuscript — W3's α, β, τ fixed, only γ on natdev. Pairs against W3 on the same row."""
    pll = C.pll()
    dev, full = C.ar('natdev'), C.ar('full')
    pred, predr = {}, {}
    res = dict(rows={}, pairs={}, rows_refit={}, pairs_refit={},
               rule_refit='W3 최종식 μ · β · τ 고정 · 다른 W 는 ω 만 natdev(ablation_rows.refit_omega)')
    W = {}
    for w, tag in SIZES:
        if w in ('W7', 'W9') and not store.window_done(tag):
            print(w, 'window runs not complete — skipped', flush=True)
            continue
        W[w] = (C.window_delta(f'{tag}_natdev', dev), C.window_delta(f'{tag}_full', full))
    F3 = S.final_params(dev, W['W3'][0], pll)
    for w, (Wd, Wt) in W.items():
        F = F3 if w == 'W3' else S.final_params(dev, Wd, pll)
        r, pred[w] = AB.ablation(full, Wt, pll, F)
        res['rows'][w] = dict(params=F, rows=r['rows'])
        Fr, dc = (F3, None) if w == 'W3' else S.refit_omega(dev, Wd, pll, F3)
        rr, predr[w] = AB.ablation(full, Wt, pll, Fr)
        res['rows_refit'][w] = dict(params=Fr, dev_final=dc, rows=rr['rows'])
    for key, P_ in (('pairs', pred), ('pairs_refit', predr)):
        for nm in REREAD_ROWS:
            for w in [x for x in W if x != 'W3']:
                res[key][f'{w} 대 W3 · {nm}'] = dict(pair(full, P_['W3'][nm], P_[w][nm]), groups=groups(full, P_[w][nm], P_['W3'][nm]))
    C.dump(f'{RES}/wsize.json', res)


# window construction arms: (row name, tag, the `window_score.run` options that differ from the default)
WCON = (('W3(기본)', DEFAULT, {}), ('W1', f'{DEFAULT}_w1', dict(hops=0)), ('W5', f'{DEFAULT}_w5', dict(hops=2)),
        ('W7', f'{DEFAULT}_w7', dict(hops=3)), ('W9', f'{DEFAULT}_w9', dict(hops=4)),
        ('영역 없음', f'{DEFAULT}_rgnone', dict(region='none')),
        ('창 점수 합(겹침 가중 없음)', f'{DEFAULT}_sum', dict(aggregation='sum')))
# rotation, boundary variants, watershed and the original p_tl run on natdev + CB 6 / harder UB 6 only (tags <arm>_cbub6) —
# rows of the supplementary factor table (`window_factors`), not of this table; the TRER arm is not part of the manuscript


def method_version(options):
    """Version string of the window construction of an arm (`tools.rescore.window.method_version` — its keyword defaults
    are the earlier definition, so the default definition is spelled out and the arm's options laid over it)."""
    from tools.rescore import window
    o = dict(region='trw15', hops=1, ptl='refined', segmentation='nearest', rotate=False, boundary=False, aggregation='coverage')
    o.update(options)
    o['region'] = {'trw15': 'band_wide15'}.get(o['region'], o['region'])
    return window.method_version(**o)


def wconstruct():
    """Window construction: the default windows against one change each (neighbor range, region, rotation, boundary
    substrings, segmentation, p_tl, aggregation). The default final combination's α, β, τ are kept and only γ is chosen
    per arm on natdev; pairs on the same row. Arms whose window runs are not complete are listed in `skipped`."""
    pll = C.pll()
    dev, full = C.ar('natdev'), C.ar('full')
    res = dict(rule='기본 최종식 μ · β · τ 고정 · 갈래마다 ω 만 natdev(ablation_rows.refit_omega)', arms={}, pairs={}, skipped=[])
    pred, F0 = {}, None
    for name, tag, env in WCON:
        if not store.window_done(tag):
            res['skipped'].append(name)
            print(name, 'window runs not complete — skipped', flush=True)
            continue
        Wd, Wt = C.window_delta(f'{tag}_natdev', dev), C.window_delta(f'{tag}_full', full)
        if F0 is None:
            F, dc = S.final_params(dev, Wd, pll), None
            F0 = F
        else:
            F, dc = S.refit_omega(dev, Wd, pll, F0)
        r, pred[name] = AB.ablation(full, Wt, pll, F)
        res['arms'][name] = dict(tags=[f'{tag}_natdev', f'{tag}_full'], options=env, method_version=method_version(env), params=F,
                                 dev_final=dc, rows=r['rows'])
    base = WCON[0][0]
    for name in pred:
        if name == base:
            continue
        for nm in REREAD_ROWS:
            res['pairs'][f'{name} 대 {base} · {nm}'] = dict(pair(full, pred[base][nm], pred[name][nm]),
                                                            groups=groups(full, pred[name][nm], pred[base][nm]))
    C.dump(f'{RES}/wconstruct.json', res)


def lmswap():
    """ByT5-XL replaced by Gemma-4-31B string log P (FP32, case-insensitive; with / without the final newline): the window
    final combination's β, τ, γ fixed, only α on natdev. Pairs against ByT5 on the same row."""
    byt5 = C.pll()
    dev, full = C.ar('natdev'), C.ar('full')
    Wd, Wt = C.window_delta(f'{DEFAULT}_natdev', dev), C.window_delta(f'{DEFAULT}_full', full)
    F = S.final_params(dev, Wd, byt5)
    rb, pb = AB.ablation(full, Wt, byt5, F)
    res = dict(rule='창 최종식 β · τ · ω 고정 · 문맥 모델 무게 μ 만 natdev(refit_mu)', byt5=dict(params=F, rows=rb['rows']), gemma={})
    for field in ('ci', 'ci_ne'):
        G = store.pll_table('gemma4-31b_fp32', field, 'strings_ar.json')
        if G is None:
            print('no Gemma table', flush=True)
            return
        allc = [y for _, b, _, c in full.items for y in [b] + [x[0] for x in c]]
        cover = sum(y in G for y in allc) / max(1, sum(1 + len(c) for _, _, _, c in full.items))
        Fg, dc = S.refit_mu(dev, Wd, G, F)
        r, pg = AB.ablation(full, Wt, G, Fg)
        r.update(cover=cover, dev_final=dc, pairs_vs_byt5={nm: dict(pair(full, pb[nm], pg[nm]), groups=groups(full, pg[nm], pb[nm]))
                                                          for nm in ('+ ByT5', '+ ByT5 + re-read')})
        res['gemma'][field] = r
    C.dump(f'{RES}/lmswap.json', res)


STAGES = {'main': main, 'ablation': ablation, 'wsize': wsize, 'wconstruct': wconstruct, 'lmswap': lmswap}

if __name__ == '__main__':
    STAGES[(sys.argv[1:] or ['main'])[0]]()

R = RES
REPRODUCE = (
    ('ar_main', 'main', {'WTAG': DEFAULT}, (), f'{R}/results.json'),
    ('ar_abl', 'ablation', {'WTAG': DEFAULT}, (), f'{R}/ablation.json'),
    ('ar_abl_w1', 'ablation', {'WTAG': f'{DEFAULT}_w1'}, (), f'{R}/ablation_{DEFAULT}_w1.json'),
    ('ar_abl_w5', 'ablation', {'WTAG': f'{DEFAULT}_w5'}, (), f'{R}/ablation_{DEFAULT}_w5.json'),
    ('ar_abl_w7', 'ablation', {'WTAG': f'{DEFAULT}_w7'}, (), f'{R}/ablation_{DEFAULT}_w7.json'),
    ('ar_abl_w9', 'ablation', {'WTAG': f'{DEFAULT}_w9'}, (), f'{R}/ablation_{DEFAULT}_w9.json'),
    ('ar_wsize', 'wsize', {}, (), f'{R}/wsize.json'),
    ('ar_wcons', 'wconstruct', {}, (), f'{R}/wconstruct.json'),
    ('ar_lmswap', 'lmswap', {}, (), f'{R}/lmswap.json'),
)
