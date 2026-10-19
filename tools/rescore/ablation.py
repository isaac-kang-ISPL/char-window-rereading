"""Rows of the main table — the final coefficients fixed, terms dropped.

Only the final combination (ByT5 + window) is chosen on the validation set (`select.final_params`). Every other row
keeps those coefficients and sets the dropped terms to 0; 'Beam' is the beam top 1 (β = τ = 0). The ILM rows choose
only δ on the validation set, with α, β, τ of the final combination. Row and pair names are the keys of the stored
results. clusters=True adds the reading-cluster rows of earlier results ('무리 + …', final combination with clusters).
"""
from __future__ import annotations

from tools.rescore.decide import batch
from tools.rescore.metrics import accuracy, groups, pair
from tools.rescore.select import select_lam

FINAL = '+ ByT5 + W 창'
ROWS = ('Beam', '+ re-read', '+ ByT5', '+ ByT5 + re-read')
CLUSTER_ROWS = ('무리 + ByT5', '무리 + re-read', '무리 + ByT5 + re-read')
PAIRS = (('+ re-read', 'Beam'), ('+ ByT5', 'Beam'), ('+ ByT5 + re-read', '+ ByT5'), ('+ ByT5 + re-read', '+ re-read'),
         ('무리 + ByT5 + re-read', '+ ByT5 + re-read'), ('무리 + ByT5 + re-read', '무리 + ByT5'),
         ('− ILM + ByT5', '+ ByT5'), ('+ ByT5 + re-read', '− ILM + ByT5'), ('− ILM', 'Beam'))
RULE = '최종 계수 고정 · 뺀 항만 0 · Beam 은 빔 1위(β 0 · τ 0) · ILM 은 λ 만 개발셋에서'


def ablation(full, Wt, lm, F, dev=None, ilm=None, clusters=False):
    """→ (res, pred). res = dict(params, rule, rows{name: params · cb · ub · correct}, pairs); pred = {name: outputs}.
    Wt = {key: {y: ΔW}} against a; with ilm ({string: log P}) and dev, the − ILM rows; clusters=True, the cluster rows."""
    B = batch(full)
    T = B.with_window(Wt)
    mu, om, beta, tau = F['mu'], F['omega'], F['beta'], F['tau']
    spec = (('Beam', T, 0., 0., 0., 0.), ('+ re-read', T, 0., beta, tau, om), ('+ ByT5', T, mu, beta, tau, 0.),
            ('+ ByT5 + re-read', T, mu, beta, tau, om))
    if clusters:
        TM = T.merged()
        spec += (('무리 + ByT5', TM, mu, beta, tau, 0.), ('무리 + re-read', TM, 0., beta, tau, om),
                 ('무리 + ByT5 + re-read', TM, mu, beta, tau, om))
    else:
        TM = None
    res = dict(params=F, rule=RULE, rows={}, pairs={})
    pred = {}
    for nm, X, m, b, t, o in spec:
        p = X.decide(lm, m, b, t, o)
        a = accuracy(full, p)
        pred[nm] = p
        res['rows'][nm] = dict(params=dict(mu=m, beta=b, tau=t, omega=o, clusters=X is TM), cb=a['cb'], ub=a['ub'], correct=a['correct'])
    if ilm is not None and dev is not None:
        for nm, m in (('− ILM', 0.), ('− ILM + ByT5', mu)):
            lam, c = select_lam(dev, lm, ilm, m, beta, tau)
            p = B.decide_ilm(lm, ilm, lam, m, beta, tau)
            a = accuracy(full, p)
            pred[nm] = p
            res['rows'][nm] = dict(params=dict(lam=lam, mu=m, beta=beta, tau=tau, omega=0., clusters=False), dev=c,
                                   cb=a['cb'], ub=a['ub'], correct=a['correct'])
    for a_, b_ in PAIRS:
        if a_ in pred and b_ in pred:
            res['pairs'][f'{a_} 대 {b_}'] = dict(pair(full, pred[b_], pred[a_]), groups=groups(full, pred[a_], pred[b_]))
    return res, pred
