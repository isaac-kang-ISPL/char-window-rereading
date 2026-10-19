"""Coefficients — one grid for every term, chosen on the validation set by the number of correct images.

Grid (`GRID`): α (window, code `omega`), γ (ByT5, code `mu`) and δ (ILM, comparison rows only) start from COEF = 0 and
doubling steps 0.0125 … 0.4; the ByT5 weight goes on to 1.6 and the window weight to 3.2 (with overlap-weighted window
scores some recognizers choose a window weight of 0.4 or more); β ∈ {0, .25, .5, 1, 2}; τ ∈ {0, .125, .25, .5, 1}.
Every term may take 0. Among settings with the same number of correct validation images the one with the smaller
coefficient sum (α + γ, or δ + γ) wins; the remaining ties go to the first setting in grid order. The chosen setting
is applied once to the evaluation images.

Partial configurations of the manuscript tables keep the final coefficients and drop terms (`ablation`). When a row
replaces the window input (another window construction, whole-image re-reading) or the language model, only that
term's weight is chosen again on the validation set (`refit_omega`, `refit_mu`). Reading clusters are off
(clusters=False); clusters=True selects on `Batch.merged()` for the rows of earlier results.
"""
from __future__ import annotations

import itertools

from tools.rescore.decide import batch

COEF = (0., .0125, .025, .05, .1, .2, .4)
GRID = dict(mu=COEF + (.8, 1.6), beta=(0., .25, .5, 1., 2.), tau=(0., .125, .25, .5, 1.), omega=COEF + (.8, 1.6, 3.2))
LAM = COEF

# configurations of the main table, by the names of the stored results: which coefficients are held at 0
ARMS = {'빔 1위': None,                                     # beam top 1: β = τ = 0, no terms
        '+ ByT5': dict(omega=(0.,)), '+ W 창': dict(mu=(0.,)), '+ ByT5 + W 창': {}}
FINAL_ARM = '+ ByT5 + W 창'


def n_correct(b, lm, mu, beta, tau, omega=0.):
    return int(b.decide(lm, mu, beta, tau, omega).ok.sum())


def select(items, lm, fixed=None, grid=None):
    """Best (γ, β, τ, α) on items — fixed = {name: values} restricts a coefficient. → dict(mu=γ, beta, tau, omega=α)."""
    return select_counted(items, lm, fixed, grid)[0]


def select_counted(items, lm, fixed=None, grid=None):
    """select() and the number of correct images of the chosen setting."""
    b = batch(items)
    g = {k: (fixed or {}).get(k, v) for k, v in (grid or GRID).items()}
    best = None
    for mu, beta, tau, om in itertools.product(g['mu'], g['beta'], g['tau'], g['omega']):
        c = n_correct(b, lm, mu, beta, tau, om)
        key = (c, -abs(mu) - abs(om))
        if best is None or key > best[0]:
            best = (key, dict(mu=mu, beta=beta, tau=tau, omega=om), c)
    return best[1], best[2]


def select_arm(items, lm, arm):
    """Coefficients of a main-table configuration (ARMS)."""
    if ARMS[arm] is None:
        return dict(mu=0., beta=0., tau=0., omega=0.)
    return select(items, lm, ARMS[arm])


def _dev(dev, Wd, clusters):
    D = batch(dev).with_window(Wd)
    return D.merged() if clusters else D


def final_params(dev, Wd, lm, clusters=False):
    """The final combination (ByT5 + window) on the validation items with ΔW table Wd."""
    return select(_dev(dev, Wd, clusters), lm)


def refit_omega(dev, Vd, lm, F, grid=None, clusters=False):
    """Final coefficients F with another window input Vd — only α (omega) is chosen again (ties: smaller α). → (F', correct)."""
    D = _dev(dev, Vd, clusters)
    best = None
    for om in (grid or GRID['omega']):
        c = n_correct(D, lm, F['mu'], F['beta'], F['tau'], om)
        if best is None or (c, -om) > best[0]:
            best = ((c, -om), om, c)
    return dict(F, omega=best[1]), best[2]


def refit_mu(dev, Wd, lm, F, clusters=False):
    """Final coefficients F with another language model lm — only γ (mu) is chosen again (ties: smaller γ). → (F', correct)."""
    D = _dev(dev, Wd, clusters)
    best = None
    for mu in GRID['mu']:
        c = n_correct(D, lm, mu, F['beta'], F['tau'], F['omega'])
        if best is None or (c, -mu) > best[0]:
            best = ((c, -mu), mu, c)
    return dict(F, mu=best[1]), best[2]


def select_lam(dev, lm, ilm, mu, beta, tau, lams=LAM):
    """δ of the ILM comparison rows with γ, β, τ fixed (ties: smaller δ). → (δ, correct)."""
    b = batch(dev)
    best = None
    for lam in lams:
        c = int(b.decide_ilm(lm, ilm, lam, mu, beta, tau).ok.sum())
        if best is None or (c, -lam) > best[0]:
            best = ((c, -lam), lam, c)
    return best[1], best[2]
