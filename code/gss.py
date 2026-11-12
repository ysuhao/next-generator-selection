"""Generator-wise sufficient statistics (GSS) for continual spoof detection.

A *group* is one spoof generator or one chunk of bona fide speech.  For each
group only (n, sum x, sum x x^T) is kept.  From these the weighted ridge /
regularised-LDA detector, its leave-one-generator-out (LOGO) variants and the
Gaussian estimate of the LOGO equal error rate are all available in closed
form, without access to any utterance.
"""
import math
import numpy as np
import torch

DT = torch.float64


def dev():
    return "cuda" if torch.cuda.is_available() else "cpu"


class Group:
    __slots__ = ("name", "kind", "n", "s", "S", "fold")

    def __init__(self, name, kind, d, fold=None, device=None):
        self.name, self.kind, self.fold = name, kind, fold if fold is not None else name
        self.n = 0
        self.s = torch.zeros(d, dtype=DT, device=device or dev())
        self.S = torch.zeros(d, d, dtype=DT, device=device or dev())

    def add(self, X):
        X = torch.as_tensor(np.asarray(X, dtype=np.float64), dtype=DT, device=self.s.device)
        self.n += X.shape[0]
        self.s += X.sum(0)
        self.S += X.T @ X
        return self

    @property
    def mu(self):
        return self.s / self.n

    @property
    def M(self):  # second moment
        return self.S / self.n

    def nbytes(self, dtype_bytes=4):
        d = self.s.numel()
        return dtype_bytes * (d * (d + 1) // 2 + d) + 8


def group_weights(groups, spoof_weighting="balanced", alpha=None):
    """Class-balanced weights (bona fide 1/2, spoof 1/2).
    spoof_weighting: 'pooled' (proportional to n_g), 'balanced' (equal per generator)
    or 'custom' with alpha = dict name -> weight."""
    w = {}
    bona = [g for g in groups if g.kind == "bona"]
    spoof = [g for g in groups if g.kind == "spoof"]
    nb = sum(g.n for g in bona)
    for g in bona:
        w[g.name] = 0.5 * g.n / nb
    if spoof_weighting == "pooled":
        ns = sum(g.n for g in spoof)
        for g in spoof:
            w[g.name] = 0.5 * g.n / ns
    elif spoof_weighting == "balanced":
        for g in spoof:
            w[g.name] = 0.5 / len(spoof)
    else:
        tot = sum(alpha[g.name] for g in spoof)
        for g in spoof:
            w[g.name] = 0.5 * alpha[g.name] / tot
    return w


def moments(groups, weights):
    d = groups[0].s.numel()
    A = torch.zeros(d, d, dtype=DT, device=groups[0].s.device)
    m = torch.zeros(d, dtype=DT, device=A.device)
    c = torch.zeros(d, dtype=DT, device=A.device)
    for g in groups:
        wk = weights[g.name]
        y = 1.0 if g.kind == "bona" else -1.0
        A += wk * g.M
        m += wk * g.mu
        c += wk * y * g.mu
    return A, m, c


def solve(groups, lambdas, spoof_weighting="balanced", alpha=None):
    """Weighted ridge regression to targets +1 (bona fide) / -1 (spoof).
    Returns W [n_lambda, d], b [n_lambda].  lambdas are relative to tr(C)/d."""
    weights = group_weights(groups, spoof_weighting, alpha)
    A, m, c = moments(groups, weights)
    C = A - torch.outer(m, m)
    C = 0.5 * (C + C.T)
    evals, V = torch.linalg.eigh(C)
    evals = evals.clamp_min(0)
    scale = evals.mean()
    lam = torch.as_tensor(lambdas, dtype=DT, device=C.device) * scale
    r = V.T @ c                                   # ybar = 0 because the classes are balanced
    W = (r[None, :] / (evals[None, :] + lam[:, None])) @ V.T
    b = -(W @ m)
    return W, b


def score_stats(W, b, g):
    """Mean and variance of the detector score on group g, from its statistics only."""
    mu = g.mu
    mean = W @ mu + b
    Sigma = g.M - torch.outer(mu, mu)
    var = ((W @ Sigma) * W).sum(1).clamp_min(1e-12)
    return mean, var


def gauss_eer(mb, vb, ms, vs):
    """EER of two Gaussian score distributions (bona fide higher).  Elementwise."""
    z = (mb - ms) / (vb.sqrt() + vs.sqrt())
    return 0.5 * torch.erfc(z / math.sqrt(2.0))


def logo_forecast(groups, lambdas, spoof_weighting="balanced", alpha=None):
    """Closed-form leave-one-generator-out EER forecast.

    Every fold id that contains a spoof group is held out in turn together with
    the bona fide groups sharing that fold id.  Returns dict fold -> tensor
    [n_lambda] of Gaussian EERs, computed without any utterance."""
    folds = sorted({g.fold for g in groups if g.kind == "spoof"})
    out = {}
    for f in folds:
        tr = [g for g in groups if g.fold != f]
        te = [g for g in groups if g.fold == f]
        te_b = [g for g in te if g.kind == "bona"]
        te_s = [g for g in te if g.kind == "spoof"]
        if not te_b or not any(g.kind == "bona" for g in tr) or not any(g.kind == "spoof" for g in tr):
            continue
        W, b = solve(tr, lambdas, spoof_weighting, alpha)
        mb, vb = pooled_stats(W, b, te_b)
        ms, vs = pooled_stats(W, b, te_s)
        out[f] = gauss_eer(mb, vb, ms, vs)
    return out


def pooled_stats(W, b, gs):
    """Score mean/variance of the union of several groups (statistics only)."""
    n = sum(g.n for g in gs)
    s = sum(g.s for g in gs)
    S = sum(g.S for g in gs)
    mu = s / n
    mean = W @ mu + b
    Sigma = S / n - torch.outer(mu, mu)
    var = ((W @ Sigma) * W).sum(1).clamp_min(1e-12)
    return mean, var


# ----------------------------------------------------------------- empirical
def eer(bona_scores, spoof_scores):
    """Empirical equal error rate (bona fide should score higher)."""
    b = np.asarray(bona_scores, dtype=np.float64)
    s = np.asarray(spoof_scores, dtype=np.float64)
    scores = np.concatenate([b, s])
    labels = np.concatenate([np.ones(len(b)), np.zeros(len(s))])
    order = np.argsort(scores, kind="mergesort")
    labels = labels[order]
    frr = np.concatenate([[0.0], np.cumsum(labels) / len(b)])            # bona rejected below thr
    far = np.concatenate([[1.0], 1.0 - np.cumsum(1 - labels) / len(s)])  # spoof accepted above thr
    i = np.argmin(np.abs(frr - far))
    return float(0.5 * (frr[i] + far[i]))


def scores(W, b, X):
    X = torch.as_tensor(np.asarray(X, dtype=np.float64), dtype=DT, device=W.device)
    return (X @ W.T + b[None, :]).cpu().numpy()   # [n, n_lambda]


def eer_torch(bona, spoof):
    """Vectorised empirical EER on the current device.  bona [nb, J], spoof [ns, J] -> [J]."""
    nb, ns = bona.shape[0], spoof.shape[0]
    sc = torch.cat([bona, spoof], 0)
    lab = torch.cat([torch.ones(nb, dtype=DT, device=sc.device), torch.zeros(ns, dtype=DT, device=sc.device)])
    order = torch.argsort(sc, dim=0, stable=True)
    ls = lab[order]                                           # [N, J]
    frr = torch.cumsum(ls, 0) / nb
    far = 1.0 - torch.cumsum(1.0 - ls, 0) / ns
    zero = torch.zeros(1, sc.shape[1], dtype=DT, device=sc.device)
    frr = torch.cat([zero, frr], 0); far = torch.cat([zero + 1.0, far], 0)
    i = torch.argmin((frr - far).abs(), 0, keepdim=True)
    return 0.5 * (frr.gather(0, i) + far.gather(0, i)).squeeze(0)
