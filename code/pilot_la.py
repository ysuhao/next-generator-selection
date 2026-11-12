"""Pilot (go / no-go) on ASVspoof 2019 LA.

Statistics of A01-A06 (train partition) only.  For a grid layer x lambda x weighting:
  (a) closed-form Gaussian LOGO EER (statistics only)
  (b) empirical LOGO EER (raw features, same folds)            -> validates the Gaussian approximation
  (c) in-distribution validation EER on the dev partition (A01-A06, unseen speakers)
  (d) EER on the eval partition, A07-A19                       -> truly unseen generators
Question: does (a) rank configurations for (d) better than (c)?
"""
import argparse, json, os, time
import numpy as np
import torch
from scipy.stats import spearmanr
import gss
from data import Feats, speaker_chunks, ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="wavlm-large")
ap.add_argument("--pool", default="mean")
ap.add_argument("--out", default="pilot_la")
args = ap.parse_args()

F = Feats("asvspoof19", args.model, args.pool)
meta = F.meta
LAMBDAS = [10.0 ** e for e in np.arange(-5, 2.01, 0.5)]
GENS = ["A01", "A02", "A03", "A04", "A05", "A06"]
EVAL_GENS = [f"A{i:02d}" for i in range(7, 20)]

tr_b = F.where(split="train", system="bonafide")
chunks = speaker_chunks(meta.speaker.values[tr_b], len(GENS), seed=0)
dev_b = F.where(split="dev", system="bonafide")
dev_s = F.where(split="dev", label="spoof")
ev_b = F.where(split="eval", system="bonafide")
ev_idx = {g: F.where(split="eval", system=g) for g in EVAL_GENS}

rows = []
t0 = time.time()
for l in range(F.L):
    Xtr = {g: F.layer(l, F.where(split="train", system=g)) for g in GENS}
    Xb = [F.layer(l, tr_b[np.isin(meta.speaker.values[tr_b], list(c))]) for c in chunks]
    groups = []
    for g, c in zip(GENS, Xb):
        groups.append(gss.Group(g, "spoof", F.D, fold=g).add(Xtr[g]))
        groups.append(gss.Group("bona_" + g, "bona", F.D, fold=g).add(c))
    Xdev_b, Xdev_s = F.layer(l, dev_b), F.layer(l, dev_s)
    Xev_b = F.layer(l, ev_b)
    Xev = {g: F.layer(l, ev_idx[g]) for g in EVAL_GENS}
    for wt in ["pooled", "balanced"]:
        fc = gss.logo_forecast(groups, LAMBDAS, wt)                       # (a)
        fc_mat = torch.stack([fc[g] for g in GENS]).cpu().numpy()          # [6, n_lam]
        emp = np.zeros_like(fc_mat)                                        # (b)
        for gi, g in enumerate(GENS):
            W, b = gss.solve([x for x in groups if x.fold != g], LAMBDAS, wt)
            sb, ss = gss.scores(W, b, Xb[gi]), gss.scores(W, b, Xtr[g])
            emp[gi] = [gss.eer(sb[:, j], ss[:, j]) for j in range(len(LAMBDAS))]
        W, b = gss.solve(groups, LAMBDAS, wt)
        sdb, sds = gss.scores(W, b, Xdev_b), gss.scores(W, b, Xdev_s)     # (c)
        seb = gss.scores(W, b, Xev_b)                                      # (d)
        ses = {g: gss.scores(W, b, Xev[g]) for g in EVAL_GENS}
        for j, lam in enumerate(LAMBDAS):
            per = {g: gss.eer(seb[:, j], ses[g][:, j]) for g in EVAL_GENS}
            rows.append(dict(layer=l, lam=lam, weighting=wt,
                             logo_gauss_mean=float(fc_mat[:, j].mean()), logo_gauss_max=float(fc_mat[:, j].max()),
                             logo_emp_mean=float(emp[:, j].mean()), logo_emp_max=float(emp[:, j].max()),
                             logo_gauss=fc_mat[:, j].tolist(), logo_emp=emp[:, j].tolist(),
                             dev_eer=gss.eer(sdb[:, j], sds[:, j]),
                             eval_pooled=gss.eer(seb[:, j], np.concatenate([ses[g][:, j] for g in EVAL_GENS])),
                             eval_mean=float(np.mean(list(per.values()))), eval_max=float(np.max(list(per.values()))),
                             eval_per=per))
    print(f"layer {l} done ({time.time()-t0:.0f}s)", flush=True)

os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
json.dump(rows, open(os.path.join(ROOT, "results", f"{args.out}_{args.model}_{args.pool}.json"), "w"))

import pandas as pd
df = pd.DataFrame(rows)
def rho(a, b):
    return spearmanr(df[a], df[b]).correlation
print("\n== rank correlation over", len(df), "configurations ==")
flat_g = np.concatenate(df.logo_gauss.values); flat_e = np.concatenate(df.logo_emp.values)
print("Gaussian vs empirical LOGO EER, per fold:   rho=%.3f" % spearmanr(flat_g, flat_e).correlation)
print("Gaussian vs empirical LOGO EER, fold mean:  rho=%.3f" % rho("logo_gauss_mean", "logo_emp_mean"))
for tgt in ["eval_pooled", "eval_mean", "eval_max"]:
    print(f"target {tgt}: rho(dev_eer)={rho('dev_eer', tgt):.3f}  rho(logo_gauss_mean)={rho('logo_gauss_mean', tgt):.3f}  "
          f"rho(logo_gauss_max)={rho('logo_gauss_max', tgt):.3f}  rho(logo_emp_mean)={rho('logo_emp_mean', tgt):.3f}")
print("\n== selected configuration and resulting EER on unseen generators (A07-A19) ==")
for name, col in [("ID dev validation", "dev_eer"), ("LOGO Gaussian mean (stats only)", "logo_gauss_mean"),
                  ("LOGO Gaussian worst (stats only)", "logo_gauss_max"), ("LOGO empirical mean (needs data)", "logo_emp_mean"),
                  ("oracle (eval pooled)", "eval_pooled")]:
    r = df.loc[df[col].idxmin()]
    print(f"{name:36s} layer={r.layer:2d} lam={r.lam:.1e} {r.weighting:8s} | sel={r[col]*100:6.2f}  "
          f"eval pooled={r.eval_pooled*100:5.2f} mean={r.eval_mean*100:5.2f} worst={r.eval_max*100:5.2f}  dev={r.dev_eer*100:5.2f}")
