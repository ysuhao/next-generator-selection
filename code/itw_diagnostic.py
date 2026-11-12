"""Why does the last layer win on In-the-Wild?  A per-layer diagnostic on cached WavLM-Large features.

For every layer, one detector is built from ALL adaptation data of the 17 ASVspoof generators
(pooled weighting, lambda = 1e-2, i.e. the 'Fixed' recipe applied to each layer).  We then report
  eer_la      : mean EER over the 17 generators' test sets (studio conditions, all generators seen)
  eer_itw     : EER on In-the-Wild
  shift_bona  : (mean score of In-the-Wild bona fide - mean score of ASVspoof bona fide test) / std of ASVspoof bona fide test scores
  shift_spoof : the same for In-the-Wild spoof against ASVspoof spoof test scores
  feat_shift  : ||mu_ITW_bona - mu_LA_bona||^2 / tr(Sigma_LA_bona)   (shift of bona fide speech in feature space)
In-the-Wild is only scored, never used to fit anything.  Writes results/itw_diagnostic.json.
"""
import json, os
import numpy as np
import torch
import gss
from pools import la_pool
from data import Feats, ROOT

P = la_pool(); F = P.F; names = P.names()
I = Feats("itw", "wavlm-large", "mean")
ib, isp = I.where(label="bonafide"), I.where(label="spoof")
dev = gss.dev(); T = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=gss.DT, device=dev)
domains = sorted({P.gens[g]["domain"] for g in names})
bt = {d: next(P.gens[g]["bona_test"] for g in names if P.gens[g]["domain"] == d) for d in domains}
out = []
for l in range(F.L):
    groups = []
    for g in names:
        groups.append(gss.Group(g, "spoof", F.D, fold=g).add(F.layer(l, P.gens[g]["adapt"])))
        groups.append(gss.Group("b_" + g, "bona", F.D, fold=g).add(F.layer(l, P.gens[g]["bona_adapt"])))
    W, b = gss.solve(groups, [1e-2], "pooled")
    sc = lambda X: (T(X) @ W.T + b)[:, 0]
    sb = {d: sc(F.layer(l, bt[d])) for d in domains}
    eers, ss_all = [], []
    for g in names:
        ss = sc(F.layer(l, P.gens[g]["test"])); ss_all.append(ss)
        eers.append(float(gss.eer_torch(sb[P.gens[g]["domain"]][:, None], ss[:, None])[0]))
    la_b = torch.cat(list(sb.values())); la_s = torch.cat(ss_all)
    Xib, Xis = I.layer(l, ib), I.layer(l, isp)
    s_ib, s_is = sc(Xib), sc(Xis)
    la_bona_feat = np.concatenate([F.layer(l, bt[d]) for d in domains]).astype(np.float64)
    mu_la = la_bona_feat.mean(0); var_la = la_bona_feat.var(0).sum()
    out.append(dict(layer=l, eer_la=float(np.mean(eers)), eer_itw=float(gss.eer_torch(s_ib[:, None], s_is[:, None])[0]),
                    shift_bona=float((s_ib.mean() - la_b.mean()) / la_b.std()), shift_spoof=float((s_is.mean() - la_s.mean()) / la_s.std()),
                    sep_la=float((la_b.mean() - la_s.mean()) / (0.5 * (la_b.std() + la_s.std()))),
                    sep_itw=float((s_ib.mean() - s_is.mean()) / (0.5 * (s_ib.std() + s_is.std()))),
                    feat_shift=float(((Xib.astype(np.float64).mean(0) - mu_la) ** 2).sum() / var_la)))
    r = out[-1]
    print(f"layer {l:2d}  EER LA {r['eer_la']*100:5.2f}  ITW {r['eer_itw']*100:5.2f}  bona score shift {r['shift_bona']:+6.2f} sd  spoof shift {r['shift_spoof']:+6.2f} sd  feat shift {r['feat_shift']:.2f}", flush=True)
json.dump(out, open(os.path.join(ROOT, "results", "itw_diagnostic.json"), "w"), indent=1)
