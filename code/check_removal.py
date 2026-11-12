"""Numerical check: removing a generator's statistics equals recomputing without it; order of arrival is irrelevant.
CPU only (one layer of cached ASVspoof 2019 LA features)."""
import json, os, numpy as np, torch
import gss
from pools import la_pool
from data import ROOT

P = la_pool(); F = P.F; l = 10; names = P.names()
lam = [1e-3, 1e-2, 1e-1]
def groups(order):
    gs = []
    for g in order:
        gs.append(gss.Group(g, "spoof", F.D, fold=g, device="cpu").add(F.layer(l, P.gens[g]["adapt"])))
        gs.append(gss.Group("b_" + g, "bona", F.D, fold=g, device="cpu").add(F.layer(l, P.gens[g]["bona_adapt"])))
    return gs
rng = np.random.RandomState(0)
o1 = list(names); o2 = [names[i] for i in rng.permutation(len(names))]
g1, g2 = groups(o1), groups(o2)
out = {}
for wt in ["pooled", "balanced"]:
    W1, b1 = gss.solve(g1, lam, wt); W2, b2 = gss.solve(g2, lam, wt)
    out[f"order_invariance_{wt}"] = float((W1 - W2).abs().max() / W1.abs().max())
    # removal of A10: delete its two triples vs. accumulate the remaining 16 generators from scratch
    kept = [g for g in g1 if g.fold != "A10"]
    scratch = groups([n for n in o2 if n != "A10"])
    Wa, ba = gss.solve(kept, lam, wt); Wb, bb = gss.solve(scratch, lam, wt)
    out[f"removal_{wt}"] = float((Wa - Wb).abs().max() / Wa.abs().max())
    # joint equivalence: incremental accumulation in 7 batches vs one pass
    g = gss.Group("A01", "spoof", F.D, device="cpu"); X = F.layer(l, P.gens["A01"]["adapt"])
    for part in np.array_split(X, 7):
        g.add(part)
    out["incremental_vs_batch_S"] = float((g.S - g1[0].S).abs().max() / g1[0].S.abs().max())
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(ROOT, "results", "check_removal.json"), "w"), indent=1)
