"""Extra closed-form baselines on the same streams (single fixed layer, pooled statistics only):
  ncm     : nearest class mean (bona fide mean vs pooled spoof mean), Euclidean
  ranpac  : random ReLU projection (M dims) of standardised features + pooled ridge;
            lambda chosen by statistics-only 2-fold in-distribution CV (as RanPAC tunes on in-distribution data)
Both keep only pooled class statistics, i.e. what a learner without generator bookkeeping stores.
"""
import argparse, json, os, time
import numpy as np
import torch
import gss
from pools import get_pool, make_stream
from data import ROOT

LAMBDAS = [10.0 ** e for e in np.arange(-5, 2.01, 0.5)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="wavlm-large")
    ap.add_argument("--poolname", default="la")
    ap.add_argument("--chrono", action="store_true")
    ap.add_argument("--orders", type=int, default=10)
    ap.add_argument("--base", type=int, default=3)
    ap.add_argument("--condition", default="full")
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--M", type=int, default=4096)
    ap.add_argument("--itw", action="store_true")
    args = ap.parse_args()
    dev = gss.dev()
    P = get_pool(args.poolname, args.model, "mean")
    F = P.F
    l = args.layer if args.layer >= 0 else F.L - 1
    names = P.names()
    domains = sorted({P.gens[g]["domain"] for g in names})
    T = lambda a: torch.as_tensor(a, dtype=gss.DT, device=dev)
    Xb = {d: T(F.layer(l, next(P.gens[g]["bona_test"] for g in names if P.gens[g]["domain"] == d))) for d in domains}
    Xs = {g: T(F.layer(l, P.gens[g]["test"])) for g in names}
    itw = None
    if args.itw:
        from data import Feats
        I = Feats("itw", args.model, "mean")
        itw = (T(I.layer(l, I.where(label="bonafide"))), T(I.layer(l, I.where(label="spoof"))))
    gen = torch.Generator(device="cpu").manual_seed(0)
    Rproj = torch.randn(F.D, args.M, generator=gen, dtype=gss.DT).to(dev) / np.sqrt(F.D)
    out_path = os.path.join(ROOT, "results", f"extra_{args.poolname}_{args.model}_{args.condition}{'_chrono' if args.chrono else ''}_L{l}.jsonl")
    fout = open(out_path, "a")
    for order_id in range(args.orders):
        order, steps, arrive, rng = make_stream(P, order_id, args.condition, args.base, args.chrono)
        seen = []
        # standardisation for the random projection is fixed after the first step (no later data needed)
        first = np.concatenate([np.concatenate(arrive[g]) for g in steps[0]])
        X0 = T(F.layer(l, first)); mu0, sd0 = X0.mean(0), X0.std(0) + 1e-6
        phi = lambda X: torch.relu(((X - mu0) / sd0) @ Rproj)
        raw = {k: gss.Group(k, "bona" if k[0] == "b" else "spoof", F.D, fold="all") for k in ["b", "s"]}
        rp = {(k, h): gss.Group(f"{k}#{h}", "bona" if k == "b" else "spoof", args.M, fold="all") for k in ["b", "s"] for h in (0, 1)}
        t0 = time.time()
        for t, new in enumerate(steps):
            seen += new
            for g in new:
                for k, idx in zip(["s", "b"], arrive[g]):
                    X = T(F.layer(l, idx))
                    raw[k].add(X.cpu().numpy())
                    perm = torch.as_tensor(rng.permutation(len(idx)), device=dev)
                    Z = phi(X)
                    for h in (0, 1):
                        part = Z[perm[h::2]]
                        grp = rp[(k, h)]
                        grp.n += part.shape[0]; grp.s += part.sum(0); grp.S += part.T @ part
            res = {}
            # --- NCM
            w = (raw["b"].mu - raw["s"].mu)[None, :]
            b = -(w @ (0.5 * (raw["b"].mu + raw["s"].mu)))
            res["ncm"] = (w, b, lambda X: X)
            # --- RanPAC-style
            crit = []
            for h in (0, 1):
                W, bb = gss.solve([rp[("b", h)], rp[("s", h)]], LAMBDAS, "pooled")
                mb, vb = gss.score_stats(W, bb, rp[("b", 1 - h)]); ms, vs = gss.score_stats(W, bb, rp[("s", 1 - h)])
                crit.append(gss.gauss_eer(mb, vb, ms, vs))
            j = int(torch.stack(crit).mean(0).argmin())
            full = []
            for k in ["b", "s"]:
                m = gss.Group(k, "bona" if k == "b" else "spoof", args.M, fold="all")
                for h in (0, 1):
                    m.n += rp[(k, h)].n; m.s += rp[(k, h)].s; m.S += rp[(k, h)].S
                full.append(m)
            W, bb = gss.solve(full, LAMBDAS, "pooled")
            res["ranpac"] = (W[j:j + 1], bb[j:j + 1], phi)
            unseen = [g for g in names if g not in seen]
            for name, (W_, b_, f) in res.items():
                sb = {d: f(Xb[d]) @ W_.T + b_ for d in domains}
                per = {g: float(gss.eer_torch(sb[P.gens[g]["domain"]], f(Xs[g]) @ W_.T + b_)[0]) for g in names}
                rec = dict(method=name, order_id=order_id, step=t, seen=list(seen), per_gen=per,
                           seen_eer=float(np.mean([per[g] for g in seen])),
                           unseen_eer=float(np.mean([per[g] for g in unseen])) if unseen else None,
                           all_eer=float(np.mean(list(per.values()))),
                           itw=float(gss.eer_torch(f(itw[0]) @ W_.T + b_, f(itw[1]) @ W_.T + b_)[0]) if itw else None,
                           domain={g: P.gens[g]["domain"] for g in names}, condition=args.condition, layer=l,
                           lam=LAMBDAS[j] if name == "ranpac" else None)
                fout.write(json.dumps(rec) + "\n"); fout.flush()
        print(f"order {order_id} done: ncm all {res and 0}  ({time.time()-t0:.0f}s)", flush=True)
    fout.close()


if __name__ == "__main__":
    main()
