"""Generator-incremental stream with closed-form (statistics-only) detectors.

At every step one new generator arrives with a chunk of bona fide speech.
The learner keeps only per-group sufficient statistics (each group additionally
split in two random halves, which makes statistics-only in-distribution
cross-validation possible as a baseline for LOGO).

Model-selection rules compared at each step (all choose layer x lambda x weighting):
  logo      : statistics-only leave-one-generator-out Gaussian EER (mean over folds)   [proposed]
  logo_max  : same, worst fold
  idcv      : statistics-only 2-fold in-distribution cross-validation Gaussian EER
  fixed     : last layer, lambda = 1e-2, pooled weighting (plain analytic learner)
  oracle_unseen / oracle_all : best configuration in hindsight (reference only)
For every configuration the true EER on every generator's test set is logged, so
all rank correlations can be computed afterwards.
"""
import argparse, json, os, time
import numpy as np
import torch
import gss
from pools import get_pool, make_stream
from data import ROOT

LAMBDAS = [10.0 ** e for e in np.arange(-5, 2.01, 0.5)]
WEIGHTINGS = ["pooled", "balanced"]


def halves_groups(name, kind, X, fold, rng):
    """Two random halves of one arriving batch as two Group objects."""
    perm = rng.permutation(len(X))
    out = []
    for h in range(2):
        part = X[perm[h::2]]
        g = gss.Group(f"{name}#{h}", kind, X.shape[1], fold=fold, device="cpu")
        if len(part):
            g.add(part)
        out.append(g)
    return [g for g in out if g.n > 0]


def on_device(groups, device):
    out = []
    for g in groups:
        c = gss.Group(g.name, g.kind, 1, fold=g.fold, device="cpu")
        c.n, c.s, c.S = g.n, g.s.to(device), g.S.to(device)
        out.append(c)
    return out


def merged(groups):
    """Merge the halves back: one group per (fold, kind)."""
    out = {}
    for g in groups:
        key = (g.fold, g.kind)
        if key not in out:
            m = gss.Group(f"{g.fold}|{g.kind}", g.kind, g.s.numel(), fold=g.fold, device=g.s.device)
            out[key] = m
        m = out[key]
        m.n += g.n; m.s += g.s; m.S += g.S
    return list(out.values())


def idcv_forecast(groups, lambdas, weighting):
    """Statistics-only 2-fold in-distribution CV: train on halves #h, test on halves #(1-h).
    Returns tensor [n_folds_total, n_lambda] of per-generator Gaussian EERs."""
    res = []
    for h in (0, 1):
        tr = [g for g in groups if g.name.endswith(f"#{h}")]
        te = [g for g in groups if g.name.endswith(f"#{1-h}")]
        if not any(g.kind == "spoof" for g in tr) or not any(g.kind == "bona" for g in tr):
            continue
        # merge names so that weighting is per generator
        W, b = gss.solve(merged(tr), lambdas, weighting)
        te_b = [g for g in te if g.kind == "bona"]
        if not te_b:
            continue
        mb, vb = gss.pooled_stats(W, b, te_b)
        for g in te:
            if g.kind == "spoof":
                ms, vs = gss.score_stats(W, b, g)
                res.append(gss.gauss_eer(mb, vb, ms, vs))
    return torch.stack(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="wavlm-large")
    ap.add_argument("--pool", default="mean")
    ap.add_argument("--orders", type=int, default=10)
    ap.add_argument("--base", type=int, default=3, help="generators in the first step")
    ap.add_argument("--condition", default="full", help="full | imb | shotN (N utterances per new generator)")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--tag", default="")
    ap.add_argument("--itw", action="store_true", help="also score In-the-Wild at every step")
    ap.add_argument("--poolname", default="la")
    ap.add_argument("--chrono", action="store_true")
    ap.add_argument("--order_start", type=int, default=0)
    ap.add_argument("--save_T", action="store_true", help="store per-generator EER of every configuration")
    args = ap.parse_args()

    P = get_pool(args.poolname, args.model, args.pool)
    F = P.F
    layers = list(range(F.L)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    names = P.names()
    dev = gss.dev()
    out_path = os.path.join(ROOT, "results", f"stream_{args.poolname}_{args.model}_{args.pool}_{args.condition}{'_chrono' if args.chrono else ''}{args.tag}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # ---- test features per layer on CPU (float16) : spoof test per generator, bona test per domain
    domains = sorted({P.gens[g]["domain"] for g in names})
    bona_test_idx = {d: next(P.gens[g]["bona_test"] for g in names if P.gens[g]["domain"] == d) for d in domains}
    itw = None
    if args.itw:
        from data import Feats
        itw = Feats("itw", args.model, args.pool)
        itw_b, itw_s = itw.where(label="bonafide"), itw.where(label="spoof")

    fout = open(out_path, "a")
    test_cache = {}
    DT64 = gss.DT
    for order_id in range(args.order_start, args.order_start + args.orders):
        order, steps, arrive, rng = make_stream(P, order_id, args.condition, args.base, args.chrono)
        # statistics for every layer are accumulated step by step
        stats = {l: [] for l in layers}
        seen = []
        t0 = time.time()
        for t, new in enumerate(steps):
            seen += new
            unseen = [g for g in names if g not in seen]
            crit = {}       # (layer, weighting) -> dict of arrays over lambda
            test = {}       # (layer, weighting) -> [n_gen, n_lambda] true EER
            itw_eer = {}
            for l in layers:
                for g in new:
                    a, b = arrive[g]
                    stats[l] += halves_groups(g, "spoof", F.layer(l, a), g, rng)
                    stats[l] += halves_groups("bona_" + g, "bona", F.layer(l, b), g, rng)
                layer_groups = on_device(stats[l], dev)     # only this layer's statistics are on the GPU
                full_groups = merged(layer_groups)
                if l not in test_cache:      # test features never change: keep them in RAM (float32)
                    test_cache[l] = dict(
                        b={d: torch.from_numpy(F.layer(l, bona_test_idx[d])) for d in domains},
                        s={g: torch.from_numpy(F.layer(l, P.gens[g]["test"])) for g in names},
                        ib=torch.from_numpy(itw.layer(l, itw_b)) if itw is not None else None,
                        is_=torch.from_numpy(itw.layer(l, itw_s)) if itw is not None else None)
                tc = test_cache[l]
                Xb = {d: tc["b"][d].to(dev, DT64) for d in domains}
                Xs = {g: tc["s"][g].to(dev, DT64) for g in names}
                if itw is not None:
                    Xib, Xis = tc["ib"].to(dev, DT64), tc["is_"].to(dev, DT64)
                for wt in WEIGHTINGS:
                    fc = gss.logo_forecast(full_groups, LAMBDAS, wt)
                    fcm = torch.stack([fc[k] for k in sorted(fc)])
                    idc = idcv_forecast(layer_groups, LAMBDAS, wt)
                    crit[(l, wt)] = dict(logo=fcm.mean(0).cpu().numpy(), logo_max=fcm.max(0).values.cpu().numpy(),
                                         idcv=idc.mean(0).cpu().numpy(), idcv_max=idc.max(0).values.cpu().numpy())
                    W, b = gss.solve(full_groups, LAMBDAS, wt)
                    sb = {d: Xb[d] @ W.T + b for d in domains}
                    E = torch.stack([gss.eer_torch(sb[P.gens[g]["domain"]], Xs[g] @ W.T + b) for g in names])
                    test[(l, wt)] = E.cpu().numpy()
                    if itw is not None:
                        itw_eer[(l, wt)] = gss.eer_torch(Xib @ W.T + b, Xis @ W.T + b).cpu().numpy()
                del Xb, Xs, layer_groups, full_groups
                torch.cuda.empty_cache()
            # ---- flatten the grid
            keys = [(l, wt, j) for l in layers for wt in WEIGHTINGS for j in range(len(LAMBDAS))]
            seen_i = [names.index(g) for g in seen]; un_i = [names.index(g) for g in unseen]
            def col(name):
                return np.array([crit[(l, wt)][name][j] for (l, wt, j) in keys])
            T = np.stack([test[(l, wt)][:, j] for (l, wt, j) in keys])          # [n_cfg, n_gen]
            seen_mean = T[:, seen_i].mean(1)
            unseen_mean = T[:, un_i].mean(1) if un_i else np.full(len(keys), np.nan)
            all_mean = T.mean(1)
            itw_all = np.array([itw_eer[(l, wt)][j] for (l, wt, j) in keys]) if itw is not None else None
            fixed = keys.index((layers[-1], "pooled", LAMBDAS.index(1e-2)))
            picks = dict(logo=int(col("logo").argmin()), logo_max=int(col("logo_max").argmin()),
                         idcv=int(col("idcv").argmin()), idcv_max=int(col("idcv_max").argmin()), fixed=fixed,
                         oracle_all=int(all_mean.argmin()),
                         oracle_unseen=int(unseen_mean.argmin()) if un_i else int(all_mean.argmin()))
            rec = dict(order_id=order_id, order=order, step=t, seen=list(seen), n_seen=len(seen),
                       condition=args.condition, model=args.model, layers=layers,
                       n_arrive={g: int(len(arrive[g][0])) for g in new}, picks={}, names=names,
                       grid=dict(logo=col("logo").tolist(), logo_max=col("logo_max").tolist(),
                                 idcv=col("idcv").tolist(), seen_mean=seen_mean.tolist(),
                                 unseen_mean=unseen_mean.tolist(), all_mean=all_mean.tolist(),
                                 itw=itw_all.tolist() if itw_all is not None else None))
            if args.save_T:
                rec["grid"]["T"] = np.round(T, 5).tolist()
                rec["grid"]["keys"] = [[l, wt, LAMBDAS[j]] for (l, wt, j) in keys]
            rec["domain"] = {g: P.gens[g]["domain"] for g in names}
            rec["family"] = {g: P.gens[g]["family"] for g in names}
            for name, k in picks.items():
                l, wt, j = keys[k]
                rec["picks"][name] = dict(layer=l, weighting=wt, lam=LAMBDAS[j], crit_logo=float(col("logo")[k]),
                                          crit_idcv=float(col("idcv")[k]),
                                          per_gen={g: float(T[k, i]) for i, g in enumerate(names)},
                                          seen=float(seen_mean[k]), unseen=float(unseen_mean[k]), all=float(all_mean[k]),
                                          itw=float(itw_all[k]) if itw_all is not None else None)
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            p = rec["picks"]
            print(f"order {order_id} step {t:2d} seen {len(seen):2d} | unseen EER%: logo {p['logo']['unseen']*100:5.2f} "
                  f"idcv {p['idcv']['unseen']*100:5.2f} fixed {p['fixed']['unseen']*100:5.2f} oracle {p['oracle_unseen']['unseen']*100:5.2f}"
                  f" | seen: logo {p['logo']['seen']*100:5.2f} idcv {p['idcv']['seen']*100:5.2f} fixed {p['fixed']['seen']*100:5.2f}"
                  f" | logo cfg L{p['logo']['layer']} {p['logo']['weighting']} {p['logo']['lam']:.0e}; idcv cfg L{p['idcv']['layer']} {p['idcv']['lam']:.0e}"
                  f" | {time.time()-t0:.0f}s", flush=True)
    fout.close()


if __name__ == "__main__":
    main()
