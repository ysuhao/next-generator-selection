"""Generator-incremental stream with closed-form (statistics-only) detectors — fast version.

Same protocol, same criteria and same output format as stream.py.  The only difference is where the
linear algebra runs: every leave-one-generator-out fold, every in-distribution CV fold and the full
solve is one job in a CPU thread pool (LAPACK releases the GIL), using running totals so that a fold's
moments are obtained by subtracting the held-out group from the totals.  Test scoring stays on the GPU.

Selection rules at each step (all choose layer x lambda x weighting):
  logo / logo_max : statistics-only leave-one-generator-out Gaussian EER (mean / worst fold)   [proposed]
  idcv / idcv_max : statistics-only 2-fold in-distribution cross-validation Gaussian EER
  fixed           : last layer, lambda = 1e-2, pooled weighting (plain analytic learner)
  oracle_*        : best configuration in hindsight (reference only)
"""
import argparse, json, os, time, math
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from scipy.special import erfc
from threadpoolctl import threadpool_limits
import gss
from pools import get_pool, make_stream
from data import ROOT

LAMBDAS = [10.0 ** e for e in np.arange(-5, 2.01, 0.5)]
LAM = np.array(LAMBDAS)
WEIGHTINGS = ["pooled", "balanced"]
GPU_EIGH = False
IDCV_PAIRED = False


class Tot:
    """Running totals of one class in one layer."""

    def __init__(self, d):
        self.n = 0; self.s = np.zeros(d); self.S = np.zeros((d, d))
        self.G = 0; self.sum_mu = np.zeros(d); self.sum_M = np.zeros((d, d))

    def add(self, n, s, S):
        self.n += n; self.s += s; self.S += S
        self.G += 1; self.sum_mu += s / n; self.sum_M += S / n


def class_moments(tot, weighting, minus=None):
    """Mean and second moment of a class mixture, optionally without one group (n, s, S)."""
    if weighting == "pooled":
        if minus is None:
            return tot.s / tot.n, tot.S / tot.n
        n = tot.n - minus[0]
        return (tot.s - minus[1]) / n, (tot.S - minus[2]) / n
    if minus is None:
        return tot.sum_mu / tot.G, tot.sum_M / tot.G
    return (tot.sum_mu - minus[1] / minus[0]) / (tot.G - 1), (tot.sum_M - minus[2] / minus[0]) / (tot.G - 1)


def solve_np(mu_b, M_b, mu_s, M_s):
    A = 0.5 * M_b + 0.5 * M_s
    m = 0.5 * mu_b + 0.5 * mu_s
    c = 0.5 * mu_b - 0.5 * mu_s
    C = A - np.outer(m, m)
    C = 0.5 * (C + C.T)
    if GPU_EIGH:                      # optional: decomposition on the GPU (same float64 arithmetic)
        e_t, V_t = torch.linalg.eigh(torch.as_tensor(C, device="cuda"))
        e, V = e_t.cpu().numpy(), V_t.cpu().numpy()
    else:
        e, V = np.linalg.eigh(C)
    e = np.clip(e, 0, None)
    lam = LAM * e.mean()
    r = V.T @ c
    W = (r[None, :] / (e[None, :] + lam[:, None])) @ V.T
    return W, -(W @ m)


def stat_scores(W, b, g):
    n, s, S = g
    mu = s / n
    mean = W @ mu + b
    var = np.einsum("ij,ij->i", W @ (S / n - np.outer(mu, mu)), W)
    return mean, np.clip(var, 1e-12, None)


def gauss_eer(mb, vb, ms, vs):
    return 0.5 * erfc((mb - ms) / (np.sqrt(vb) + np.sqrt(vs)) / math.sqrt(2.0))


def job_logo(L, wt, g):
    sp, bo = L["sp"][g], L["bo"][g]
    mu_b, M_b = class_moments(L["tb"], "pooled", bo)
    mu_s, M_s = class_moments(L["ts"], wt, sp)
    W, b = solve_np(mu_b, M_b, mu_s, M_s)
    return gauss_eer(*stat_scores(W, b, bo), *stat_scores(W, b, sp))


def job_idcv(L, wt, h, seen):
    mu_b, M_b = class_moments(L["tbh"][h], "pooled")
    mu_s, M_s = class_moments(L["tsh"][h], wt)
    W, b = solve_np(mu_b, M_b, mu_s, M_s)
    if IDCV_PAIRED:     # variant: each generator's held-out half against the held-out half of ITS OWN bona fide batch (as LOGO pairs them)
        return [gauss_eer(*stat_scores(W, b, L["boh"][g][1 - h]), *stat_scores(W, b, L["sph"][g][1 - h]))
                for g in seen if L["sph"][g][1 - h][0] > 0 and L["boh"][g][1 - h][0] > 0]
    o = L["tbh"][1 - h]
    mb, vb = stat_scores(W, b, (o.n, o.s, o.S))
    return [gauss_eer(mb, vb, *stat_scores(W, b, L["sph"][g][1 - h])) for g in seen if L["sph"][g][1 - h][0] > 0]


def job_full(L, wt):
    return solve_np(*class_moments(L["tb"], "pooled"), *class_moments(L["ts"], wt))


def stats3(X):
    X = np.asarray(X, dtype=np.float64)
    return (X.shape[0], X.sum(0), X.T @ X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="wavlm-large")
    ap.add_argument("--pool", default="mean")
    ap.add_argument("--orders", type=int, default=10)
    ap.add_argument("--order_start", type=int, default=0)
    ap.add_argument("--base", type=int, default=3)
    ap.add_argument("--condition", default="full")
    ap.add_argument("--layers", default="all", help="'all', 'odd' (every second layer incl. last) or comma list")
    ap.add_argument("--tag", default="")
    ap.add_argument("--itw", action="store_true")
    ap.add_argument("--poolname", default="la")
    ap.add_argument("--chrono", action="store_true")
    ap.add_argument("--save_T", action="store_true")
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    ap.add_argument("--gpu_eigh", action="store_true")
    ap.add_argument("--exclude_family", default=None, help="drop all generators of this family from the pool (e.g. TTS-VC)")
    ap.add_argument("--bona_split", default="speaker", choices=["speaker", "utt"], help="how bona fide speech is cut into arriving chunks")
    ap.add_argument("--family_last", default=None, help="generators of this family arrive last (e.g. VC)")
    ap.add_argument("--idcv_paired", action="store_true", help="ID-CV variant: same split for all layers, paired bona fide halves")
    args = ap.parse_args()
    global GPU_EIGH, IDCV_PAIRED
    GPU_EIGH = args.gpu_eigh
    IDCV_PAIRED = args.idcv_paired
    if GPU_EIGH:      # initialise the CUDA linear-algebra backend once, before worker threads use it
        torch.linalg.eigh(torch.eye(4, dtype=torch.float64, device="cuda"))

    P = get_pool(args.poolname, args.model, args.pool, bona_split=args.bona_split)
    if args.exclude_family:
        for g in [g for g in P.names() if P.gens[g]["family"] == args.exclude_family]:
            del P.gens[g]
    F = P.F
    if args.layers == "all":
        layers = list(range(F.L))
    elif args.layers == "odd":
        layers = list(range(0, F.L, 2))
    else:
        layers = [int(x) for x in args.layers.split(",")]
    names = P.names()
    dev = gss.dev()
    DT = gss.DT
    out_path = os.path.join(ROOT, "results", f"stream_{args.poolname}_{args.model}_{args.pool}_{args.condition}{'_chrono' if args.chrono else ''}{('_last' + args.family_last) if args.family_last else ''}{('_no' + args.exclude_family) if args.exclude_family else ''}{'_bonautt' if args.bona_split == 'utt' else ''}{'_idcvpaired' if args.idcv_paired else ''}{args.tag}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    domains = sorted({P.gens[g]["domain"] for g in names})
    bona_test_idx = {d: next(P.gens[g]["bona_test"] for g in names if P.gens[g]["domain"] == d) for d in domains}
    itw = None
    if args.itw:
        from data import Feats
        itw = Feats("itw", args.model, args.pool)
        itw_b, itw_s = itw.where(label="bonafide"), itw.where(label="spoof")
    # test features on the GPU once, in half precision (the cached features are float16, so this is lossless)
    test = {}
    for l in layers:
        test[l] = dict(b={d: torch.from_numpy(F.layer(l, bona_test_idx[d])).half().to(dev) for d in domains},
                       s={g: torch.from_numpy(F.layer(l, P.gens[g]["test"])).half().to(dev) for g in names},
                       ib=torch.from_numpy(itw.layer(l, itw_b)).half().to(dev) if itw is not None else None,
                       is_=torch.from_numpy(itw.layer(l, itw_s)).half().to(dev) if itw is not None else None)
    print("test features cached", flush=True)
    ex = ThreadPoolExecutor(args.workers)
    fout = open(out_path, "a")
    with threadpool_limits(limits=1):
        for order_id in range(args.order_start, args.order_start + args.orders):
            order, steps, arrive, rng = make_stream(P, order_id, args.condition, args.base, args.chrono, args.family_last)
            Ls = {l: dict(sp={}, bo={}, sph={}, boh={}, ts=Tot(F.D), tb=Tot(F.D),
                          tsh=[Tot(F.D), Tot(F.D)], tbh=[Tot(F.D), Tot(F.D)]) for l in layers}
            seen = []
            t0 = time.time()
            for t, new in enumerate(steps):
                seen += new
                unseen = [g for g in names if g not in seen]

                def ingest(l):
                    L = Ls[l]
                    r = np.random.RandomState(hash((order_id, t) if IDCV_PAIRED else (order_id, l, t)) % (2 ** 31))
                    for g in new:
                        for kind, idx, full, halves, tot, toth in (("sp", arrive[g][0], L["sp"], L["sph"], L["ts"], L["tsh"]),
                                                                    ("bo", arrive[g][1], L["bo"], L["boh"], L["tb"], L["tbh"])):
                            X = F.layer(l, idx).astype(np.float64)
                            perm = r.permutation(len(X))
                            hs = [stats3(X[perm[h::2]]) for h in (0, 1)]
                            full[g] = (hs[0][0] + hs[1][0], hs[0][1] + hs[1][1], hs[0][2] + hs[1][2])
                            halves[g] = hs
                            tot.add(*full[g])
                            for h in (0, 1):
                                if hs[h][0] > 0:
                                    toth[h].add(*hs[h])
                list(ex.map(ingest, layers))
                # ---- submit all linear algebra of this step
                fut_logo = {(l, wt, g): ex.submit(job_logo, Ls[l], wt, g) for l in layers for wt in WEIGHTINGS for g in seen}
                fut_id = {(l, wt, h): ex.submit(job_idcv, Ls[l], wt, h, list(seen)) for l in layers for wt in WEIGHTINGS for h in (0, 1)}
                fut_full = {(l, wt): ex.submit(job_full, Ls[l], wt) for l in layers for wt in WEIGHTINGS}
                crit, tst, itw_eer = {}, {}, {}
                for l in layers:
                    for wt in WEIGHTINGS:
                        fcm = np.stack([fut_logo[(l, wt, g)].result() for g in seen])
                        idc = np.stack(fut_id[(l, wt, 0)].result() + fut_id[(l, wt, 1)].result())
                        crit[(l, wt)] = dict(logo=fcm.mean(0), logo_max=fcm.max(0), idcv=idc.mean(0), idcv_max=idc.max(0))
                        W, b = fut_full[(l, wt)].result()
                        W = torch.as_tensor(W, dtype=DT, device=dev); b = torch.as_tensor(b, dtype=DT, device=dev)
                        tc = test[l]
                        sb = {d: tc["b"][d].to(DT) @ W.T + b for d in domains}
                        E = torch.stack([gss.eer_torch(sb[P.gens[g]["domain"]], tc["s"][g].to(DT) @ W.T + b) for g in names])
                        tst[(l, wt)] = E.cpu().numpy()
                        if itw is not None:
                            itw_eer[(l, wt)] = gss.eer_torch(tc["ib"].to(DT) @ W.T + b, tc["is_"].to(DT) @ W.T + b).cpu().numpy()
                # ---- flatten the grid
                keys = [(l, wt, j) for l in layers for wt in WEIGHTINGS for j in range(len(LAMBDAS))]
                seen_i = [names.index(g) for g in seen]; un_i = [names.index(g) for g in unseen]
                col = lambda name: np.array([crit[(l, wt)][name][j] for (l, wt, j) in keys])
                T = np.stack([tst[(l, wt)][:, j] for (l, wt, j) in keys])
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
                                     unseen_mean=[None if np.isnan(x) else float(x) for x in unseen_mean], all_mean=all_mean.tolist(),
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
                                              seen=float(seen_mean[k]), unseen=None if not un_i else float(unseen_mean[k]), all=float(all_mean[k]),
                                              itw=float(itw_all[k]) if itw_all is not None else None)
                fout.write(json.dumps(rec) + "\n"); fout.flush()
                p = rec["picks"]
                u = lambda r: (float("nan") if p[r]["unseen"] is None else p[r]["unseen"]) * 100
                print(f"order {order_id} step {t:2d} seen {len(seen):2d} | unseen EER%: logo {u('logo'):5.2f} idcv {u('idcv'):5.2f} "
                      f"fixed {u('fixed'):5.2f} oracle {u('oracle_unseen'):5.2f} | seen: logo {p['logo']['seen']*100:5.2f} "
                      f"idcv {p['idcv']['seen']*100:5.2f} fixed {p['fixed']['seen']*100:5.2f} | logo L{p['logo']['layer']} "
                      f"{p['logo']['weighting']} {p['logo']['lam']:.0e}; idcv L{p['idcv']['layer']} {p['idcv']['lam']:.0e} | {time.time()-t0:.0f}s", flush=True)
    fout.close()


if __name__ == "__main__":
    main()
