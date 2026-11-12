"""Summarise stream_*.jsonl (closed-form learners) and grad_*.jsonl (gradient baselines).

Prints, per condition:
  * time-averaged EER on unseen / seen generators and final EER on all generators, mean +- std over orders
  * paired Wilcoxon signed-rank test LOGO vs ID-CV over orders (time-averaged unseen EER)
  * Spearman correlation between each criterion and the true unseen-generator EER across the configuration grid
  * forgetting of gradient baselines
Writes results/summary_<name>.json with every number that goes into the paper.
"""
import argparse, glob, json, os, sys
import numpy as np
from scipy.stats import spearmanr, wilcoxon
from data import ROOT


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def ms(x):
    x = np.asarray(x, dtype=float)
    return dict(mean=float(np.mean(x)), std=float(np.std(x, ddof=1)) if len(x) > 1 else 0.0, n=int(len(x)))


def stream_summary(path):
    recs = load(path)
    orders = sorted({r["order_id"] for r in recs})
    rules = list(recs[0]["picks"].keys())
    out = dict(file=os.path.basename(path), n_orders=len(orders), rules={}, corr={}, tests={}, chosen={})
    per_order = {rule: dict(unseen=[], seen=[], final_all=[], itw=[]) for rule in rules}
    for o in orders:
        rs = sorted([r for r in recs if r["order_id"] == o], key=lambda r: r["step"])
        for rule in rules:
            un = [r["picks"][rule]["unseen"] for r in rs if r["n_seen"] < len(r["names"])]
            per_order[rule]["unseen"].append(np.mean(un))
            per_order[rule]["seen"].append(np.mean([r["picks"][rule]["seen"] for r in rs]))
            per_order[rule]["final_all"].append(rs[-1]["picks"][rule]["all"])
            if rs[0]["picks"][rule].get("itw") is not None:
                per_order[rule]["itw"].append(np.mean([r["picks"][rule]["itw"] for r in rs]))
    for rule in rules:
        out["rules"][rule] = {k: ms(v) for k, v in per_order[rule].items() if len(v)}
    for a, b in [("logo", "idcv"), ("logo", "fixed"), ("logo_max", "idcv"), ("idcv", "fixed")]:
        if a in rules and b in rules:
            d = np.array(per_order[a]["unseen"]) - np.array(per_order[b]["unseen"])
            p = float(wilcoxon(d).pvalue) if len(d) >= 6 and np.any(d != 0) else None
            out["tests"][f"{a}_vs_{b}_unseen"] = dict(mean_diff=float(d.mean()), wins=int((d < 0).sum()), n=int(len(d)), wilcoxon_p=p)
    # rank correlations per step
    cors = {k: [] for k in ["logo", "logo_max", "idcv"]}
    cors_itw = {k: [] for k in ["logo", "idcv"]}
    for r in recs:
        g = r["grid"]
        if r["n_seen"] < len(r["names"]):
            for k in cors:
                cors[k].append(spearmanr(g[k], g["unseen_mean"]).correlation)
        if g.get("itw") is not None:
            for k in cors_itw:
                cors_itw[k].append(spearmanr(g[k], g["itw"]).correlation)
    out["corr"] = {k: ms(v) for k, v in cors.items()}
    out["corr_itw"] = {k: ms(v) for k, v in cors_itw.items() if len(v)}
    # by number of seen generators
    by_n = {}
    for r in recs:
        if r["n_seen"] < len(r["names"]):
            d = by_n.setdefault(r["n_seen"], {rule: [] for rule in rules})
            for rule in rules:
                d[rule].append(r["picks"][rule]["unseen"])
    out["unseen_by_nseen"] = {int(n): {rule: float(np.mean(v)) for rule, v in d.items()} for n, d in sorted(by_n.items())}
    by_n = {}
    for r in recs:
        d = by_n.setdefault(r["n_seen"], {rule: [] for rule in rules})
        for rule in rules:
            d[rule].append(r["picks"][rule]["seen"])
    out["seen_by_nseen"] = {int(n): {rule: float(np.mean(v)) for rule, v in d.items()} for n, d in sorted(by_n.items())}
    # what gets chosen
    for rule in ["logo", "idcv", "oracle_unseen"]:
        if rule in rules:
            layers = [r["picks"][rule]["layer"] for r in recs]
            lam = [np.log10(r["picks"][rule]["lam"]) for r in recs]
            bal = [r["picks"][rule]["weighting"] == "balanced" for r in recs]
            out["chosen"][rule] = dict(layer_mean=float(np.mean(layers)), layer_hist=np.bincount(layers, minlength=25).tolist(),
                                       log10_lam_mean=float(np.mean(lam)), balanced_frac=float(np.mean(bal)))
    return out


def grad_summary(path):
    recs = load(path)
    methods = sorted({r["method"] for r in recs})
    out = dict(file=os.path.basename(path), methods={})
    for m in methods:
        orders = sorted({r["order_id"] for r in recs if r["method"] == m})
        un, se, fa, fo, itw = [], [], [], [], []
        for o in orders:
            rs = sorted([r for r in recs if r["method"] == m and r["order_id"] == o], key=lambda r: r["step"])
            un.append(np.mean([r["unseen_eer"] for r in rs if r["unseen_eer"] is not None]))
            se.append(np.mean([r["seen_eer"] for r in rs]))
            fa.append(rs[-1]["all_eer"])
            if rs[0].get("itw") is not None:
                itw.append(np.mean([r["itw"] for r in rs]))
            # forgetting: final EER minus best EER since the generator was first seen
            f = []
            for g in rs[-1]["seen"]:
                hist = [r["per_gen"][g] for r in rs if g in r["seen"]]
                f.append(hist[-1] - min(hist))
            fo.append(np.mean(f))
        out["methods"][m] = dict(unseen=ms(un), seen=ms(se), final_all=ms(fa), forgetting=ms(fo), n_orders=len(orders))
        if itw:
            out["methods"][m]["itw"] = ms(itw)
    return out


def stream_forgetting(path, rule):
    """Forgetting of a closed-form rule (non-zero only because the selected configuration can change)."""
    recs = load(path)
    fo = []
    for o in sorted({r["order_id"] for r in recs}):
        rs = sorted([r for r in recs if r["order_id"] == o], key=lambda r: r["step"])
        f = []
        for g in rs[-1]["seen"]:
            hist = [r["picks"][rule]["per_gen"][g] for r in rs if g in r["seen"]]
            f.append(hist[-1] - min(hist))
        fo.append(np.mean(f))
    return ms(fo)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    args = ap.parse_args()
    files = args.files or sorted(glob.glob(os.path.join(ROOT, "results", "stream_*.jsonl")) + glob.glob(os.path.join(ROOT, "results", "grad_*.jsonl")))
    for fp in files:
        name = os.path.basename(fp)
        if name.startswith("stream_"):
            s = stream_summary(fp)
            for rule in ["logo", "idcv", "fixed"]:
                s.setdefault("forgetting", {})[rule] = stream_forgetting(fp, rule)
            print(f"\n=== {name}  ({s['n_orders']} orders) ===")
            print(f"{'rule':14s} {'unseen(avg)':>14s} {'seen(avg)':>14s} {'final all':>14s} {'ITW(avg)':>14s}")
            for rule, v in s["rules"].items():
                f = lambda k: f"{v[k]['mean']*100:6.2f}±{v[k]['std']*100:4.2f}" if k in v else "      -     "
                print(f"{rule:14s} {f('unseen'):>14s} {f('seen'):>14s} {f('final_all'):>14s} {f('itw'):>14s}")
            for k, v in s["tests"].items():
                print(f"  {k}: mean diff {v['mean_diff']*100:+.2f} pts, wins {v['wins']}/{v['n']}, Wilcoxon p={v['wilcoxon_p']}")
            print("  Spearman(criterion, true unseen EER) over grid, mean over steps:",
                  {k: round(v['mean'], 3) for k, v in s["corr"].items()})
            if s["corr_itw"]:
                print("  Spearman(criterion, ITW EER):", {k: round(v['mean'], 3) for k, v in s["corr_itw"].items()})
            print("  chosen:", {k: (round(v['layer_mean'], 1), round(v['log10_lam_mean'], 2), round(v['balanced_frac'], 2)) for k, v in s["chosen"].items()})
            print("  forgetting:", {k: round(v['mean'] * 100, 3) for k, v in s["forgetting"].items()})
        else:
            s = grad_summary(fp)
            print(f"\n=== {name} ===")
            print(f"{'method':14s} {'unseen(avg)':>14s} {'seen(avg)':>14s} {'final all':>14s} {'forgetting':>14s} {'ITW':>14s}")
            for m, v in s["methods"].items():
                f = lambda k: f"{v[k]['mean']*100:6.2f}±{v[k]['std']*100:4.2f}" if k in v else "      -     "
                print(f"{m:14s} {f('unseen'):>14s} {f('seen'):>14s} {f('final_all'):>14s} {f('forgetting'):>14s} {f('itw'):>14s}  (n={v['n_orders']})")
        json.dump(s, open(os.path.join(ROOT, "results", "summary_" + name.replace(".jsonl", ".json")), "w"), indent=1)
