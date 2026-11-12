"""Break stream results down by corpus (ASVspoof LA vs DFADD) and by generator family (TTS / VC / hybrid).

For every rule: EER on unseen generators of each domain / family, averaged over the steps at which at
least one such generator is unseen.  For chronological streams additionally: EER on the (all unseen)
DFADD generators at the step where all 17 ASVspoof generators have been seen, and the rank correlation
between each criterion and that EER over the configuration grid (needs --save_T).
"""
import json, os, sys, glob
import numpy as np
from scipy.stats import spearmanr, wilcoxon
from data import ROOT


def ms(x):
    x = np.asarray(x, dtype=float)
    return dict(mean=float(np.mean(x)), std=float(np.std(x, ddof=1)) if len(x) > 1 else 0.0, n=int(len(x)))


def run(path):
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if "domain" not in recs[0]:
        return None
    names = recs[0]["names"]; dom = recs[0]["domain"]; fam = recs[0]["family"]
    dom2 = {g: ("DFADD" if dom[g] == "DFADD" else "LA") for g in names}
    rules = list(recs[0]["picks"].keys())
    orders = sorted({r["order_id"] for r in recs})
    out = dict(file=os.path.basename(path), by_domain={}, by_family={}, handover={}, handover_corr={}, tests={})
    for label, key in [("by_domain", dom2), ("by_family", fam)]:
        groups = sorted(set(key.values()))
        for grp in groups:
            per_rule = {rule: [] for rule in rules}
            for o in orders:
                rs = [r for r in recs if r["order_id"] == o]
                for rule in rules:
                    vals = []
                    for r in rs:
                        un = [g for g in names if g not in r["seen"] and key[g] == grp]
                        if un:
                            vals.append(np.mean([r["picks"][rule]["per_gen"][g] for g in un]))
                    if vals:
                        per_rule[rule].append(np.mean(vals))
            out[label][grp] = {rule: ms(v) for rule, v in per_rule.items() if v}
            if len(per_rule["logo"]) >= 6:
                d = np.array(per_rule["logo"]) - np.array(per_rule["idcv"])
                out["tests"][f"{label}:{grp}:logo_vs_idcv"] = dict(mean_diff=float(d.mean()), wins=int((d < 0).sum()), n=len(d),
                                                                  p=float(wilcoxon(d).pvalue) if np.any(d != 0) else None)
                d = np.array(per_rule["logo"]) - np.array(per_rule["fixed"])
                out["tests"][f"{label}:{grp}:logo_vs_fixed"] = dict(mean_diff=float(d.mean()), wins=int((d < 0).sum()), n=len(d),
                                                                   p=float(wilcoxon(d).pvalue) if np.any(d != 0) else None)
    # hand-over step: every LA generator seen, no DFADD generator seen
    la = [g for g in names if dom2[g] == "LA"]; df = [g for g in names if dom2[g] == "DFADD"]
    hand = [r for r in recs if set(r["seen"]) == set(la)]
    if hand and df:
        for rule in rules:
            out["handover"][rule] = dict(dfadd=ms([np.mean([r["picks"][rule]["per_gen"][g] for g in df]) for r in hand]),
                                         itw=ms([r["picks"][rule]["itw"] for r in hand]) if hand[0]["picks"][rule].get("itw") is not None else None,
                                         layers=[r["picks"][rule]["layer"] for r in hand])
        d = np.array([np.mean([r["picks"]["logo"]["per_gen"][g] for g in df]) for r in hand]) - np.array([np.mean([r["picks"]["idcv"]["per_gen"][g] for g in df]) for r in hand])
        out["tests"]["handover:logo_vs_idcv"] = dict(mean_diff=float(d.mean()), wins=int((d < 0).sum()), n=len(d), p=float(wilcoxon(d).pvalue) if len(d) >= 6 and np.any(d != 0) else None)
        if "T" in hand[0]["grid"]:
            di = [names.index(g) for g in df]
            cc = {k: [] for k in ["logo", "logo_max", "idcv"]}
            for r in hand:
                tgt = np.array(r["grid"]["T"])[:, di].mean(1)
                for k in cc:
                    cc[k].append(spearmanr(r["grid"][k], tgt).correlation)
            out["handover_corr"] = {k: ms(v) for k, v in cc.items()}
    # correlation with unseen-DFADD EER at every step where a DFADD generator is unseen
    if "T" in recs[0]["grid"] and df:
        cc = {k: [] for k in ["logo", "idcv"]}
        for r in recs:
            un = [names.index(g) for g in df if g not in r["seen"]]
            if un:
                tgt = np.array(r["grid"]["T"])[:, un].mean(1)
                for k in cc:
                    cc[k].append(spearmanr(r["grid"][k], tgt).correlation)
        out["corr_unseen_dfadd"] = {k: ms(v) for k, v in cc.items()}
    return out


if __name__ == "__main__":
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(ROOT, "results", "stream_*.jsonl")))
    for fp in files:
        s = run(fp)
        if s is None:
            continue
        print("\n===", s["file"])
        for label in ["by_domain", "by_family"]:
            for grp, v in s[label].items():
                print(f"  unseen {grp:8s}: " + "  ".join(f"{rule} {x['mean']*100:.2f}±{x['std']*100:.2f}" for rule, x in v.items() if rule in ("logo", "idcv", "fixed", "oracle_unseen")))
        if s["handover"]:
            print("  hand-over (all LA seen, DFADD unseen): " + "  ".join(f"{rule} DFADD {v['dfadd']['mean']*100:.2f}±{v['dfadd']['std']*100:.2f}" + (f" ITW {v['itw']['mean']*100:.2f}" if v['itw'] else "") for rule, v in s["handover"].items() if rule in ("logo", "idcv", "fixed", "oracle_unseen")))
            print("  hand-over corr:", {k: round(v['mean'], 3) for k, v in s["handover_corr"].items()})
        if "corr_unseen_dfadd" in s:
            print("  corr with unseen-DFADD EER:", {k: round(v['mean'], 3) for k, v in s["corr_unseen_dfadd"].items()})
        for k, v in s["tests"].items():
            print(f"  test {k}: diff {v['mean_diff']*100:+.2f} wins {v['wins']}/{v['n']} p={v['p']}")
        json.dump(s, open(os.path.join(ROOT, "results", "domain_" + s["file"].replace(".jsonl", ".json")), "w"), indent=1)
