"""Post-hoc re-analysis of stored criteria (no new training): how often LOGO and ID-CV disagree,
which of them is right when they do, and how a rank-average of the two criteria would have performed.

EXPLORATORY: the combined rule was formulated after the results of both family-shift experiments were
known and is evaluated on the same streams; it is reported as such in the paper.

Disagreement: the layers selected by the two criteria differ by at least four (the same threshold for
25-layer grids and for grids with every second layer).  States that are identical in every order (all
earlier generators seen, i.e. the hand-over state) are excluded from the disagreement counts, because
they are not independent across orders; the remaining steps of one order are still not independent of
each other, so the counts are descriptive.
Writes results/combined.json.
"""
import json, os, sys
import numpy as np
from scipy.stats import rankdata, wilcoxon
from data import ROOT

R = os.path.join(ROOT, "results")
NL = 15
THR = 4


def load(name):
    p = os.path.join(R, name)
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else None


def ms(x):
    x = np.asarray(x, dtype=float)
    return [float(x.mean() * 100), float(x.std(ddof=1) * 100) if len(x) > 1 else 0.0]


def wil(a, b):
    d = np.asarray(a) - np.asarray(b)
    return dict(diff=float(d.mean() * 100), wins=int((d < 0).sum()), n=int(len(d)), p=float(wilcoxon(d).pvalue) if len(d) >= 6 and np.any(d != 0) else None)


def analyse(S, target=None, late=None):
    """target: optional set of generator names; if given (needs grid.T) the error is the mean EER on the
    generators of that set, restricted to steps where none of them has been seen.
    late: (lo, hi) range of n_seen for the 'late' summary (target analyses only)."""
    layers = S[0]["layers"]
    lay_of = np.repeat(layers, 2 * NL)
    names = S[0]["names"]
    orders = sorted({r["order_id"] for r in S})
    # states (sets of seen generators) that occur in every order are deterministic repeats
    occ = {}
    for r in S:
        occ.setdefault(frozenset(r["seen"]), set()).add(r["order_id"])
    repeated = {k for k, v in occ.items() if len(v) == len(orders)}
    rules = ["logo", "idcv", "rankavg", "oracle"]
    per = {k: {} for k in rules}; per_late = {k: {} for k in rules}; hand = {k: [] for k in rules}
    dis = dict(n=0, n_dis=0, logo_better=0, idcv_better=0, gap=[], gap_dis=[], gap_agree=[])
    n_other = len(names) - (len(target) if target else 0)
    for r in S:
        if r["n_seen"] >= len(names):
            continue
        g = r["grid"]
        if target is not None:
            if any(x in target for x in r["seen"]):
                continue
            ti = [names.index(x) for x in names if x in target]
            err = np.array(g["T"])[:, ti].mean(1)
        else:
            err = np.array(g["unseen_mean"], dtype=float)
        lo, ic = np.array(g["logo"]), np.array(g["idcv"])
        pick = dict(logo=int(lo.argmin()), idcv=int(ic.argmin()), rankavg=int((rankdata(lo) + rankdata(ic)).argmin()), oracle=int(err.argmin()))
        for k, i in pick.items():
            per[k].setdefault(r["order_id"], []).append(err[i])
            if target is not None and late and late[0] <= r["n_seen"] <= late[1]:
                per_late[k].setdefault(r["order_id"], []).append(err[i])
            if target is not None and r["n_seen"] == n_other:
                hand[k].append(err[i])
        if frozenset(r["seen"]) in repeated:
            continue
        d = abs(int(lay_of[pick["logo"]]) - int(lay_of[pick["idcv"]]))
        gap = abs(err[pick["logo"]] - err[pick["idcv"]])
        dis["n"] += 1; dis["gap"].append(gap)
        if d >= THR:
            dis["n_dis"] += 1; dis["gap_dis"].append(gap)
            dis["logo_better"] += int(err[pick["logo"]] < err[pick["idcv"]]); dis["idcv_better"] += int(err[pick["idcv"]] < err[pick["logo"]])
        else:
            dis["gap_agree"].append(gap)
    po = {k: np.array([np.mean(per[k][o]) for o in orders]) for k in per}
    out = dict(n_orders=len(orders), eer={k: ms(v) for k, v in po.items()},
               rankavg_vs_logo=wil(po["rankavg"], po["logo"]), rankavg_vs_idcv=wil(po["rankavg"], po["idcv"]), logo_vs_idcv=wil(po["logo"], po["idcv"]),
               steps=dis["n"], disagree_frac=dis["n_dis"] / max(dis["n"], 1), logo_better_when_disagree=dis["logo_better"], idcv_better_when_disagree=dis["idcv_better"],
               mean_abs_gap_all=float(np.mean(dis["gap"]) * 100) if dis["gap"] else None,
               mean_abs_gap_disagree=float(np.mean(dis["gap_dis"]) * 100) if dis["gap_dis"] else None,
               mean_abs_gap_agree=float(np.mean(dis["gap_agree"]) * 100) if dis["gap_agree"] else None, threshold=THR)
    if target is not None and late:
        pl = {k: np.array([np.mean(per_late[k][o]) for o in orders]) for k in rules}
        out["late"] = dict(range=list(late), eer={k: ms(v) for k, v in pl.items()}, rankavg_vs_logo=wil(pl["rankavg"], pl["logo"]), rankavg_vs_idcv=wil(pl["rankavg"], pl["idcv"]))
        out["handover"] = {k: ms(v) for k, v in hand.items()}
    return out


if __name__ == "__main__":
    out = {}
    jobs = [("ASVspoof stream (WavLM, 25 layers)", "stream_la_wavlm-large_mean_full.jsonl", None, None),
            ("ASVspoof->DFADD chronological (WavLM, 25 layers)", "stream_la+dfadd_wavlm-large_mean_full_chrono.jsonl", None, None),
            ("ASVspoof+DFADD random order (WavLM, 13 layers)", "stream_la+dfadd_wavlm-large_mean_full.jsonl", None, None),
            ("ASVspoof stream (XLS-R, 13 layers)", "stream_la_wav2vec2-xls-r-300m_mean_full.jsonl", None, None),
            ("ASVspoof->DFADD chronological (XLS-R, 13 layers)", "stream_la+dfadd_wav2vec2-xls-r-300m_mean_full_chrono.jsonl", None, None),
            ("TTS+hybrid->VC, unseen VC before any VC arrives (WavLM, 25 layers)", "stream_la_wavlm-large_mean_full_lastVC_L25.jsonl", "VC", (8, 12)),
            ("pure TTS->VC, unseen VC before any VC arrives (WavLM, 25 layers)", "stream_la_wavlm-large_mean_full_lastVC_noTTS-VC_L25.jsonl", "VC", (6, 9)),
            ("TTS+hybrid->VC, unseen VC before any VC arrives (XLS-R, 25 layers)", "stream_la_wav2vec2-xls-r-300m_mean_full_lastVC_L25.jsonl", "VC", (8, 12)),
            ("pure TTS->VC, unseen VC before any VC arrives (XLS-R, 25 layers)", "stream_la_wav2vec2-xls-r-300m_mean_full_lastVC_noTTS-VC_L25.jsonl", "VC", (6, 9))]
    for label, f, fam, late in jobs:
        S = load(f)
        if S is None:
            continue
        last = max(r["step"] for r in S)
        done = {o for o in {r["order_id"] for r in S} if any(r["order_id"] == o and r["step"] == last for r in S)}
        S = [r for r in S if r["order_id"] in done]
        for o in done:   # a resumed run appends to the file: every complete order must hold each step exactly once
            assert sorted(r["step"] for r in S if r["order_id"] == o) == list(range(last + 1)), (f, o)
        target = {g for g, fm in S[0]["family"].items() if fm == fam} if fam else None
        out[label] = analyse(S, target, late)
        v = out[label]
        print(f"{label}: orders {v['n_orders']}  " + "  ".join(f"{k} {e[0]:.2f}+-{e[1]:.2f}" for k, e in v["eer"].items()))
        print(f"    rankavg vs logo {v['rankavg_vs_logo']}  vs idcv {v['rankavg_vs_idcv']}")
        print(f"    layer disagreement >={THR} in {v['disagree_frac']*100:.0f}% of {v['steps']} steps; LOGO better {v['logo_better_when_disagree']}, ID-CV better {v['idcv_better_when_disagree']}; |gap| disagree {v['mean_abs_gap_disagree']} agree {v['mean_abs_gap_agree']}")
        if "late" in v:
            print(f"    late {v['late']['range']}: " + "  ".join(f"{k} {e[0]:.2f}+-{e[1]:.2f}" for k, e in v["late"]["eer"].items()), "| rankavg vs idcv", v["late"]["rankavg_vs_idcv"], "| hand-over:", {k: round(e[0], 2) for k, e in v["handover"].items()})
    json.dump(out, open(os.path.join(R, "combined.json"), "w"), indent=1)
