"""Build every table and figure of the paper from results/ (no number is typed by hand).

Outputs: paper/tables/*.tex, paper/figs/*.pdf, paper/numbers.json (values quoted in the running text).
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon
from data import ROOT

R = os.path.join(ROOT, "results")
T = os.path.join(ROOT, "paper", "tables"); FG = os.path.join(ROOT, "paper", "figs")
os.makedirs(T, exist_ok=True); os.makedirs(FG, exist_ok=True)
NUM = {}
LAMBDAS = [10.0 ** e for e in np.arange(-5, 2.01, 0.5)]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
C = dict(logo="#1f5fa8", idcv="#d9822b", fixed="#7a7a7a", oracle_unseen="#2e8b57")
NAME = dict(logo="LOGO (proposed)", logo_max="LOGO, worst fold", idcv="ID-CV", idcv_max="ID-CV, worst fold", fixed="Fixed",
            oracle_unseen="Oracle", oracle_all="Oracle (all)")


def load(name):
    p = os.path.join(R, name)
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else None


def pm(x, scale=100, d=2):
    x = np.asarray(x, dtype=float) * scale
    return f"{x.mean():.{d}f} $\\pm$ {x.std(ddof=1):.{d}f}" if len(x) > 1 else f"{x.mean():.{d}f}"


def per_order(recs, rule, field):
    """Time-averaged value of picks[rule][field] per order (unseen: only steps with unseen generators)."""
    out = []
    for o in sorted({r["order_id"] for r in recs}):
        rs = [r for r in recs if r["order_id"] == o]
        if field == "final":
            out.append(max(rs, key=lambda r: r["step"])["picks"][rule]["all"])
        else:
            v = [r["picks"][rule][field] for r in rs if r["picks"][rule].get(field) is not None]
            out.append(np.mean(v))
    return np.array(out)


def per_order_grad(recs, method, field):
    out = []
    for o in sorted({r["order_id"] for r in recs if r["method"] == method}):
        rs = sorted([r for r in recs if r["order_id"] == o and r["method"] == method], key=lambda r: r["step"])
        if field == "final":
            out.append(rs[-1]["all_eer"])
        elif field == "forgetting":
            out.append(np.mean([[r["per_gen"][g] for r in rs if g in r["seen"]][-1] - min(r["per_gen"][g] for r in rs if g in r["seen"]) for g in rs[-1]["seen"]]))
        else:
            key = dict(unseen="unseen_eer", seen="seen_eer", itw="itw")[field]
            out.append(np.mean([r[key] for r in rs if r.get(key) is not None]))
    return np.array(out)


def forgetting_stream(recs, rule):
    out = []
    for o in sorted({r["order_id"] for r in recs}):
        rs = sorted([r for r in recs if r["order_id"] == o], key=lambda r: r["step"])
        out.append(np.mean([[r["picks"][rule]["per_gen"][g] for r in rs if g in r["seen"]][-1] - min(r["picks"][rule]["per_gen"][g] for r in rs if g in r["seen"]) for g in rs[-1]["seen"]]))
    return np.array(out)


def wil(a, b):
    d = np.asarray(a) - np.asarray(b)
    return dict(diff=float(d.mean() * 100), wins=int((d < 0).sum()), n=int(len(d)), p=float(wilcoxon(d).pvalue) if len(d) >= 6 and np.any(d != 0) else None)


def fmt_p(p):
    return "--" if p is None else ("$<$0.001" if p < 0.001 else f"{p:.3f}")


# ------------------------------------------------------------------ pilot
def pilot(model, tag):
    p = os.path.join(R, f"pilot_la_{model}_mean.json")
    if not os.path.exists(p):
        return None
    rows = json.load(open(p))
    col = lambda k: np.array([r[k] for r in rows])
    out = {}
    for name, k in [("ID validation (dev partition)", "dev_eer"), ("LOGO, statistics only", "logo_gauss_mean"),
                    ("LOGO, worst fold, statistics only", "logo_gauss_max"), ("LOGO, empirical (needs data)", "logo_emp_mean"), ("Oracle", "eval_pooled")]:
        c = col(k)
        tie = np.nonzero(c == c.min())[0]
        # pooled and balanced weighting coincide when all generators have the same size: count distinct (layer, lambda)
        distinct = sorted({(rows[i]["layer"], rows[i]["lam"]) for i in tie})
        ev = col("eval_pooled")[tie] * 100; evm = col("eval_max")[tie] * 100
        out[name] = dict(n_tied=len(distinct), layers=sorted({rows[i]["layer"] for i in tie}), lam=sorted({rows[i]["lam"] for i in tie}),
                         dev=float(col("dev_eer")[tie].mean() * 100), eval_pooled=float(ev.mean()), eval_lo=float(ev.min()), eval_hi=float(ev.max()),
                         eval_max=float(evm.mean()), first=rows[int(tie[0])])
    NUM[f"pilot_{tag}"] = dict(
        n_cfg=len(rows), n_dev_below_0p05=int((col("dev_eer") < 0.0005).sum()),
        n_cfg_distinct=len({(r["layer"], r["lam"]) for r in rows}),
        n_dev_below_0p05_distinct=len({(r["layer"], r["lam"]) for r in rows if r["dev_eer"] < 0.0005}),
        eval_range_dev_below_0p05=[float(col("eval_pooled")[col("dev_eer") < 0.0005].min() * 100), float(col("eval_pooled")[col("dev_eer") < 0.0005].max() * 100)],
        rho_gauss_emp_foldmean=float(spearmanr(col("logo_gauss_mean"), col("logo_emp_mean")).correlation),
        rho_dev_eval=float(spearmanr(col("dev_eer"), col("eval_pooled")).correlation),
        rho_logo_eval=float(spearmanr(col("logo_gauss_mean"), col("eval_pooled")).correlation),
        rho_logomax_eval=float(spearmanr(col("logo_gauss_max"), col("eval_pooled")).correlation),
        picks={k: {kk: vv for kk, vv in v.items() if kk != "first"} for k, v in out.items()})
    return rows, out


def pilot_table():
    BS = chr(92)
    lines = [BS + "begin{tabular}{llrrrrrr}", BS + "toprule",
             "Encoder & Selection rule & Tied & Layer & $" + BS + "lambda$ & Dev & " + BS + "multicolumn{2}{c}{Eval, A07--A19} " + BS * 2,
             " & & config. & & & EER & pooled & worst " + BS * 2, BS + "midrule"]
    for model, tag, label in [("wavlm-large", "wavlm", "WavLM-Large"), ("wav2vec2-xls-r-300m", "xlsr", "XLS-R 300M")]:
        res = pilot(model, tag)
        if res is None:
            continue
        rows, out = res
        for i, (name, r) in enumerate(out.items()):
            lay = str(r["layers"][0]) if len(r["layers"]) == 1 else "%d layers" % len(r["layers"])
            lam = ("$10^{%.1f}$" % np.log10(r["lam"][0])) if r["n_tied"] == 1 else "--"
            ev = ("%.2f" % r["eval_pooled"]) if r["n_tied"] == 1 else ("%.2f [%.2f, %.2f]" % (r["eval_pooled"], r["eval_lo"], r["eval_hi"]))
            lines.append((label if i == 0 else "") + " & " + name + " & %d & %s & %s & %.2f & %s & %.2f " % (r["n_tied"], lay, lam, r["dev"], ev, r["eval_max"]) + BS * 2)
        lines.append(BS + "midrule")
    lines[-1] = BS + "bottomrule"; lines.append(BS + "end{tabular}")
    open(os.path.join(T, "pilot.tex"), "w").write(chr(10).join(lines))
    res = pilot("wavlm-large", "wavlm")
    if res:
        rows = res[0]
        col = lambda k: np.array([r[k] for r in rows]) * 100
        fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
        for a, k, xl in [(ax[0], "dev_eer", "In-distribution validation EER (%)"), (ax[1], "logo_gauss_mean", "LOGO criterion (%), statistics only")]:
            sc = a.scatter(np.maximum(col(k), 0.005), col("eval_pooled"), c=[r["layer"] for r in rows], cmap="viridis", s=7, alpha=0.8, linewidths=0)
            a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel(xl)
            a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            best = int(col(k).argmin()); a.scatter([max(col(k)[best], 0.005)], [col("eval_pooled")[best]], marker="*", s=120, c="crimson", zorder=5, label="selected")
            a.legend(frameon=False, loc="upper left")
        ax[0].set_ylabel("EER on unseen generators A07–A19 (%)")
        cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02); cb.set_label("encoder layer")
        fig.savefig(os.path.join(FG, "pilot_scatter.pdf"), bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ main stream tables
def stream_table(fname_stream, fname_grad, fname_extra, out, key):
    S = load(fname_stream)
    if S is None:
        return
    G = load(fname_grad) or []; E = load(fname_extra) or []
    has_itw = S[0]["picks"]["logo"].get("itw") is not None
    lines = ["\\begin{tabular}{llrrrr" + ("r" if has_itw else "") + "}", "\\toprule",
             "Family & Method & Unseen & Seen & Final (all) & Forgetting" + (" & In-the-Wild" if has_itw else "") + " \\\\", "\\midrule"]
    NUM[key] = dict(n_orders=len({r["order_id"] for r in S}), rules={}, grad={})
    first = True
    for rule in ["logo", "idcv", "fixed", "oracle_unseen"]:
        v = {f: per_order(S, rule, f) for f in ["unseen", "seen", "final"] + (["itw"] if has_itw else [])}
        fo = forgetting_stream(S, rule)
        NUM[key]["rules"][rule] = {f: [float(x.mean() * 100), float(x.std(ddof=1) * 100)] for f, x in v.items()} | {"forgetting": [float(fo.mean() * 100), float(fo.std(ddof=1) * 100)]}
        lines.append(f"{'Closed form' if first else ''} & {NAME[rule]} & {pm(v['unseen'])} & {pm(v['seen'])} & {pm(v['final'])} & {pm(fo)}" + (f" & {pm(v['itw'], d=1)}" if has_itw else "") + " \\\\")
        first = False
    for recs, fam in [(E, "Closed form, pooled"), (G, "Gradient (head)")]:
        if not recs:
            continue
        lines.append("\\midrule"); first = True
        order = [m for m in ["ncm", "ranpac", "ft", "ewc", "lwf", "replay100", "replay500", "joint"] if any(r["method"] == m for r in recs)]
        label = dict(ncm="Nearest class mean", ranpac="RanPAC-style", ft="Fine-tuning", ewc="EWC", lwf="LwF", replay100="Replay (100/gen.)", replay500="Replay (500/gen.)", joint="Joint retraining$^\\dagger$")
        for m in order:
            v = {f: per_order_grad(recs, m, f) for f in ["unseen", "seen", "final", "forgetting"] + (["itw"] if has_itw and recs[0].get("itw") is not None else [])}
            NUM[key]["grad"][m] = {f: [float(x.mean() * 100), float(x.std(ddof=1) * 100) if len(x) > 1 else 0.0, int(len(x))] for f, x in v.items()}
            lines.append(f"{fam if first else ''} & {label[m]} & {pm(v['unseen'])} & {pm(v['seen'])} & {pm(v['final'])} & {pm(v['forgetting'])}" + ((f" & {pm(v['itw'], d=1)}" if "itw" in v else " & --") if has_itw else "") + " \\\\")
            first = False
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(T, out), "w").write("\n".join(lines))
    a, b, c = per_order(S, "logo", "unseen"), per_order(S, "idcv", "unseen"), per_order(S, "fixed", "unseen")
    NUM[key]["tests"] = dict(logo_vs_idcv=wil(a, b), logo_vs_fixed=wil(a, c), idcv_vs_fixed=wil(b, c))
    if has_itw:
        NUM[key]["tests"]["itw_logo_vs_idcv"] = wil(per_order(S, "logo", "itw"), per_order(S, "idcv", "itw"))
        NUM[key]["tests"]["itw_logo_vs_fixed"] = wil(per_order(S, "logo", "itw"), per_order(S, "fixed", "itw"))
    # rank correlations of criteria with true error over the grid
    cor = {k: [] for k in ["logo", "idcv"]}; cor_itw = {k: [] for k in ["logo", "idcv"]}
    for r in S:
        g = r["grid"]
        if r["n_seen"] < len(r["names"]):
            for k in cor:
                cor[k].append(spearmanr(g[k], g["unseen_mean"]).correlation)
        if g.get("itw") is not None:
            for k in cor_itw:
                cor_itw[k].append(spearmanr(g[k], g["itw"]).correlation)
    NUM[key]["corr_unseen"] = {k: [float(np.mean(v)), float(np.std(v, ddof=1))] for k, v in cor.items()}
    NUM[key]["corr_itw"] = {k: [float(np.mean(v)), float(np.std(v, ddof=1))] for k, v in cor_itw.items() if v}
    NUM[key]["chosen_layer"] = {rule: np.bincount([r["picks"][rule]["layer"] for r in S], minlength=25).tolist() for rule in ["logo", "idcv", "oracle_unseen"]}
    NUM[key]["balanced_frac"] = {rule: float(np.mean([r["picks"][rule]["weighting"] == "balanced" for r in S])) for rule in ["logo", "idcv", "oracle_unseen"]}
    return S


def unseen_dom(r, rule, dom):
    """Mean EER of rule's detector on the not-yet-seen generators of one corpus (None if there are none)."""
    un = [g for g in r["names"] if g not in r["seen"] and (r["domain"][g] == "DFADD") == (dom == "DFADD")]
    return float(np.mean([r["picks"][rule]["per_gen"][g] for g in un])) if un else None


def curves(S, out, title=None, handover=None, dom=None):
    """Unseen-generator EER as a function of the number of seen generators."""
    fig, ax = plt.subplots(1, 2 if S[0]["picks"]["logo"].get("itw") is not None else 1, figsize=(6.6, 2.6))
    ax = np.atleast_1d(ax)
    ns = sorted({r["n_seen"] for r in S})
    for a, field, yl in zip(ax, ["unseen", "itw"], [("EER on unseen DFADD generators (%)" if dom else "EER on unseen generators (%)"), "EER on In-the-Wild (%)"]):
        for rule in ["fixed", "idcv", "logo", "oracle_unseen"]:
            xs, m, lo, hi = [], [], [], []
            for n in ns:
                if field == "unseen" and dom:
                    v = [unseen_dom(r, rule, dom) for r in S if r["n_seen"] == n]
                    v = [x for x in v if x is not None]
                else:
                    v = [r["picks"][rule][field] for r in S if r["n_seen"] == n and r["picks"][rule].get(field) is not None]
                if v:
                    v = np.array(v) * 100; xs.append(n); m.append(v.mean()); se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
                    lo.append(v.mean() - se); hi.append(v.mean() + se)
            a.plot(xs, m, color=C[rule], label=NAME[rule], lw=1.6, ls="--" if rule == "oracle_unseen" else "-")
            a.fill_between(xs, lo, hi, color=C[rule], alpha=0.15, lw=0)
        a.set_xlabel("generators seen"); a.set_ylabel(yl)
        if field == "unseen":
            a.set_yscale("log"); a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); a.set_yticks([0.3, 1, 3, 10]); a.set_yticklabels(["0.3", "1", "3", "10"])
        if handover:
            a.axvline(handover, color="k", lw=0.6, ls=":")
    ax[0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout(); fig.savefig(os.path.join(FG, out), bbox_inches="tight"); plt.close(fig)


def prehandover(S, key="prehandover_dfadd", table=True):
    """Steps of the chronological stream before any DFADD generator has arrived: the seen set is a random
    subset of the ASVspoof generators (different in every order), DFADD is entirely unseen."""
    res = {}
    for lo, hi, tag in [(3, 16, "3-16"), (3, 8, "3-8"), (9, 16, "9-16"), (17, 17, "17")]:
        per = {rule: [] for rule in ["logo", "idcv", "fixed", "oracle_unseen"]}
        for o in sorted({r["order_id"] for r in S}):
            rs = [r for r in S if r["order_id"] == o and lo <= r["n_seen"] <= hi and all(r["domain"][g] != "DFADD" for g in r["seen"])]
            for rule in per:
                per[rule].append(np.mean([unseen_dom(r, rule, "DFADD") for r in rs]))
        res[tag] = dict(eer={k: [float(np.mean(v) * 100), float(np.std(v, ddof=1) * 100)] for k, v in per.items()},
                        logo_vs_idcv=wil(per["logo"], per["idcv"]), logo_vs_fixed=wil(per["logo"], per["fixed"]))
    NUM[key] = res
    if not table:
        return
    BS = "\\"
    lines = [BS + "begin{tabular}{lrrrrrr}", BS + "toprule",
             "ASVspoof generators seen & LOGO & ID-CV & Fixed & Oracle & " + BS + "multicolumn{2}{c}{LOGO vs ID-CV} " + BS * 2, " & & & & & wins & $p$ " + BS * 2, BS + "midrule"]
    for tag, lab in [("3-8", "3--8"), ("9-16", "9--16"), ("17", "all 17$^{" + BS + "ast}$")]:
        e = res[tag]["eer"]; t = res[tag]["logo_vs_idcv"]
        tst = " & -- & -- " if tag == "17" else " & %d/%d & %s " % (t["wins"], t["n"], fmt_p(t["p"]))
        lines.append(lab + " & " + " & ".join("%.2f $%spm$ %.2f" % (e[k][0], BS, e[k][1]) for k in ["logo", "idcv", "fixed", "oracle_unseen"]) + tst + BS * 2)
    lines += [BS + "bottomrule", BS + "end{tabular}"]
    open(os.path.join(T, "prehandover.tex"), "w").write(chr(10).join(lines))


def layer_hist(key, out):
    if key not in NUM:
        return
    fig, ax = plt.subplots(figsize=(4.2, 2.3))
    w = 0.28
    for i, rule in enumerate(["idcv", "logo", "oracle_unseen"]):
        h = np.array(NUM[key]["chosen_layer"][rule], dtype=float); h /= h.sum()
        ax.bar(np.arange(25) + (i - 1) * w, h, w, color=C[rule], label=NAME[rule])
    ax.set_xlabel("selected encoder layer"); ax.set_ylabel("fraction of steps"); ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout(); fig.savefig(os.path.join(FG, out), bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ secondary conditions
def conditions_table():
    rows = [("ASVspoof, full", "stream_la_wavlm-large_mean_full.jsonl"), ("ASVspoof, imbalanced", "stream_la_wavlm-large_mean_imb.jsonl"),
            ("ASVspoof, 100-shot", "stream_la_wavlm-large_mean_shot100.jsonl"),
            ("ASVspoof+DFADD, random order", "stream_la+dfadd_wavlm-large_mean_full.jsonl"),
            ("ASVspoof+DFADD, chronological", "stream_la+dfadd_wavlm-large_mean_full_chrono.jsonl"),
            ("ASVspoof+DFADD, chrono., imbalanced", "stream_la+dfadd_wavlm-large_mean_imb_chrono.jsonl"),
            ("XLS-R: ASVspoof, full", "stream_la_wav2vec2-xls-r-300m_mean_full.jsonl"),
            ("XLS-R: ASVspoof+DFADD, chrono.", "stream_la+dfadd_wav2vec2-xls-r-300m_mean_full_chrono.jsonl")]
    BS = chr(92)
    lines = [BS + "begin{tabular}{lrrrrrrrr}", BS + "toprule",
             "Stream & Orders & Layers & " + BS + "multicolumn{4}{c}{Unseen-generator EER (" + BS + "%)} & " + BS + "multicolumn{2}{c}{LOGO vs ID-CV} " + BS * 2,
             " & & & LOGO & ID-CV & Fixed & Oracle & wins & $p$ " + BS * 2, BS + "midrule"]
    NUM["conditions"] = {}
    for label, f in rows:
        S = load(f)
        if S is None:
            continue
        v = {rule: per_order(S, rule, "unseen") for rule in ["logo", "idcv", "fixed", "oracle_unseen"]}
        t = wil(v["logo"], v["idcv"]); t2 = wil(v["logo"], v["fixed"])
        itw = {rule: per_order(S, rule, "itw") for rule in ["logo", "idcv", "fixed"]} if S[0]["picks"]["logo"].get("itw") is not None else None
        NUM["conditions"][label] = dict(n=len(v["logo"]), unseen={k: [float(x.mean() * 100), float(x.std(ddof=1) * 100)] for k, x in v.items()}, logo_vs_idcv=t, logo_vs_fixed=t2,
                                        itw={k: [float(x.mean() * 100), float(x.std(ddof=1) * 100)] for k, x in itw.items()} if itw else None,
                                        itw_logo_vs_idcv=wil(itw["logo"], itw["idcv"]) if itw else None, n_layers=len(S[0]["layers"]))
        lines.append(label + " & %d & %d & " % (len(v["logo"]), len(S[0]["layers"])) + " & ".join(pm(v[k]) for k in ["logo", "idcv", "fixed", "oracle_unseen"]) + " & %d/%d & %s " % (t["wins"], t["n"], fmt_p(t["p"])) + chr(92) * 2)
    lines += [chr(92) + "bottomrule", chr(92) + "end{tabular}"]
    open(os.path.join(T, "conditions.tex"), "w").write("\n".join(lines))


def domain_tables():
    """Hand-over table (chronological stream) and family table, from analyze_domain output."""
    p = os.path.join(R, "domain_stream_la+dfadd_wavlm-large_mean_full_chrono.json")
    if os.path.exists(p):
        d = json.load(open(p)); NUM["handover"] = dict(handover=d["handover"], corr=d.get("handover_corr"), corr_unseen_dfadd=d.get("corr_unseen_dfadd"), tests=d["tests"], by_domain=d["by_domain"])
        lines = ["\\begin{tabular}{lrr}", "\\toprule", "Selection rule & DFADD generators (unseen) & In-the-Wild \\\\", "\\midrule"]
        for rule in ["logo", "idcv", "fixed", "oracle_unseen"]:
            h = d["handover"][rule]
            lines.append(f"{NAME[rule]} & {h['dfadd']['mean']*100:.2f} $\\pm$ {h['dfadd']['std']*100:.2f} & {h['itw']['mean']*100:.1f} $\\pm$ {h['itw']['std']*100:.1f} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        open(os.path.join(T, "handover.tex"), "w").write("\n".join(lines))
    p = os.path.join(R, "domain_stream_la_wavlm-large_mean_full.json")
    if os.path.exists(p):
        d = json.load(open(p)); NUM["family"] = dict(by_family=d["by_family"], tests=d["tests"])
        lines = ["\\begin{tabular}{lrrrr}", "\\toprule", "Unseen generators & LOGO & ID-CV & Fixed & Oracle \\\\", "\\midrule"]
        for fam, lab in [("TTS", "TTS (10)"), ("VC", "VC (4)"), ("TTS-VC", "TTS--VC hybrid (3)")]:
            v = d["by_family"][fam]
            lines.append(f"{lab} & " + " & ".join(f"{v[r]['mean']*100:.2f} $\\pm$ {v[r]['std']*100:.2f}" for r in ["logo", "idcv", "fixed", "oracle_unseen"]) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        open(os.path.join(T, "family.tex"), "w").write("\n".join(lines))


def layer_subset_ablation(fname, key):
    """Post-hoc: restrict the candidate set (memory) and re-select with the stored criteria."""
    S = load(fname)
    if S is None:
        return
    layers = S[0]["layers"]; nl = len(LAMBDAS)
    subsets = {"all 25 layers": layers, "13 layers (every 2nd)": layers[::2], "7 layers (every 4th)": layers[::4], "4 layers (6,12,18,24)": [6, 12, 18, 24],
               "last layer only": [24], "balanced weighting only": None, "pooled weighting only": None}
    idx_of = {}
    k = 0
    for l in layers:
        for wt in ["pooled", "balanced"]:
            for j in range(nl):
                idx_of[(l, wt, j)] = k; k += 1
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule", "Candidate set & Stored per generator & LOGO & ID-CV & Oracle \\\\", "\\midrule"]
    NUM[key] = {}
    for name, sub in subsets.items():
        if sub is None:
            wts = ["balanced"] if name.startswith("balanced") else ["pooled"]; sub_l = layers
        else:
            wts = ["pooled", "balanced"]; sub_l = sub
        cand = np.array([idx_of[(l, wt, j)] for l in sub_l for wt in wts for j in range(nl)])
        res = {r: [] for r in ["logo", "idcv", "oracle"]}
        for o in sorted({r["order_id"] for r in S}):
            acc = {r: [] for r in res}
            for r in S:
                if r["order_id"] != o or r["n_seen"] >= len(r["names"]):
                    continue
                g = r["grid"]; un = np.array(g["unseen_mean"], dtype=float)
                for rule, crit in [("logo", np.array(g["logo"])), ("idcv", np.array(g["idcv"])), ("oracle", un)]:
                    acc[rule].append(un[cand[np.argmin(crit[cand])]])
            for rule in res:
                res[rule].append(np.mean(acc[rule]))
        mb = len(sub_l) * (1024 * 1025 / 2 + 1025) * 4 * 2 / 1e6       # spoof + bona fide triple, float32
        NUM[key][name] = dict(mb_per_generator=mb, **{r: [float(np.mean(v) * 100), float(np.std(v, ddof=1) * 100)] for r, v in res.items()})
        lines.append(f"{name} & {mb:.1f} MB & {pm(res['logo'])} & {pm(res['idcv'])} & {pm(res['oracle'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(T, f"{key}.tex"), "w").write("\n".join(lines))


def unseen_set(r, rule, target):
    un = [g for g in r["names"] if g not in r["seen"] and g in target]
    return float(np.mean([r["picks"][rule]["per_gen"][g] for g in un])) if un else None


def pre_shift(S, target, bins, key):
    """EER on a target set of generators while none of them has been seen, by number of other generators seen."""
    res = {}
    for lo, hi, tag in bins:
        per = {rule: [] for rule in ["logo", "logo_max", "idcv", "fixed", "oracle_unseen"]}
        for o in sorted({r["order_id"] for r in S}):
            rs = [r for r in S if r["order_id"] == o and lo <= r["n_seen"] <= hi and not any(g in target for g in r["seen"])]
            if not rs:
                continue
            for rule in per:
                per[rule].append(np.mean([unseen_set(r, rule, target) for r in rs]))
        res[tag] = dict(n=len(per["logo"]), eer={k: [float(np.mean(v) * 100), float(np.std(v, ddof=1) * 100) if len(v) > 1 else 0.0] for k, v in per.items()},
                        logo_vs_idcv=wil(per["logo"], per["idcv"]), logo_vs_fixed=wil(per["logo"], per["fixed"]))
    NUM[key] = res
    return res


def complete_orders(S):
    last = max(r["step"] for r in S)
    done = {o for o in {r["order_id"] for r in S} if any(r["order_id"] == o and r["step"] == last for r in S)}
    out = [r for r in S if r["order_id"] in done]
    for o in done:   # a resumed run appends to the file: every complete order must hold each step exactly once
        assert sorted(r["step"] for r in out if r["order_id"] == o) == list(range(last + 1)), "duplicated or missing steps in order %d" % o
    return out


def followups():
    BS = chr(92)
    # (1) family shift inside ASVspoof on all 25 layers: (a) TTS + hybrid first, VC last; (b) pure TTS first, VC last (hybrids removed)
    blocks = []
    for key, f, label in [("shift_vc_pure", "stream_la_wavlm-large_mean_full_lastVC_noTTS-VC_L25.jsonl", "TTS generators seen (hybrids removed)"),
                          ("shift_vc", "stream_la_wavlm-large_mean_full_lastVC_L25.jsonl", "TTS and hybrid generators seen")]:
        S = load(f)
        if not S:
            continue
        S = complete_orders(S)
        if not S:
            continue
        vc = {g for g, fm in S[0]["family"].items() if fm == "VC" and g in S[0]["names"]}
        n_other = len(S[0]["names"]) - len(vc)
        mid = 7 if n_other > 10 else 5
        res = pre_shift(S, vc, [(3, mid, "3-%d" % mid), (mid + 1, n_other - 1, "%d-%d" % (mid + 1, n_other - 1)), (n_other, n_other, "all")], key)
        H = [r for r in S if r["n_seen"] == n_other]          # one record per order; identical statistics in every order
        hc = {}
        for rule in ["logo", "logo_max", "idcv", "fixed", "oracle_unseen"]:
            lays = [r["picks"][rule]["layer"] for r in H]; lams = [r["picks"][rule]["lam"] for r in H]; wts = [r["picks"][rule]["weighting"] for r in H]
            hc[rule] = dict(layers=sorted(set(lays)), layer_counts={int(l): lays.count(l) for l in sorted(set(lays))}, layer=int(max(set(lays), key=lays.count)),
                            lam=sorted(set(lams)), weighting=sorted(set(wts)), crit_logo=float(np.mean([r["picks"][rule]["crit_logo"] for r in H])),
                            per_gen={g: float(np.mean([r["picks"][rule]["per_gen"][g] for r in H]) * 100) for g in sorted(vc)}, n=len(H))
        NUM[key + "_meta"] = dict(n_orders=len({r["order_id"] for r in S}), layers=S[0]["layers"], n_other=n_other, handover_cfg=hc)
        blocks.append((key, label, res, n_other))
    old7 = load("stream_la_wavlm-large_mean_full_lastVC.jsonl")
    if old7:
        vc = {g for g, fm in old7[0]["family"].items() if fm == "VC"}
        pre_shift(old7, vc, [(3, 7, "3-7"), (8, 12, "8-12"), (13, 13, "all")], "shift_vc_7layer")
    if blocks:
        lines = [BS + "begin{tabular}{llrrrrrr}", BS + "toprule",
                 "Stream & Seen & LOGO & ID-CV & Fixed & Oracle & " + BS + "multicolumn{2}{c}{LOGO vs ID-CV} " + BS * 2, " & & & & & & wins & $p$ " + BS * 2, BS + "midrule"]
        for bi, (key, label, res, n_other) in enumerate(blocks):
            for ti, (tag, v) in enumerate(res.items()):
                e = v["eer"]; t = v["logo_vs_idcv"]
                lab = ("all %d$^{" % n_other + BS + "ast}$") if tag == "all" else tag.replace("-", "--")
                tst = " & -- & -- " if tag == "all" else " & %d/%d & %s " % (t["wins"], t["n"], fmt_p(t["p"]))
                lines.append((("TTS + hybrid $" + BS + "to$ VC" if key == "shift_vc" else "TTS $" + BS + "to$ VC") if ti == 0 else "") + " & " + lab + " & " +
                             " & ".join("%.2f $%spm$ %.2f" % (e[k][0], BS, e[k][1]) for k in ["logo", "idcv", "fixed", "oracle_unseen"]) + tst + BS * 2)
            if bi < len(blocks) - 1:
                lines.append(BS + "midrule")
        lines += [BS + "bottomrule", BS + "end{tabular}"]
        open(os.path.join(T, "shift_vc.tex"), "w").write(chr(10).join(lines))
        # per-generator EER at the hand-over
        lines = [BS + "begin{tabular}{lllllrrrr}", BS + "toprule", "Stream & Rule & Layer & $" + BS + "lambda$ & Weights & A05 & A06 & A17 & A18 " + BS * 2, BS + "midrule"]
        for bi, (key, label, res, n_other) in enumerate(blocks):
            hc = NUM[key + "_meta"]["handover_cfg"]
            for ri, (rule, nm) in enumerate([("logo", "LOGO"), ("idcv", "ID-CV$^{" + BS + "dagger}$"), ("fixed", "Fixed"), ("oracle_unseen", "Oracle")]):
                c = hc[rule]
                lay = str(c["layers"][0]) if len(c["layers"]) == 1 else "%d--%d" % (c["layers"][0], c["layers"][-1])
                lam = ("$10^{%.1f}$" % np.log10(c["lam"][0])) if len(c["lam"]) == 1 else "$10^{%.1f}$--$10^{%.1f}$" % (np.log10(c["lam"][0]), np.log10(c["lam"][-1]))
                wt = c["weighting"][0] if len(c["weighting"]) == 1 else "both"
                lines.append((("TTS + hybrid $" + BS + "to$ VC" if key == "shift_vc" else "TTS $" + BS + "to$ VC") if ri == 0 else "") + " & " + nm + " & " + lay + " & " + lam + " & " + wt + " & " +
                             " & ".join("%.2f" % c["per_gen"][g] for g in ["A05", "A06", "A17", "A18"]) + " " + BS * 2)
            if bi < len(blocks) - 1:
                lines.append(BS + "midrule")
        lines += [BS + "bottomrule", BS + "end{tabular}"]
        open(os.path.join(T, "shift_vc_pergen.tex"), "w").write(chr(10).join(lines))
    # (2) ID-CV variant with one split for all layers and paired bona fide halves (chronological stream, 7 layers)
    S = load("stream_la+dfadd_wavlm-large_mean_full_chrono_idcvpaired_g.jsonl") or load("stream_la+dfadd_wavlm-large_mean_full_chrono_idcvpaired.jsonl")
    if S:
        df = {g for g, d in S[0]["domain"].items() if d == "DFADD"}
        pre_shift(S, df, [(3, 8, "3-8"), (9, 16, "9-16"), (17, 17, "17")], "idcv_paired_pre")
        v = {rule: per_order(S, rule, "unseen") for rule in ["logo", "idcv", "fixed", "oracle_unseen"]}
        NUM["idcv_paired"] = dict(n_orders=len(v["logo"]), layers=S[0]["layers"], unseen={k: [float(x.mean() * 100), float(x.std(ddof=1) * 100)] for k, x in v.items()},
                                  logo_vs_idcv=wil(v["logo"], v["idcv"]))
    # (3) control: bona fide chunks stratified over utterances instead of speaker-disjoint (chronological stream, 7 layers)
    S = load("stream_la+dfadd_wavlm-large_mean_full_chrono_bonautt.jsonl")
    if S:
        S = complete_orders(S)
    if S:
        df = {g for g, d in S[0]["domain"].items() if d == "DFADD"}
        pre_shift(S, df, [(3, 8, "3-8"), (9, 16, "9-16"), (17, 17, "17")], "bona_utt_pre")
        v = {rule: per_order(S, rule, "unseen") for rule in ["logo", "idcv", "fixed", "oracle_unseen"]}
        NUM["bona_utt"] = dict(n_orders=len(v["logo"]), layers=S[0]["layers"], unseen={k: [float(x.mean() * 100), float(x.std(ddof=1) * 100)] for k, x in v.items()},
                               logo_vs_idcv=wil(v["logo"], v["idcv"]),
                               logo_layers=np.bincount([r["picks"]["logo"]["layer"] for r in S], minlength=25).tolist())
    # (4) exploratory combined criterion and disagreement statistics (code/analyze_combined.py)
    p = os.path.join(R, "combined.json")
    if os.path.exists(p):
        C = json.load(open(p)); NUM["combined"] = C
        lines = [BS + "begin{tabular}{lrrrrrr}", BS + "toprule", "Stream & Orders & LOGO & ID-CV & Rank average & Oracle & Disagree " + BS * 2, BS + "midrule"]
        for label, v in C.items():
            e = v["eer"]
            lines.append(label.replace("->", "$" + BS + "to$").replace("+", "+") + " & %d & " % v["n_orders"] + " & ".join("%.2f $%spm$ %.2f" % (e[k][0], BS, e[k][1]) for k in ["logo", "idcv", "rankavg", "oracle"]) +
                         " & %d" % round(v["disagree_frac"] * 100) + BS + "% " + BS * 2)
        lines += [BS + "bottomrule", BS + "end{tabular}"]
        open(os.path.join(T, "combined.tex"), "w").write(chr(10).join(lines))


def followups_xlsr():
    """TTS -> VC family shift repeated with the second encoder (features re-extracted locally, 25 layers)."""
    BS = chr(92)
    blocks = []
    for key, f in [("shift_vc_pure_xlsr", "stream_la_wav2vec2-xls-r-300m_mean_full_lastVC_noTTS-VC_L25.jsonl"),
                   ("shift_vc_xlsr", "stream_la_wav2vec2-xls-r-300m_mean_full_lastVC_L25.jsonl")]:
        S = load(f)
        if not S:
            continue
        S = complete_orders(S)
        if not S:
            continue
        vc = {g for g, fm in S[0]["family"].items() if fm == "VC" and g in S[0]["names"]}
        n_other = len(S[0]["names"]) - len(vc)
        mid = 7 if n_other > 10 else 5
        res = pre_shift(S, vc, [(3, mid, "3-%d" % mid), (mid + 1, n_other - 1, "%d-%d" % (mid + 1, n_other - 1)), (n_other, n_other, "all")], key)
        H = [r for r in S if r["n_seen"] == n_other]
        hc = {}
        for rule in ["logo", "idcv", "fixed", "oracle_unseen"]:
            lays = [r["picks"][rule]["layer"] for r in H]
            hc[rule] = dict(layers=sorted(set(lays)), layer=int(max(set(lays), key=lays.count)), crit_logo=float(np.mean([r["picks"][rule]["crit_logo"] for r in H])),
                            per_gen={g: float(np.mean([r["picks"][rule]["per_gen"][g] for r in H]) * 100) for g in sorted(vc)})
        NUM[key + "_meta"] = dict(n_orders=len({r["order_id"] for r in S}), layers=S[0]["layers"], n_other=n_other, handover_cfg=hc)
        blocks.append((key, res, n_other))
    if not blocks:
        return
    lines = [BS + "begin{tabular}{llrrrrrr}", BS + "toprule",
             "Stream & Seen & LOGO & ID-CV & Fixed & Oracle & " + BS + "multicolumn{2}{c}{LOGO vs ID-CV} " + BS * 2, " & & & & & & wins & $p$ " + BS * 2, BS + "midrule"]
    for bi, (key, res, n_other) in enumerate(blocks):
        for ti, (tag, v) in enumerate(res.items()):
            e = v["eer"]; t = v["logo_vs_idcv"]
            lab = ("all %d$^{" % n_other + BS + "ast}$") if tag == "all" else tag.replace("-", "--")
            tst = " & -- & -- " if tag == "all" else " & %d/%d & %s " % (t["wins"], t["n"], fmt_p(t["p"]))
            lines.append((("TTS + hybrid $" + BS + "to$ VC" if key == "shift_vc_xlsr" else "TTS $" + BS + "to$ VC") if ti == 0 else "") + " & " + lab + " & " +
                         " & ".join("%.2f $%spm$ %.2f" % (e[k][0], BS, e[k][1]) for k in ["logo", "idcv", "fixed", "oracle_unseen"]) + tst + BS * 2)
        if bi < len(blocks) - 1:
            lines.append(BS + "midrule")
    lines += [BS + "bottomrule", BS + "end{tabular}"]
    open(os.path.join(T, "shift_vc_xlsr.tex"), "w").write(chr(10).join(lines))


def itw_diag():
    p = os.path.join(R, "itw_diagnostic.json")
    if not os.path.exists(p):
        return
    D = json.load(open(p))
    L = np.array([d["layer"] for d in D]); la = np.array([d["eer_la"] for d in D]) * 100; itw = np.array([d["eer_itw"] for d in D]) * 100
    fs = np.array([d["feat_shift"] for d in D]); sb = np.array([d["shift_bona"] for d in D]); ss = np.array([d["shift_spoof"] for d in D])
    NUM["itw_diag"] = dict(best_itw_layer=int(L[itw.argmin()]), best_itw=float(itw.min()), best_la_layer=int(L[la.argmin()]), best_la=float(la.min()),
                           itw_at_best_la=float(itw[la.argmin()]), la_at_best_itw=float(la[itw.argmin()]),
                           itw_l10=float(itw[10]), itw_l24=float(itw[24]), la_l10=float(la[10]), la_l24=float(la[24]),
                           feat_shift_first=float(fs[0]), feat_shift_l8=float(fs[8]), feat_shift_last=float(fs[-1]),
                           rho_featshift_itw=float(spearmanr(fs, itw).correlation), rho_featshift_la=float(spearmanr(fs, la).correlation),
                           shift_bona_min=float(sb.min()), shift_bona_max=float(sb.max()), shift_spoof_min=float(ss.min()), shift_spoof_max=float(ss.max()))
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
    ax[0].plot(L, itw, color=C["logo"], lw=1.6, label="In-the-Wild")
    ax[0].set_xlabel("encoder layer"); ax[0].set_ylabel("EER on In-the-Wild (%)", color=C["logo"])
    a2 = ax[0].twinx(); a2.plot(L, la, color=C["idcv"], lw=1.6, label="ASVspoof test"); a2.set_yscale("log")
    a2.set_ylabel("EER on ASVspoof test (%)", color=C["idcv"]); a2.spines["right"].set_visible(True)
    ax[1].plot(L, fs, color="k", lw=1.6)
    ax[1].set_xlabel("encoder layer"); ax[1].set_ylabel("shift of bona fide speech\n(In-the-Wild vs ASVspoof)")
    fig.tight_layout(); fig.savefig(os.path.join(FG, "itw_diag.pdf"), bbox_inches="tight"); plt.close(fig)


def macros():
    """paper/numbers.tex: every number quoted in the running text, as a LaTeX macro."""
    M = {}
    def pmm(v, d=2):
        return "%.*f\,$\pm$\,%.*f" % (d, v[0], d, v[1])
    def pv(p):
        return "<0.001" if p < 0.001 else "=%.2f" % p if p >= 0.01 else "=%.3f" % p
    pc = chr(92) + "%"
    p = NUM.get("pilot_wavlm")
    if p:
        k = p["picks"]
        M.update(PilotNdevlow=str(p["n_dev_below_0p05"]), PilotEvalLo="%.2f" % p["eval_range_dev_below_0p05"][0] + pc, PilotEvalHi="%.2f" % p["eval_range_dev_below_0p05"][1] + pc,
                 PilotIdEval="%.2f" % k["ID validation (dev partition)"]["eval_pooled"] + pc, PilotIdTies=str(k["ID validation (dev partition)"]["n_tied"]),
                 PilotLogoEval="%.2f" % k["LOGO, statistics only"]["eval_pooled"] + pc,
                 PilotOracleEval="%.2f" % k["Oracle"]["eval_pooled"] + pc, PilotRhoGaussEmp="%.3f" % p["rho_gauss_emp_foldmean"],
                 PilotRhoDev="%.2f" % p["rho_dev_eval"], PilotRhoLogo="%.2f" % p["rho_logo_eval"])
    p = NUM.get("pilot_xlsr")
    if p:
        k = p["picks"]; i_ = k["ID validation (dev partition)"]
        M.update(PilotXIdEval="%.2f" % i_["eval_pooled"] + pc, PilotXIdLo="%.2f" % i_["eval_lo"] + pc, PilotXIdHi="%.2f" % i_["eval_hi"] + pc, PilotXIdTies=str(i_["n_tied"]),
                 PilotXIdLayers=str(len(i_["layers"])), PilotXLogoEval="%.2f" % k["LOGO, statistics only"]["eval_pooled"] + pc,
                 PilotXLogoMaxEval="%.2f" % k["LOGO, worst fold, statistics only"]["eval_pooled"] + pc, PilotXOracleEval="%.2f" % k["Oracle"]["eval_pooled"] + pc)
    for key, pre in [("main_la", "MainLa"), ("main_chrono", "MainCh")]:
        m = NUM.get(key)
        if not m:
            continue
        r = m["rules"]; t = m["tests"]["logo_vs_idcv"]
        M[pre + "Orders"] = str(m["n_orders"])
        for rule, nm in [("logo", "Logo"), ("idcv", "Idcv"), ("fixed", "Fixed"), ("oracle_unseen", "Oracle")]:
            M[pre + nm + "Unseen"] = pmm(r[rule]["unseen"])
            if "itw" in r[rule]:
                M[pre + nm + "Itw"] = pmm(r[rule]["itw"], 1)
        M[pre + "Wins"] = str(t["wins"])
        M[pre + "P"] = pv(t["p"]) if pre == "MainCh" else ("%.2f" % t["p"])
        if "joint" in m["grad"]:
            M[pre + "JointUnseen"] = pmm(m["grad"]["joint"]["unseen"])
    m = NUM.get("main_chrono")
    if m:
        M.update(CorrItwLogo="%.2f" % m["corr_itw"]["logo"][0], CorrItwIdcv="%.2f" % m["corr_itw"]["idcv"][0])
    h = NUM.get("prehandover_dfadd")
    if h:
        M.update(PreEarlyLogo=pmm(h["3-8"]["eer"]["logo"]), PreEarlyIdcv=pmm(h["3-8"]["eer"]["idcv"]), PreEarlyP="%.2f" % h["3-8"]["logo_vs_idcv"]["p"],
                 PreLateLogo=pmm(h["9-16"]["eer"]["logo"]), PreLateIdcv=pmm(h["9-16"]["eer"]["idcv"]), PreLateWins=str(h["9-16"]["logo_vs_idcv"]["wins"]), PreLateP=pv(h["9-16"]["logo_vs_idcv"]["p"]),
                 HandLogo="%.2f" % h["17"]["eer"]["logo"][0], HandIdcv=pmm(h["17"]["eer"]["idcv"]), HandFixed="%.2f" % h["17"]["eer"]["fixed"][0], HandOracle="%.2f" % h["17"]["eer"]["oracle_unseen"][0])
    c = NUM.get("conditions", {})
    for label, pre in [("XLS-R: ASVspoof, full", "XLa"), ("XLS-R: ASVspoof+DFADD, chrono.", "XCh"), ("ASVspoof+DFADD, random order", "Rnd"),
                       ("ASVspoof+DFADD, chrono., imbalanced", "ImbCh"), ("ASVspoof, 100-shot", "Shot"), ("ASVspoof, imbalanced", "Imb")]:
        if label in c:
            v = c[label]
            M[pre + "Logo"] = pmm(v["unseen"]["logo"]); M[pre + "Idcv"] = pmm(v["unseen"]["idcv"]); M[pre + "Fixed"] = pmm(v["unseen"]["fixed"])
            M[pre + "P"] = "%.2f" % v["logo_vs_idcv"]["p"] if v["logo_vs_idcv"]["p"] >= 0.01 else "%.3f" % v["logo_vs_idcv"]["p"]
            M[pre + "Wins"] = str(v["logo_vs_idcv"]["wins"]); M[pre + "N"] = str(v["n"])
            if v.get("itw"):
                M[pre + "LogoItw"] = pmm(v["itw"]["logo"], 1); M[pre + "IdcvItw"] = pmm(v["itw"]["idcv"], 1); M[pre + "FixedItw"] = pmm(v["itw"]["fixed"], 1)
    hx = NUM.get("prehandover_dfadd_xlsr")
    if hx:
        M.update(XHandLogo="%.2f" % hx["17"]["eer"]["logo"][0], XHandIdcv="%.2f" % hx["17"]["eer"]["idcv"][0], XHandFixed="%.2f" % hx["17"]["eer"]["fixed"][0],
                 XPreLateLogo=pmm(hx["9-16"]["eer"]["logo"]), XPreLateIdcv=pmm(hx["9-16"]["eer"]["idcv"]), XPreLateP="%.2f" % hx["9-16"]["logo_vs_idcv"]["p"])
    a = NUM.get("ablation_chrono")
    if a:
        for name, pre in [("all 25 layers", "AblAll"), ("13 layers (every 2nd)", "AblThirteen"), ("7 layers (every 4th)", "AblSeven"), ("4 layers (6,12,18,24)", "AblFour"), ("last layer only", "AblLast")]:
            M[pre + "Logo"] = pmm(a[name]["logo"]); M[pre + "Idcv"] = pmm(a[name]["idcv"])
    for key, pre in [("main_la", "MainLa"), ("main_chrono", "MainCh")]:
        m = NUM.get(key)
        if m:
            for meth, nm in [("ranpac", "Ranpac"), ("replay100", "Replay"), ("ft", "Ft"), ("ncm", "Ncm")]:
                if meth in m["grad"]:
                    M[pre + nm + "Unseen"] = pmm(m["grad"][meth]["unseen"])
                    if "itw" in m["grad"][meth]:
                        M[pre + nm + "Itw"] = pmm(m["grad"][meth]["itw"], 1)
            M[pre + "LogoSeen"] = pmm(m["rules"]["logo"]["seen"]); M[pre + "FixedSeen"] = pmm(m["rules"]["fixed"]["seen"])
            M[pre + "CorrLogo"] = "%.2f" % m["corr_unseen"]["logo"][0]; M[pre + "CorrIdcv"] = "%.2f" % m["corr_unseen"]["idcv"][0]
    f_ = NUM.get("family")
    if f_:
        for fam, nm in [("TTS", "Tts"), ("VC", "Vc")]:
            M["Fam" + nm + "Logo"] = "%.2f" % (f_["by_family"][fam]["logo"]["mean"] * 100); M["Fam" + nm + "Fixed"] = "%.2f" % (f_["by_family"][fam]["fixed"]["mean"] * 100)
    for key, pre in [("shift_vc", "Vc"), ("shift_vc_pure", "Pure")]:
        sv = NUM.get(key)
        if not sv:
            continue
        meta = NUM[key + "_meta"]; tags = list(sv); early, late = tags[0], tags[1]
        hc = meta["handover_cfg"]
        M[pre + "Orders"] = str(meta["n_orders"]); M[pre + "Layers"] = str(len(meta["layers"])); M[pre + "Other"] = str(meta["n_other"])
        M[pre + "EarlyRange"] = early.replace("-", "--"); M[pre + "LateRange"] = late.replace("-", "--")
        M[pre + "EarlyLogo"] = pmm(sv[early]["eer"]["logo"]); M[pre + "EarlyIdcv"] = pmm(sv[early]["eer"]["idcv"])
        M[pre + "EarlyP"] = "%.2f" % sv[early]["logo_vs_idcv"]["p"]; M[pre + "EarlyWins"] = str(sv[early]["logo_vs_idcv"]["wins"])
        M[pre + "LateLogo"] = pmm(sv[late]["eer"]["logo"]); M[pre + "LateIdcv"] = pmm(sv[late]["eer"]["idcv"]); M[pre + "LateFixed"] = pmm(sv[late]["eer"]["fixed"])
        pl = sv[late]["logo_vs_idcv"]["p"]
        M[pre + "LateP"] = ("%.2f" % pl) if pl >= 0.01 else ("%.3f" % pl)
        M[pre + "LateWins"] = str(sv[late]["logo_vs_idcv"]["wins"]); M[pre + "LateN"] = str(sv[late]["logo_vs_idcv"]["n"])
        M[pre + "AllLogo"] = "%.2f" % sv["all"]["eer"]["logo"][0]; M[pre + "AllIdcv"] = pmm(sv["all"]["eer"]["idcv"])
        M[pre + "AllFixed"] = "%.2f" % sv["all"]["eer"]["fixed"][0]; M[pre + "AllOracle"] = "%.2f" % sv["all"]["eer"]["oracle_unseen"][0]
        M[pre + "AllLogoLayer"] = str(hc["logo"]["layer"]); M[pre + "AllOracleLayer"] = str(hc["oracle_unseen"]["layer"])
        M[pre + "AllIdcvLayerLo"] = str(hc["idcv"]["layers"][0]); M[pre + "AllIdcvLayerHi"] = str(hc["idcv"]["layers"][-1])
        M[pre + "AllLogoCrit"] = "%.1f" % (hc["logo"]["crit_logo"] * 1e4)
        for g in ["A05", "A06", "A17", "A18"]:
            nm = {"A05": "Five", "A06": "Six", "A17": "Seventeen", "A18": "Eighteen"}[g]
            M[pre + "Logo" + nm] = "%.2f" % hc["logo"]["per_gen"][g]; M[pre + "Idcv" + nm] = "%.2f" % hc["idcv"]["per_gen"][g]
    for key, pre in [("shift_vc_xlsr", "XVc"), ("shift_vc_pure_xlsr", "XPure")]:
        sv = NUM.get(key)
        if not sv:
            continue
        meta = NUM[key + "_meta"]; tags = list(sv); early, late = tags[0], tags[1]; hc = meta["handover_cfg"]
        M[pre + "Orders"] = str(meta["n_orders"]); M[pre + "LateRange"] = late.replace("-", "--")
        assert len(hc["logo"]["layers"]) == 1 and len(hc["oracle_unseen"]["layers"]) == 1
        for tg, nm in [(early, "Early"), (late, "Late")]:
            t = sv[tg]["logo_vs_idcv"]
            M[pre + nm + "Logo"] = pmm(sv[tg]["eer"]["logo"]); M[pre + nm + "Idcv"] = pmm(sv[tg]["eer"]["idcv"]); M[pre + nm + "Fixed"] = pmm(sv[tg]["eer"]["fixed"])
            M[pre + nm + "Wins"] = str(t["wins"]); M[pre + nm + "N"] = str(t["n"])
            M[pre + nm + "P"] = "n/a" if t["p"] is None else (("%.2f" % t["p"]) if t["p"] >= 0.01 else ("%.3f" % t["p"]))
        M[pre + "AllLogo"] = "%.2f" % sv["all"]["eer"]["logo"][0]; M[pre + "AllIdcv"] = pmm(sv["all"]["eer"]["idcv"])
        M[pre + "AllFixed"] = "%.2f" % sv["all"]["eer"]["fixed"][0]; M[pre + "AllOracle"] = "%.2f" % sv["all"]["eer"]["oracle_unseen"][0]
        M[pre + "AllLogoLayer"] = str(hc["logo"]["layer"]); M[pre + "AllOracleLayer"] = str(hc["oracle_unseen"]["layer"])
        M[pre + "AllIdcvLayerLo"] = str(hc["idcv"]["layers"][0]); M[pre + "AllIdcvLayerHi"] = str(hc["idcv"]["layers"][-1])
    s7 = NUM.get("shift_vc_7layer")
    if s7:
        M.update(VcSevenAllLogo="%.2f" % s7["all"]["eer"]["logo"][0], VcSevenAllIdcv=pmm(s7["all"]["eer"]["idcv"]),
                 VcSevenLateLogo=pmm(s7["8-12"]["eer"]["logo"]), VcSevenLateIdcv=pmm(s7["8-12"]["eer"]["idcv"]))
    bu = NUM.get("bona_utt")
    if bu:
        pre_ = NUM["bona_utt_pre"]
        M.update(BonaOrders=str(bu["n_orders"]), BonaLogo=pmm(bu["unseen"]["logo"]), BonaIdcv=pmm(bu["unseen"]["idcv"]),
                 BonaP="%.3f" % bu["logo_vs_idcv"]["p"] if bu["logo_vs_idcv"]["p"] is not None else "--", BonaWins=str(bu["logo_vs_idcv"]["wins"]),
                 BonaLateLogo=pmm(pre_["9-16"]["eer"]["logo"]), BonaLateIdcv=pmm(pre_["9-16"]["eer"]["idcv"]),
                 BonaHandLogo="%.2f" % pre_["17"]["eer"]["logo"][0], BonaHandIdcv=pmm(pre_["17"]["eer"]["idcv"]))
    cb = NUM.get("combined")
    if cb:
        for label, pre in [("ASVspoof stream (WavLM, 25 layers)", "CbLa"), ("ASVspoof->DFADD chronological (WavLM, 25 layers)", "CbCh"),
                           ("ASVspoof->DFADD chronological (XLS-R, 13 layers)", "CbXCh"), ("ASVspoof stream (XLS-R, 13 layers)", "CbXLa"),
                           ("TTS+hybrid->VC, unseen VC before any VC arrives (WavLM, 25 layers)", "CbVc"), ("pure TTS->VC, unseen VC before any VC arrives (WavLM, 25 layers)", "CbPure"),
                           ("TTS+hybrid->VC, unseen VC before any VC arrives (XLS-R, 25 layers)", "CbXVc"), ("pure TTS->VC, unseen VC before any VC arrives (XLS-R, 25 layers)", "CbXPure")]:
            if label in cb:
                v = cb[label]
                M[pre + "Rank"] = pmm(v["eer"]["rankavg"]); M[pre + "Logo"] = pmm(v["eer"]["logo"]); M[pre + "Idcv"] = pmm(v["eer"]["idcv"])
                M[pre + "Dis"] = "%d" % round(v["disagree_frac"] * 100)
                M[pre + "DisLogo"] = str(v["logo_better_when_disagree"]); M[pre + "DisIdcv"] = str(v["idcv_better_when_disagree"])
                for other in ["logo", "idcv"]:
                    pp = v["rankavg_vs_" + other]["p"]
                    M[pre + "RankP" + other.capitalize()] = "--" if pp is None else (("%.2f" % pp) if pp >= 0.01 else ("%.3f" % pp))
                if "late" in v:
                    M[pre + "LateRank"] = pmm(v["late"]["eer"]["rankavg"]); M[pre + "LateIdcvCb"] = pmm(v["late"]["eer"]["idcv"]); M[pre + "LateLogoCb"] = pmm(v["late"]["eer"]["logo"])
                    pp = v["late"]["rankavg_vs_idcv"]["p"]
                    M[pre + "LateRankP"] = ("%.2f" % pp) if pp >= 0.01 else ("%.3f" % pp)
                    M[pre + "LateRankWins"] = str(v["late"]["rankavg_vs_idcv"]["wins"])
                    M[pre + "HandRank"] = "%.2f" % v["handover"]["rankavg"][0]
    bf = [NUM[k]["balanced_frac"][r] for k in ("main_la", "main_chrono") if k in NUM for r in ("logo", "idcv", "oracle_unseen")]
    if bf:
        M["BalMin"] = "%d" % round(min(bf) * 100); M["BalMax"] = "%d" % round(max(bf) * 100)
    for label, pre in [("ASVspoof, 100-shot", "Shot"), ("XLS-R: ASVspoof, full", "XLa"), ("XLS-R: ASVspoof+DFADD, chrono.", "XCh"), ("ASVspoof, imbalanced", "Imb")]:
        v = NUM.get("conditions", {}).get(label)
        if v and v.get("itw_logo_vs_idcv") and v["itw_logo_vs_idcv"]["p"] is not None:
            M[pre + "ItwP"] = "%.2f" % v["itw_logo_vs_idcv"]["p"]
    pw = NUM.get("pilot_wavlm")
    if pw:
        M["PilotDistinct"] = str(pw["n_cfg_distinct"]); M["PilotNdevlowDistinct"] = str(pw["n_dev_below_0p05_distinct"])
    ip = NUM.get("idcv_paired")
    if ip:
        pre = NUM["idcv_paired_pre"]
        M.update(PairedOrders=str(ip["n_orders"]), PairedLogo=pmm(ip["unseen"]["logo"]), PairedIdcv=pmm(ip["unseen"]["idcv"]),
                 PairedP="%.3f" % ip["logo_vs_idcv"]["p"] if ip["logo_vs_idcv"]["p"] is not None else "--", PairedWins=str(ip["logo_vs_idcv"]["wins"]),
                 PairedLateLogo=pmm(pre["9-16"]["eer"]["logo"]), PairedLateIdcv=pmm(pre["9-16"]["eer"]["idcv"]),
                 PairedHandLogo="%.2f" % pre["17"]["eer"]["logo"][0], PairedHandIdcv=pmm(pre["17"]["eer"]["idcv"]))
    dg = NUM.get("itw_diag")
    if dg:
        M.update(DgBestItwLayer=str(dg["best_itw_layer"]), DgBestItw="%.1f" % dg["best_itw"], DgBestLaLayer=str(dg["best_la_layer"]), DgBestLa="%.2f" % dg["best_la"],
                 DgItwAtBestLa="%.1f" % dg["itw_at_best_la"], DgLaAtBestItw="%.2f" % dg["la_at_best_itw"], DgItwTen="%.1f" % dg["itw_l10"], DgItwLast="%.1f" % dg["itw_l24"],
                 DgShiftFirst="%.1f" % dg["feat_shift_first"], DgShiftEight="%.1f" % dg["feat_shift_l8"], DgShiftLast="%.2f" % dg["feat_shift_last"],
                 DgRhoItw="%.2f" % dg["rho_featshift_itw"], DgBonaMin="%.1f" % abs(dg["shift_bona_max"]), DgBonaMax="%.1f" % abs(dg["shift_bona_min"]),
                 DgSpoofMax="%.1f" % max(abs(dg["shift_spoof_min"]), abs(dg["shift_spoof_max"])))
    for key, pre in [("main_la", "MainLa"), ("main_chrono", "MainCh")]:
        m = NUM.get(key)
        if m and m.get("corr_itw"):
            M[pre + "CorrItwLogo"] = "%.2f" % m["corr_itw"]["logo"][0]; M[pre + "CorrItwIdcv"] = "%.2f" % m["corr_itw"]["idcv"][0]
    m = NUM.get("main_chrono")
    if m:
        h_ = np.array(m["chosen_layer"]["oracle_unseen"], dtype=float)
        M["OracleLateShare"] = "%d" % round(100 * h_[23] / h_.sum())
    p_ = os.path.join(R, "itw_diagnostic.json")
    if os.path.exists(p_):
        D_ = json.load(open(p_))
        mid = max(D_[14:20], key=lambda d: d["eer_itw"])
        M["DgMidPeak"] = "%.1f" % (mid["eer_itw"] * 100); M["DgMidPeakLayer"] = str(mid["layer"])
    with open(os.path.join(ROOT, "paper", "numbers.tex"), "w") as f:
        for k in sorted(M):
            f.write(chr(92) + "newcommand{" + chr(92) + k + "}{" + M[k] + "}" + chr(10))
    return M


if __name__ == "__main__":
    pilot_table()
    S1 = stream_table("stream_la_wavlm-large_mean_full.jsonl", "grad_la_wavlm-large_full.jsonl", "extra_la_wavlm-large_full_L24.jsonl", "main_la.tex", "main_la")
    S2 = stream_table("stream_la+dfadd_wavlm-large_mean_full_chrono.jsonl", "grad_la+dfadd_wavlm-large_full_chrono.jsonl", "extra_la+dfadd_wavlm-large_full_chrono_L24.jsonl", "main_chrono.tex", "main_chrono")
    if S1:
        curves(S1, "curves_la.pdf"); layer_hist("main_la", "layers_la.pdf")
    if S2:
        curves(S2, "curves_chrono.pdf", handover=17, dom="DFADD"); prehandover(S2)
    SX = load("stream_la+dfadd_wav2vec2-xls-r-300m_mean_full_chrono.jsonl")
    if SX:
        prehandover(SX, "prehandover_dfadd_xlsr", table=False)
        NUM["xlsr_layers"] = {rule: np.bincount([r["picks"][rule]["layer"] for r in SX], minlength=25).tolist() for rule in ["logo", "idcv", "oracle_unseen"]}; layer_hist("main_chrono", "layers_chrono.pdf")
    conditions_table(); domain_tables(); followups(); followups_xlsr(); itw_diag()
    layer_subset_ablation("stream_la_wavlm-large_mean_full.jsonl", "ablation_la")
    layer_subset_ablation("stream_la+dfadd_wavlm-large_mean_full_chrono.jsonl", "ablation_chrono")
    p = os.path.join(R, "check_removal.json")
    if os.path.exists(p):
        NUM["exactness"] = json.load(open(p))
    macros()
    json.dump(NUM, open(os.path.join(ROOT, "paper", "numbers.json"), "w"), indent=1)
    print(json.dumps(NUM, indent=1)[:6000])
