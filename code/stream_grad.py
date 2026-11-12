"""Gradient-based continual-learning baselines on the same cached SSL features.

Head: softmax-weighted sum of all SSL layers -> LayerNorm -> Linear(256) -> ReLU -> Linear(2).
Methods: ft (sequential fine-tuning), ewc (online EWC), lwf (distillation on new data),
replay (feature replay: a fixed random sample of R utterances, half spoofed and half bona fide, kept from
every seen generator and mixed with the new data; not a reservoir),
joint (retrain from scratch on everything seen so far; upper bound, not continual).
Streams, arrivals and test sets are identical to stream.py.
"""
import argparse, json, os, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import gss
from pools import get_pool, make_stream
from data import ROOT


class Head(nn.Module):
    def __init__(self, L, D, h=256):
        super().__init__()
        self.lw = nn.Parameter(torch.zeros(L))
        self.norm = nn.LayerNorm(D)
        self.fc1 = nn.Linear(D, h)
        self.fc2 = nn.Linear(h, 2)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):                      # x [B, L, D]
        w = torch.softmax(self.lw, 0)
        z = (x * w[None, :, None]).sum(1)
        return self.fc2(self.drop(torch.relu(self.fc1(self.norm(z)))))


def train(model, X, y, iters, lr, dev, old=None, method="ft", fisher=None, star=None, ewc_lam=0.0, bs=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    cw = torch.tensor([0.5 / max((y == 0).float().mean().item(), 1e-6), 0.5 / max((y == 1).float().mean().item(), 1e-6)], device=dev)
    model.train()
    n = len(y)
    for it in range(iters):
        idx = torch.randint(0, n, (min(bs, n),), generator=g)
        xb = X[idx].to(dev).float(); yb = y[idx].to(dev)
        out = model(xb)
        loss = Fn.cross_entropy(out, yb, weight=cw)
        if method == "lwf" and old is not None:
            with torch.no_grad():
                t = old(xb)
            loss = loss + Fn.kl_div(Fn.log_softmax(out / 2, 1), Fn.softmax(t / 2, 1), reduction="batchmean") * 4
        if method == "ewc" and fisher is not None:
            pen = sum((fisher[k] * (p - star[k]) ** 2).sum() for k, p in model.named_parameters())
            loss = loss + 0.5 * ewc_lam * pen
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()


def fisher_diag(model, X, y, dev, n=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(y), generator=g)[:n]
    F = {k: torch.zeros_like(p) for k, p in model.named_parameters()}
    for i in idx:
        model.zero_grad()
        out = model(X[i:i + 1].to(dev).float())
        Fn.cross_entropy(out, y[i:i + 1].to(dev)).backward()
        for k, p in model.named_parameters():
            F[k] += p.grad.detach() ** 2
    return {k: v / len(idx) for k, v in F.items()}


@torch.no_grad()
def score(model, X, dev, bs=2048):
    out = []
    for i in range(0, len(X), bs):
        o = model(X[i:i + bs].to(dev).float())
        out.append((o[:, 1] - o[:, 0]).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="wavlm-large")
    ap.add_argument("--orders", type=int, default=10)
    ap.add_argument("--order_start", type=int, default=0)
    ap.add_argument("--base", type=int, default=3)
    ap.add_argument("--condition", default="full")
    ap.add_argument("--methods", default="ft,ewc,lwf,replay,joint")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ewc_lam", type=float, default=1000.0)
    ap.add_argument("--replay_per_gen", type=int, default=100)
    ap.add_argument("--tag", default="")
    ap.add_argument("--itw", action="store_true")
    ap.add_argument("--poolname", default="la")
    ap.add_argument("--chrono", action="store_true")
    args = ap.parse_args()
    dev = "cuda"
    P = get_pool(args.poolname, args.model, "mean")
    F = P.F
    names = P.names()

    def load(idx):                                # float16 tensor [n, L, D]
        return torch.from_numpy(F.rows(idx))

    domains = sorted({P.gens[g]["domain"] for g in names})
    Xb_test = {d: load(next(P.gens[g]["bona_test"] for g in names if P.gens[g]["domain"] == d)) for d in domains}
    Xs_test = {g: load(P.gens[g]["test"]) for g in names}
    itw = None
    if args.itw:
        from data import Feats
        I = Feats("itw", args.model, "mean")
        itw = (torch.from_numpy(np.asarray(I.X[I.where(label="bonafide")])), torch.from_numpy(np.asarray(I.X[I.where(label="spoof")])))
    print("test features loaded", flush=True)

    out_path = os.path.join(ROOT, "results", f"grad_{args.poolname}_{args.model}_{args.condition}{'_chrono' if args.chrono else ''}{args.tag}.jsonl")
    fout = open(out_path, "a")
    for order_id in range(args.order_start, args.order_start + args.orders):
        order, steps, arrive, rng = make_stream(P, order_id, args.condition, args.base, args.chrono)
        data = {g: (load(arrive[g][0]), load(arrive[g][1])) for g in names}     # spoof, bona
        for method in args.methods.split(","):
            method_name = method
            torch.manual_seed(order_id)
            model = Head(F.L, F.D).to(dev)
            old, fisher, star = None, None, None
            buf_X, buf_y = [], []
            seen = []
            t0 = time.time()
            for t, new in enumerate(steps):
                seen += new
                Xn = torch.cat([torch.cat(data[g]) for g in new])
                yn = torch.cat([torch.cat([torch.zeros(len(data[g][0]), dtype=torch.long), torch.ones(len(data[g][1]), dtype=torch.long)]) for g in new])
                if method == "joint":
                    torch.manual_seed(order_id * 100 + t)
                    model = Head(F.L, F.D).to(dev)
                    Xtr = torch.cat([torch.cat(data[g]) for g in seen])
                    ytr = torch.cat([torch.cat([torch.zeros(len(data[g][0]), dtype=torch.long), torch.ones(len(data[g][1]), dtype=torch.long)]) for g in seen])
                    train(model, Xtr, ytr, args.iters * 3, args.lr, dev, seed=order_id * 100 + t)
                elif method == "replay":
                    Xtr = torch.cat([Xn] + buf_X); ytr = torch.cat([yn] + buf_y)
                    train(model, Xtr, ytr, args.iters, args.lr, dev, seed=order_id * 100 + t)
                    for g in new:           # R/2 spoof + R/2 bona fide per generator
                        r = args.replay_per_gen // 2
                        s_, b_ = data[g]
                        buf_X += [s_[torch.randperm(len(s_))[:r]], b_[torch.randperm(len(b_))[:r]]]
                        buf_y += [torch.zeros(min(r, len(s_)), dtype=torch.long), torch.ones(min(r, len(b_)), dtype=torch.long)]
                    method_name = f"replay{args.replay_per_gen}"
                else:
                    train(model, Xn, yn, args.iters, args.lr, dev, old=old, method=method, fisher=fisher, star=star,
                          ewc_lam=args.ewc_lam, seed=order_id * 100 + t)
                    if method == "lwf":
                        old = copy.deepcopy(model).eval()
                    if method == "ewc":
                        f_new = fisher_diag(model, Xn, yn, dev, seed=order_id * 100 + t)
                        fisher = f_new if fisher is None else {k: 0.5 * fisher[k] + 0.5 * f_new[k] for k in f_new}   # online EWC
                        star = {k: p.detach().clone() for k, p in model.named_parameters()}
                sb = {d: score(model, Xb_test[d], dev) for d in domains}
                per = {g: gss.eer(sb[P.gens[g]["domain"]], score(model, Xs_test[g], dev)) for g in names}
                unseen = [g for g in names if g not in seen]
                rec = dict(method=method_name, order_id=order_id, step=t, seen=list(seen), per_gen=per,
                           seen_eer=float(np.mean([per[g] for g in seen])),
                           unseen_eer=float(np.mean([per[g] for g in unseen])) if unseen else None,
                           all_eer=float(np.mean(list(per.values()))),
                           itw=gss.eer(score(model, itw[0], dev), score(model, itw[1], dev)) if itw else None,
                           domain={g: P.gens[g]['domain'] for g in names}, condition=args.condition, iters=args.iters, lr=args.lr, ewc_lam=args.ewc_lam)
                fout.write(json.dumps(rec) + "\n"); fout.flush()
            print(f"order {order_id} {method_name:10s} final seen {rec['seen_eer']*100:5.2f} all {rec['all_eer']*100:5.2f} ({time.time()-t0:.0f}s)", flush=True)
    fout.close()


if __name__ == "__main__":
    main()
