"""Extract frozen SSL features (per-layer temporal mean and std) for every utterance.

Audio is read straight from the downloaded archives (zip / parquet), so nothing
is unpacked.  Output per (dataset, model):
    feats/<model>/<dataset>.mean.npy   float16 [N, L+1, D]
    feats/<model>/<dataset>.std.npy    float16 [N, L+1, D]
    feats/<model>/<dataset>.meta.csv   utt, speaker, system, label, split, dur
"""
import argparse, csv, io, os, sys, time, zipfile, glob
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import soundfile as sf
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_SEC = 8.0
SR = 16000


# ----------------------------------------------------------------- item lists
def items_asvspoof19():
    zp = os.path.join(ROOT, "data", "asvspoof2019", "LA.zip")
    z = zipfile.ZipFile(zp)
    # archive root differs between copies ("LA/..." in the official zip): detect it
    anchor = "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    pre = [n for n in z.namelist() if n.endswith(anchor)][0][: -len(anchor)]
    out = []
    for split, proto in [("train", "ASVspoof2019.LA.cm.train.trn.txt"),
                         ("dev", "ASVspoof2019.LA.cm.dev.trl.txt"),
                         ("eval", "ASVspoof2019.LA.cm.eval.trl.txt")]:
        txt = z.read(f"{pre}ASVspoof2019_LA_cm_protocols/{proto}").decode()
        for line in txt.strip().splitlines():
            spk, utt, _, system, key = line.split()
            system = "bonafide" if key == "bonafide" else system
            out.append(dict(utt=utt, speaker=spk, system=system, label=key, split=split,
                            ref=f"{pre}ASVspoof2019_LA_{split}/flac/{utt}.flac"))
    return zp, out


def items_itw():
    zp = os.path.join(ROOT, "data", "in_the_wild", "release_in_the_wild.zip")
    z = zipfile.ZipFile(zp)
    names = z.namelist()
    meta_name = [n for n in names if n.endswith("meta.csv")][0]
    base = meta_name[: -len("meta.csv")]
    rows = list(csv.DictReader(io.StringIO(z.read(meta_name).decode("utf-8"))))
    out = []
    for r in rows:
        key = "bonafide" if r["label"].strip() == "bona-fide" else "spoof"
        out.append(dict(utt=r["file"], speaker=r["speaker"], system=("bonafide" if key == "bonafide" else "itw_spoof"),
                        label=key, split="test", ref=base + r["file"]))
    return zp, out


class ZipReader:
    def __init__(self, zp):
        self.zp = zp
        self.local = {}

    def read(self, ref):
        import threading
        tid = threading.get_ident()
        if tid not in self.local:
            self.local[tid] = zipfile.ZipFile(self.zp)
        return self.local[tid].read(ref)


def decode(raw):
    wav, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    wav = wav.mean(1)
    if sr != SR:
        import torchaudio
        wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, SR).numpy()
    dur = len(wav) / SR
    wav = wav[: int(MAX_SEC * SR)]
    if len(wav) < 1600:  # pad very short clips to 0.1 s
        wav = np.pad(wav, (0, 1600 - len(wav)))
    return wav, dur


# ----------------------------------------------------------------- extraction
@torch.no_grad()
def run_batch(model, wavs, device):
    lens = torch.tensor([len(w) for w in wavs])
    T = int(lens.max())
    x = torch.zeros(len(wavs), T)
    mask = torch.zeros(len(wavs), T, dtype=torch.long)
    for i, w in enumerate(wavs):
        w = torch.from_numpy(w)
        w = (w - w.mean()) / (w.std() + 1e-7)  # same as Wav2Vec2FeatureExtractor(do_normalize=True)
        x[i, : len(w)] = w
        mask[i, : len(w)] = 1
    x, mask = x.to(device), mask.to(device)
    with torch.autocast("cuda", dtype=torch.float16):
        hs = model(x, attention_mask=mask, output_hidden_states=True).hidden_states
    flen = model._get_feat_extract_output_lengths(lens).to(device)
    F = hs[0].shape[1]
    fm = (torch.arange(F, device=device)[None, :] < flen[:, None]).float()[:, :, None]
    n = fm.sum(1)
    means, stds = [], []
    for h in hs:
        h = h.float()
        mu = (h * fm).sum(1) / n
        var = ((h - mu[:, None, :]) ** 2 * fm).sum(1) / n
        means.append(mu)
        stds.append(var.sqrt())
    return torch.stack(means, 1).half().cpu().numpy(), torch.stack(stds, 1).half().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["asvspoof19", "itw", "dfadd"])
    ap.add_argument("--model", default="microsoft/wavlm-large")
    ap.add_argument("--max_batch_sec", type=float, default=160.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoModel
    device = "cuda"
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    tag = args.model.split("/")[-1]
    outdir = os.path.join(ROOT, "feats", tag)
    os.makedirs(outdir, exist_ok=True)

    if args.dataset == "asvspoof19":
        zp, items = items_asvspoof19(); reader = ZipReader(zp).read
    elif args.dataset == "itw":
        zp, items = items_itw(); reader = ZipReader(zp).read
    else:
        from dfadd_items import items_dfadd
        items, reader = items_dfadd()
    if args.limit:
        rng = np.random.RandomState(0)
        items = [items[i] for i in rng.permutation(len(items))[: args.limit]]
    N = len(items)
    L = model.config.num_hidden_layers + 1
    D = model.config.hidden_size
    suffix = f".limit{args.limit}" if args.limit else ""
    mean_mm = np.lib.format.open_memmap(os.path.join(outdir, f"{args.dataset}{suffix}.mean.npy"), mode="w+", dtype=np.float16, shape=(N, L, D))
    std_mm = np.lib.format.open_memmap(os.path.join(outdir, f"{args.dataset}{suffix}.std.npy"), mode="w+", dtype=np.float16, shape=(N, L, D))
    durs = np.zeros(N, dtype=np.float32)

    pool = ThreadPoolExecutor(8)
    CH = 512
    t0 = time.time()
    for s in range(0, N, CH):
        idx = list(range(s, min(N, s + CH)))
        dec = list(pool.map(lambda i: decode(reader(items[i]["ref"])), idx))
        order = np.argsort([len(d[0]) for d in dec])[::-1]
        b = []
        def flush():
            if not b:
                return
            mu, sd = run_batch(model, [dec[j][0] for j in b], device)
            for k, j in enumerate(b):
                mean_mm[idx[j]] = mu[k]; std_mm[idx[j]] = sd[k]
            b.clear()
        for j in order:
            longest = len(dec[b[0]][0]) if b else len(dec[j][0])
            if b and (len(b) + 1) * longest / SR > args.max_batch_sec:
                flush()
            b.append(j)
            durs[idx[j]] = dec[j][1]
        flush()
        done = idx[-1] + 1
        el = time.time() - t0
        print(f"{done}/{N}  {done/el:.1f} utt/s  eta {(N-done)/(done/el)/60:.1f} min", flush=True)
    mean_mm.flush(); std_mm.flush()
    with open(os.path.join(outdir, f"{args.dataset}{suffix}.meta.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["utt", "speaker", "system", "label", "split", "dur"])
        for it, d in zip(items, durs):
            w.writerow([it["utt"], it["speaker"], it["system"], it["label"], it["split"], f"{d:.3f}"])
    print("done", N, "items in", (time.time() - t0) / 60, "min")


if __name__ == "__main__":
    main()
