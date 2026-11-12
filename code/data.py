"""Loading of cached features and metadata."""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Feats:
    def __init__(self, dataset, model="wavlm-large", pool="mean"):
        base = os.path.join(ROOT, "feats", model, dataset)
        self.X = np.load(base + f".{pool}.npy", mmap_mode="r")      # [N, L, D] float16
        self.meta = pd.read_csv(base + ".meta.csv", dtype=str, keep_default_na=False)
        assert len(self.meta) == self.X.shape[0]
        self.L, self.D = self.X.shape[1], self.X.shape[2]

    def layer(self, l, idx=None):
        """float32 [n, D] for layer l (idx: integer positions)."""
        if idx is None:
            return np.asarray(self.X[:, l, :], dtype=np.float32)
        idx = np.asarray(idx)
        order = np.argsort(idx)
        out = np.empty((len(idx), self.D), dtype=np.float32)
        out[order] = np.asarray(self.X[idx[order], l, :], dtype=np.float32)
        return out

    def rows(self, idx):
        idx = np.asarray(idx); o = np.argsort(idx)
        out = np.empty((len(idx), self.L, self.D), dtype=np.float16)
        out[o] = self.X[idx[o]]
        return out

    def where(self, **conds):
        m = np.ones(len(self.meta), dtype=bool)
        for k, v in conds.items():
            col = self.meta[k].values
            m &= np.isin(col, v) if isinstance(v, (list, tuple, set, np.ndarray)) else (col == v)
        return np.nonzero(m)[0]


def speaker_chunks(speakers, k, seed=0):
    """Split a list of unique speakers into k chunks (deterministic)."""
    rng = np.random.RandomState(seed)
    spk = np.array(sorted(set(speakers)))
    rng.shuffle(spk)
    return [set(c.tolist()) for c in np.array_split(spk, k)]


class MultiFeats:
    """Several Feats objects addressed with one global row index."""

    def __init__(self, feats):
        self.parts = feats
        self.off = np.cumsum([0] + [len(f.meta) for f in feats])
        self.meta = pd.concat([f.meta for f in feats], ignore_index=True)
        self.L, self.D = feats[0].L, feats[0].D

    def _split(self, idx):
        idx = np.asarray(idx)
        which = np.searchsorted(self.off, idx, side="right") - 1
        return idx, which

    def layer(self, l, idx):
        idx, which = self._split(idx)
        out = np.empty((len(idx), self.D), dtype=np.float32)
        for k, f in enumerate(self.parts):
            m = which == k
            if m.any():
                out[m] = f.layer(l, idx[m] - self.off[k])
        return out

    def rows(self, idx):
        """float16 [n, L, D]"""
        idx, which = self._split(idx)
        out = np.empty((len(idx), self.L, self.D), dtype=np.float16)
        for k, f in enumerate(self.parts):
            m = np.nonzero(which == k)[0]
            if len(m):
                loc = idx[m] - self.off[k]
                o = np.argsort(loc)
                out[m[o]] = f.X[loc[o]]
        return out

    def where(self, **conds):
        m = np.ones(len(self.meta), dtype=bool)
        for k, v in conds.items():
            col = self.meta[k].values
            m &= np.isin(col, v) if isinstance(v, (list, tuple, set, np.ndarray)) else (col == v)
        return np.nonzero(m)[0]
