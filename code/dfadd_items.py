"""DFADD (HF isjwdu/DFADD) items read from the downloaded parquet shards.

File names look like p227_164_GradTTS.flac (spoof) or p228_290.wav (bona fide, VCTK).
Only the shards present in data/dfadd are used; the official split column is kept.
"""
import glob, os, re, threading
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_PER_SYSTEM_TRAIN = 3000     # cap per (split, system) to keep extraction short; deterministic (first rows)


def items_dfadd():
    files = sorted(glob.glob(os.path.join(ROOT, "data", "dfadd", "*.parquet")))
    items, count = [], {}
    for fp in files:
        pf = pq.ParquetFile(fp)
        row0 = 0
        for rg in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(rg, columns=["audio_name", "split", "label"])
            for k, (name, split, label) in enumerate(zip(t.column("audio_name").to_pylist(), t.column("split").to_pylist(), t.column("label").to_pylist())):
                stem = os.path.splitext(name)[0]
                parts = stem.split("_")
                system = "bonafide" if label == "real" else "DF_" + "_".join(parts[2:])
                key = (split, system)
                count[key] = count.get(key, 0) + 1
                if split == "train" and count[key] > MAX_PER_SYSTEM_TRAIN:
                    continue
                items.append(dict(utt=name, speaker=parts[0], system=system,
                                  label="bonafide" if label == "real" else "spoof", split=split, ref=(fp, rg, k)))
    cache = threading.local()

    def reader(ref):
        fp, rg, k = ref
        if getattr(cache, "key", None) != (fp, rg):
            cache.tab = pq.ParquetFile(fp).read_row_group(rg, columns=["audio"]).column("audio").to_pylist()
            cache.key = (fp, rg)
        return cache.tab[k]["bytes"]

    return items, reader


if __name__ == "__main__":
    import collections
    items, _ = items_dfadd()
    c = collections.Counter((i["split"], i["system"]) for i in items)
    for k in sorted(c):
        print(k, c[k], len({i["speaker"] for i in items if (i["split"], i["system"]) == k}))
