"""Generator pools: which utterances a continual learner may adapt on, and which are test only.

ASVspoof 2019 LA pool (17 distinct generators; A16 and A19 are re-runs of A04 and A06
on other speakers and are left out of the stream):
  A01-A06 : adapt = train partition,            test = dev partition
  A07-A18 : adapt = half of the eval speakers,  test = the other half
Bona fide speech of a partition is cut into speaker-disjoint chunks, one chunk
arriving together with each generator of that partition.
"""
import numpy as np
from data import Feats, MultiFeats

FAMILY = {"A01": "TTS", "A02": "TTS", "A03": "TTS", "A04": "TTS", "A05": "VC", "A06": "VC",
          "A07": "TTS", "A08": "TTS", "A09": "TTS", "A10": "TTS", "A11": "TTS", "A12": "TTS",
          "A13": "TTS-VC", "A14": "TTS-VC", "A15": "TTS-VC", "A17": "VC", "A18": "VC"}
LA_TRAIN_GENS = ["A01", "A02", "A03", "A04", "A05", "A06"]
LA_EVAL_GENS = ["A07", "A08", "A09", "A10", "A11", "A12", "A13", "A14", "A15", "A17", "A18"]


class Pool:
    """gens: name -> dict(adapt=idx, test=idx, bona_adapt=idx, bona_test=idx, family, domain)
    All idx are row positions in self.F (one Feats object or a stacked view)."""

    def __init__(self, F):
        self.F = F
        self.gens = {}

    def names(self):
        return list(self.gens.keys())


DFADD_GENS = ["DF_GradTTS", "DF_NaturalSpeech2", "DF_StyleTTS2", "DF_matcha", "DF_pflow"]


def get_pool(name, model="wavlm-large", pool="mean", bona_split="speaker"):
    if name == "la":
        return la_pool(model, pool, bona_split=bona_split)
    if name == "la+dfadd":
        return la_pool(model, pool, with_dfadd=True, bona_split=bona_split)
    raise ValueError(name)


def la_pool(model="wavlm-large", pool="mean", split_seed=0, with_dfadd=False, bona_split="speaker"):
    F = Feats("asvspoof19", model, pool)
    if with_dfadd:
        F = MultiFeats([F, Feats("dfadd", model, pool)])
    m = F.meta
    spk = m.speaker.values
    P = Pool(F)
    rng = np.random.RandomState(split_seed)

    def chunk_by_speaker(idx, k):
        sp = np.array(sorted(set(spk[idx])))
        rng.shuffle(sp)          # (drawn in both modes so that all other random choices stay identical)
        if bona_split == "utt":  # control: every chunk contains utterances of all adaptation speakers
            perm = np.random.RandomState(split_seed + 7).permutation(len(idx))
            return [np.sort(idx[perm[j::k]]) for j in range(k)]
        return [idx[np.isin(spk[idx], c)] for c in np.array_split(sp, k)]

    # --- train/dev partitions
    tr_b = F.where(split="train", system="bonafide")
    if with_dfadd:                      # DFADD also has a "train" split: keep ASVspoof rows only here
        tr_b = tr_b[tr_b < F.off[1]]
    dev_b = F.where(split="dev", system="bonafide")
    for g, cb in zip(LA_TRAIN_GENS, chunk_by_speaker(tr_b, len(LA_TRAIN_GENS))):
        P.gens[g] = dict(adapt=F.where(split="train", system=g), test=F.where(split="dev", system=g),
                         bona_adapt=cb, bona_test=dev_b, family=FAMILY[g], domain="LA-dev")
    # --- eval partition, speaker-disjoint halves
    ev = F.where(split="eval")
    ev_spk = np.array(sorted(set(spk[ev])))
    rng.shuffle(ev_spk)
    adapt_spk = set(ev_spk[: len(ev_spk) // 2].tolist())
    is_adapt = np.array([s in adapt_spk for s in spk])
    ev_b = F.where(split="eval", system="bonafide")
    ev_b_adapt, ev_b_test = ev_b[is_adapt[ev_b]], ev_b[~is_adapt[ev_b]]
    for g, cb in zip(LA_EVAL_GENS, chunk_by_speaker(ev_b_adapt, len(LA_EVAL_GENS))):
        idx = F.where(split="eval", system=g)
        P.gens[g] = dict(adapt=idx[is_adapt[idx]], test=idx[~is_adapt[idx]],
                         bona_adapt=cb, bona_test=ev_b_test, family=FAMILY[g], domain="LA-eval")
    if with_dfadd:
        # DFADD: adapt = (subset of) train split, test = official valid + test splits (4 speakers)
        is_df = np.zeros(len(m), dtype=bool); is_df[F.off[1]:] = True
        d_b = F.where(split="train", system="bonafide"); d_b = d_b[is_df[d_b]]
        d_bt = F.where(split=["valid", "test"], system="bonafide"); d_bt = d_bt[is_df[d_bt]]
        for g, cb in zip(DFADD_GENS, chunk_by_speaker(d_b, len(DFADD_GENS))):
            P.gens[g] = dict(adapt=F.where(split="train", system=g), test=F.where(split=["valid", "test"], system=g),
                             bona_adapt=cb, bona_test=d_bt, family="TTS-diff", domain="DFADD")
    return P


def make_stream(P, order_id, condition="full", base=3, chrono=False, family_last=None):
    """Deterministic stream: generator order, step composition and arriving utterances.
    chrono=True: all ASVspoof 2019 generators (random order) before the DFADD generators (random order).
    Returns (order, steps, arrive, rng); rng continues to be used by the caller."""
    names = P.names()
    rng = np.random.RandomState(1000 + order_id)
    order = [names[i] for i in rng.permutation(len(names))]
    if chrono:
        order = [g for g in order if P.gens[g]["domain"] != "DFADD"] + [g for g in order if P.gens[g]["domain"] == "DFADD"]
    if family_last:      # all generators of one family (e.g. "VC") arrive after all the others
        order = [g for g in order if P.gens[g]["family"] != family_last] + [g for g in order if P.gens[g]["family"] == family_last]
    steps = [order[:base]] + [[g] for g in order[base:]]
    arrive = {}
    for g in names:
        a, b = P.gens[g]["adapt"], P.gens[g]["bona_adapt"]
        if condition == "full":
            n = len(a)
        elif condition == "imb":
            n = int(round(10 ** rng.uniform(np.log10(30), np.log10(len(a)))))
        elif condition.startswith("shot"):
            n = int(condition[4:])
        else:
            raise ValueError(condition)
        a = a[rng.permutation(len(a))[:n]]
        if condition.startswith("shot"):   # equally few bona fide utterances
            b = b[rng.permutation(len(b))[:n]]
        arrive[g] = (np.sort(a), np.sort(b))
    return order, steps, arrive, rng
