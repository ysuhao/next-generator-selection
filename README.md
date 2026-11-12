# Choosing for the next generator — code, result files and logs

Material accompanying the manuscript "Choosing for the next generator: when does statistics-only leave-one-generator-out selection help continual speech deepfake detection?" (submitted to Computer Speech & Language).

**Licence:** MIT (see `LICENSE`). The speech corpora are not redistributed.

## Contents
- `code/`
  - `extract_features.py`, `dfadd_items.py`, `data.py` — frozen-encoder features (per-layer temporal mean, float16) for ASVspoof 2019 LA, DFADD and In-the-Wild, read straight from the downloaded archives.
  - `gss.py` — generator-wise sufficient statistics, the closed-form detector, the statistics-only leave-one-generator-out (LOGO) criterion and EER.
  - `pools.py` — generator pools, families, speaker-disjoint adaptation/test splits and stream orders.
  - `stream2.py` — the generator-incremental stream with LOGO, in-distribution (ID-CV), fixed and oracle selection (`stream.py` is the earlier, slower implementation that gives identical results).
  - `stream_grad.py`, `stream_extra.py` — gradient-trained head baselines (fine-tuning, EWC, LwF, replay, joint) and NCM / RanPAC-style baselines.
  - `pilot_la.py` — static test under the official ASVspoof 2019 LA protocol.
  - `analyze_stream.py`, `analyze_domain.py`, `analyze_combined.py`, `itw_diagnostic.py`, `check_removal.py` — analyses.
  - `make_assets.py` — produces every table, figure and number of the paper from `results/`.
- `results/` — one JSON line per stream step for every run (`stream_*.jsonl`, `grad_*.jsonl`, `extra_*.jsonl`), including the selection criteria of all 750 configurations per step, and derived JSON summaries. `results/slim/` holds compact per-step summaries of the two XLS-R voice-conversion streams.
- `logs/` — run logs, including the logs of runs that were interrupted and resumed.

## Not included
- ASVspoof 2019 LA (https://datashare.ed.ac.uk/handle/10283/3336), DFADD (https://huggingface.co/datasets/isjwdu/DFADD), In-the-Wild (https://deepfake-total.com/in_the_wild). Place the archives under `data/` as expected by `code/extract_features.py`.
- Encoder weights: `microsoft/wavlm-large`, `facebook/wav2vec2-xls-r-300m` (Hugging Face).
- Cached features (about 12 GB per encoder for ASVspoof 2019 LA).

## Reproducing
    pip install -r requirements.txt
    python code/extract_features.py --dataset asvspoof19 --model microsoft/wavlm-large
    python code/extract_features.py --dataset dfadd      --model microsoft/wavlm-large
    python code/extract_features.py --dataset itw        --model microsoft/wavlm-large
    python code/pilot_la.py
    python code/stream2.py --poolname la --orders 20 --itw
    python code/stream2.py --poolname la+dfadd --chrono --orders 20 --itw
    python code/stream2.py --poolname la --family_last VC --exclude_family TTS-VC --orders 8 --tag _L25 --save_T
    python code/analyze_combined.py && python code/itw_diagnostic.py
    python code/make_assets.py     # writes paper/tables, paper/figs, paper/numbers.tex

The options of each run are encoded in the name of its result file (`stream_<pool>_<encoder>_<pooling>_<condition>[_chrono][_lastVC][_noTTS-VC][_bonautt][_idcvpaired][tag].jsonl`); see the argument list of `code/stream2.py`. Secondary runs used fewer orders or every second layer, as stated in the paper.
