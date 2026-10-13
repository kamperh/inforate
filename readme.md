# Inforate

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](license.md)


## Overview

The following units are compared in the **main comparison**. All of the units are
represented as constituent phones, e.g. every word is written as the phones of which it
is composed.

- phone: 39 unit types
- subwordphnx-BPE-5000: Subword BPE units are obtained at the phone level, i.e. common
  co-occurring phone sequences are merged. Merging across word boundaries is not
  allowed.
- syllable: Represented as phones, e.g. "sorry" is represented as "S-AA R-IY".
- wordphn: The word "sorry" is represented as "S-AA-R-IY". Each word type is therefore
  determined by its phonetic realisation. This means that "to", "too" and "two" are all
  represented as "T-UW".
- wordphn-BPE-200: As above for wordphn, but common multi-word combinations are now
  merged.
- wordphn-BPE-1000
- wordphn-BPE-5000

There is also several **sanity checks** using different units:

- subword-BPE: Merging occurs at the character level rather than the phone level as
  above for subwordphnx-BPE.
- word: Now words are represented according to their orthography.
- word-BPE-5000: Multi-word combinations are now merged according to word orthography.

Finally, two Qwen models, Qwen2.5-1.5B and Qwen2.5-3B, are also applied to the data
using its own unicode BPE tokenisation.



## Environment

Set up the base environment:

```bash
conda create -n inforate python=3.14 -y
conda run -n inforate pip install tgt tqdm numpy matplotlib
conda activate inforate
```

For the neural models, you also need to install the following in the environment:

```bash
pip install transformers tokenizers wandb
pip install "torch==2.13.0+cu126" --index-url https://download.pytorch.org/whl/cu126
```

For logging training to [Weights & Biases](https://wandb.ai/site/) you will have to have
an account and log in:

```bash
wandb login
```

You only have to do this once on your machine. You can also set `wandb_mode="offline"`
in a script's `TrainConfig` if you prefer not to do this.



## Data

Download LibriSpeech TextGrid alignments from
[here](https://github.com/s-malan/evaluation/releases/tag/v0.1.0) and extract it into
`data/librispeech-alignments/`. As an example, you should have a
`data/librispeech-alignments/dev-clean/` directory after completing this step.

To get the data statistics, run:

```bash
python ngram/librispeech_stats.py  # writes ngram/librispeech_stats.md
```

A subset of the statistics:

| set | utterances | speech duration | word tokens | word vocab |
|---|---:|---:|---:|---:|
| LibriSpeech 960h | 281,239 | 769.8h (2771365s) | 9,403,427 | 89,048 |
| dev-clean | 2,703 | 4.4h (15846s) | 54,402 | 8,332 |



## N-gram language models

N-gram models of order 1 to 5 are trained on 25%, 50%, 75%, 100% of LibriSpeech 960h.
Bitrates and perplexities are evaluated on dev-clean. For each of the different units,
a particular script would be run.

**Phones.** Stress is stripped, resulting in the 39-symbol ARPABET set.

```bash
python ngram/ngram_phone_sweep.py
```

**Syllables.** Out-of-vocabulary syllables are spelled out via their constituent phones.

```bash
python ngram/ngram_syllable_sweep.py
```

**Words.** Out-of-vocabulary words are spelled out via their constituent characters.

```bash
python ngram/ngram_word_sweep.py
```

**Word-BPE.** Can be done at different levels.

```bash
python ngram/ngram_bpe_sweep.py --n-merges 200
python ngram/ngram_bpe_sweep.py --n-merges 1000
python ngram/ngram_bpe_sweep.py --n-merges 5000
```

To inspect and view merges, the following can be used:

```bash
python utils/inspect_bpe.py                               # first 30 of 200 merges
python utils/inspect_bpe.py --n-merges 5000 --top 50
python utils/inspect_bpe.py --n-merges 1000 --all
python utils/inspect_bpe.py --n-merges 5000 --longest 15  # deepest merge chains
```



## Neural language models

For each neural model, data is first prepared and then the model is trained.

**Phones**

```bash
python llm/phone_prepare_mmap.py
python llm/phone_train.py
```

**Syllables**

```bash
python llm/syllable_prepare_mmap.py
python llm/syllable_train.py
```

**Subword BPE.** This is not something I am really interested in, but I included it as a
sanity check. Here BPE is performed within words, as is done normally. You can also
sample from this model.

```bash
python llm/subword_bpe_prepare_mmap.py
python llm/subword_bpe_train.py
python llm/subword_bpe_sample.py   # optional: sample continuations from a checkpoint
```

**Words.** You can also sample from this model.

```bash
python llm/word_prepare_mmap.py
python llm/word_train.py
python llm/word_sample.py   # optional: sample continuations from a checkpoint
```

**Word-BPE.** Again this is done at multiple levels.

```bash
for n in 200 1000 5000; do
  python llm/word_bpe_prepare_mmap.py --n-merges $n
  python llm/word_bpe_train.py --n-merges $n
done
```



## Evaluation

Every unit is scored agains an existing checkpoint, specified in `UNITS` at the top of
the script. The script writes `llm/eval_bitrate.md`.

```bash
python llm/eval_bitrate.py
```



## AI disclosure

The code for this project was written using AI (Claude). I conceived of all the
experiments and carefully laid out a plan; AI then implemented it. I reviewed the code,
found and fixed bugs, and edited it to improve clarity. The documentation produced and
maintained by AI is included with this repository (`design.md`, `results.md`,
`reproduce.md`, `CLAUDE.md`).
