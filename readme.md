# Inforate

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](license.md)


## Overview

This is the repository for the paper:

- H. Kamper, "On the information rate of phones, syllables and words," 2026. [[arXiv](https://www.tocomplete.com)]

The following units are compared in the **main comparison**. All of the units are
represented as constituent phones, e.g. every word is written as the phones of which it
is composed.

- phone: 39-symbol ARPABET set
- subwordphnx-BPE-2000: Subword BPE units are obtained at the phone level, i.e. common
  co-occurring phone sequences are merged. Merging across word boundaries is not
  allowed. The `-2000` indicates the final vocabulary size after merging.
- syllable: Represented as phones, e.g. "sorry" is represented as "S-AA R-IY".
- wordphn: The word "sorry" is represented as "S-AA-R-IY". Each word is therefore
  determined by its phonetic realisation. This means that too" and "two" are both
  represented as "T-UW".
- wordphn-BPE-200: As above for wordphn, but common multi-word combinations are now
  merged.
- wordphn-BPE-1000
- wordphn-BPE-5000

There are also several **sanity checks** using different units:

- subword-BPE: Merging occurs at the character level rather than the phone level as
  above for subwordphnx-BPE.
- word: Now words are represented according to their orthography.
- word-BPE-5000: Multi-word combinations are now merged according to word orthography.

Finally, two Qwen models, Qwen2.5-1.5B and Qwen2.5-3B, are also applied to the data
using its own byte-level BPE tokenisation.



## Environment

Set up the base environment:

```bash
conda create -n inforate python=3.14 -y
conda run -n inforate pip install tgt tqdm numpy
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

**Subword-BPE (phone level).** Common co-occurring phones are merged to form new units.
Merges across word boundaries are not allowed. Can be done at different levels.

```bash
python ngram/ngram_subwordphnx_bpe_sweep.py --vocab-size 200
python ngram/ngram_subwordphnx_bpe_sweep.py --vocab-size 2000
python ngram/ngram_subwordphnx_bpe_sweep.py --vocab-size 5000
```

**Syllables.** Out-of-vocabulary syllables are spelled out via their constituent phones.

```bash
python ngram/ngram_syllable_sweep.py
```

**Words (phonetic realisations).** Out-of-vocabulary words are spelled out via their
constituent phones.

```bash
python ngram/ngram_wordphn_sweep.py
```

**Word-BPE (phonetic realisations).** Can be done at different levels.

```bash
python ngram/ngram_wordphn_bpe_sweep.py --n-merges 200
python ngram/ngram_wordphn_bpe_sweep.py --n-merges 1000
python ngram/ngram_wordphn_bpe_sweep.py --n-merges 5000
```

To inspect and view merges, the following can be used:

```bash
python utils/inspect_bpe.py                               # first 30 of 200 merges
python utils/inspect_bpe.py --n-merges 5000 --top 50
python utils/inspect_bpe.py --n-merges 1000 --all
python utils/inspect_bpe.py --n-merges 5000 --longest 15  # deepest merge chains
```



## Neural language models

For each neural model, data is first prepared and then the model is trained. Directly
below are the models that form the main comparison.

**Phones**

```bash
python llm/phone_prepare_mmap.py
python llm/phone_train.py
```

**Subword-BPE (phone level)**

```bash
python llm/subwordphnx_bpe_prepare_mmap.py --vocab-size 5000
python llm/subwordphnx_bpe_train.py --vocab-size 5000
python llm/eval_bitrate.py --units subwordphnx-bpe-5000
```

**Syllables**

```bash
python llm/syllable_prepare_mmap.py
python llm/syllable_train.py
```

**Words (phonetic realisations)**

```bash
python llm/wordphn_prepare_mmap.py
python llm/wordphn_train.py
python llm/eval_bitrate.py --units wordphn
python utils/bootstrap.py --units neural:wordphn neural:word  # per-utterance samples
```

**Word-BPE (phonetic realisations).** Includes a loop over the different levels.

```bash
for n in 200 1000 5000; do
    python llm/wordphn_bpe_prepare_mmap.py --n-merges $n
    python llm/wordphn_bpe_train.py --n-merges $n
done
python llm/eval_bitrate.py --units wordphn-bpe-200 wordphn-bpe-1000 wordphn-bpe-5000
python utils/bootstrap.py --units neural:word \
    neural:wordphn-bpe-200 neural:wordphn-bpe-1000 neural:wordphn-bpe-5000
```

The models below are sanity checks.

**Subword BPE.** This is not something I am really interested in, but I included it as a
sanity check. Here BPE is performed within words on characters, as is done normally. You
can also sample from this model.

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

Every unit is scored using an existing checkpoint, specified in `UNITS` at the top of
the script. The script writes `llm/eval_bitrate.md`.

```bash
python llm/eval_bitrate.py
```

I also trained things for multiple seeds. This can be done e.g. as follows:

```bash
s=1
python llm/word_bpe_train.py --n-merges 5000 --seed $s --tag seed$s
python llm/eval_bitrate.py
python utils/bootstrap.py --seeds
```



## Qwen models

Bitrates for two large models are considered:

```bash
python qwen/eval_bitrate.py --model Qwen/Qwen2.5-1.5B
python qwen/eval_bitrate.py --model Qwen/Qwen2.5-3B
```



## AI disclosure

The code for this project was written using AI (Claude Opus). I conceived of all the
experiments and carefully laid out a plan; AI then implemented it. I reviewed the code,
found and fixed bugs, and edited it to improve clarity. The documentation produced and
maintained by AI is included with this repository (`design.md`, `results.md`,
`reproduce.md`). I wrote this `readme.md` as well as the paper that accompany this work.
