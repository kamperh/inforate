"""Pretrained-LM bit rate on dev-clean, at the subword-bpe "level" (Qwen's own
pretrained tokenizer stands in for the unit -- no training, no vocab choice on
our side). Ported from the predecessor project's estimator comparison
(docs/assets/2026-08-30/paper_2.md Sec 4.6, "An estimator comparison, not a
correction"): the point isn't to add another unit to the controlled OPT-20M
comparison in results.md, it's to show how far the absolute bit rate moves when
the *model class* changes drastically -- an extreme point on the (unit, model
class, data budget) triple this whole project is about.

**Interpretational limit, carried over from the predecessor (not fixable
here)**: Qwen has seen an unmeasurable quantity of pretraining text, so this is
not a controlled, held-out estimate in the same sense as the rest of this
project's models -- it answers "how much does the estimator matter", not
"what is the true information rate of English".

**The primer.** Scored cold (no context), a pretrained LM is more surprised by
LibriSpeech's transcript style (lowercase, no punctuation) than it would be by
ordinary web text, and the predecessor found this penalty is *larger* for
bigger models (more sharply tuned to well-formed text) -- large enough (7.5
bits/s for Qwen2.5-3B) to swamp the effect being measured, and even flip the
1.5B-vs-3B ordering. Their fix: prepend a short primer in the same lowercase/
unpunctuated style before every scored utterance, so the model conditions on a
few examples of "this is the register we're in" -- held out of the scored set,
its own tokens never counted. `qwen.data_common.primer_utterances()` reproduces
this, with one deliberate deviation: the primer is drawn from train-clean-100,
not dev-clean (the predecessor drew its own primer from the same pool it
scored). This keeps the *full*, unmodified dev-clean set as the
scored/denominator set, matching every other unit's bit rate in this project.

**Scoring convention.** Mirrors the predecessor's "beginning-of-sequence token
prepended, end-of-sequence token scored" (their formula 6), extended with the
primer as shared context: `primer_ids + [sep] + utterance_ids + [sep]`, where
`sep = tokenizer.eos_token_id` (Qwen2.5's bos and eos are the same token,
`<|endoftext|>`, id 151643) doubles as both the separator after the primer and
the trailing scored end-of-utterance token. Loss is counted for predicting
`utterance_ids + [sep]` (the primer's own tokens are never a prediction
target) -- the same number of scored positions per utterance as the
non-primed convention would give, just conditioned on more context. This
*matched* llm/eval_bitrate.py's convention since the utterance-boundary fix,
which made llm/ train on and score a real end-of-utterance token too. What still
differs is the primer, which conditions on extra context llm/ has no analogue
for; bit rate stays comparable either way (a duration ratio), while perplexity
remains not comparable across scripts, on top of the already-standing
"perplexity is not comparable across units" caveat.

Byte-level BPE means Qwen's tokenizer has no OOV case: every input string is
representable, so none of this project's own hybrid-backoff/spell-out
machinery (llm/tokenisers.py) is needed here.

The primer is encoded once and cached (`past_key_values`), then reused as a
prefix for every utterance's own forward pass, avoiding a full re-encode of
the primer per utterance. `batch_size=1` throughout: cross_entropy's softmax
buffer scales with `batch_size x seq_len x vocab_size`, and Qwen2.5's
151,936-token vocab is larger than any unit tried elsewhere in this project,
so batching multiple utterances together at once is a likely OOM risk not
worth taking for a first pass, especially for Qwen2.5-3B on this machine's
8GB GPU.

Before the real runs, the `DynamicCache`/`attention_mask` plumbing in
`score_dev_clean` was smoke-tested with a tiny, randomly-initialized model
built with Qwen2.5-1.5B's real vocab_size/bos/eos but 2 layers and
hidden_size=32, combined with the real tokenizer, run through this exact
unmodified function against a handful of real dev-clean utterances. That
caught a real bug: `DynamicCache` has no public `.copy()` in this transformers
version, so a `hasattr(primer_cache, "copy")` fallback was silently reusing
(and cumulatively mutating) the same cache object across every utterance
instead of a fresh copy of the primer-only prefix; fixed with `copy.deepcopy`,
verified independent (`primer_cache.get_seq_length()` stays fixed across
repeated scorings).
"""

import argparse
import copy
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from data_common import DEV_SPLIT, primer_utterances, read_tier

OUT_PATH = Path(__file__).resolve().parent / "eval_bitrate.md"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_primer_text():
    """Primer utterances' word tokens, space-joined per utterance, utterances
    newline-separated (no punctuation is introduced, since none occurs in the
    target lowercase/unpunctuated style being primed for).
    """
    utterances = primer_utterances()
    sentences = [" ".join(tokens) for tokens in utterances.values()]
    return "\n".join(sentences)


def dev_clean_utterances():
    """{key: (word tokens, duration)} for all of dev-clean, silence-interval
    empty strings dropped from the token list (duration is unaffected -- see
    data_common.read_tier's own docstring; matches
    llm/data.py's TokenizedUnitsUtteranceDataset convention).
    """
    raw = read_tier([DEV_SPLIT], "words")
    return {
        key: ([t for t in tokens if t], duration)
        for key, (tokens, duration) in raw.items()
    }


def score_dev_clean(model, tokenizer):
    """Total (nats, scored target tokens, dev-clean speech-only seconds) over
    all of dev-clean, teacher-forced, each utterance scored as
    `primer + sep + utterance + sep` with the primer's own tokens never a
    prediction target (see module docstring). Summed, not averaged, so bit
    rate can be computed as one ratio of two totals, matching every other
    bit-rate calculation in this project. Duration is summed from the same
    `dev_clean_utterances()` call already needed for scoring, rather than a
    second, separate walk of dev-clean's TextGrids.
    """
    sep_id = tokenizer.eos_token_id

    primer_ids = tokenizer(build_primer_text(), add_special_tokens=False)["input_ids"]
    primer_tensor = torch.tensor([primer_ids], device=DEVICE)
    with torch.no_grad():
        primer_out = model(input_ids=primer_tensor, past_key_values=DynamicCache(), use_cache=True)
    primer_cache = primer_out.past_key_values
    primer_len = primer_tensor.shape[1]

    utterances = dev_clean_utterances()
    total_nats = 0.0
    total_tokens = 0
    total_duration = 0.0
    with torch.no_grad():
        for tokens, duration in utterances.values():
            total_duration += duration
            utt_ids = tokenizer(" ".join(tokens), add_special_tokens=False)["input_ids"]
            new_ids = [sep_id, *utt_ids, sep_id]
            new_tensor = torch.tensor([new_ids], device=DEVICE)
            attention_mask = torch.ones(
                (1, primer_len + new_tensor.shape[1]), dtype=torch.long, device=DEVICE
            )
            # Fresh copy of the primer cache per utterance: a Cache object is
            # mutated in place by each forward call, so the shared primer
            # prefix must not be extended cumulatively across utterances.
            # DynamicCache has no public .copy()/.clone() in this transformers
            # version (checked directly); copy.deepcopy is the verified-correct
            # way to get an independent cache (confirmed: mutating the deep
            # copy leaves the original's seq_length unchanged).
            cache = copy.deepcopy(primer_cache)
            out = model(
                input_ids=new_tensor,
                past_key_values=cache,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = out.logits[:, :-1, :]  # drop the last position: no target beyond it
            targets = new_tensor[:, 1:]  # utt_ids + trailing sep, primer excluded
            nats = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="sum"
            )
            total_nats += nats.item()
            total_tokens += targets.numel()

    return total_nats, total_tokens, total_duration


def evaluate(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.to(DEVICE)
    model.eval()

    total_nats, total_tokens, duration = score_dev_clean(model, tokenizer)
    total_bits = total_nats / math.log(2)

    return {
        "bit_rate": total_bits / duration,
        "perplexity": 2 ** (total_bits / total_tokens),
        "total_tokens": total_tokens,
        "duration_s": duration,
    }


def write_report(model_name, result):
    header = "| model | bit rate (bits/s) | perplexity | dev-clean tokens |"
    sep = "|---|---:|---:|---:|"
    row = (
        f"| {model_name} | {result['bit_rate']:.2f} | {result['perplexity']:.2f} | "
        f"{result['total_tokens']:,} |"
    )

    prior_lines = OUT_PATH.read_text().splitlines() if OUT_PATH.exists() else []
    existing_rows = [
        line
        for line in prior_lines
        if line.startswith("| ")
        and not line.startswith(("| model", "|---", f"| {model_name} |"))
    ]

    lines = [
        "# Qwen pretrained-LM bit rate on dev-clean\n",
        "One row per `--model` invocation; see qwen/eval_bitrate.py "
        "for the exact scoring convention and its differences from "
        "llm/eval_bitrate.py's.\n",
        "\n".join([header, sep, *existing_rows, row]) + "\n",
    ]
    print("Writing:", OUT_PATH)
    OUT_PATH.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B",
        help="HuggingFace model id, e.g. Qwen/Qwen2.5-1.5B or Qwen/Qwen2.5-3B "
        "(both already cached locally from the predecessor project). "
        "Default: Qwen/Qwen2.5-1.5B.",
    )
    args = parser.parse_args()

    print(f"{args.model}: loading and scoring dev-clean")
    result = evaluate(args.model)
    print(
        f"  bit rate {result['bit_rate']:.2f} bits/s, perplexity {result['perplexity']:.2f}, "
        f"{result['total_tokens']:,} dev-clean tokens, {result['duration_s']:.0f}s speech"
    )
    write_report(args.model, result)


if __name__ == "__main__":
    main()
