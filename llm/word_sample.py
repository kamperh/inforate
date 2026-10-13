"""Sample words from a trained checkpoint, continuing a word prompt.

The prompt is plain words (e.g. "the cat sat"), split on whitespace and encoded with
`is_split_into_words=True`, matching the convention used everywhere else in this unit.
Loads the saved vocab file `llm/word_prepare_mmap.py` built, not a freshly-recomputed
one (see `WordTokenizer`'s docstring in llm/tokenisers.py).

Generation itself uses `WordTokenizer`'s deterministic in-vocab/spell-out rule only,
same as validation/evaluation: no random augmentation, since that's a training-only
pre-processing step in llm/word_prepare_mmap.py, never inside `WordTokenizer` itself.

A diagnostic, not part of any result: nothing here is scored, and no number in
`results.md` comes from this file.
"""

from pathlib import Path

import torch
from transformers import OPTConfig, OPTForCausalLM

from tokenisers import WordTokenizer

# Change these
run_name = "opt125m-word-librispeech"  # Must match word_train.py's run_name, which
# takes a `-<tag>` suffix when `--tag` was passed, so a seeded run lives under e.g.
# opt125m-word-seed1-librispeech
checkpoint = (
    Path(__file__).resolve().parent
    / "checkpoints"
    / run_name
    / f"{run_name}-best.pt"  # Or f"{run_name}.pt" for the final (not best-val) checkpoint
)
vocab_path = Path(__file__).resolve().parent / "data" / "word-mmap" / "vocab.json"
device = "cuda" if torch.cuda.is_available() else "cpu"
prompt = "the cat sat"  # Plain words; <bos> gets added below via tokenizer.encode
max_new_tokens = 80
temperature = 1.0
top_p = 0.85

tokenizer = WordTokenizer(vocab_path)
ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
model = OPTForCausalLM(OPTConfig.from_dict(ckpt["cfg"]))
model.load_state_dict(ckpt["model"])
model.to(device)
model.eval()

prompt_ids = tokenizer.encode(prompt.split(), is_split_into_words=True)
input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
out = model.generate(
    input_ids,
    max_new_tokens=max_new_tokens,
    do_sample=True,
    temperature=temperature,
    top_p=top_p,
    pad_token_id=tokenizer.pad_token_id,
    # Model config sets eos_token_id=bos_token_id (see word_train.py). Since the
    # utterance-boundary fix, the model is actually trained to emit this token at real
    # utterance ends (not meaningless noise, as before that fix) -- letting generate()
    # stop there naturally is the correct behaviour now, capped by max_new_tokens either
    # way. Pass eos_token_id=None explicitly for an unbounded continuation that ignores
    # utterance boundaries. See https://huggingface.co/docs/transformers/en/llm_tutorial
    # for more sampling techniques
)
# Tokens rather than `tokenizer.decode(out[0], skip_special_tokens=True)`, because a
# spelled-out word comes back as `~`-marked character tokens and `_decode` strips the
# marker (see `_MarkedFallbackTokenizer`), which would hide where the fallback fired
print(" ".join(tokenizer.convert_ids_to_tokens(out[0])))
