"""Sample subword BPE tokens from a trained checkpoint, continuing a word prompt.

The prompt is plain words, split on whitespace and encoded with
`is_split_into_words=True`, which is how `SubwordBpeTokenizer` is called everywhere
else. Loads the saved tokeniser `llm/subword_bpe_prepare_mmap.py` wrote, never a
freshly-trained one, so the ids mean the same thing they did during training.

A diagnostic, not part of any result: nothing here is scored, and no number in
`results.md` comes from this file.
"""

from pathlib import Path

import torch
from transformers import OPTConfig, OPTForCausalLM

from tokenisers import SubwordBpeTokenizer

# Change these
run_name = "opt125m-subword-bpe-librispeech"  # Must match subword_bpe_train.py's
# run_name, which takes a `-<tag>` suffix when `--tag` was passed, so a seeded run
# lives under e.g. opt125m-subword-bpe-seed1-librispeech
checkpoint = (
    Path(__file__).resolve().parent
    / "checkpoints"
    / run_name
    / f"{run_name}-best.pt"  # Or f"{run_name}.pt" for the final (not best-val) checkpoint
)
tokenizer_path = (
    Path(__file__).resolve().parent / "data" / "subword-bpe-mmap" / "tokenizer.json"
)
device = "cuda" if torch.cuda.is_available() else "cpu"
prompt = "the cat sat"  # Plain words; <bos> gets added below via tokenizer.encode
max_new_tokens = 80
temperature = 1.0
top_p = 0.85

tokenizer = SubwordBpeTokenizer(tokenizer_path)
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
    # Model config sets eos_token_id=bos_token_id (see subword_bpe_train.py). Since the
    # utterance-boundary fix, the model is actually trained to emit this token at real
    # utterance ends (not meaningless noise, as before that fix) -- letting generate()
    # stop there naturally is the correct behaviour now, capped by max_new_tokens either
    # way. Pass eos_token_id=None explicitly for an unbounded continuation that ignores
    # utterance boundaries. See https://huggingface.co/docs/transformers/en/llm_tutorial
    # for more sampling techniques
)
print(tokenizer.decode(out[0], skip_special_tokens=True))
