# Qwen pretrained-LM bit rate on dev-clean

One row per `--model` invocation; see qwen/eval_bitrate.py for the exact scoring convention and its differences from llm/eval_bitrate.py's.

| model | bit rate (bits/s) | perplexity | dev-clean tokens |
|---|---:|---:|---:|
| Qwen/Qwen2.5-1.5B | 25.90 | 94.37 | 62,551 |
| Qwen/Qwen2.5-3B | 25.26 | 84.34 | 62,551 |
