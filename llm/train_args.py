"""The command-line flags every `<unit>_train.py` shares.

These five were byte-identical in all five scripts, so they lived in five places
and had to be edited in five places: `--epochs` and `--lr-max` were added that
way, then `--seed`, each time by patching every script. Defining them once
removes that.

Only argument *parsing* is shared. Everything that decides what a run actually
does -- tokeniser, tier, mmap dtype, batch size, accumulation steps -- stays in
each unit's own script, because those genuinely differ per unit and are what a
reader goes there to find. `word_bpe_train.py` adds its own `--n-merges` on top.

Defaults are the canonical configuration: 20 epochs, `lr_max=1e-3`, `seed = 0`.
Changing one here changes it for every unit, which is the point.
"""

import argparse


def build_parser(description):
    """An `ArgumentParser` carrying the five flags shared by every unit."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--tag",
        default="",
        help="Optional short tag distinguishing this run from the default "
        "hyperparameters (e.g. lr5e-4), appended to run_name/checkpoint_dir so "
        "a variant run can't collide with another run. Empty by default.",
    )
    parser.add_argument(
        "--lr-max",
        type=float,
        default=1e-3,
        help="Peak learning rate (default: 1e-3, the canonical value). Pass a "
        "different value together with --tag to run a hyperparameter-sweep "
        "variant without touching the canonical checkpoint.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Training length in epochs (default: 20, the canonical value); "
        "max_steps and warmup_steps are derived from it. Pass a different value "
        "together with --tag to run a sweep variant.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0, the canonical value). Seeds torch, numpy "
        "and python, so it varies both initialisation and the random training "
        "window sampling. Pass a different value together with --tag to measure "
        "run-to-run variance without touching the canonical checkpoint. Note it "
        "does not vary the augmentation draw, which is fixed when the mmap is "
        "built, nor any learned BPE merges.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing checkpoint_dir. Without this (or "
        "--tag), the script refuses to start if checkpoint_dir already has files.",
    )
    return parser
