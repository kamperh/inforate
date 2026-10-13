"""Confidence intervals on bit rate, by bootstrapping dev-clean.

Every number in this project is one point estimate, and the conclusions now turn
on differences of well under a bit per second. This script attaches intervals to
them, reading the per-utterance `{key: [bits, duration]}` dumps that
`llm/eval_bitrate.py` and `ngram/per_utterance.py` write. Both stages produce the
same shape, so one loader serves both and cross-stage comparisons work.

Bit rate is a ratio of sums, `sum(bits) / sum(duration)`, so a resample means
drawing utterances with replacement and recomputing the ratio, not averaging
per-utterance rates (which would weight a half-second utterance the same as a
thirty-second one).

**Two things this script is careful about, because getting them wrong is the easy
mistake here.**

1. **Comparisons use a paired interval, not two marginal ones.** Every unit scores
   the identical 2,703 utterances, so per-utterance difficulty is shared and the
   two estimates are strongly correlated. Asking whether their marginal intervals
   overlap is badly over-conservative and can hide a real difference. Instead the
   difference `rate_A - rate_B` is recomputed on *the same* resample, and it is
   that distribution which is reported.
2. **The headline interval resamples speakers, not utterances.** Utterances from
   one speaker are not independent, so an i.i.d.-over-utterances bootstrap
   understates the uncertainty. dev-clean has 40 speakers and no speaker exceeds
   2.8% of total duration, so a cluster bootstrap over speakers is stable. Both are
   reported; the speaker-level one is the honest number and the utterance-level one
   is the optimistic bound.

What this does **not** measure is training variance: each checkpoint is treated as
fixed, so it cannot say whether a retrain at another seed would land elsewhere. For
`ngram/` that is a complete answer, since an n-gram fit is deterministic given its
training data. For the neural units it is not, which is what the seed runs measure
(`--seeds`).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # utils/ -> project root
SOURCES = {"neural": ROOT / "llm" / "per_utterance",
           "ngram": ROOT / "ngram" / "per_utterance"}
B = 10000
SEED = 0
ALPHA = 5.0  # percent, for a 95% percentile interval


def load(path):
    """{key: (bits, duration)} from one dump."""
    return {k: (float(b), float(d)) for k, (b, d) in json.loads(path.read_text()).items()}


def discover():
    """{label: {key: (bits, duration)}} for every dump found, labelled by stage."""
    out = {}
    for stage, d in SOURCES.items():
        for f in sorted(d.glob("*.json")) if d.exists() else []:
            out[f"{stage}:{f.stem}"] = load(f)
    return out


def aligned(datasets):
    """(keys, bits matrix, duration vector) over the utterances every dataset has.

    Restricting to the shared keys is what makes the pairing exact; they should
    already be identical across units, and the caller asserts as much.
    """
    common = sorted(set.intersection(*(set(d) for d in datasets.values())))
    bits = np.array([[datasets[name][k][0] for k in common] for name in datasets])
    first = next(iter(datasets))
    durations = np.array([datasets[first][k][1] for k in common])
    # Durations come from the alignment, so every unit must report the same
    # value for a given utterance. Taking them from one dataset is only valid
    # because of that; check it rather than assume, since a mismatch would make
    # the ratio statistic quietly incomparable across units.
    for name, d in datasets.items():
        if not np.allclose([d[k][1] for k in common], durations, atol=1e-9):
            raise ValueError(f"{name} disagrees with {first} on utterance durations")
    return common, bits, durations


def resample_indices(keys, rng, by_speaker):
    """One bootstrap resample, as an index array into `keys`.

    Utterance keys are `speaker-chapter-utterance`, so the speaker is the first
    field. Resampling speakers draws whole clusters with replacement and keeps
    every utterance belonging to a drawn speaker, so the resample size varies a
    little between draws; that is inherent to a cluster bootstrap and is fine for
    a ratio statistic.
    """
    n = len(keys)
    if not by_speaker:
        return rng.integers(0, n, size=n)
    groups = defaultdict(list)
    for i, k in enumerate(keys):
        groups[k.split("-")[0]].append(i)
    speakers = list(groups)
    drawn = rng.integers(0, len(speakers), size=len(speakers))
    return np.concatenate([groups[speakers[s]] for s in drawn])


def bootstrap(datasets, by_speaker, b=B, seed=SEED):
    """(point estimates, resampled rates) for every dataset, on shared resamples.

    `rates` is (B, n_datasets): row i is every unit's bit rate under the same
    resample i, which is what makes a paired difference well defined.
    """
    names = list(datasets)
    keys, bits, durations = aligned(datasets)
    point = bits.sum(axis=1) / durations.sum()
    rng = np.random.default_rng(seed)
    rates = np.empty((b, len(names)))
    for i in range(b):
        idx = resample_indices(keys, rng, by_speaker)
        rates[i] = bits[:, idx].sum(axis=1) / durations[idx].sum()
    return dict(zip(names, point)), rates, names


def ci(samples):
    lo, hi = np.percentile(samples, [ALPHA / 2, 100 - ALPHA / 2])
    return lo, hi


def report(datasets, comparisons, by_speaker):
    label = "speaker-level (cluster)" if by_speaker else "utterance-level (i.i.d.)"
    point, rates, names = bootstrap(datasets, by_speaker)
    print(f"\n=== {label} bootstrap, B={B}, 95% percentile intervals ===\n")
    print(f"  {'unit':<28} {'bit rate':>9}  {'95% CI':>18}  {'width':>6}")
    for j, n in enumerate(names):
        lo, hi = ci(rates[:, j])
        print(f"  {n:<28} {point[n]:>9.2f}  [{lo:>7.2f}, {hi:>7.2f}]  {hi - lo:>6.2f}")

    if not comparisons:
        return
    print(f"\n  paired differences (A - B), same resample:\n")
    print(f"  {'comparison':<44} {'diff':>7}  {'95% CI':>18}  signif")
    for a, b_ in comparisons:
        if a not in names or b_ not in names:
            continue
        d = rates[:, names.index(a)] - rates[:, names.index(b_)]
        lo, hi = ci(d)
        sig = "yes" if (lo > 0 or hi < 0) else "no"
        print(f"  {a + ' - ' + b_:<44} {point[a] - point[b_]:>7.2f}  "
              f"[{lo:>7.2f}, {hi:>7.2f}]  {sig}")


def seed_summary(found):
    """Across-seed spread for any unit that has `<unit>-seed<N>` dumps alongside
    its canonical one, and what that implies for each comparison.

    The quantity to judge a gap against is not one unit's SD but the SD of a
    *difference* between two independently trained units, `sqrt(sa^2 + sb^2)`,
    since the two runs share no randomness. With four runs per unit the SD has
    three degrees of freedom, so treat it as an order of magnitude rather than a
    precise figure.
    """
    def rate(d):
        return sum(b for b, _ in d.values()) / sum(x for _, x in d.values())

    units = {}
    for label, data in found.items():
        if not label.startswith("neural:"):
            continue
        name = label.split(":", 1)[1]
        base, _, tail = name.rpartition("-seed")
        if tail.isdigit():
            units.setdefault(base, {})[int(tail)] = rate(data)
        elif f"neural:{name}" in found:
            units.setdefault(name, {})[0] = rate(data)
    units = {u: v for u, v in units.items() if len(v) > 1}
    if not units:
        print("\n  (no seed variants found)")
        return {}

    print(f"\n=== across-seed spread ===\n")
    print(f"  {'unit':<18} {'seeds':>28}  {'mean':>7} {'sd':>6} {'range':>7}")
    sds = {}
    for u, by_seed in sorted(units.items()):
        vals = [by_seed[k] for k in sorted(by_seed)]
        mean = sum(vals) / len(vals)
        sd = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
        sds[u] = sd
        shown = " ".join(f"{v:.2f}" for v in vals)
        print(f"  {u:<18} {shown:>28}  {mean:>7.3f} {sd:>6.3f} "
              f"{max(vals) - min(vals):>7.3f}")

    print(f"\n  gaps against the noise floor (sd of a difference = "
          f"sqrt(sa^2+sb^2)):\n")
    print(f"  {'comparison':<38} {'gap':>7} {'sd_diff':>8}  {'gap/sd':>7}  verdict")
    names = sorted(sds)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ra = sum(units[a].values()) / len(units[a])
            rb = sum(units[b].values()) / len(units[b])
            sd = (sds[a] ** 2 + sds[b] ** 2) ** 0.5
            gap = abs(ra - rb)
            ratio = gap / sd if sd else float("inf")
            verdict = "clear" if ratio >= 3 else ("marginal" if ratio >= 2 else "WITHIN NOISE")
            print(f"  {a + ' vs ' + b:<38} {gap:>7.3f} {sd:>8.3f}  {ratio:>7.1f}  {verdict}")
    return sds


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--units", nargs="+", help="Restrict to these labels (stage:name).")
    ap.add_argument("--list", action="store_true", help="List available dumps and exit.")
    ap.add_argument("--seeds", action="store_true",
                    help="Report across-seed spread and the implied noise floor.")
    args = ap.parse_args()

    found = discover()
    if args.list or not found:
        for k, v in found.items():
            print(f"  {k}  ({len(v):,} utterances)")
        if not found:
            print("  no dumps found; run llm/eval_bitrate.py and ngram/per_utterance.py")
        return
    if args.units:
        missing = [u for u in args.units if u not in found]
        if missing:
            raise SystemExit(f"Unknown: {', '.join(missing)}. Try --list.")
        found = {u: found[u] for u in args.units}

    sizes = {len(v) for v in found.values()}
    assert len(sizes) == 1, f"dumps disagree on utterance count: {sizes}"

    # Comparisons worth an interval: word against every other unit within each
    # stage, plus the same unit across stages. `ngram/` labels carry an
    # "-order3-100%" suffix that `llm/`'s do not, so the baseline is matched on
    # the stripped unit name rather than an exact label.
    def unit_of(label):
        return label.split(":", 1)[1].split("-order")[0]

    comparisons = []
    for stage in ("neural", "ngram"):
        same_stage = [n for n in found if n.startswith(stage + ":")]
        baseline = next((n for n in same_stage if unit_of(n) == "word"), None)
        if baseline:
            comparisons += [(baseline, o) for o in same_stage if o != baseline]
    for n in found:
        if n.startswith("ngram:"):
            other = f"neural:{unit_of(n)}"
            if other in found:
                comparisons.append((n, other))

    if args.seeds:
        seed_summary(found)
        return

    for by_speaker in (False, True):
        report(found, comparisons, by_speaker)


if __name__ == "__main__":
    main()
