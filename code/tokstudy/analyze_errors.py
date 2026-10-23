"""
How do trained models fail on the long tiers? Compares several runs (e.g. seeds) on
the same problems.

For each run and split it reports:
  - error types: wrong answer length (digit dropped or added) vs right length with
    wrong digits, and for the latter whether it is a single carry error
    (|pred - true| = 10^p)
  - the place (counted from the units digit) of the FIRST wrong digit
  - accuracy by the shorter operand's length (both operands long vs one short)
  - accuracy by longest carry run
  - which problems each run gets wrong, and how much the runs overlap
  - example failures with the wrong places marked

Usage (from the repo root):
    python tokstudy/analyze_errors.py tokstudy/out/<run1> tokstudy/out/<run2> [--splits ood,far_ood]
"""
import argparse
import os
from collections import Counter

import torch

from data import load_split, longest_carry_run
from eval import generate_answers
from model import GPT, GPTConfig
from schemes import Scheme


def load_run(out_dir, device):
    ck = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ck['model_cfg'])).to(device)
    model.load_state_dict(ck['model'])
    return model, Scheme(ck['run_cfg']['scheme'])


def predicted_digits(scheme, toks):
    """Digit string (normal writing) from the number tokens before the first special,
    even if the whole output is malformed."""
    chunks = []
    for t in toks:
        if not scheme.is_number_id(t):
            break
        chunks.append(scheme.itos[t])
    if not chunks:
        return ''
    if scheme.kind == 'gpt2':
        s = ''.join(chunks)
        return s[::-1] if scheme.order == 'rev' else s
    return ''.join(chunks[::-1] if scheme.order == 'rev' else chunks)


def first_wrong_place(pred, true):
    """Place index from the units digit (0 = units) of the first mismatch."""
    p, t = pred[::-1], true[::-1]
    for i in range(max(len(p), len(t))):
        if i >= len(p) or i >= len(t) or p[i] != t[i]:
            return i
    return None


def classify(pred, true):
    if pred == true:
        return 'correct'
    if len(pred) != len(true):
        return f'length {len(pred) - len(true):+d}'
    d = abs(int(pred) - int(true)) if pred else None
    if d and str(d).rstrip('0') == '1':  # |pred - true| is a power of ten
        return 'single carry error'
    n_wrong = sum(a != b for a, b in zip(pred, true))
    return f'{n_wrong} wrong digit' + ('s' if n_wrong > 1 else '') if n_wrong <= 2 else '3+ wrong digits'


def mark(pred, true):
    """Pred with wrong places shown in brackets, aligned to the units digit."""
    width = max(len(pred), len(true))
    p, t = pred.rjust(width, '_'), true.rjust(width, '_')
    return ''.join(c if c == d else f'[{c}]' for c, d in zip(p, t))


def hist(counter, total, keys=None, fmt='{:.0%}'):
    keys = sorted(counter) if keys is None else keys
    return '  '.join(f'{k}:{fmt.format(counter[k] / total)}' for k in keys if counter[k])


def failure_profile(model, scheme, device, splits=('gap', 'mid', 'ood', 'far_ood', 'carry_ood')):
    """Compact per-split summary of HOW a model fails, stored with every run's results:
    error-type shares (answer too long = did not stop, too short, single carry error,
    other wrong digits) and the most common places where the first wrong digit is."""
    out = {}
    for split in splits:
        probs = load_split(split)
        trues = [str(a + b) for a, b in probs]
        _, raw = generate_answers(model, scheme, probs, device, return_tokens=True)
        preds = [predicted_digits(scheme, t) for t in raw]
        wrong = [i for i in range(len(probs)) if preds[i] != trues[i]]
        kinds = Counter()
        for i in wrong:
            k = classify(preds[i], trues[i])
            kinds['too_long' if k.startswith('length +') else 'too_short' if k.startswith('length -')
                  else 'single_carry' if k == 'single carry error' else 'wrong_digits'] += 1
        fw = Counter(first_wrong_place(preds[i], trues[i]) for i in wrong)
        out[split] = {'n': len(probs), 'n_wrong': len(wrong),
                      'error_types': {k: v / len(wrong) for k, v in kinds.items()} if wrong else {},
                      'first_wrong_place_top5': fw.most_common(5)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--splits', default='ood,far_ood,carry_ood')
    ap.add_argument('--n_examples', type=int, default=8)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    names = [os.path.basename(r.rstrip('/\\')) for r in args.runs]
    for split in args.splits.split(','):
        probs = load_split(split)
        trues = [str(a + b) for a, b in probs]
        print(f'\n{"=" * 100}\n{split}: {len(probs)} problems\n{"=" * 100}')
        wrong_sets = {}
        for name, run in zip(names, args.runs):
            model, scheme = load_run(run, args.device)
            _, raw = generate_answers(model, scheme, probs, args.device, return_tokens=True)
            preds = [predicted_digits(scheme, toks) for toks in raw]
            kinds = [classify(p, t) for p, t in zip(preds, trues)]
            wrong = [i for i, k in enumerate(kinds) if k != 'correct']
            wrong_sets[name] = set(wrong)
            n = len(probs)
            print(f'\n--- {name}: accuracy {1 - len(wrong) / n:.3f}')
            if not wrong:
                continue
            kc = Counter(kinds[i] for i in wrong)
            print('  error types (share of errors): ' + hist(kc, len(wrong), keys=[k for k, _ in kc.most_common()]))
            fw = Counter(first_wrong_place(preds[i], trues[i]) for i in wrong)
            print('  first wrong place (0 = units): ' + hist(fw, len(wrong)))
            # where errors start relative to the answer length: last place = top digit
            rel = Counter(len(trues[i]) - 1 - first_wrong_place(preds[i], trues[i]) for i in wrong)
            print('  ...counted from the TOP digit (0 = top): ' + hist(rel, len(wrong)))
            short = Counter(min(len(str(a)), len(str(b))) for a, b in probs)
            short_ok = Counter(min(len(str(probs[i][0])), len(str(probs[i][1]))) for i in range(n) if kinds[i] == 'correct')
            buckets = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 26)]
            acc_short = []
            for lo, hi in buckets:
                tot = sum(short[k] for k in range(lo, hi + 1))
                ok = sum(short_ok[k] for k in range(lo, hi + 1))
                if tot:
                    acc_short.append(f'{lo}-{hi}:{ok / tot:.2f} (n={tot})')
            print('  accuracy by SHORTER operand length: ' + '  '.join(acc_short))
            cr = Counter(min(longest_carry_run(a, b), 10) for a, b in probs)
            cr_ok = Counter(min(longest_carry_run(*probs[i]), 10) for i in range(n) if kinds[i] == 'correct')
            print('  accuracy by longest carry run: ' + '  '.join(f'{k}:{cr_ok[k] / cr[k]:.2f}' for k in sorted(cr)))
            print('  examples (a + b: true | pred, wrong places in brackets):')
            for i in wrong[:args.n_examples]:
                a, b = probs[i]
                print(f'    {len(str(a))}+{len(str(b))} digits, carry run {longest_carry_run(a, b)}: '
                      f'{trues[i]} | {mark(preds[i], trues[i])}  ({kinds[i]})')
        if len(wrong_sets) >= 2:
            sets = list(wrong_sets.values())
            both = set.intersection(*sets)
            anyw = set.union(*sets)
            print(f'\n  overlap: wrong in all runs {len(both)}, wrong in at least one {len(anyw)}; ' +
                  '  '.join(f'only {nm}: {len(s - set.union(*(o for m, o in wrong_sets.items() if m != nm)))}'
                            for nm, s in wrong_sets.items()))


if __name__ == '__main__':
    main()
