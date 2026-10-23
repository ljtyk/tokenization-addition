"""
How far does a model's length generalization actually reach? Free-run exact-match
accuracy on random-digit problems with a long operand of L digits (short operand
of 1, L/2 or L digits), for L far beyond the test tiers.

The decode budget is raised past eval.MAX_NEW_TOKENS (32, enough only for the test
tiers) and batches are kept small so long sequences fit in GPU memory.

Usage (from the repo root):
    python tokstudy/probe_length.py tokstudy/out/<run> [tokstudy/out/<run2> ...] [--lengths 26,40,60]
"""
import argparse
import os
import random

import torch

import eval as ev
from analyze_errors import load_run, predicted_digits
from probe_wall import make_problem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--lengths', default='26,34,40,50,60,70,80,100')
    ap.add_argument('--n', type=int, default=50, help='problems per (L, short-length) cell')
    ap.add_argument('--batch_size', type=int, default=48)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    lengths = [int(x) for x in args.lengths.split(',')]
    ev.MAX_NEW_TOKENS = max(lengths) + 8

    rng = random.Random(0)
    probs, cell = [], []
    for L in lengths:
        for S in sorted({1, max(1, L // 2), L}):
            for _ in range(args.n):
                probs.append(make_problem(rng, L, S, carry_free=False))
                cell.append(L)
    trues = [str(a + b) for a, b in probs]
    print(f'random-digit problems, decode budget {ev.MAX_NEW_TOKENS} tokens; accuracy by long operand length L')
    print(f'{"run":44s}' + ''.join(f'{L:>7}' for L in lengths))
    for run in args.runs:
        model, scheme = load_run(run, args.device)
        _, raw = ev.generate_answers(model, scheme, probs, args.device, batch_size=args.batch_size, return_tokens=True)
        preds = [predicted_digits(scheme, t) for t in raw]
        acc = {}
        for i, L in enumerate(cell):
            acc.setdefault(L, []).append(preds[i] == trues[i])
        print(f'{os.path.basename(run.rstrip("/")):44s}' + ''.join(f'{sum(acc[L]) / len(acc[L]):7.2f}' for L in lengths),
              flush=True)


if __name__ == '__main__':
    main()
