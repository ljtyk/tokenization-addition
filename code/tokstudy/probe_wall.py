"""
What does the "position wall" depend on? (see analyze_errors.py: seed 0 of the
k1-rev scaled-RoPE + place-value EMA runs breaks around answer place 21-22)

Controlled synthetic problems, no training. Each problem has a LONG operand of L
digits and a SHORT operand of S digits (randomly assigned to a or b). Three
candidate explanations make different predictions about where errors start:

  absolute place   errors start at place ~21-22 whatever L and S are
  answer end       errors start at the top of the answer (place L-1 or L),
                   i.e. the model cannot tell when to stop, once L is large
  short-operand    errors start at place S, where the short operand runs out
  end              and the model must notice "no digit here, treat it as 0"

Two variants:
  carry-free   every column sums to <= 9: no carries at all, answer length = L,
               so any error is about position, not arithmetic
  random       uniformly random digits (carries included)

Usage (from the repo root):
    python tokstudy/probe_wall.py tokstudy/out/<run> [tokstudy/out/<run2> ...]
"""
import argparse
import os
import random
from collections import Counter

import torch

from analyze_errors import first_wrong_place, load_run, predicted_digits
from eval import generate_answers

L_LIST = [12, 16, 20, 21, 22, 23, 24, 26]
S_LIST = [1, 4, 8, 12, 16, 20, 21, 22, 24, 26]
N_PER_CELL = 100


def make_problem(rng, L, S, carry_free):
    long_d = [rng.randint(0, 9) for _ in range(L - 1)] + [rng.randint(1, 9)]
    short_d = [rng.randint(0, 9) for _ in range(S - 1)] + [rng.randint(1, 9)] if S > 1 else [rng.randint(1, 9)]
    if carry_free:
        for i in range(S):
            lo = 1 if i == L - 1 else 0                       # long operand's top digit stays nonzero
            short_lo = 1 if i == S - 1 else 0                 # short operand's top digit stays nonzero
            short_d[i] = rng.randint(short_lo, 9 - lo)
            long_d[i] = rng.randint(lo, 9 - short_d[i])
    a = int(''.join(map(str, reversed(long_d))))
    b = int(''.join(map(str, reversed(short_d))))
    return (a, b) if rng.random() < 0.5 else (b, a)


def classify_start(fw, L, S):
    if fw is None:
        return 'correct'
    if S < L and fw == S:
        return 'at short end (place S)'
    if fw >= L - 1:
        return 'at answer top (place >= L-1)'
    if 21 <= fw <= 23:
        return 'at place 21-23'
    return 'elsewhere'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    cells = [(L, S) for L in L_LIST for S in S_LIST if S <= L]
    for variant in ('carry-free', 'random'):
        rng = random.Random(0)
        probs, cell_of = [], []
        for L, S in cells:
            for _ in range(N_PER_CELL):
                probs.append(make_problem(rng, L, S, variant == 'carry-free'))
                cell_of.append((L, S))
        trues = [str(a + b) for a, b in probs]
        for run in args.runs:
            name = os.path.basename(run.rstrip('/\\'))
            model, scheme = load_run(run, args.device)
            _, raw = generate_answers(model, scheme, probs, args.device, return_tokens=True)
            preds = [predicted_digits(scheme, t) for t in raw]
            print(f'\n{"=" * 90}\n{name} | {variant} problems | accuracy by long length L (rows) x short length S (cols)')
            print('L\\S  ' + ''.join(f'{S:>6}' for S in S_LIST))
            acc, starts, fw_by_L = {}, Counter(), {}
            for i, (L, S) in enumerate(cell_of):
                ok = preds[i] == trues[i]
                acc.setdefault((L, S), []).append(ok)
                if not ok:
                    fw = first_wrong_place(preds[i], trues[i])
                    starts[classify_start(fw, L, S)] += 1
                    fw_by_L.setdefault(L, Counter())[fw] += 1
            for L in L_LIST:
                row = ''.join(f'{sum(acc[(L, S)]) / len(acc[(L, S)]):>6.2f}' if (L, S) in acc else '     .'
                              for S in S_LIST)
                print(f'{L:>4} {row}')
            n_err = sum(starts.values())
            if n_err:
                print(f'where errors START ({n_err} errors): ' +
                      '  '.join(f'{k}: {v / n_err:.0%}' for k, v in starts.most_common()))
                print('first wrong place, per long length L (most common 4):')
                for L in L_LIST:
                    if L in fw_by_L:
                        top = ', '.join(f'place {p}: {c}' for p, c in fw_by_L[L].most_common(4))
                        print(f'   L={L:>2}: {top}')


if __name__ == '__main__':
    main()
