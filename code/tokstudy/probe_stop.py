"""
Teacher-forced decision probe: which single decision breaks, and is it tied to
absolute index values or to the span of a number?

The model is given the TRUE answer so far, and each next-token prediction is scored
on its own, so one early mistake can't cascade. Carry-free problems (every column
sums to <= 9), so the answer has exactly L digits and every decision is about
position, not arithmetic. Each answer step j (= place j for k1-rev) is one of:

  both   j < S        both operands have a digit at this place
  one    S <= j < L   the short operand has run out (must treat it as 0)
  stop   j == L       both have run out: the right output is the end token

plus 'early stop': predicting the end token at any j < L.

Each model is probed under several place-value index settings: offset (all
indices + offset) and stride (index = 1 + offset + stride * chunk). If a failure
stays at the same place when the offset changes, it is about the number's span; if
it moves, it is tied to absolute index values. Stride 2-3 asks whether a
stride-trained model only learned to work at the strides it saw most.

Only for k=1 schemes (answer step j = place j).

Usage (from the repo root):
    python tokstudy/probe_stop.py tokstudy/out/<run> [tokstudy/out/<run2> ...]
"""
import argparse
import os
import random
from collections import defaultdict

import torch

from analyze_errors import load_run
from data import load_split, place_index
from eval import generate_answers, score
from probe_wall import make_problem

L_LIST = [4, 8, 12, 16, 20, 21, 22, 23, 24, 26]
N_PER = 40
SETTINGS = [(0, 1), (20, 1), (50, 1), (0, 2), (0, 3)]  # (index offset, index stride)
BUCKETS = [(0, 10), (11, 20), (21, 30)]


def build_problems():
    rng = random.Random(0)
    probs = []
    for L in L_LIST:
        for S in sorted({1, max(1, L // 2), L}):
            probs += [make_problem(rng, L, S, carry_free=True) for _ in range(N_PER)]
    return probs


@torch.no_grad()
def teacher_forced(model, scheme, probs, offset, stride, device, batch_size=512):
    """Returns per-decision records (case, place, correct, early_stop)."""
    model.eval()
    recs = []
    sep = getattr(model.cfg, 'sep_index', False)
    for s in range(0, len(probs), batch_size):
        chunk = probs[s:s + batch_size]
        encs = [scheme.encode(a, b) for a, b in chunk]
        T = max(len(ids) for ids, _, _ in encs) - 1
        x = torch.full((len(chunk), T), scheme.pad_id, dtype=torch.long)
        idx = torch.zeros((len(chunk), T), dtype=torch.long)
        for r, (ids, cidx, _) in enumerate(encs):
            n = len(ids) - 1
            x[r, :n] = torch.tensor(ids[:-1])
            idx[r, :n] = torch.tensor(place_index(cidx[:-1], offset, stride, sep,
                                                  getattr(model.cfg, 'index_rope', False)))
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
            logits, _ = model(x.to(device), idx.to(device))  # right padding: causal mask suffices
        pred = logits.float().argmax(-1).cpu()
        for r, ((a, b), (ids, _, ans_start)) in enumerate(zip(chunk, encs)):
            L, S = max(len(str(a)), len(str(b))), min(len(str(a)), len(str(b)))
            for j in range(len(ids) - ans_start):  # answer steps, the last one is the end token
                t = ans_start - 1 + j               # position whose output predicts answer step j
                target, p = ids[ans_start + j], int(pred[r, t])
                case = 'stop' if target == scheme.eos_id else ('both' if j < S else 'one')
                recs.append((case, j, p == target, p == scheme.eos_id and case != 'stop'))
    model.train()
    return recs


def summarize(recs):
    acc = defaultdict(lambda: [0, 0])
    early = [0, 0]
    for case, j, ok, es in recs:
        for lo, hi in BUCKETS:
            if lo <= j <= hi:
                acc[(case, lo)][0] += ok
                acc[(case, lo)][1] += 1
        if case != 'stop':
            early[0] += es
            early[1] += 1
    cols = []
    for case in ('both', 'one', 'stop'):
        for lo, hi in BUCKETS:
            c, n = acc[(case, lo)]
            cols.append(f'{c / n:5.2f}' if n else '    .')
    return cols, early[0] / max(early[1], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    probs = build_problems()
    ood = load_split('ood')
    hdr = ''.join(f'{c + "@" + f"{lo}-{hi}":>12}' for c in ('both', 'one', 'stop') for lo, hi in BUCKETS)
    for run in args.runs:
        model, scheme = load_run(run, args.device)
        assert scheme.kind == 'chunk' and scheme.k == 1, 'probe assumes k=1'
        print(f'\n{os.path.basename(run.rstrip("/\\\\"))}  (teacher-forced accuracy per decision type @ place range)')
        print(f'{"offset,stride":>14}{hdr}  early-stop  free-run OOD')
        for off, st in SETTINGS:
            cols, early = summarize(teacher_forced(model, scheme, probs, off, st, args.device))
            free = score(ood, generate_answers(model, scheme, ood, args.device, index_offset=off, index_stride=st))['acc']
            print(f'{f"{off},{st}":>14}' + ''.join(f'{c:>12}' for c in cols) + f'{early:>11.3f}{free:>13.3f}')


if __name__ == '__main__':
    main()
