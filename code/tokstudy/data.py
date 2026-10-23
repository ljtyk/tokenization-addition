"""
Online problem sampler and batch collation (design doc section 6).

- Bucketed lengths: d ~ U{1..max_digits}; one operand has exactly d digits, the
  other a length ~ U{1..d}, randomly assigned to a or b.
- carry_frac of problems are carry-heavy: a carry is GENERATED at a random place
  p (a column summing to >= 10) and then PROPAGATED through a run of columns that
  sum to exactly 9. Unlike nanoGPT/data/addition/generate.py (trailing 9s,
  triggered by the units digit), the chain can sit anywhere, so it crosses chunk
  boundaries at every offset.
- Problems that appear in any test file are rejected, so the test tiers stay
  held out despite fresh sampling.
"""
import os
import random

import torch

TEST_DIR = os.path.join(os.path.dirname(__file__), '..', 'nanoGPT', 'data', 'addition')
TEST_SPLITS = {
    'indist': 'test_indist.txt',             # 1-10
    'gap': 'test_gap.txt',                   # 11-13
    'mid': 'test_hint.txt',                  # 14-18 (former hint tier, now never trained)
    'ood': 'test_ood.txt',                   # 19-21
    'far_ood': 'test_far_ood.txt',           # 24-26
    'carry_gap': 'test_carry_heavy_gap.txt',  # 11-13, all carry-heavy
    'carry_ood': 'test_carry_heavy_ood.txt',  # 19-21, all carry-heavy
}


def load_split(name):
    pairs = []
    with open(os.path.join(TEST_DIR, TEST_SPLITS[name])) as f:
        for line in f:
            line = line.strip()
            if line:
                lhs = line.split('=')[0]
                a, b = lhs.split('+')
                pairs.append((int(a), int(b)))
    return pairs


def load_test_pairs():
    held_out = set()
    for name in TEST_SPLITS:
        held_out.update(load_split(name))
    return held_out


def _rand_digits(rng, n):
    """n digits, least-significant first, with a nonzero leading digit (n=1 allows 0)."""
    if n == 1:
        return [rng.randint(0, 9)]
    return [rng.randint(0, 9) for _ in range(n - 1)] + [rng.randint(1, 9)]


def _from_digits(ds):
    return int(''.join(map(str, reversed(ds))))


def _lengths(rng, max_digits):
    d = rng.randint(1, max_digits)
    other = rng.randint(1, d)
    return (d, other) if rng.random() < 0.5 else (other, d)


def sample_natural(rng, max_digits):
    la, lb = _lengths(rng, max_digits)
    return _from_digits(_rand_digits(rng, la)), _from_digits(_rand_digits(rng, lb))


def sample_carry_heavy(rng, max_digits):
    """Carry generated at place p < short_len, then propagated through c columns
    summing to 9 (places p+1..p+c). c ~ U{0..long_len-1-p}; c reaching the top place
    forces an overflow into an extra answer digit."""
    while True:
        la, lb = _lengths(rng, max_digits)
        L, S = max(la, lb), min(la, lb)
        long_d, short_d = _rand_digits(rng, L), _rand_digits(rng, S)
        p = rng.randint(0, S - 1)
        x = rng.randint(1, 9)                          # long digit at p
        long_d[p], short_d[p] = x, rng.randint(10 - x, 9)  # sum >= 10: generates a carry
        c = rng.randint(0, L - 1 - p)
        for i in range(p + 1, p + 1 + c):              # propagate: column sum == 9
            if i < S:
                short_d[i] = rng.randint(0, 9)
                long_d[i] = 9 - short_d[i]
            else:
                long_d[i] = 9
        if (L > 1 and long_d[-1] == 0) or (S > 1 and short_d[-1] == 0):
            continue  # a propagate column produced a leading zero; resample
        long_v, short_v = _from_digits(long_d), _from_digits(short_d)
        return (long_v, short_v) if la == L else (short_v, long_v)


class ProblemSampler:
    def __init__(self, seed, max_digits=10, carry_frac=0.3, held_out=None, long_frac=0.0, long_min=20, long_max=40):
        self.rng = random.Random(seed)
        self.max_digits = max_digits
        self.carry_frac = carry_frac
        self.held_out = load_test_pairs() if held_out is None else held_out
        # optional long problems (side experiment, chat 2026-10-19): with probability long_frac the
        # long operand has long_min..long_max digits, the short one 1..that many, random digits
        self.long_frac, self.long_min, self.long_max = long_frac, long_min, long_max

    def sample(self):
        if self.long_frac and self.rng.random() < self.long_frac:
            L = self.rng.randint(self.long_min, self.long_max)
            S = self.rng.randint(1, L)
            a, b = _from_digits(_rand_digits(self.rng, L)), _from_digits(_rand_digits(self.rng, S))
            return (a, b) if self.rng.random() < 0.5 else (b, a)
        while True:
            if self.rng.random() < self.carry_frac:
                ab = sample_carry_heavy(self.rng, self.max_digits)
            else:
                ab = sample_natural(self.rng, self.max_digits)
            if ab not in self.held_out:
                return ab

    def batch(self, n):
        return [self.sample() for _ in range(n)]


def place_index(cidx, offset=0, stride=1, sep_index=False, sep_start=False):
    """Place-value index inputs for one sequence, from Scheme.encode's chunk_idx.

    0 = no index. Chunk j of a number gets 1 + offset + stride * j. With sep_index,
    the separator right after a number ('+' or '=') gets the index one past that
    number's last chunk, 1 + offset + stride * n_chunks, so where each operand ends
    is explicit instead of having to be inferred from a failed index lookup.
    With sep_start (index-RoPE), every separator gets offset + 1 - stride: "one step
    before the next number's first chunk", so '=' looks for operand chunk 0 at index
    difference +1 exactly like every answer token does. It carries no length."""
    out, prev = [], -1
    for j in cidx:
        if j >= 0:
            out.append(1 + offset + stride * j)
        elif sep_start:
            out.append(1 + offset - stride)
        elif sep_index and prev >= 0:
            out.append(1 + offset + stride * (prev + 1))
        else:
            out.append(0)
        prev = j
    return out


def collate(scheme, problems, offset_rng=None, max_offset=0, stride_max=1, sep_index=False, sep_start=False):
    """Pad a list of problems to one (B, T) batch.

    Returns x, y, idx, n_tokens, where y is -1 everywhere except answer tokens and
    the end token (loss only on the answer, design decision 1), and idx is the
    place-value index input (see place_index). The offset and the stride are ONE
    random draw per problem, shared by both operands and the answer, so operand
    chunk i and answer chunk i keep the same index."""
    encs = [scheme.encode(a, b) for a, b in problems]
    T = max(len(ids) for ids, _, _ in encs) - 1
    B = len(encs)
    x = torch.full((B, T), scheme.pad_id, dtype=torch.long)
    y = torch.full((B, T), -1, dtype=torch.long)
    idx = torch.zeros((B, T), dtype=torch.long)
    n_tokens = 0
    for r, (ids, cidx, ans_start) in enumerate(encs):
        off = offset_rng.randint(0, max_offset) if (offset_rng and max_offset) else 0
        stride = offset_rng.randint(1, stride_max) if (offset_rng and stride_max > 1) else 1
        n = len(ids) - 1
        n_tokens += n
        x[r, :n] = torch.tensor(ids[:-1])
        y[r, ans_start - 1:n] = torch.tensor(ids[ans_start:])
        idx[r, :n] = torch.tensor(place_index(cidx[:-1], off, stride, sep_index, sep_start))
    return x, y, idx, n_tokens


def longest_carry_run(a, b):
    best = run = carry = 0
    while a or b:
        s = a % 10 + b % 10 + carry
        carry = int(s >= 10)
        run = run + 1 if carry else 0
        best = max(best, run)
        a //= 10
        b //= 10
    return best


if __name__ == '__main__':
    from collections import Counter
    s = ProblemSampler(0)
    probs = s.batch(20000)
    runs = Counter(min(longest_carry_run(a, b), 10) for a, b in probs)
    print('longest-carry-run histogram (10 = 10+):', sorted(runs.items()))
    print('max operand len histogram:', sorted(Counter(max(len(str(a)), len(str(b))) for a, b in probs).items()))
    rng = random.Random(1)
    print('carry-heavy examples:', [f'{a}+{b}={a + b}' for a, b in (sample_carry_heavy(rng, 10) for _ in range(6))])
