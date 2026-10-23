"""
Greedy exact-match evaluation over every test tier (design doc section 7).

Each answer is classified as:
  correct   - the canonical encoding of the true sum
  wrong     - the canonical encoding of some OTHER number
  malformed - not the canonical encoding of any number (bad boundary, leading
              zero, stray special token, or no end token within the budget)
Accuracy counts only "correct"; the malformed rate is reported separately.

Usage:
    python tokstudy/eval.py --out_dir tokstudy/out/<run>
"""
import argparse
import json
import os
from collections import defaultdict

import torch

from data import TEST_SPLITS, load_split, longest_carry_run, place_index
from model import rope_positions
from schemes import Scheme

MAX_NEW_TOKENS = 32  # far-OOD answers are at most 27 digits, plus the end token


def prompt_index(cidx, sep_index=False, offset=0, stride=1, sep_start=False):
    """Place-value index for prompt tokens at eval: offset 0 and stride 1 unless a
    probe asks otherwise (see data.place_index)."""
    return place_index(cidx, offset, stride, sep_index, sep_start)


def generated_index(tok, step, scheme, offset=0, stride=1):
    """Place-value index for the answer token emitted at decoding step `step`."""
    return 1 + offset + stride * step if scheme.is_number_id(tok) else 0


@torch.no_grad()
def generate_answers(model, scheme, problems, device, batch_size=512, return_tokens=False,
                     index_offset=0, index_stride=1):
    """Returns a list of (value or None) per problem; with return_tokens, also the raw
    generated token ids per problem (up to and including the end token, if any).

    Prompts of different lengths share a batch via LEFT padding: padded keys are
    masked out (model.forward key_valid), and each row's RoPE positions and
    place-value indices are shifted so its first real token sits at position 0,
    exactly as in training. index_offset / index_stride shift the place-value index
    (probes only; the reported evaluation always uses 0 / 1)."""
    model.eval()
    gen_pos = torch.Generator().manual_seed(0)  # fixed draws for rope_pos='random', so eval is repeatable
    items = []
    for i, (a, b) in enumerate(problems):
        ids, cidx, ans_start = scheme.encode(a, b)
        items.append((i, ids[:ans_start], prompt_index(cidx[:ans_start], getattr(model.cfg, 'sep_index', False),
                                                       index_offset, index_stride,
                                                       getattr(model.cfg, 'index_rope', False))))
    items.sort(key=lambda t: len(t[1]))  # similar lengths together: less padding
    out = [None] * len(problems)
    raw = [None] * len(problems)
    for s in range(0, len(items), batch_size):
        chunk = items[s:s + batch_size]
        B, P = len(chunk), max(len(ids) for _, ids, _ in chunk)
        pad = torch.tensor([P - len(ids) for _, ids, _ in chunk], device=device)
        x = torch.tensor([[scheme.pad_id] * (P - len(ids)) + ids for _, ids, _ in chunk], device=device)
        idx = torch.tensor([[0] * (P - len(ix)) + ix for _, _, ix in chunk], device=device)
        key_valid = torch.arange(P, device=device)[None] >= pad[:, None]
        # positions for each row's real tokens, then shifted right by that row's padding
        real_pos = rope_positions(model.cfg, B, P + MAX_NEW_TOKENS, False, device, gen_pos)
        cols = (torch.arange(P + MAX_NEW_TOKENS, device=device)[None] - pad[:, None]).clamp(min=0)
        pos = real_pos.gather(1, cols)
        gen = torch.empty((B, 0), dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        for step in range(MAX_NEW_TOKENS):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
                logits, _ = model(x, idx, pos=pos[:, :x.size(1)], key_valid=key_valid)
            nxt = logits[:, -1].float().argmax(-1)
            gen = torch.cat([gen, nxt[:, None]], dim=1)
            done |= nxt == scheme.eos_id
            if done.all():
                break
            nidx = torch.where(nxt >= scheme.n_special, 1 + index_offset + index_stride * step, 0)  # = generated_index
            x = torch.cat([x, nxt[:, None]], dim=1)
            idx = torch.cat([idx, nidx[:, None]], dim=1)
            key_valid = torch.cat([key_valid, torch.ones(B, 1, dtype=torch.bool, device=device)], dim=1)
        for r, (i, _, _) in enumerate(chunk):
            row = gen[r].tolist()
            if scheme.eos_id in row:
                row = row[:row.index(scheme.eos_id) + 1]
                out[i] = scheme.decode_answer(row[:-1])
            # else: no end token -> malformed (None)
            raw[i] = row
    model.train()
    return (out, raw) if return_tokens else out


def score(problems, preds):
    res = {'n': len(problems), 'correct': 0, 'malformed': 0,
           'by_len': defaultdict(lambda: [0, 0]), 'by_carry_run': defaultdict(lambda: [0, 0])}
    for (a, b), p in zip(problems, preds):
        ok = p == a + b
        res['correct'] += ok
        res['malformed'] += p is None
        L = max(len(str(a)), len(str(b)))
        cr = min(longest_carry_run(a, b), 10)
        for key, k in (('by_len', L), ('by_carry_run', cr)):
            res[key][k][0] += ok
            res[key][k][1] += 1
    n = max(res['n'], 1)
    return {
        'n': res['n'],
        'acc': res['correct'] / n,
        'malformed': res['malformed'] / n,
        'by_len': {k: c / t for k, (c, t) in sorted(res['by_len'].items())},
        'by_carry_run': {k: c / t for k, (c, t) in sorted(res['by_carry_run'].items())},
    }


def evaluate_all(model, scheme, device, splits=None, limit=None):
    results = {}
    for name in (splits or TEST_SPLITS):
        probs = load_split(name)
        if limit:
            probs = probs[:limit]
        results[name] = score(probs, generate_answers(model, scheme, probs, device))
    return results


def print_summary(results):
    print(f"{'tier':<10} {'n':>6} {'acc':>7} {'malformed':>10}")
    for name, r in results.items():
        print(f"{name:<10} {r['n']:>6} {r['acc']:>7.3f} {r['malformed']:>10.3f}")


if __name__ == '__main__':
    from model import GPT, GPTConfig
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--ckpt', default='ckpt.pt')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    ck = torch.load(os.path.join(args.out_dir, args.ckpt), map_location=args.device, weights_only=False)
    model = GPT(GPTConfig(**ck['model_cfg'])).to(args.device)
    model.load_state_dict(ck['model'])
    scheme = Scheme(ck['run_cfg']['scheme'])
    results = evaluate_all(model, scheme, args.device)
    print_summary(results)
    with open(os.path.join(args.out_dir, 'eval_results.json'), 'w') as f:
        json.dump(results, f, indent=1)
