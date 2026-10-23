"""
Training-vs-eval input check.

1. Construction: for every scheme, the (token, place-value index) sequence that
   eval feeds the model while decoding the TRUE answer must equal the training
   row built by data.collate with offset 0, token for token.
2. Decoding: with a trained checkpoint, batched greedy decoding
   (eval.generate_answers) must give the same answers as a plain one-problem-at-a-
   time reference decode.

Usage (from the repo root):
    python tokstudy/check_inputs.py [--out_dir tokstudy/out/<run>]
"""
import argparse
import os
import random

import torch

from data import collate, load_split
from eval import MAX_NEW_TOKENS, generate_answers, generated_index, prompt_index
from schemes import ALL_GRID, Scheme


def check_construction(sep_index, sep_start=False):
    rng = random.Random(0)
    probs = [(12345, 678), (923456, 111111), (0, 0), (9, 1), (99999999999999999999, 1)]
    probs += [(rng.randint(0, 10 ** rng.randint(1, 26)), rng.randint(0, 10 ** rng.randint(1, 26))) for _ in range(200)]
    n = 0
    for name in ALL_GRID:
        s = Scheme(name)
        for a, b in probs:
            x, y, idx, _ = collate(s, [(a, b)], sep_index=sep_index, sep_start=sep_start)
            ids, cidx, ans_start = s.encode(a, b)
            ev_x, ev_idx = list(ids[:ans_start]), prompt_index(cidx[:ans_start], sep_index, sep_start=sep_start)
            for step, tok in enumerate(ids[ans_start:-1]):  # teacher-force the true answer
                ev_x.append(tok)
                ev_idx.append(generated_index(tok, step, s))
            assert ev_x == x[0].tolist(), (name, a, b)
            assert ev_idx == idx[0].tolist(), (name, a, b, ev_idx, idx[0].tolist())
            # targets: exactly the answer tokens + end token, starting at the '=' position
            tgt = [t for t in y[0].tolist() if t != -1]
            assert tgt == ids[ans_start:], (name, a, b)
            assert y[0].tolist().index(tgt[0]) == ans_start - 1
            n += 1
    print(f'construction (sep_index={sep_index}, sep_start={sep_start}): {n} (scheme, problem) pairs match exactly')


@torch.no_grad()
def check_decoding(out_dir, device):
    from model import GPT, GPTConfig
    ck = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ck['model_cfg'])).to(device).eval()
    model.load_state_dict(ck['model'])
    s = Scheme(ck['run_cfg']['scheme'])
    for split in ('indist', 'ood'):
        probs = load_split(split)[:300]
        preds = generate_answers(model, s, probs, device)
        mismatches = checked = 0
        for (a, b), p in zip(probs, preds):
            ids, cidx, ans_start = s.encode(a, b)
            x, ix = list(ids[:ans_start]), prompt_index(cidx[:ans_start], getattr(model.cfg, 'sep_index', False),
                                                        sep_start=getattr(model.cfg, 'index_rope', False))
            gen = []
            for step in range(MAX_NEW_TOKENS):  # unbatched reference decode, full re-forward each step
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
                    logits, _ = model(torch.tensor([x], device=device), torch.tensor([ix], device=device))
                t = int(logits[0, -1].float().argmax())
                gen.append(t)
                if t == s.eos_id:
                    break
                x.append(t)
                ix.append(generated_index(t, step, s))
            ref = s.decode_answer(gen[:-1]) if gen[-1] == s.eos_id else None
            checked += 1
            mismatches += ref != p
        print(f'decoding [{split}]: batched greedy vs unbatched reference differ on {mismatches}/{checked} problems')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    check_construction(False)
    check_construction(True)
    check_construction(False, sep_start=True)
    if args.out_dir:
        check_decoding(args.out_dir, args.device)
