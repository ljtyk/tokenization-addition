"""
Train one cell of the tokenization grid.

Logs both fairness axes (design doc section 5): cumulative problems seen and
cumulative training FLOPs. Every eval_interval it runs a quick greedy in-dist
check, used to find the plateau in the stage-1 budget pilot.

Usage (from the repo root):
    python -u tokstudy/train.py --scheme k1-rev --pos pv --seed 0 --max_iters 2000
"""
import argparse
import copy
import json
import math
import os
import random
import time
from dataclasses import asdict

import torch

from analyze_errors import failure_profile
from data import ProblemSampler, collate, load_split
from eval import evaluate_all, generate_answers, print_summary, score
from model import GPT, GPTConfig, index_rope_scale, rope_positions
from schemes import Scheme

MAX_CHUNKS = 64  # chunk positions per number covered by the index table (far-OOD k=1 needs 27)


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scheme', required=True)
    ap.add_argument('--pos', choices=['rope', 'pv', 'nope', 'nope_pv', 'irope'], required=True,
                    help='rope = RoPE only; pv = RoPE + place-value index table; nope / nope_pv = the same without '
                         'RoPE; irope = half the RoPE pairs driven by sequence position, half by place-value index')
    ap.add_argument('--index_rope_scale_max', type=float, default=4.0,
                    help='irope: index rotation stretched by s ~ U[1, max] per sequence in training (1 = off)')
    ap.add_argument('--rope_pos', choices=['plain', 'scale', 'random'], default='plain',
                    help="RoPE positions: plain 0,1,2..; scale = one random stretch s~U[1,rope_scale_max] per "
                         "sequence (= scaling every angle); random = sorted random positions (Ruoss et al.)")
    ap.add_argument('--rope_scale_max', type=float, default=4.0)
    ap.add_argument('--rope_rand_range', type=int, default=256)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_layer', type=int, default=6)
    ap.add_argument('--n_head', type=int, default=6)
    ap.add_argument('--n_embd', type=int, default=384)
    ap.add_argument('--batch_size', type=int, default=256, help='problems per step')
    ap.add_argument('--max_iters', type=int, default=2000)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--min_lr', type=float, default=1e-4)
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--weight_decay', type=float, default=0.1)
    ap.add_argument('--max_offset', type=int, default=100, help='random place-value offset range (training only)')
    ap.add_argument('--sep_index', action='store_true',
                    help="separators '+' / '=' carry the index one past their number's last chunk")
    ap.add_argument('--index_stride_max', type=int, default=1,
                    help='training only: place-value index stride s ~ U{1..max} per problem (1 = off)')
    ap.add_argument('--max_digits', type=int, default=10)
    ap.add_argument('--carry_frac', type=float, default=0.3)
    ap.add_argument('--long_frac', type=float, default=0.0,
                    help='fraction of training problems drawn long (--long_min..--long_max digits); 0 = off')
    ap.add_argument('--long_min', type=int, default=20)
    ap.add_argument('--long_max', type=int, default=40)
    ap.add_argument('--eval_interval', type=int, default=250)
    ap.add_argument('--eval_n', type=int, default=300, help='problems per split in the periodic check')
    ap.add_argument('--check_splits', default='indist,ood,far_ood', help='test splits in the periodic check')
    ap.add_argument('--log_interval', type=int, default=50)
    ap.add_argument('--ema_decay', type=float, default=0.999, help='weight EMA decay; 0 disables EMA')
    ap.add_argument('--check_raw', action='store_true',
                    help='also run the periodic check on the raw (non-EMA) weights; off by default to save time')
    ap.add_argument('--late_frac', type=float, default=0.25,
                    help='periodic checks in the last late_frac of training are averaged into late_avg')
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--init_from', default=None,
                    help='warm start: run dir of a finished run with the SAME model config. Raw and EMA weights, '
                         'history and the problems/FLOPs counters carry over; training continues from its last '
                         'iteration to --max_iters (total) with a fresh warmup + cosine over the continuation, a '
                         're-initialized optimizer (its state is not saved), and new data seeds')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return ap.parse_args()


def lr_at(it, a, it0=0):
    """Linear warmup then cosine to min_lr, over iterations it0..max_iters (it0 > 0 for a
    warm start: the schedule restarts for the continuation)."""
    k = it - it0
    if k < a.warmup:
        return a.lr * (k + 1) / a.warmup
    t = (k - a.warmup) / max(1, a.max_iters - it0 - a.warmup)
    return a.min_lr + 0.5 * (a.lr - a.min_lr) * (1 + math.cos(math.pi * min(t, 1.0)))


def main():
    a = get_args()
    pos_tag = a.pos if a.rope_pos == 'plain' else f'{a.pos}-{a.rope_pos}'
    pos_tag += '-sep' if a.sep_index else ''
    pos_tag += f'-st{a.index_stride_max}' if a.index_stride_max > 1 else ''
    pos_tag += f'-is{a.index_rope_scale_max:g}' if a.pos == 'irope' else ''
    run_name = f'{a.scheme}_{pos_tag}_L{a.n_layer}_s{a.seed}'
    out_dir = a.out_dir or os.path.join(os.path.dirname(__file__), 'out', run_name)
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    scheme = Scheme(a.scheme)
    cfg = GPTConfig(vocab_size=scheme.vocab_size, n_layer=a.n_layer, n_head=a.n_head, n_embd=a.n_embd,
                    rope=a.pos in ('rope', 'pv', 'irope'), rope_pos=a.rope_pos, rope_scale_max=a.rope_scale_max,
                    rope_rand_range=a.rope_rand_range, place_value=a.pos in ('pv', 'nope_pv'),
                    n_index=1 + a.max_offset + a.index_stride_max * (MAX_CHUNKS + 1),
                    sep_index=a.sep_index, index_stride_max=a.index_stride_max,
                    index_rope=a.pos == 'irope', index_rope_scale_max=a.index_rope_scale_max)
    model = GPT(cfg).to(a.device)
    N = model.num_params()
    print(f'{run_name}: vocab {scheme.vocab_size}, {N / 1e6:.2f}M non-embedding params, '
          f'{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M total', flush=True)

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and not n.startswith('wie')]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2 or n.startswith('wie')]
    # EMA of the weights: length generalization swings between checkpoints (budget pilot,
    # design doc section 5), so the reported model is a smoothed average of recent weights
    ema = copy.deepcopy(model).eval() if a.ema_decay > 0 else None
    if ema is not None:
        for p in ema.parameters():
            p.requires_grad_(False)
    models = {'raw': model, **({'ema': ema} if ema is not None else {})}
    check_models = models if (a.check_raw or ema is None) else {'ema': ema}

    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': a.weight_decay},
                             {'params': no_decay, 'weight_decay': 0.0}],
                            lr=a.lr, betas=(0.9, 0.99), fused=(a.device == 'cuda'))

    it0, problems_seen, tokens_seen, flops, history = 0, 0, 0, 0.0, []
    if a.init_from:
        ck = torch.load(os.path.join(a.init_from, 'ckpt.pt'), map_location=a.device, weights_only=False)
        assert ck['model_cfg'] == asdict(cfg), f'model config differs from {a.init_from}'
        model.load_state_dict(ck['model_raw'] if 'model_raw' in ck else ck['model'])
        if ema is not None:
            ema.load_state_dict(ck['model'])
        history = ck['history']
        last = history[-1]
        it0, problems_seen, tokens_seen, flops = last['iter'], last['problems'], last['tokens'], last['flops']
        assert it0 < a.max_iters, f'{a.init_from} already reached iter {it0}'
        print(f'warm start from {a.init_from} at iter {it0} ({problems_seen} problems seen)', flush=True)
    # data seeds differ per segment, so a continuation does not replay the first segment's problems
    dseed = a.seed if it0 == 0 else a.seed * 1_000_003 + it0
    sampler = ProblemSampler(seed=dseed, max_digits=a.max_digits, carry_frac=a.carry_frac,
                             long_frac=a.long_frac, long_min=a.long_min, long_max=a.long_max)
    offset_rng = random.Random(dseed + 10_000)
    pos_gen = torch.Generator().manual_seed(dseed + 20_000)
    # fixed random sample per split (the files are ordered by length, so a prefix would be one length only)
    check_sets = {}
    for name in a.check_splits.split(','):
        pairs = load_split(name)
        check_sets[name] = random.Random(0).sample(pairs, min(a.eval_n, len(pairs)))

    t0 = time.time()
    for it in range(it0, a.max_iters + 1):
        if (it % a.eval_interval == 0 or it == a.max_iters) and not (it == it0 and it0 > 0):  # it0 is in history
            rec = dict(iter=it, problems=problems_seen, tokens=tokens_seen, flops=flops)
            msg = []
            for tag, m in check_models.items():
                msg.append(f'| {tag}')
                for name, probs in check_sets.items():
                    r = score(probs, generate_answers(m, scheme, probs, a.device))
                    rec[f'{tag}_{name}_acc'], rec[f'{tag}_{name}_malformed'] = r['acc'], r['malformed']
                    msg.append(f'{name} {r["acc"]:.3f}')
            history.append(rec)
            print(f'[eval] iter {it} problems {problems_seen} flops {flops:.3e} ' + ' '.join(msg), flush=True)
        if it == a.max_iters:
            break

        for g in opt.param_groups:
            g['lr'] = lr_at(it, a, it0)
        probs = sampler.batch(a.batch_size)
        x, y, idx, n_tok = collate(scheme, probs, offset_rng,
                                   a.max_offset if (cfg.place_value or cfg.index_rope) else 0,
                                   a.index_stride_max, a.sep_index, sep_start=cfg.index_rope)
        x, y, idx = x.to(a.device), y.to(a.device), idx.to(a.device)
        pos = rope_positions(cfg, x.size(0), x.size(1), True, a.device, pos_gen)
        isc = index_rope_scale(cfg, x.size(0), True, a.device, pos_gen) if cfg.index_rope else None
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(a.device == 'cuda')):
            _, loss = model(x, idx, y, pos=pos, idx_scale=isc)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if ema is not None:
            with torch.no_grad():
                ema_p, raw_p = list(ema.parameters()), list(model.parameters())
                torch._foreach_lerp_(ema_p, raw_p, 1.0 - a.ema_decay)

        # FLOPs over real (non-pad) tokens: 6N per token for the dense layers, plus causal
        # attention 6 * n_layer * n_embd * len^2 per sequence (fwd + bwd)
        lens = (x != scheme.pad_id).sum(1).float()
        flops += 6 * N * n_tok + 6 * a.n_layer * a.n_embd * float((lens ** 2).sum())
        problems_seen += len(probs)
        tokens_seen += n_tok
        if it % a.log_interval == 0:
            print(f'iter {it} loss {loss.item():.4f} lr {lr_at(it, a, it0):.2e} '
                  f'{(time.time() - t0) / (it - it0 + 1) * 1000:.0f}ms/it', flush=True)

    # late_avg: mean of each periodic-check metric over the last late_frac of training
    late = [h for h in history if h['iter'] > 0 and h['iter'] >= (1 - a.late_frac) * a.max_iters]
    late_avg = {k: sum(h[k] for h in late) / len(late) for k in late[0] if k.endswith('_acc')} if late else {}
    print(f'late_avg over {len(late)} checks: ' + ' '.join(f'{k} {v:.3f}' for k, v in late_avg.items()), flush=True)

    # 'model' is what eval.py loads: the EMA weights when EMA is on
    final_model = ema if ema is not None else model
    torch.save({'model': final_model.state_dict(), 'model_raw': model.state_dict(), 'model_cfg': asdict(cfg),
                'run_cfg': vars(a), 'history': history}, os.path.join(out_dir, 'ckpt.pt'))
    finals = {tag: evaluate_all(m, scheme, a.device) for tag, m in models.items()}
    for tag, res in finals.items():
        print(f'final ({tag}):')
        print_summary(res)
    profile = failure_profile(final_model, scheme, a.device)
    print('failure profile: ' + '  '.join(
        f'{s}: ' + ','.join(f'{k} {v:.0%}' for k, v in p['error_types'].items()) for s, p in profile.items()), flush=True)
    with open(os.path.join(out_dir, 'eval_results.json'), 'w') as f:
        json.dump({'run_cfg': vars(a), 'history': history, 'late_avg': late_avg,
                   'final': finals.get('ema', finals['raw']), 'final_raw': finals['raw'],
                   'failure_profile': profile, 'train_seconds': time.time() - t0}, f, indent=1)
    print(f'done in {time.time() - t0:.0f}s -> {out_dir}', flush=True)


if __name__ == '__main__':
    main()
