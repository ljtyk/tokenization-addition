"""
Plain fixed-depth GPT decoder with RoPE, plus an optional place-value
(chunk-index) embedding (design doc sections 1 and 3).

The block and RoPE code follow nanoGPT/model.py, without the loop-research
machinery. Sequences are right-padded, so causal attention alone keeps real
tokens from ever attending to padding.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256          # RoPE table length; far-OOD sequences are < 100 tokens
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    rope: bool = True              # False = NoPE (causal mask is the only positional signal)
    rope_pos: str = 'plain'        # RoPE positions in TRAINING: 'plain' = 0,1,2,...; 'scale' = s*(0,1,2,...),
                                   # one random s per sequence; 'random' = sorted random positions (Ruoss et al. 2023)
    rope_scale_max: float = 4.0    # 'scale': s ~ U[1, rope_scale_max] in training, s = 1 at eval
    rope_rand_range: int = 256     # 'random': positions drawn from 0..rope_rand_range-1, in training AND eval
    place_value: bool = False      # add the chunk-index embedding
    n_index: int = 0               # size of the chunk-index table (slot 0 = no index)
    sep_index: bool = False        # separators '+' / '=' get the index one past their number's last chunk
    index_stride_max: int = 1      # training only: index stride s ~ U{1..index_stride_max} per problem
    index_rope: bool = False       # place-value index drives RoPE on half of each head's dim pairs (the other
                                   # half keeps sequence-position RoPE); replaces the additive index table
    index_rope_scale_max: float = 4.0  # training only: index rotation stretched by s ~ U[1, max] per sequence


def _rope_inv_freq(head_dim, base=10000.0):
    return 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))


def _rope_cos_sin(pos, inv_freq, idx_pos=None, inv_freq_idx=None):
    """pos: (B, T) float positions (non-integer allowed). Returns cos, sin of shape
    (B, 1, T, head_dim), broadcasting over heads.

    With idx_pos (index-RoPE), the first dim pairs rotate by sequence position and the
    rest by place-value index, so q.k depends on the position DIFFERENCE through the
    first group and on the index DIFFERENCE through the second."""
    freqs = pos[..., None] * inv_freq            # angle = position * speed, per dim pair
    if idx_pos is not None:
        freqs = torch.cat([freqs, idx_pos[..., None] * inv_freq_idx], dim=-1)
    emb = torch.cat([freqs, freqs], dim=-1)[:, None]
    return emb.cos(), emb.sin()


def index_rope_scale(cfg, B, training, device, gen=None):
    """(B, 1) stretch of the index rotation: s ~ U[1, index_rope_scale_max] per
    sequence in training (so large index differences are seen), 1 at eval."""
    if training and cfg.index_rope_scale_max > 1:
        s = 1.0 + (cfg.index_rope_scale_max - 1.0) * torch.rand(B, 1, generator=gen)
    else:
        s = torch.ones(B, 1)
    return s.to(device)


def rope_positions(cfg, B, T, training, device, gen=None):
    """(B, T) RoPE positions for one batch, per cfg.rope_pos.

    'scale' stretches every position of a row by one s, which is identical to scaling
    every RoPE angle by s. 'random' draws T distinct positions from the larger range
    and sorts them, so order is kept but gaps are irregular; it is used at eval too,
    since the model never learned what an evenly spaced sequence means."""
    ar = torch.arange(T, dtype=torch.float)
    if cfg.rope_pos == 'plain' or (cfg.rope_pos == 'scale' and not training):
        pos = ar[None].expand(B, T)
    elif cfg.rope_pos == 'scale':
        s = 1.0 + (cfg.rope_scale_max - 1.0) * torch.rand(B, 1, generator=gen)
        pos = s * ar[None]
    elif cfg.rope_pos == 'random':
        assert T <= cfg.rope_rand_range
        pos = torch.stack([torch.randperm(cfg.rope_rand_range, generator=gen)[:T].sort().values
                           for _ in range(B)]).float()
    else:
        raise ValueError(cfg.rope_pos)
    return pos.to(device)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, rope=None, attn_mask=None):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        if rope is not None:
            cos, sin = rope
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=attn_mask is None,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=False)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=False)
        self.mlp = MLP(cfg)

    def forward(self, x, rope=None, attn_mask=None):
        x = x + self.attn(self.ln_1(x), rope, attn_mask)
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wie = nn.Embedding(cfg.n_index, cfg.n_embd, padding_idx=0) if cfg.place_value else None
        self.drop = nn.Dropout(cfg.dropout)
        self.h = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=False)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight  # weight tying, as in nanoGPT
        head_dim = cfg.n_embd // cfg.n_head
        if cfg.index_rope:
            assert cfg.rope and not cfg.place_value, 'index_rope needs sequence RoPE and replaces the index table'
            n_seq = head_dim // 4                  # half of the head_dim/2 rotation pairs
            n_idx = head_dim // 2 - n_seq
            self.register_buffer('inv_freq', 1.0 / (10000.0 ** (torch.arange(n_seq).float() / n_seq)), persistent=False)
            self.register_buffer('inv_freq_idx', 1.0 / (10000.0 ** (torch.arange(n_idx).float() / n_idx)), persistent=False)
        else:
            self.register_buffer('inv_freq', _rope_inv_freq(head_dim), persistent=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))
        if self.wie is not None:
            with torch.no_grad():
                self.wie.weight[0].zero_()

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wte.weight.numel()
            if self.wie is not None:
                n -= self.wie.weight.numel()
        return n

    def forward(self, x, idx=None, targets=None, pos=None, key_valid=None, idx_scale=None):
        """pos: optional (B, T) RoPE positions; default 0..T-1 for every row.
        idx: (B, T) place-value indices (additive table, or index-RoPE positions).
        idx_scale: optional (B, 1) stretch of the index rotation (index-RoPE training).
        key_valid: optional (B, T) bool, False at LEFT padding (used by batched eval).
        Real tokens never attend to padding; each pad attends only to itself, so its
        hidden state stays finite and its exactly-zero attention weight cannot leak NaN."""
        attn_mask = None
        if key_valid is not None:
            T = x.size(1)
            causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            eye = torch.eye(T, dtype=torch.bool, device=x.device)
            attn_mask = ((causal & key_valid[:, None, :]) | eye)[:, None]  # (B, 1, T, T)
        rope = None
        if self.cfg.rope:
            if pos is None:
                pos = torch.arange(x.size(1), device=x.device, dtype=torch.float)[None].expand(x.size(0), -1)
            if self.cfg.index_rope:
                ipos = idx.float() * (1.0 if idx_scale is None else idx_scale)
                rope = _rope_cos_sin(pos.float(), self.inv_freq, ipos, self.inv_freq_idx)
            else:
                rope = _rope_cos_sin(pos.float(), self.inv_freq)
        h = self.wte(x)
        if self.wie is not None:
            h = h + self.wie(idx)
        h = self.drop(h)
        for block in self.h:
            h = block(h, rope, attn_mask)
        logits = self.lm_head(self.ln_f(h))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss
