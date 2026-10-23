"""
Tokenization schemes for the addition study (see docs/tokenization-study-design.md).

A scheme turns a problem (a, b) into the token sequence

    <a chunks> + <b chunks> = <answer chunks> \\n

Operands and answer share one cut and one emission order (design decision "(a)").

Scheme names:
    k1-fwd, k1-rev                      single digits (cut is irrelevant)
    k{2,3}-{left,right}-{fwd,rev}       fixed-size chunks
    gpt2-fwd, gpt2-rev                  stock GPT-2 BPE segmentation, remapped to a
                                        compact vocab (gpt2-rev = GPT-2 applied to the
                                        digit-reversed string, as emitted)

Inside a fixed-size chunk the digits are always written normally ("[023]"), so a
reversed scheme reverses the ORDER of chunks, not the digits within them. The
GPT-2 scheme is the exception: it tokenizes whatever string is emitted, so its
reversed tokens carry reversed digits.
"""
import re

PAD, PLUS, EQ, EOS = '<pad>', '+', '=', '\n'
SPECIALS = [PAD, PLUS, EQ, EOS]

_gpt2_enc = None


def _gpt2():
    global _gpt2_enc
    if _gpt2_enc is None:
        import tiktoken
        _gpt2_enc = tiktoken.get_encoding('gpt2')
    return _gpt2_enc


def gpt2_digit_tokens():
    """Every pure-digit token in the GPT-2 vocab (994 of them). Mapping all of them
    up front means no test number can ever hit an out-of-vocab token."""
    enc = _gpt2()
    out = []
    for i in range(enc.n_vocab):
        s = enc.decode_single_token_bytes(i)
        if re.fullmatch(rb'[0-9]+', s):
            out.append(s.decode())
    return sorted(out, key=lambda s: (len(s), s))


def _all_digit_strings(max_len):
    out = []
    for n in range(1, max_len + 1):
        out += [str(i).zfill(n) for i in range(10 ** n)]
    return out


class Scheme:
    def __init__(self, name):
        self.name = name
        parts = name.split('-')
        if parts[0] == 'gpt2':
            assert len(parts) == 2, name
            self.kind, self.k, self.cut, self.order = 'gpt2', None, None, parts[1]
            number_tokens = gpt2_digit_tokens()
        else:
            self.kind = 'chunk'
            self.k = int(parts[0][1:])
            if self.k == 1:
                assert len(parts) == 2, name
                self.cut, self.order = 'right', parts[1]
            else:
                assert len(parts) == 3, name
                self.cut, self.order = parts[1], parts[2]
            assert self.cut in ('left', 'right'), name
            number_tokens = _all_digit_strings(self.k)
        assert self.order in ('fwd', 'rev'), name

        self.itos = SPECIALS + number_tokens
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.vocab_size = len(self.itos)
        self.pad_id, self.plus_id, self.eq_id, self.eos_id = (self.stoi[s] for s in SPECIALS)
        self.n_special = len(SPECIALS)

    def is_number_id(self, i):
        return i >= self.n_special

    # -- number <-> chunk strings ------------------------------------------------

    def chunks(self, n):
        """Token strings for integer n, in emission order."""
        s = str(n)
        if self.kind == 'gpt2':
            if self.order == 'rev':
                s = s[::-1]
            enc = _gpt2()
            return [enc.decode([t]) for t in enc.encode(s)]
        k = self.k
        if self.cut == 'right':
            head = len(s) % k
            out = ([s[:head]] if head else []) + [s[i:i + k] for i in range(head, len(s), k)]
        else:
            out = [s[i:i + k] for i in range(0, len(s), k)]
        return out[::-1] if self.order == 'rev' else out

    def parse(self, chunk_strs):
        """Inverse of chunks(). Returns the integer, or None if the chunk sequence is
        not the canonical encoding of any number (e.g. a non-canonical boundary or a
        leading zero), so malformed outputs are told apart from wrong values."""
        if not chunk_strs:
            return None
        if self.kind == 'gpt2':
            s = ''.join(chunk_strs)
            if self.order == 'rev':
                s = s[::-1]
        else:
            ordered = chunk_strs[::-1] if self.order == 'rev' else chunk_strs
            s = ''.join(ordered)
        if len(s) > 1 and s[0] == '0':
            return None
        n = int(s)
        return n if self.chunks(n) == list(chunk_strs) else None

    # -- full problems -------------------------------------------------------------

    def encode(self, a, b):
        """Returns (ids, chunk_idx, ans_start).

        chunk_idx[t] is the 0-based position of token t within its own number, in
        emission order, or -1 for +, =, and the end token (the place-value embedding
        adds the offset and shifts so -1 maps to the reserved "no index" slot).
        ans_start is the index of the first answer token in ids."""
        ids, idx = [], []
        for n, sep in ((a, self.plus_id), (b, self.eq_id)):
            for j, c in enumerate(self.chunks(n)):
                ids.append(self.stoi[c])
                idx.append(j)
            ids.append(sep)
            idx.append(-1)
        ans_start = len(ids)
        for j, c in enumerate(self.chunks(a + b)):
            ids.append(self.stoi[c])
            idx.append(j)
        ids.append(self.eos_id)
        idx.append(-1)
        return ids, idx, ans_start

    def decode_answer(self, ids):
        """ids = generated answer tokens up to (not including) the end token.
        Returns the integer, or None if malformed."""
        if any(not self.is_number_id(i) for i in ids):
            return None
        return self.parse([self.itos[i] for i in ids])

    def render(self, ids):
        return ''.join(f'[{self.itos[i]}]' if self.is_number_id(i) else
                       (' ' + self.itos[i] + ' ' if self.itos[i] in (PLUS, EQ) else '')
                       for i in ids)


ALL_STAGE1 = ['k1-fwd', 'k1-rev', 'gpt2-fwd', 'gpt2-rev']
ALL_GRID = (['k1-fwd', 'k1-rev']
            + [f'k{k}-{c}-{o}' for k in (2, 3) for c in ('left', 'right') for o in ('fwd', 'rev')]
            + ['gpt2-fwd', 'gpt2-rev'])  # trained-BPE cells to be added once specified
