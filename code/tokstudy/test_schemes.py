"""Every worked example in docs/tokenization-study-design.md, as asserts.

Run: python tokstudy/test_schemes.py   (or pytest)"""
import random

from schemes import Scheme, gpt2_digit_tokens


def answer(name, a, b):
    s = Scheme(name)
    ids, _, ans_start = s.encode(a, b)
    return [s.itos[i] for i in ids[ans_start:-1]]


def operands(name, a, b):
    s = Scheme(name)
    return s.chunks(a), s.chunks(b)


def test_section_2_1_answers():
    # A: 123456 + 111111 = 234567, B: 923456 + 111111 = 1034567
    A, B = (123456, 111111), (923456, 111111)
    assert answer('k3-right-rev', *A) == ['567', '234']
    assert answer('k3-right-rev', *B) == ['567', '034', '1']
    assert answer('k3-right-fwd', *A) == ['234', '567']
    assert answer('k3-right-fwd', *B) == ['1', '034', '567']
    assert answer('k3-left-fwd', *A) == ['234', '567']
    assert answer('k3-left-fwd', *B) == ['103', '456', '7']
    assert answer('k3-left-rev', *A) == ['567', '234']
    assert answer('k3-left-rev', *B) == ['7', '456', '103']


def test_section_2_2_operands():
    # 12345 + 678 = 13023, k=3: operands share the answer's cut and order
    assert operands('k3-right-rev', 12345, 678) == (['345', '12'], ['678'])
    assert answer('k3-right-rev', 12345, 678) == ['023', '13']
    assert operands('k3-right-fwd', 12345, 678) == (['12', '345'], ['678'])
    assert answer('k3-right-fwd', 12345, 678) == ['13', '023']
    assert operands('k3-left-fwd', 12345, 678) == (['123', '45'], ['678'])
    assert answer('k3-left-fwd', 12345, 678) == ['130', '23']
    assert operands('k3-left-rev', 12345, 678) == (['45', '123'], ['678'])
    assert answer('k3-left-rev', 12345, 678) == ['23', '130']


def test_section_2_4_widths():
    assert Scheme('k3-left-fwd').chunks(12305) == ['123', '05']
    assert Scheme('k3-right-fwd').vocab_size == 4 + 1110


def test_section_3_chunk_index():
    s = Scheme('k3-right-rev')
    ids, idx, ans_start = s.encode(12345, 678)
    # [345][12] + [678] = [023][13] \n
    assert idx == [0, 1, -1, 0, -1, 0, 1, -1]
    assert ans_start == 5


def test_k1():
    assert answer('k1-fwd', 12345, 678) == list('13023')
    assert answer('k1-rev', 12345, 678) == list('32031')


def test_gpt2_segmentation_and_remap():
    toks = gpt2_digit_tokens()
    assert len(toks) == 994
    s = Scheme('gpt2-fwd')
    assert s.chunks(1034567) == ['103', '45', '67']
    assert s.chunks(999999) == ['9999', '99']
    assert s.chunks(2025) == ['20', '25']
    assert s.vocab_size == 4 + 994
    r = Scheme('gpt2-rev')
    assert r.chunks(1034567) == ['765', '43', '01']  # GPT-2 applied to "7654301"


def test_roundtrip_and_malformed():
    rng = random.Random(0)
    names = ['k1-fwd', 'k1-rev', 'gpt2-fwd', 'gpt2-rev'] + \
            [f'k{k}-{c}-{o}' for k in (2, 3) for c in ('left', 'right') for o in ('fwd', 'rev')]
    for name in names:
        s = Scheme(name)
        for _ in range(300):
            n = rng.randint(0, 10 ** rng.randint(1, 27))
            ids = [s.stoi[c] for c in s.chunks(n)]
            assert s.decode_answer(ids) == n, (name, n)
    # non-canonical boundaries are malformed, not just wrong
    s = Scheme('k3-right-fwd')
    assert s.decode_answer([s.stoi['10'], s.stoi['34']]) is None    # should be [1][034]
    assert s.decode_answer([s.stoi['034'], s.stoi['567']]) is None  # leading zero
    assert s.decode_answer([s.stoi['1'], s.eq_id]) is None          # stray special
    assert s.decode_answer([]) is None


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
