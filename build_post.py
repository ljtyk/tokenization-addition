"""
Builds index.html (the post) from the result files in data/.

    python build_post.py

Inputs (produced by the scripts in code/tokstudy/; see README):
    data/tiers.json                       final (EMA) accuracy per run on the fixed test tiers
    data/logs/logs_probe_length_*.txt     reach of the place-value cells (probe_length.py)
    data/reach_k3_*.json                  k=3 reach with the attention fix and its baseline
    data/logs/logs_cells_k3_left_fix.txt  left-cut reach split by problem shape (analyze_cells.py)
    data/logs/logs_reach_longmix_*.txt    the long-problem data mix and its controls
"""
import json
import os

from postlib import (f2, legend, line_chart, page, parse_reach_log, pct, table)

HERE = os.path.dirname(os.path.abspath(__file__))
D = lambda *p: os.path.join(HERE, 'data', *p)

# ---------------------------------------------------------------------------------------------------------
# data

T = json.load(open(D('tiers.json')))


def seeds(cell, tier):
    return ' / '.join(f2(T[f'{cell}_s{s}'][tier]) for s in range(3))


ROWS = [  # (label, order, encoding, cell)
    ('<strong>k=1</strong>', 'reversed', 'place-value', 'k1-rev_irope'),
    ('k=1', 'reversed', 'standard', 'k1-rev_rope'),
    ('k=1', 'forward', 'place-value', 'k1-fwd_irope'),
    ('k=1', 'forward', 'standard', 'k1-fwd_rope'),
    ('<strong>k=3 right-cut</strong>', 'reversed', 'place-value', 'k3-right-rev_irope'),
    ('k=3 right-cut', 'reversed', 'standard', 'k3-right-rev_rope'),
    ('k=3 right-cut', 'forward', 'place-value', 'k3-right-fwd_irope'),
    ('k=3 right-cut', 'forward', 'standard', 'k3-right-fwd_rope'),
    ('k=3 left-cut (GPT-style)', 'forward', 'place-value', 'k3-left-fwd_irope'),
    ('k=3 left-cut', 'forward', 'standard', 'k3-left-fwd_rope'),
    ('k=3 left-cut', 'reversed', 'place-value', 'k3-left-rev_irope'),
    ('k=3 left-cut', 'reversed', 'standard', 'k3-left-rev_rope'),
    ('GPT-2 BPE', 'reversed', 'place-value', 'gpt2-rev_irope'),
    ('GPT-2 BPE', 'reversed', 'standard', 'gpt2-rev_rope'),
    ('GPT-2 BPE', 'forward', 'place-value', 'gpt2-fwd_irope'),
    ('GPT-2 BPE', 'forward', 'standard', 'gpt2-fwd_rope'),
]
STEPS = {c: T[f'{c}_s0']['steps'] for _, _, _, c in ROWS}

tier_rows = []
for label, order, enc, cell in ROWS:
    good = cell in ('k1-rev_irope', 'k3-right-rev_irope')
    ood, far = seeds(cell, 'ood'), seeds(cell, 'far_ood')
    tier_rows.append([label, order, enc, seeds(cell, 'indist'), seeds(cell, 'gap'),
                      f'<strong class="hi">{ood}</strong>' if good else ood,
                      f'<strong class="hi">{far}</strong>' if good else far])

k1_reach = parse_reach_log(D('logs', 'logs_probe_length_k1_16k.txt'))
k3_reach = parse_reach_log(D('logs', 'logs_probe_length_k3_33k.txt'))
RL = [26, 40, 50, 60]
reach_rows = [['k=1 reversed, place-value'] + [' / '.join(f2(k1_reach[f'k1-rev_irope_s{s}'][L]) for s in range(3)) for L in RL],
              ['k=3 right-cut reversed, place-value'] + [' / '.join(f2(k3_reach[f'k3-right-rev_irope_s{s}'][L]) for s in range(3)) for L in RL]]

base = json.load(open(D('reach_k3_baseline.json')))
fixr = json.load(open(D('reach_k3-right-rev_fix.json')))
fixl = json.load(open(D('reach_k3-left-rev_fix.json')))
FL = base['lengths']


def series(J, prefix):
    return [[J['results'][f'{prefix}_s{s}'][str(L)] for L in FL] for s in range(3)]


b_right, b_left = series(base, 'k3-right-rev_irope'), series(base, 'k3-left-rev_irope')
f_right, f_left = series(fixr, 'k3-right-rev_irc-is4-decay-sq-snc'), series(fixl, 'k3-left-rev_irc-is4-decay-sq-snc')
mean = lambda ss: [sum(c) / len(c) for c in zip(*ss)]
at = lambda ss, L: ' / '.join(f2(s[FL.index(L)]) for s in ss)

# left-cut split by problem shape
cells = {}
cur = None
for line in open(D('logs', 'logs_cells_k3_left_fix.txt'), encoding='utf-8'):
    if line.startswith('k3-left-rev'):
        cur = line.strip()[-2:]
        cells[cur] = {}
    elif cur and line.strip():
        key, val = line.rsplit(None, 2)[0].strip(), float(line.rsplit(None, 2)[1])
        cells[cur][key] = val
CK = lambda key: ' / '.join(f2(cells[f's{s}'][key]) for s in range(3))

# long-problem mix
lm = {}
for name in ('logs_reach_longmix_base.txt', 'logs_reach_longmix_control12.txt', 'logs_reach_longmix_mix.txt'):
    lm.update(parse_reach_log(D('logs', name)))
ML = [40, 60, 80, 120, 160, 200, 400]
lmv = lambda run, L: f2(lm[run][L]) if L in lm[run] else '&ndash;'
mix_rows = [['baseline (16k steps)', '10'] + [' / '.join(lmv(f'k1-rev_irope_s{s}', L) for s in range(3)) for L in ML],
            ['control (24k, normal data)', '10'] + [' / '.join(lmv(f'k1-rev_irope_control_s{s}', L) for s in range(3)) for L in ML],
            ['<strong>10% long problems (24k)</strong>', '40'] + [' / '.join(lmv(f'k1-rev_irope_mix_s{s}', L) for s in range(3)) for L in ML]]

# ---------------------------------------------------------------------------------------------------------
# figures

fig_fix = legend([('right-cut, index-RoPE + stretch (mean)', '--ctx', ''), ('right-cut, 2D RoPE + decay + position-free sink', '--s1', ''),
                  ('left-cut, same fix (mean)', '--s2', '6 4')]) + line_chart(
    [dict(name='right-cut, index-RoPE (mean)', xs=FL, ys=mean(b_right), var='--ctx', label='index-RoPE'),
     *[dict(name=f'right-cut + fix, seed {s}', xs=FL, ys=f_right[s], var='--s1', width=2.5 if s < 2 else 1.5,
            dash='' if s < 2 else '2 3', markers=s < 2, label=f'fix, seed {s}') for s in range(3)],
     dict(name='left-cut + fix (mean)', xs=FL, ys=mean(f_left), var='--s2', dash='6 4', label='left-cut + fix')],
    (FL[0], FL[-1]), (0, 1.02), FL, [0, 0.25, 0.5, 0.75, 1.0], 'operand length (digits); trained on ≤ 10',
    'exact-match accuracy', 'k=3 reach: index-RoPE falls to zero by 60 digits; with the attention fix two right-cut seeds stay '
    'near 100% to 200 digits; left-cut stays low.', ytickfmt=pct)

LX = [26, 40, 60, 80, 120, 160, 200, 300, 400]
fig_mix = legend([('trained on ≤ 10 digits (mean of 3 seeds)', '--ctx', ''), ('10% of problems at 20–40 digits (mean)', '--s1', '')]) + \
    line_chart([dict(name='baseline (mean)', xs=LX, ys=[sum(lm[f'k1-rev_irope_s{s}'][L] for s in range(3)) / 3 for L in LX],
                     var='--ctx', label='≤ 10 digits'),
                dict(name='long mix (mean)', xs=LX, ys=[sum(lm[f'k1-rev_irope_mix_s{s}'][L] for s in range(3)) / 3 for L in LX],
                     var='--s1', width=2.5, markers=True, label='with long problems')],
               (LX[0], LX[-1]), (0, 1.02), [26, 40, 80, 160, 200, 300, 400], [0, 0.25, 0.5, 0.75, 1.0],
               'operand length (digits)', 'exact-match accuracy',
               'Training on longer problems moves the cliff from about 40 to about 160 digits: four times the longest '
               'training length both times.', ytickfmt=pct, vline=160)

# ---------------------------------------------------------------------------------------------------------
# page

body = f'''
<div class="page">

  <div class="kicker">research notebook &middot; tokenization and arithmetic in transformers</div>
  <h1 class="title">How Tokenization Decides Whether a Transformer Can Add</h1>
  <p class="dek">The same small transformer, trained on the same additions, either extends far past its training length or fails soon after it, depending only on how the numbers are cut into tokens, the order the answer is written in, and whether position follows place value. A controlled grid of 16 tokenization &times; encoding cells, three seeds each, at matched training compute.</p>
  <div class="byline">
    <span>research log</span>
    <span>trained on operands of &le; 10 digits</span>
    <span>written with Claude (Anthropic) as research assistant</span>
  </div>

  <div class="stat-duel">
    <div class="stat-cell fail">
      <div class="label">GPT-style tokens (3-digit chunks cut from the left), forward</div>
      <div class="num">{pct(sum(T[f"k3-left-fwd_irope_s{s}"]["ood"] for s in range(3)) / 3)}</div>
      <div class="sub">average exact-match at 19&ndash;21 digits, with the place-value encoding; {pct(sum(T[f"k3-left-fwd_irope_s{s}"]["indist"] for s in range(3)) / 3)} in-distribution.</div>
    </div>
    <div class="stat-cell ok">
      <div class="label">Same chunks cut from the right, written units-first</div>
      <div class="num">{pct(sum(T[f"k3-right-rev_irope_s{s}"]["ood"] for s in range(3)) / 3)}</div>
      <div class="sub">at 19&ndash;21 digits, same model, data and compute. Only the cut and the order changed.</div>
    </div>
  </div>

  <p class="lede">How numbers are split into tokens is usually treated as a preprocessing detail. For arithmetic it is not: it decides which algorithm is easy to learn, and whether that algorithm keeps working on numbers longer than any seen in training. This log measures that directly, with everything else held fixed.</p>

  <div class="ledger"></div>

  <h2>The setup</h2>
  <p>Addition problems with operands of up to 10 digits, freshly sampled, 30% of them built to contain long carry chains. Small GPT-style decoders (6 layers, 384 wide, ~10.6M parameters), EMA weights, three seeds per cell. Each tokenization is three independent choices:</p>
  <ul>
    <li><strong>Chunk size k:</strong> 1 or 3 digits per token, or GPT-2's BPE (its 994 digit tokens remapped to a compact vocabulary so model size matches).</li>
    <li><strong>Cut:</strong> chunk boundaries counted from the units digit (<em>right-cut</em>: chunk i always holds places 3i&ndash;3i+2) or from the leading digit (<em>left-cut</em>, as GPT-style tokenizers do: boundaries depend on the number's length).</li>
    <li><strong>Order:</strong> the answer written normally (most significant first) or <em>reversed</em> (units first, so each answer chunk depends only on its own column and the carry already written).</li>
  </ul>
  <p>Two position encodings: <em>standard</em> (rotary positions stretched by a random factor 1&ndash;4 per training sequence) and <em>place-value</em> (half of the rotary pairs driven by each chunk's index within its own number, also stretched, a form of position coupling). Cells are compared at matched training compute (~4.3&middot;10<sup>15</sup> FLOPs: {STEPS["k1-rev_irope"] // 1000}k steps for k=1, {STEPS["gpt2-rev_irope"] // 1000}k for GPT-2, {STEPS["k3-right-rev_irope"] // 1000}k for k=3, since longer chunks mean shorter sequences). Test tiers: in-distribution (1&ndash;10 digits), a gap tier (11&ndash;13), and two out-of-distribution tiers (19&ndash;21, 24&ndash;26).</p>

  <h2>Results</h2>
  {table('Exact-match accuracy per tier, seeds 0 / 1 / 2 (final EMA weights)',
         ['tokenization', 'order', 'encoding', 'in-dist', '11&ndash;13', '19&ndash;21', '24&ndash;26'], tier_rows)}

  <h2>What the grid says</h2>
  <ol class="steps">
    <li><strong>Extending past the training length takes three things together:</strong> reversed order, chunks aligned with place value (k=1, or right-cut k=3), and a place-value position signal. Remove any one and accuracy beyond the training length is near zero, often with perfect in-distribution accuracy. Only the two cells with all three extend.</li>
    <li><strong>Order.</strong> Writing the most significant digit first learns in-distribution but fails within a few digits of the training length, for every tokenization: the first answer token needs the carry from every column below it.</li>
    <li><strong>Cut.</strong> Left-cut fails to extend even when reversed. Its chunk boundaries move with each number's length, so the rule "add chunk i of each operand" is a different rule for every pair of lengths.</li>
    <li><strong>BPE.</strong> GPT-2's tokenizer is worst on both counts: it never reaches the fixed-chunk cells' in-distribution accuracy and barely extends in either order. Its splits are irregular (<code>1034567</code> &rarr; <code>[103][45][67]</code>).</li>
    <li><strong>The place-value signal only helps when chunks align with place value.</strong> With misaligned chunks it is useless or harmful.</li>
  </ol>
  <p>This matches what Singh &amp; Strouse found in frontier models: forcing GPT-3.5 and GPT-4 to group digits from the right (with commas) made their addition much more accurate, which they trace to alignment between the tokens of the operands and the answer.</p>

  <h2>How far the good cells reach</h2>
  {table('Accuracy on random problems far past the tiers, seeds 0 / 1 / 2 (place-value encoding)',
         ['cell', '26 digits', '40', '50', '60'], reach_rows)}
  <p>k=3 reaches <em>fewer</em> digits than k=1, not three times as many. The limit is not sequence length but coverage: the place-value rotation has only seen index differences up to about 4 chunks &times; the stretch of 4 during training, and reach ends near the edge of what it has seen. Reach depends on tokenization through how many chunks training numbers contain.</p>

  <div class="ledger"></div>

  <h2>With a better position encoding</h2>
  <p>A companion log (<a href="https://ljtyk.github.io/rope-aliasing/">Why Rotary Embeddings Stop Extrapolating</a>) traced the coverage limit to rotary aliasing and fixed it for k=1: 2D rotary positions (row &times; column), a fixed multiplicative decay with column distance, and a position-free attention sink took k=1 to 100% at 200 digits on every seed. The same fix, trained from scratch on k=3 at the same compute:</p>
  {fig_fix}
  <p class="fig-cap">{base["n_per_length"]} random problems per length. Right-cut with the fix: {at(f_right, 60)} at 60 digits and {at(f_right, 200)} at 200 (seeds 0 / 1 / 2), from {at(b_right, 60)} at 60 with index-RoPE. Seed 2's remaining failure is a separate mechanism (dilution of a weak attention sink; see the companion attention log).</p>
  <p>Left-cut with the same fix gets {at(f_left, 60)} at 60 digits and {at(f_left, 200)} at 200, more than its zero baseline. Split by problem shape, nearly all of it comes from problems that are mostly copying:</p>
  {table('k=3 left-cut reversed with the fix: accuracy by problem shape over all lengths (26&ndash;200 digits), seeds 0 / 1 / 2',
         ['problem shape', 'accuracy'],
         [['short operand 1 digit (the answer is mostly a copy of the long operand)', CK('S=1')],
          ['short operand L/2 digits', CK('S=L/2')],
          ['<strong>both operands L digits</strong> (real chunk-by-chunk addition)', f'<strong class="lo">{CK("S=L")}</strong>'],
          ['a carry-out shifts the answer&rsquo;s chunk boundaries', CK('operands same len mod 3, answer +1 digit changes mod 3')]])}
  <p>So the fix sharpens the first finding rather than changing it. With place-aligned chunks the remaining limit was the position encoding, and fixing it takes k=3 from zero at 60 digits to near-perfect at 200 on two of three seeds. With left-cut chunks the model still cannot add past its training length at all: no position encoding can line up chunks whose boundaries move with each number's length.</p>

  <h2>And with more data instead</h2>
  <p>The other way to reach further is to train on longer problems. Keeping the place-value encoding and changing only the data (10% of problems with a 20&ndash;40 digit operand, warm-started from the 16k-step models):</p>
  {table('k=1 reversed, place-value: reach by operand length, seeds 0 / 1 / 2',
         ['model', 'longest trained', '40', '60', '80', '120', '160', '200', '400'], mix_rows)}
  {fig_mix}
  <p class="fig-cap">Dashed line: 160 digits, four times the longest training length. Means over three seeds.</p>
  <p>Long data makes every seed (near-)perfect to 120 digits, but the cliff only moves: from about 40 to about 160 digits, four times the longest training length both times, which is exactly the encoding's random stretch factor. Data extends coverage; it does not teach a mechanism that works beyond it.</p>

  <h2>Limits</h2>
  <ul>
    <li>One task (addition), one model size, three seeds per cell; k=2 and trained (rather than stock) BPE were not run.</li>
    <li>The place-value index is computed from the tokenization, not learned; the "with a better position encoding" results use task structure (rows and columns).</li>
    <li>Reach is measured on 90 random problems per length.</li>
  </ul>

  <div class="verdict">
    <div class="note-label">Where this leaves things</div>
    <h3>The tokenizer decides which algorithm is learnable.</h3>
    <p>Right-cut chunks written units-first make each answer token a local function of one column and a carry, and a model trained on 10 digits can then extend to 200 with the right position encoding. Left-cut chunks, the way GPT-style tokenizers cut numbers, change the rule with every number length, and no position encoding or amount of training length tested here undoes that.</p>
  </div>

  <h2>References</h2>
  <ol class="refs">
    <li>A. K. Singh, D. J. Strouse, <em>Tokenization counts: the impact of tokenization on arithmetic in frontier LLMs</em>, arXiv 2402.14903 (2024).</li>
    <li>H. Cho et al., <em>Position Coupling: Improving Length Generalization of Arithmetic Transformers Using Task Structure</em>, NeurIPS 2024 (arXiv 2405.20671).</li>
    <li>S. McLeish et al., <em>Transformers Can Do Arithmetic with the Right Embeddings</em>, NeurIPS 2024 (arXiv 2405.17399).</li>
    <li>Y. Zhou et al., <em>Transformers Can Achieve Length Generalization But Not Robustly</em>, arXiv 2402.09371 (2024).</li>
    <li>A. Radford et al., <em>Language Models are Unsupervised Multitask Learners</em> (GPT-2), OpenAI (2019).</li>
  </ol>

  <footer>
    every number in this post comes from the result files in <a href="https://github.com/ljtyk/tokenization-addition/tree/main/data">data/</a>; the code behind them is in <a href="https://github.com/ljtyk/tokenization-addition/tree/main/code">code/</a>
  </footer>

</div>
'''

base_css = open(D('base_style.css'), encoding='utf-8').read()
html_out = page('How Tokenization Decides Whether a Transformer Can Add',
                'A controlled grid of digit tokenizations (chunk size, cut, order) and position encodings for a small addition '
                'transformer: which ones extend past the training length, and why.', body, base_css)
open(os.path.join(HERE, 'index.html'), 'w', encoding='utf-8').write(html_out)
print('wrote index.html', len(html_out), 'bytes')
