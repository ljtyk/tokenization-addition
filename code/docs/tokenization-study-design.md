# Tokenization Study: Design Decisions

How does tokenization affect a transformer's ability to learn multi-digit addition with carries? The framing is from [tokenization-compute-allocation.md](tokenization-compute-allocation.md): a tokenizer is a frozen compute-allocation scheme. This doc records the concrete experiment design as decided on 2026-10-16.

Status: stage 1 is fully specified and implemented in [tokstudy/](../tokstudy/) (§10). Position encodings were revised after diagnostics: plain RoPE was replaced by **scaled RoPE** (2026-10-16, §3.1), and the place-value index table by **index-RoPE with a random stretch** (2026-10-17, §3.2–3.5). Every run reports **EMA weights** (§3.2). Stage 1 (k=1, GPT-2 BPE) and the k=3 cells have been run at matched training compute; results are in §11. Trained-BPE details and the V-measurement method are still open (§9).

---

## 1. Base model

- A **plain fixed-depth nanoGPT decoder**. No looped combiner, thinking tokens or variable repeat from the loop-research line (v22–v38). The loop is itself an adaptive-compute mechanism, which is what tokenization stands in for, so it would hide the effect being measured.
- **6 layers × 384 wide × 6 heads, dropout 0.** That's about 10M params, the same nanoGPT scale as Lee et al. (2023). Dropout is 0 because every batch is freshly sampled (§6), so there is nothing to overfit.

---

## 2. Tokenization: cut × chunk size × order

Chunking is defined by three independent choices. They are deliberately **not** named "L2R / R2L / reversed", because "reverse the string, then chunk left-to-right" silently equals a right cut (see §2.3).

- **Chunk size k:** 1, 2 or 3 digits per token (plus BPE, §2.4).
- **Cut:** where chunk boundaries are anchored.
  - **Right-cut:** boundaries counted from the units digit. Place-aligned: chunk i always holds places [ik, ik+k−1].
  - **Left-cut:** boundaries counted from the leading digit (GPT-style). Not place-aligned.
  - With k=1, cut doesn't matter.
- **Order:** the order chunks are emitted in.
  - **Forward:** most significant chunk first (normal writing).
  - **Reversed:** units chunk first.
  - Digits *inside* a chunk are always written normally (`[345]`, not `[543]`). A chunk is one atomic token, so its internal order is only a label.

### 2.1 Worked example (k=3)

- A: `123456 + 111111 = 234567` (6-digit answer)
- B: `923456 + 111111 = 1034567` (7-digit answer, because the top place carries)

| Scheme | A answer | B answer | Needed to emit the **first** answer token |
|---|---|---|---|
| ① right-cut, reversed | `[567][234]` | `[567][034][1]` | Only `456+111`. Identical for A and B. |
| ② right-cut, forward | `[234][567]` | `[1][034][567]` | Whether the top carries (`[2…]` vs `[1]`): the whole carry chain. |
| ③ left-cut, forward (GPT-style) | `[234][567]` | `[103][456][7]` | The whole carry chain, *and* every later boundary shifts with it. |
| ④ left-cut, reversed | `[567][234]` | `[7][456][103]` | The answer's total length, i.e. the whole carry chain, before writing the units digit. |

- ① is the scheme the V analysis predicts is best.
- ③ is what GPT-style tokenizers do.
- ④ is pathological. It is kept in the grid as a **stress test**: V should concentrate on its first token.

Note how in left-cut the shared digits `34567` are tokenized completely differently in A vs B. One carry out of the top place moves every boundary.

### 2.2 Operands use the same cut and order as the answer

There's one tokenizer and one order setting for the whole sequence (decision "(a)"). Example: `12345 + 678 = 13023`, k=3.

| Scheme | Full sequence |
|---|---|
| ① right-cut, reversed | `[345][12] + [678] = [023][13]` |
| ② right-cut, forward | `[12][345] + [678] = [13][023]` |
| ③ left-cut, forward | `[123][45] + [678] = [130][23]` |
| ④ left-cut, reversed | `[45][123] + [678] = [23][130]` |

In ①/②, `[345]` and `[678]` both mean "places 0–2". In ③/④, `[45]` (places 0–1) and `[678]` (places 0–2) are misaligned before the answer even starts.

**Later diagnostic (b), not in the main grid:** operands always forward, only the answer's order varies. For example ① under (b) is `[12][345] + [678] = [023][13]`. This separates the effect of the input side from the output side.

### 2.3 Why "reverse then chunk L2R" is avoided as a definition

B's answer reversed as a string is `7654301`. Chunked left-to-right that gives `[765][430][1]`. Reading each chunk backwards gives `[567][034][1]`, which is exactly scheme ①. Direction-based names let two labels describe the same cell. Cut + order names don't.

### 2.4 Chunk widths and vocabulary

- **Right-cut:** inner chunks are always k digits and keep leading zeros (`[023]`). Only the top chunk is shorter (`[13]`). `[023]` and `[23]` are **different tokens**. For k=3 the vocab is about 1,000 + 90 + 10 ≈ 1,100 number tokens.
- **Left-cut:** the short chunk is at the end and may have a leading zero (`12305` → `[123][05]`).
- **No zero-padding** of operands, anywhere. Padding accidentally aligns left-cut chunks: `[012][345] + [000][678]` lines up, which would erase the effect being measured.

### 2.5 BPE cells

There are two BPE variants. Neither has a cut. Both are crossed with forward/reversed order.

- **(a) Stock GPT-2 BPE:** applied as-is to whatever string is emitted. This is the realistic "off-the-shelf tokenizer" baseline. On reversed strings its merges see digit patterns it never learned from.
- **(b) BPE trained on our own corpus:** one per order format, with a matched vocab budget. Details are open (§9).

**GPT-2 vocab remap.** The stock GPT-2 vocab has 50,257 entries. A 50,257 × 384 embedding is ≈ 19M params, about twice the rest of the model, so BPE cells would not be size-matched. Therefore:

1. Run GPT-2's tokenizer only to **decide segmentation**.
2. Take **every** pure-digit token in the GPT-2 vocab: **994** of them (all 10 one-digit, all 100 two-digit, 777 of the 1,000 three-digit strings, and a few longer ones, up to sixteen zeros). Covering them all up front means no test number can produce an out-of-vocab token.
3. Map them, plus `+`, `=`, newline and padding, to a compact contiguous vocab (998 IDs) and train on those IDs.

GPT-2's splits are irregular, not a clean left-cut: `1034567` → `[103][45][67]`, `999999` → `[9999][99]`, `2025` → `[20][25]`.

The segmentation stays exactly GPT-2's, and model size matches the other cells. Encode with **no spaces**, because GPT-2 treats `" 123"` and `"123"` as different tokens.

---

## 3. Position encodings

- **Standard = scaled RoPE.** Each training sequence draws one stretch s ~ U[1, 4] and uses RoPE positions s·(0, 1, 2, …). That's identical to multiplying every RoPE angle by s. At eval, s = 1. Inside a sequence the gaps stay equal, so exact distance remains usable ("answer digit i looks k tokens back"), while training already covers the long distances that OOD problems need. Plain RoPE and learned absolute positions are excluded; see §3.1.
- **Place-value = index-RoPE with stretch** (`--pos irope`).
  - Each chunk gets an index equal to its position **within its own number, counted in emission order**.
  - The index is **not** the true place value. True place value would require knowing the answer's length before writing it, which leaks the carry and gives the left-cut schemes the answer for free. Emission-order indices are computable from what has already been written, so they are fair to every scheme.
  - **How the index enters:** half of each attention head's RoPE rotation pairs turn by scaled sequence position, as in the standard condition. The other half turn by the place-value index. The attention score between two tokens therefore depends on the *difference* of their indices. "Find the operand digit at my place" is the same rule (index difference +1) at every place, and stopping is "no operand token at +1". There is **no index lookup table**.
  - **Separators** `+` and `=` get index 0, meaning "one step before the next number's first chunk". So `=` finds operand chunk 0 by the same +1 rule. It carries no length information.
  - **Stretch:** in training, the index rotation is multiplied by s ~ U[1, 4] per sequence (s = 1 at eval), so large index differences have been seen. Without it, the clocks wrap and length generalization is 0 (§3.5).
  - This is essentially *position coupling* (Cho et al. 2024) applied to half the dimensions, combined with the random stretch. It replaced an Abacus-style learned index table (McLeish et al. 2024), whose failure is analysed in §3.3–3.4.

Examples (`12345 + 678 = 13023`, k=3):

```
① right-cut, reversed:   [345] [12]  +  [678]  =  [023] [13]
                           0     1        0         0     1     <- index i ↔ same places

③ left-cut, forward:     [123] [45]  +  [678]  =  [130] [23]
                           0     1        0         0     1     <- index i, different places
```

With a bad cut the indices match by count but not by place value. The embedding cannot rescue the cut, and that is the effect being measured.

### 3.1 Why not plain RoPE: diagnostics (2026-10-16)

The first smoke run used plain RoPE + place-value on the cell predicted to be best (k=1 reversed). It was perfect in-distribution but ~0% past 15 digits. Training sequences top out at ~34 tokens, so RoPE never sees the relative distances (~42+) that a 20-digit problem needs.

A uniform offset to RoPE positions cannot fix this. RoPE is relative, so shifting every position leaves every distance unchanged. Training has to expose *long distances*. Three ways of randomizing RoPE's angle were considered:

| Variant | What it is | Exposes long distances? | Exact distance usable? |
|---|---|---|---|
| **Scaled** | one s per sequence, all angles × s (= evenly stretched positions) | yes | yes, as a multiple of the gap |
| **Randomized** (Ruoss et al. 2023) | sorted random positions from 0–255, in training and eval | yes | no, only order |
| Per-frequency scale | each dimension pair's speed × its own random s | not really (no real distance matches) | scrambled |

The per-frequency variant was skipped because it isn't a position remap.

All runs below: k=1 reversed, place-value index, 1,500 steps, seed 0. Each cell is exact-match accuracy; malformed rates were ≤ 0.08 everywhere.

| Tier | Plain RoPE | Plain RoPE, 5k steps | **Scaled RoPE** | Randomized RoPE | NoPE |
|---|---|---|---|---|---|
| In-dist (1–10) | 1.000 | 1.000 | 0.992 | 0.999 | 0.999 |
| Gap (11–13) | 0.720 | 0.790 | **0.982** | 0.973 | 0.962 |
| Mid (14–18) | 0.128 | 0.152 | **0.922** | 0.608 | 0.898 |
| OOD (19–21) | 0.003 | 0.015 | **0.845** | 0.575 | 0.638 |
| Far OOD (24–26) | 0.000 | 0.000 | **0.568** | 0.365 | 0.293 |
| Carry gap | 0.635 | 0.758 | **0.943** | 0.912 | 0.828 |
| Carry OOD | 0.013 | 0.043 | **0.738** | 0.437 | 0.392 |

Accuracy by operand length:

| Length | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 24 | 25 | 26 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Plain RoPE | .95 | .65 | .56 | .37 | .18 | .03 | .03 | .03 | .01 | .00 | .00 | .00 | .00 | .00 |
| **Scaled RoPE** | .98 | .98 | .97 | .96 | .93 | .93 | .91 | .89 | .89 | .82 | .82 | .74 | .51 | .46 |
| Randomized RoPE | .99 | .97 | .95 | .66 | .58 | .64 | .49 | .67 | .69 | .59 | .45 | .39 | .39 | .33 |
| NoPE | .97 | .97 | .94 | .95 | .94 | .95 | .92 | .72 | .83 | .76 | .33 | .29 | .29 | .29 |

Findings:
- **Plain RoPE fails by construction.** 3.3× more steps moved OOD only from 0.3% to 1.5%.
- **Scaled RoPE is best on every OOD tier.** It decays smoothly with length instead of falling off a cliff.
- **Randomized positions** learn more slowly (0.47 in-dist at step 750 vs 0.97 for scaled) and give jagged accuracy by length. Discarding exact distance appears to hurt addition, which needs a fixed "k tokens back" lookup.
- **NoPE** is solid to ~20 digits, then drops to ~0.3.
- **Caveats:** single seed, short budget. Scaled RoPE's far-OOD still slides from 0.74 to 0.46 across 24–26 digits. The randomized-vs-NoPE ordering could be seed noise; scaled vs the rest is a large gap.

A training-vs-eval input check (`tokstudy/check_inputs.py`) found no pipeline bug. Training rows and eval inputs match exactly for every scheme on 2,460 scheme-problem pairs. Batched and one-at-a-time greedy decoding agree on 300/300 in-dist problems; on OOD they differ on 4–5 of 300, presumably bf16 rounding on near-ties (not yet checked in fp32).

### 3.2 Checkpoint instability, fixed with weight EMA (2026-10-16/17)

The budget pilot (scaled RoPE + index table, k=1 reversed, 10k steps) showed that **OOD accuracy never settles; it swings between checkpoints**. In-dist stayed at 92–100% throughout. Far-OOD went 0.513 → 0.043 → 0.427 between steps 5000, 5500 and 6000, and the 10k final (OOD 0.745, far-OOD 0.352) was worse than a 1.5k run. More steps don't converge it.

**Fix:** keep an exponential moving average of the weights (decay 0.999) and report it. From step ~2250 on, EMA OOD stayed within about ±2 points per seed while raw weights swung 0.48–0.83. EMA also beat raw weights on every tier. **Protocol:** 4k steps; report the EMA final evaluation and the EMA average over the last 25% of periodic checks (`late_avg`).

### 3.3 Seed variance is a mix of failure modes (2026-10-17)

Even with EMA, seeds of the index-table setup disagreed: OOD 0.84 / 0.93 / 0.87, far-OOD 0.62 / 0.77 / 0.59. Error analysis (`analyze_errors.py`) showed the seeds **fail differently, and on different problems**: on OOD, 17 problems were wrong in both seeds out of 120 wrong in either.
- **Seed 0, "position wall":** it doesn't stop at the end, adding extra digits around answer place 21–22, and its accuracy doesn't depend on carries.
- **Seed 1, carry errors:** single dropped or added carries (|pred − true| = 10^p), growing with carry-chain length.

Controlled probes (`probe_wall.py`, carry-free and random digits, long length L × short length S) and teacher-forced probes (`probe_stop.py`) narrowed it down:
- **Every digit decision generalizes.** That includes reading the right digits and treating a missing operand digit as 0, even at places 21–30.
- **The failing decision is stopping.** The model keeps writing past the end and never stops too early.
- **Stopping depends on absolute index values, not on a number's span.** Shifting every index by an offset at eval moved the failures, differently per seed. One seed stopped 0.78 → 1.00 at offset 50; another went 0.99 → 0.60.

Cause: the index entered through a **learned lookup table with no order**. Index 22 and 23 are unrelated vectors, and "no operand token carries my index" depends on how those particular vectors were learned. Each seed had its own weak index regions, and whether the standard eval (indices 1–27) hit them was luck.

### 3.4 Two index fixes that did not work (2026-10-17)

k=1 reversed, scaled RoPE, EMA, 4k steps, 3 seeds each. Final OOD / far-OOD:

| Version | s0 | s1 | s2 |
|---|---|---|---|
| Index table (Abacus-style) | .838/.615 | .933/.772 | .868/.590 |
| + separator end-index (`+`/`=` carry "one past the last chunk") | .515/.353 | .533/.223 | .272/.012 |
| Index table, stretched (stride 1–3 per problem) | .723/.585 | .423/.518 | .980/.938 |

- **Separator end-index: harmful.** The explicit end let the model compute each operand's *length* (relative to the number's start) and rely on it. Lengths over 10 were never trained, so it broke right after the training range: carry-free L = 16–20 already failed.
- **Stretched index table: higher ceiling, worse reliability.** One seed had no wall at all; another failed to stop (81% of its errors). Its index vectors became smoother (neighbour cosine ≈ 0.3 vs ≈ 0), which plausibly helps matching but blurs "index i vs i−1". Its stopping failures also moved with the eval offset.

### 3.5 Index-RoPE + stretch: results (2026-10-17)

Same cell and protocol. `--pos irope`, as defined in §3.

| Version | Seed | In-dist | Gap | Mid | OOD | Far OOD | Carry OOD |
|---|---|---|---|---|---|---|---|
| Index table | 0 / 1 / 2 | .999 / .999 / .999 | .995 / .998 / .995 | .969 / .962 / .963 | .838 / .933 / .868 | .615 / .772 / .590 | .758 / .873 / .807 |
| **Index-RoPE + stretch** | 0 / 1 / 2 | 1.000 / 1.000 / 1.000 | .998 / 1.000 / 1.000 | 1.000 / .998 / 1.000 | **1.000 / .998 / 1.000** | **1.000 / .955 / 1.000** | **.998 / .997 / 1.000** |
| Index-RoPE, no stretch | 0 / 1 / 2 | 1.000 / 1.000 / 1.000 | .573 / .457 / .318 | .005 / .000 / .000 | .000 / .000 / .000 | .000 / .000 / .000 | .000 / .000 / .000 |

- **Offset-invariant by construction:** shifting every index changes an untrained model's logits by ~1e-6, against 1.2 for the table model. The offset rows of `probe_stop.py` are identical for every stretched seed.
- **Stopping fixed:** every teacher-forced decision, including stopping at places 21–30, is 1.00 for all stretched seeds.
- **The stretch is essential.** Without it, far-away tokens sit at index differences never seen in training (the rotation "clocks" wrap), and even low places of long problems go wrong.
- **How far it goes** (random digits, decode budget raised to 128 tokens):

  | Long operand | 34 | 40 | 50 | 60 |
  |---|---|---|---|---|
  | Stretched s0 / s1 / s2 | 1.00 / .87 / 1.00 | .99 / .63 / 1.00 | .63 / .04 / .53 | 0 / 0 / 0 |
  | Index table s1 | .35 | .06 | 0 | 0 |

  The limit, ~4–5× the training length, matches the stretch factor of 4 on both index and sequence RoPE. §3.6 tests this directly.
- **Eval note:** decoding stops after 32 new tokens (`MAX_NEW_TOKENS`), enough for the test tiers (answers ≤ 27 digits) but not for longer probes. Long probes (`probe_length.py`) raise the budget and use eval batches ≤ 48 to fit in GPU memory.

### 3.6 The reach is a coverage dial, not generalization (2026-10-17)

Concern raised: the position tricks (scaled RoPE, index stretch, offsets) extend *how far* the model works, but each one only makes long test problems look like training ones. That's coverage, not an algorithm that generalizes. Test: raise the stretch from 4 to 8 (`run_stretch8.sh`; k=1 reversed, index-RoPE, 4k steps) and measure reach (`probe_length.py`).

| Run | Accuracy at 40 / 50 / 60 / 80 / 90 / 100 digits |
|---|---|
| Both stretches 4 (s0, s2) | .99–1.00 / .57–.64 / .01 / .00 / – / .00 |
| Both stretches 8, s0 | .99 / .86 / .72 / .45 / .41 / .03 |
| Both stretches 8, s1 | .68 / .40 / .15 / .01 / .00 / .00 |
| Index stretch 8, sequence stretch 4, s0 | 1.00 / .51 / .15 / .01 / .01 / .00 |

- **The concern holds.** Doubling the stretch roughly doubles the reach. The limit is a setting we chose, not something learned.
- **The binding dial is the sequence-RoPE stretch.** Raising the index stretch alone gains nothing. Accuracy tracks the **total token count** of the problem, failing past about sequence stretch × the longest training sequence (~34 tokens): stretch 4 → ~1.0 up to 119 tokens, 0.54 at 120–139, 0.12 at 140–159. At 50 digits, a 1-digit short operand (short sequence) scores 0.93 while a 50-digit one (long sequence) scores 0.35. The model still finds operand digits partly by sequence distance.
- **A bigger dial costs reliability.** Stretch-8 seed 1 is weaker even on the standard tiers (0.79 at 26 digits).
- **Index-banded attention was considered and rejected.** Restricting attention to index differences of ±1 would remove every length-dependent quantity, but it changes the attention structure itself. That makes it an architecture study, and it would hard-wire a depth limit on forward-order carries. This study stays with position-encoding choices.

**What this means for the study:**
1. **The OOD tiers measure the rule, not position coverage.** The longest tier (26 digits) is ~82 tokens even for k=1, well inside the ~136-token coverage of stretch 4, for every scheme. So tier accuracy asks whether a tokenization lets the model learn an arithmetic rule that extends to longer numbers, given that positions are covered. It does **not** demonstrate unbounded length generalization, and results should not be phrased that way.
2. **Reach is reported separately, in digits and tokens.** Every run gets `probe_length.py` after training.
   - *Prediction made here (wrong, see §11.3):* coverage is in tokens, so coarser chunking (k=3 uses ~⅓ the tokens per number) should reach ~3× more digits from shorter sequences alone.
   - *What happened:* k=3 reaches **fewer** digits than k=1. Its sequences stay well inside the sequence-position coverage, so that limit never binds. The binding limit is the place-value index range counted **in chunks**: training numbers have at most 4 chunks, so the index stretch covers only ~12–16 chunks (~35–45 digits). Coarser chunks mean fewer chunks per training number, hence less index coverage. Still a tokenization effect, but the reverse of the prediction.

---

## 4. The grid

| Tokenization | Cut | Order | Cells |
|---|---|---|---|
| k=1 | – | fwd, rev | 2 |
| k=2 | left, right | fwd, rev | 4 |
| k=3 | left, right | fwd, rev | 4 |
| BPE: GPT-2 (remapped) and trained | – | fwd, rev | 4 |

That's **14 tokenization cells × 2 position encodings = 28 cells**, with 3 seeds each.

---

## 5. Training budget and fairness

- **Report both matched-FLOPs and matched-examples.** Tokens per problem differ a lot. For a 10+10-digit problem with an 11-digit answer:

  | Tokenization | Tokens per problem |
  |---|---|
  | k=1 | 10+1+10+1+11 = 33 |
  | k=2 | 5+1+5+1+6 = 18 |
  | k=3 | 4+1+4+1+4 = 14 |

  At matched FLOPs, k=3 sees ~2.4× more problems with less compute each. Showing both separates "more examples" from "more compute per example".
- **Budget: 4,000 steps** (batch 256, cosine LR over the run, EMA 0.999). The budget pilot showed in-dist settles by ~4k steps, while OOD only wanders with more steps (§3.2). EMA plus the late-checkpoint average handle that.
- **Seeds:** 3 per cell, reported individually, plus a **failure profile** per run (error-type shares, first-wrong-place histogram; §3.3), because how a cell fails is more informative than its mean.

---

## 6. Training data

Built on the existing generator [nanoGPT/data/addition/generate.py](../nanoGPT/data/addition/generate.py), with these changes.

- **Bucketed lengths (kept).** Pick d uniformly from 1–10. One operand has exactly d digits, the other a length uniform in 1..d. Uniform numbers would be ~90% max-length, and mismatched operand lengths are exactly where the cut matters (`12345 + 678`).
- **30% carry-heavy examples (kept), with random chain placement (new).** Under random digits, a chain of length L occurs about 10⁻ᴸ per position, so long chains are effectively never seen. The old generator only builds chains as trailing 9s triggered by the units digit (`528[999] + 4`). With chunks, *where* a chain sits relative to chunk boundaries matters:
  - `528999 + 4` → `[528][999] + [4]`: the chain fills chunk 0. All the carrying happens inside one token.
  - `5299946 + 56` → `[5][299][946] + [56]`: the chain starts at place 1 and **crosses** a chunk boundary. The carry has to pass between tokens.

  New chains start at a random place, triggered by any column summing to 10 or more.
- **Fresh sampling every batch** (new), not a fixed 40k-example set. Otherwise 1,000-token k=3 cells would be data-starved, and "k=3 is worse" would partly measure data efficiency instead of tokenization.
- **Hint tier dropped from training.** The old sparse 14–18-digit exposure was specific to the loop study. Everything past 10 digits is now truly unseen.

---

## 7. Evaluation

**Metric:** exact-match accuracy of the whole answer (not bpb).

The old test files store plain `a+b=c` lines. They are **reused and re-tokenized per scheme**, so every cell is scored on identical problems, comparable to earlier project numbers.

| Tier | Operand lengths | Exposure |
|---|---|---|
| In-dist | 1–10 | trained |
| Gap | 11–13 | never seen |
| (former hint) | 14–18 | never seen |
| OOD | 19–21 | never seen |
| Far OOD | 24–26 | never seen |
| Carry-heavy gap / OOD | 11–13 / 19–21 | never seen, every example carry-heavy |

---

## 8. Staging and run budget

1. **Stage 1 (done, §11):** k=1 vs GPT-2 BPE (remapped) × {fwd, rev} × {standard = scaled RoPE, place-value = index-RoPE + stretch} × 3 seeds = **24 runs**. As run:
   - 4k steps (`run_stage1.sh`): most cells were still improving, so the 4k results are kept only as a learning-speed snapshot (`tokstudy/out/stage1`).
   - Warm-started to **16k** steps (`run_stage1_16k.sh`, `train.py --init_from`): the continuation restarts warmup + cosine, with a fresh optimizer and new data seeds.
   - GPT-2 cells warm-started further to **31k** steps (`run_stage1_gpt2_31k.sh`), matching k1's training FLOPs at 16k: BPE sequences are shorter, so 16k GPT-2 steps were only ~52% of k1's compute.
   - **Reference cells, already run:** plain RoPE (§3.1) and the Abacus-style index table (§3.3–3.4) on k=1 reversed. Their results document why those encodings were dropped, so no new runs are needed.
2. **Main grid:** the remaining cells × 3 seeds = **60 runs**.
   - **k=3 (done, §11):** 4 cut × order schemes × 2 encodings × 3 seeds = 24 runs. 16k steps from scratch (`run_k3.sh`), then warm-started to **33k** (`run_k3_33k.sh`) to match k1's FLOPs (16k k=3 steps were ~48% of k1's compute: same 4.1M problems, ~2.1× fewer tokens).
   - **Still open:** k=2 (8 cells) and trained BPE (4 cells, §9).
3. **Depth sweep** (after the main grid): k ∈ {1, 3}, right-cut, {fwd, rev}, place-value, depth ∈ {2, 4, 6, 8}, 3 seeds = **48 runs**.
   - Why: each answer token's work grows with k. In scheme ①, k=1 computes `6+1+carry`, while k=3 computes `456+111+carry` with carries passing inside one forward pass. So "k=3 is worse at 6 layers" might just mean "not deep enough".
   - Open hypothesis: coarser chunks may *help* forward order. The carry between chunks is still one bit, but the chain is n/k tokens long instead of n.
4. **Later:** V analysis, diagnostic (b), and the loss-driven merge/split search from the summary doc.

A 4k-step run takes ~7 min on the RTX 4080 Laptop when run **one at a time**; two concurrent runs were slower in total (GPU memory full). Matched-compute runs take ~6–25 min each.

**Fairness convention used (§5):** cells are compared at **matched training FLOPs** (~4.3e15: k=1 at 16k steps, GPT-2 at 31k, k=3 at 33k), checking that in-dist has flattened. Matched FLOPs also means more optimizer steps and ~2× more problems for the shorter-sequence tokenizations. A cleaner alternative (equal steps, larger batch for shorter sequences) was noted but not run.

---

## 9. Open questions

- **Trained BPE (b):** vocab size, training corpus, and whether one BPE is trained per order format. Not needed until the main grid.
- **Measuring V_i:** ablating or truncating compute at a position vs the gradient-norm proxy. Needed after the grid.

---

## 10. Implementation (`tokstudy/`)

| File | Role |
|---|---|
| `schemes.py` | Scheme names (`k1-rev`, `k3-right-fwd`, `gpt2-rev`, …); `encode(a, b)` gives token IDs + chunk indices; `decode_answer` returns an int or `None` (malformed); GPT-2 remap |
| `test_schemes.py` | Every worked example in this doc, as asserts |
| `data.py` | Online sampler (§6), test-pair exclusion, batch collation (loss on answer + end token only; one random place-value offset per problem, shared by both operands and the answer) |
| `model.py` | Plain GPT; `--pos rope / pv / nope / nope_pv / irope`; `--rope_pos plain / scale / random`; index options `--index_rope_scale_max`, and (rejected) `--sep_index`, `--index_stride_max` |
| `train.py` | One cell; logs problems seen and FLOPs; periodic greedy in-dist check; full eval at the end |
| `eval.py` | Greedy exact-match over all tiers; accuracy by length and by longest carry run; malformed rate |
| `check_inputs.py` | Training-vs-eval input check (§3.1) |
| `analyze_errors.py` | How runs fail: error types, first wrong place, accuracy by operand length and carry run, overlap between runs; `failure_profile` is stored with every run |
| `probe_wall.py` | Controlled long length L × short length S probe, carry-free and random digits (§3.3) |
| `probe_stop.py` | Teacher-forced decisions (digit / missing digit / stop) under shifted index offsets and strides (§3.3, §3.5) |
| `run_stage1.sh` | Stage 1 queue; skips finished cells, so it can be re-run after an interruption |
| `run_stage1_16k.sh`, `run_stage1_gpt2_31k.sh`, `run_k3.sh`, `run_k3_33k.sh` | Matched-compute queues (§8); `train.py --init_from <run>` warm-starts from a finished run |
| `probe_length.py` | Reach: accuracy on random-digit problems far past the test tiers (raises the decode budget) |

Stage-1 cell, e.g.:

```
python -u tokstudy/train.py --scheme k1-rev --pos irope --rope_pos scale --seed 0 --max_iters 4000 --eval_interval 250
```

---

## 11. Results at matched training compute (2026-10-18)

All cells at ~4.3e15 training FLOPs (§8): k=1 at 16k steps, GPT-2 at 31k, k=3 at 33k. EMA weights, final evaluation. Each cell lists seeds 0 / 1 / 2. Standard = scaled RoPE only; place-value = index-RoPE + stretch (§3). Every cell has settled (in-dist gain over the last 4k steps ≤ +0.02), except GPT-2 (+0.01 to +0.03) and k=3 right-cut forward standard seed 2 (+0.06).

### 11.1 Accuracy by tier

| Tokenization | Order | Encoding | In-dist | Gap (11–13) | OOD (19–21) | Far OOD (24–26) |
|---|---|---|---|---|---|---|
| **k=1** | reversed | place-value | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | **1.00 / .99 / 1.00** | **1.00 / .92 / 1.00** |
| k=1 | reversed | standard | 1.00 / 1.00 / 1.00 | .99 / .99 / 1.00 | .07 / .01 / .04 | 0 / 0 / 0 |
| k=1 | forward | place-value | .97 / .97 / .99 | .82 / .86 / .90 | 0 / .14 / 0 | 0 / .04 / 0 |
| k=1 | forward | standard | .99 / .99 / .99 | .94 / .94 / .95 | 0 / .01 / 0 | 0 / 0 / 0 |
| **k=3 ① right-cut** | reversed | place-value | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | **1.00 / .96 / 1.00** | **.99 / .93 / 1.00** |
| k=3 ① right-cut | reversed | standard | 1.00 / 1.00 / 1.00 | .39 / .28 / .26 | 0 / 0 / 0 | 0 / 0 / 0 |
| k=3 ② right-cut | forward | place-value | .99 / .99 / .98 | .44 / .28 / .32 | 0 / 0 / 0 | 0 / 0 / 0 |
| k=3 ② right-cut | forward | standard | .99 / .99 / .74 | .22 / .25 / .15 | 0 / 0 / 0 | 0 / 0 / 0 |
| k=3 ③ left-cut (GPT-style) | forward | place-value | .98 / .99 / .99 | .03 / .07 / .05 | .01 / 0 / 0 | .01 / 0 / 0 |
| k=3 ③ left-cut | forward | standard | .99 / .99 / .98 | .24 / .16 / .12 | 0 / 0 / 0 | 0 / 0 / 0 |
| k=3 ④ left-cut | reversed | place-value | 1.00 / .99 / .99 | .31 / .51 / .52 | 0 / 0 / 0 | 0 / 0 / 0 |
| k=3 ④ left-cut | reversed | standard | 1.00 / 1.00 / 1.00 | .01 / .15 / .12 | 0 / 0 / 0 | 0 / 0 / 0 |
| GPT-2 BPE | reversed | place-value | .90 / .89 / .92 | .62 / .65 / .70 | .03 / .06 / .11 | 0 / 0 / .02 |
| GPT-2 BPE | reversed | standard | .92 / .91 / .93 | .20 / .22 / .57 | 0 / 0 / 0 | 0 / 0 / 0 |
| GPT-2 BPE | forward | place-value | .79 / .78 / .80 | .30 / .35 / .36 | 0 / 0 / .01 | 0 / 0 / 0 |
| GPT-2 BPE | forward | standard | .76 / .80 / .77 | .29 / .28 / .25 | 0 / 0 / 0 | 0 / 0 / 0 |

### 11.2 Findings

1. **Extending past the training length takes three things together:** reversed order, place-aligned chunks (k=1, or right-cut k=3), and a place-value position signal. Remove any one and OOD accuracy is ~0, often with perfect in-dist accuracy. The two cells with all three (k=1 reversed and k=3 ① right-cut reversed) are the only ones that extend, and at matched compute they are equally good on the tiers.
2. **Order.** Forward order learns in-dist (.97–.99 for k=1 and k=3 at matched compute) but fails within 3–4 digits of the training length, for every tokenization.
3. **Cut.** Left-cut fails to extend even in reversed order (④: in-dist .99–1.00, gap .01–.52, OOD 0). Its chunk boundaries depend on each number's length, so a rule learned on short numbers doesn't line up on long ones.
4. **BPE.** GPT-2 BPE is the worst on both counts. It never reaches the fixed-chunk cells' in-dist accuracy (.76–.93), and it barely extends past training in either order.
5. **The place-value index helps only when chunks align with place value.** With misaligned chunks it is useless or harmful: at 16k steps, k=3 left-cut forward was at .46 in-dist with the index vs .94 without. k=1 forward was barely affected, because single digits misalign far less than 3-digit chunks.
6. **More compute settles in-dist but doesn't create extension.** 4k → 16k/31k/33k raised forward and BPE in-dist a lot (e.g. GPT-2 forward .12–.17 → .76–.80) without moving their OOD off ~0. For k=3 ① it tightened the seeds (OOD .95 / .36 / .91 at 16k → 1.00 / .96 / 1.00 at 33k).
7. **Failure profiles** (§3.3, `failure_profile`). Just past training, most errors are single carry errors. Further out, multi-digit errors take over (lost alignment). Forward without the index also produces wrong-length answers.

### 11.3 Reach (accuracy on random-digit problems far past the tiers)

| Cell | 26 digits | 40 | 50 | 60 |
|---|---|---|---|---|
| k=1 reversed, place-value (16k) | 1.00 / .93 / 1.00 | 1.00 / .62 / 1.00 | .80 / .23 / .83 | .38 / .03 / .05 |
| k=3 ① reversed, place-value (33k) | 1.00 / .90 / .96 | .40 / .41 / .01 | .11 / .01 / 0 | 0 / 0 / 0 |

k=3 reaches **fewer** digits than k=1, not ~3× more as predicted in §3.6. For k=1 the binding limit is the sequence-position coverage (§3.6). For k=3, sequences stay far inside it; the binding limit is the place-value index range in chunks (training numbers have ≤ 4 chunks, so the stretch covers ~12–16 chunks). So reach does depend on tokenization, but through how many chunks training numbers contain, not through sequence length.

### 11.4 Side branch: lining up the columns (`tokenization-2d-rope`)

Separate branch and worktree (`../LoopResearch-2d`). k=1 reversed only, 16k steps. Motivation: a person extends addition indefinitely by writing the numbers under each other; can the position encoding impose that alignment?

| Version | Tiers (OOD / far OOD) | Reach | Finding |
|---|---|---|---|
| Index-RoPE + stretch (main study) | ~1.00 / .92–1.00 | ~40–50 digits | limit = sequence-position stretch (§3.6) |
| **2D RoPE**: half the rotation by row (a / b / answer), half by column; no sequence position (`--pos irc`, commit b8b4703) | .99–1.00 / .95–1.00, most consistent across seeds | ~40 digits (.21–.36 at 50) | the limit moves to the **column** stretch (~4 × 11 columns); a column stretch of 16 gives only a thin tail |
| 2D RoPE + **null attention**: a learned, position-free "attend to nothing" slot per head (`--null_attn`, f602be2) | 1.00 / .99–1.00 | no better (.03–.71 at 50) | the model uses the slot as an always-on head switch, not as the sink for stopping |

Diagnosis (`probe_attention.py`, teacher-forced by column):
- In 2D, the step that should stop uses the separators (column 0) as an attention **sink**: whole heads at mass 1.00 at 20 digits, 0.00 at 60. Their column difference from the stop query grows out of the trained range, so the dumped attention lands on far digits and stopping collapses (0–8% at 60+ digits).
- Far columns also disturb digit decisions, on opposite sides per seed: one seed at low columns, the other at high columns.
- Adding a length-free sink doesn't help, because in training the separator sink works just as well, so nothing pushes the model to the length-free option.

**Full analysis:** `docs/rope-aliasing-analysis.md` on the `tokenization-2d-rope` branch covers the mechanism (fast rotation pairs alias; the mid pairs' learned distance rejection flips sign past the training range; far aliases outscore the separator sink), the Fourier view (the score is a finite Fourier series over distance, which always comes back round), related work, and a monotone column-distance-penalty experiment.

**Common thread:** training on ≤ 10 digits fits several mechanisms equally well, and the model picks one that doesn't extend. Each fix so far either extended coverage (a dial we set) or offered an alternative the model didn't need. In the good cells the arithmetic rule itself is learned correctly (teacher-forced decisions perfect at any place); finding the digits and knowing when to stop is what breaks.

**Resolution on the side branch (2026-10-19, published as https://ljtyk.github.io/rope-aliasing/):** the remaining limit was RoPE aliasing, including in the separator sink's own score. 2D RoPE + a fixed multiplicative column decay + separator keys without column rotation gives k=1 reversed 1.00 on all 3 seeds at every length to 200 digits (analysis doc §6.1–6.3).

### 11.5 Does the RoPE fix change the tokenization findings? (side branch, `run_k3_fix.sh`)

The same fix for k=3, from scratch at the matched compute (33k steps), 3 seeds; reach on 90 random problems per length (analysis doc §6.4):

| Cell | Encoding | 60 digits | 200 digits |
|---|---|---|---|
| k=3 ① right-cut reversed | index-RoPE + stretch (§11) | 0 / 0 / 0 | 0 / 0 / 0 |
| k=3 ① right-cut reversed | 2D RoPE fix | 1.00 / 1.00 / .67 | .99 / 1.00 / .49 |
| k=3 ④ left-cut reversed | index-RoPE + stretch (§11) | 0 / 0 / 0 | 0 / 0 / 0 |
| k=3 ④ left-cut reversed | 2D RoPE fix | .37 / .29 / .51 | .21 / .01 / .16 |

- **It sharpens finding 1 (§11.2) rather than changing it.** With place-aligned chunks the reach limit was the position encoding, and fixing it takes k=3 from 0 at 60 digits to .99–1.00 at 200 digits on 2 of 3 seeds. Seed 2 (.49) fails from far-key dilution: the decay sends far scores to 0, not to -inf, so when the first operand has run out its far chunks take the attention; cutting attention beyond 8 columns at eval removes every seed-2 error (analysis doc §6.5).
- **Left-cut still cannot add past the training length.** Its nonzero reach comes only from copy-like problems (1-digit short operand: .92 / .20 / .60). With both operands long it is 0.00 on every seed at every length, and so is any problem whose carry-out shifts the answer's chunk boundaries. No position encoding can line up chunks whose boundaries move with each number's length.
- Not tested with the fix: forward order (its failure is the carry look-ahead, not lookup) and GPT-2 BPE (no consistent column to align).

### 11.6 Can the model learn robust extension from data alone? (`run_longmix.sh`)

The side branch fixed extension by designing the attention (decay, a position-free sink); every eval-time variant of those fixes traded one failure for another (analysis doc §6.6). Question: with **no attention changes**, does training data that makes the long-range mechanisms necessary let the model learn them itself? k=1 reversed, index-RoPE + stretch (this study's place-value encoding), warm start 16k → 24k, 3 seeds:
- **long mix:** 10% of training problems have a 20–40 digit long operand (`--long_frac 0.1`);
- **control:** the same continuation on the normal 1–10 digit data.

Reach (90 random problems per length, seeds 0 / 1 / 2):

| Model | Longest trained | 40 | 60 | 80 | 120 | 160 | 200 | 300–400 |
|---|---|---|---|---|---|---|---|---|
| baseline (16k) | 10 | 1.00 / .67 / 1.00 | .31 / 0 / .02 | .04 / 0 / .01 | 0 | 0 | 0 | 0 |
| control (24k) | 10 | 1.00 / .72 / 1.00 | .01 / 0 / .01 | ~0 | 0 | 0 | – | – |
| long mix (24k) | 40 | 1.00 ×3 | 1.00 ×3 | 1.00 ×3 | 1.00 / 1.00 / .98 | 1.00 / .82 / .94 | **0 / .03 / 0** | 0 |

- **A coverage dial, not a learned mechanism.** Reach went from ~40 to ~160 digits, i.e. 4x the longest training length both times, which is exactly the stretch factor (s ~ U[1, 4] on both the sequence and the index rotations). The cliff is sharp (1.00 at 160, 0 at 200): the edge of the rotation angles seen in training, as in §3.6.
- **Inside the covered range the long data does help:** every seed is perfect to 120 digits and the baseline's seed variance is gone. So data fixed reliability within coverage (the sink and aliasing failures that appear at the edge of the training range), but nothing in it pushed the model to a mechanism that works beyond the angles it has seen. Matched-step control: more steps alone change nothing.
- **Reading:** with rotary encodings, self-learning covers what the data (times the stretch) shows; extension beyond it needs a prior in the architecture (the side branch's decay + position-free sink: 1.00 at 200 digits = 20x from 10-digit training). The choice is between scaling data with the target length (what LLMs do: long-context training) and building the prior in.
