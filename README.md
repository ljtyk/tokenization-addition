# How Tokenization Decides Whether a Transformer Can Add

A controlled study of how digit tokenization affects a small transformer learning addition: chunk size (1 digit, 3 digits, GPT-2 BPE), where chunks are cut (from the units digit or from the leading digit), the order the answer is written in, and whether the position encoding follows place value. 16 cells × 3 seeds at matched training compute. The post is `index.html` (see the GitHub Pages site).

**Short version.** Extending past the training length takes three things together: the answer written units-first, chunks aligned with place value, and a place-value position signal. Remove any one and accuracy just past the training length is near zero. GPT-style left-cut chunks cannot be rescued by a better position encoding, and training on longer problems only moves the length limit to a fixed multiple of the longest training length.

Companion posts: [Why Rotary Embeddings Stop Extrapolating](https://github.com/ljtyk/rope-aliasing) (the position-encoding fix used in one section) and [What Attention Is Missing](https://github.com/ljtyk/attention-addressing).

## Layout

- `index.html`: the post. `build_post.py` (with `postlib.py`) regenerates it from `data/`.
- `data/tiers.json`: final (EMA) accuracy of every run on the fixed test tiers. `data/reach_*.json` and `data/logs/`: reach probes, the left-cut problem-shape breakdown and the long-problem runs.
- `code/tokstudy/`: tokenization schemes, data, model, training, evaluation and probes (PyTorch). `run_stage1*.sh`, `run_k3*.sh` and `run_longmix.sh` are the experiment queues. The k=3 runs with the 2D-RoPE fix use the code in the attention-addressing repo (`run_k3_fix.sh` there).
- `code/docs/tokenization-study-design.md`: the full design and results notes.
- `code/nanoGPT/data/addition/`: the fixed test sets.

Requirements: Python 3.10+, PyTorch 2.x, `tiktoken` for the GPT-2 schemes. The `PY=` line in the run scripts points at a local virtual environment; adjust it. Trained checkpoints are not included.

The model code follows [nanoGPT](https://github.com/karpathy/nanoGPT) (MIT, see `code/nanoGPT/LICENSE`).

Written with Claude (Anthropic) as research assistant.
