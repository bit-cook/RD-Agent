You are working in the `SebiSebi/AI2-Reasoning-Challenge-ARC` repo.

Goal: make the AttentiveRanker evaluation runnable on macOS arm64 using TensorFlow 2.x + `tf.keras` (no standalone `keras` dependency), with minimal code changes.

Acceptance (must pass):

- `/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 tools/arc_smoke.py --datasets easy_test,challenge_test`

It must exit with code 0 and print parsed accuracies for BOTH datasets.

Do NOT commit. Do NOT push.

Implement:

1) Patch `AttentiveRanker/src/arch.py`
- Replace `keras.*` imports with `tensorflow.keras.*` equivalents.
- Keep behavior and API the same.

2) Patch `AttentiveRanker/src/kv_attention.py`
- Implement `KVAttention` as a `tf.keras.layers.Layer`.
- Remove the `keras.backend` backend check.
- Replace deprecated `tf.nn.xw_plus_b` with `tf.matmul(x, w) + b`.
- Keep `return_attention_scores` behavior.

3) Add a reproducible runner `tools/arc_smoke.py`
- Must be idempotent and safe to rerun.
- Use cache dir: `~/.autorl_bench/cache/ai2_arc_attentiveranker`.
- Download required assets from Google Drive file id `1QK9rWNGF-7iKIolIykhcJrWZDqjWyC3i` using `python -m gdown ...` into the cache if missing.
- The downloaded file is a ZIP (it starts with `PK`). Extract using `zipfile.ZipFile`.
  - Be robust if the cache already contains the file under a different extension (eg `.tar.gz` but it is actually ZIP); detect by header and handle it.
- Extract into the cache, producing `data/` and `final_models/`.
- Create/refresh symlinks in `AttentiveRanker/src/`:
  - `AttentiveRanker/src/data` -> `<cache>/data`
  - `AttentiveRanker/src/final_models` -> `<cache>/final_models`
  (Avoid extracting large files into the git working tree.)
- Create/use a venv at `<cache>/venv` using the current interpreter.
- Install deps into that venv (only if missing):
  - `gdown`
  - TensorFlow: `tensorflow-macos==2.16.2` on darwin, else `tensorflow==2.16.2`.
- Run `AttentiveRanker/src/eval.py -d <dataset>` for each dataset requested via `--datasets` (default: `easy_test,challenge_test`).
- Parse stdout lines like `Accuracy: 72.3044%` and collect results.
- Write `arc_eval_results.json` at repo root with accuracies (percent and fraction), plus basic metadata.
- Exit nonzero if any dataset fails or accuracy cannot be parsed.

Keep changes minimal and focused.

