# Models 3–4 real Transformer smoke validation

This record covers only `chandar-lab/NeoBERT` and
`jhu-clsp/mmBERT-small`. All metrics are integration smoke evidence, not final
benchmark results.

## Environment and dependency change

The clean `.venv-transformer-smoke` environment from Models 1–2 was reused:

- Python 3.12.14
- PyTorch 2.14.0+cpu
- Transformers 5.17.0
- Accelerate 1.15.0
- PEFT 0.20.0
- safetensors 0.8.0
- scikit-learn 1.9.0
- huggingface-hub 1.31.0
- tokenizers 0.23.2
- xformers 0.0.35

The only addition was xformers: absent → 0.0.35. It was installed with
`--no-deps`, so no existing package version changed. NeoBERT's pinned
`model.py` imports `xformers.ops.SwiGLU` unconditionally. The compiled
CUDA extensions do not match this CPU/Python environment, but xFormers' PyTorch
fallback completed a CPU forward and backward pass. `pip check` remained clean.

CUDA was unavailable, so execution used CPU and FP32.

## Immutable revisions

| Model | Model revision | Tokenizer revision |
|---|---|---|
| NeoBERT | `5424c8efeea6491b151d62dee55a752165407430` | same |
| mmBERT Small | `abc32620dd4f6ab06f5fbe905dc25f310618e09f` | same |

EuroBERT and Ettin pins were not modified.

## NeoBERT remote-code review

The pinned `config.json` maps `AutoModelForSequenceClassification` to
`model.NeoBERTForSequenceClassification`, which is not native to Transformers;
therefore `trust_remote_code=True` is required. Registry validation permits it
only for the pinned NeoBERT key. Tokenizer/model loads use the same immutable
revision, and offline reload trusts only code saved inside the local checkpoint.

Reviewed executable files and SHA-256 hashes:

- `model.py` (15,780 bytes):
  `7a1a127d343604fe1f168f72cedb16c5899e316a3c7916a6434b3333e8cf8617`
- `rotary.py` (2,582 bytes):
  `aaf408678fb7509fc69ad0f2207b10c14009a30b95a7eaf53d995f3004f22945`
- `config.json` (928 bytes):
  `e8a7e7aa346814e780523dcf227655f86d8e46661cd9ea63a995ea70af4fa5d2`

The code imports PyTorch, NumPy, Transformers, xFormers SwiGLU, the local
rotary helper, and optional FlashAttention. No filesystem mutation, network,
subprocess, shell, pickle, `eval`, or `exec` operation was found. Ordinary
padded batches use PyTorch scaled-dot-product attention. FlashAttention is
required only for the optional packed-sequence GPU path, which was not used.

The `model.py` and `rotary.py` copies persisted in the local checkpoint exactly
match the reviewed hashes.

## Locked smoke protocol

- Dataset: IMDb
- Existing split manifest hash:
  `fba6491bca30fa9dcf2a8c48a9d945b3d7042f98f5d61e3415e24664022cb327`
- Locked memberships: 7,200 train / 800 validation / 2,000 test
- Balanced smoke subsets: 4 train / 4 validation / 2 test
- Shared preprocessing version: 1.0
- Seed: 42
- Maximum length: 256 with dynamic per-batch padding
- Full fine-tuning, CPU FP32
- AdamW, learning rate `2e-5`, weight decay `0.01`
- Linear scheduler, warmup ratio 0.10
- Physical batch size 2; gradient accumulation 1
- Two real optimizer steps (one pass over four training examples)
- Checkpoint selection used validation metrics only
- Test results: **SMOKE TEST — NOT FINAL BENCHMARK**

## Executed results

| Check | NeoBERT | mmBERT Small |
|---|---:|---:|
| Native/custom class | `NeoBERTForSequenceClassification` | `ModernBertForSequenceClassification` |
| Trainable / total parameters | 222,257,666 / 222,257,666 | 140,642,306 / 140,642,306 |
| FP32 parameter bytes | 889,030,664 | 562,569,224 |
| Tokenizer vocabulary | 30,522 | 256,000 |
| Optimizer steps | 2 | 2 |
| Training time | 40.507 s | 41.856 s |
| Validation logits | `[4, 2]` | `[4, 2]` |
| Validation accuracy | 0.50 | 0.50 |
| Validation weighted F1 | 0.333333 | 0.333333 |
| Validation loss | 0.723178 | 1.585768 |
| Smoke-test logits | `[2, 2]` | `[2, 2]` |
| Smoke-test accuracy | 0.50 | 0.50 |
| Checkpoint bytes | 889,069,939 | 562,585,565 |
| Safetensors-only checkpoint | yes | yes |
| Local and forced-offline reload | yes | yes |

Observed Windows working-set peaks were approximately 3.39 GB for NeoBERT and
2.54 GB for mmBERT across the real attempts. These are process-level sampled
peaks, not controlled benchmark measurements. mmBERT's 256K × 384 embedding is
the dominant part of its parameter footprint.

The first two mmBERT persistence attempts reached training/validation but failed
at serialization because Windows temporarily expanded its pagefile and exhausted
disk space. Incomplete outputs were deleted, storage recovered, and the reported
clean run completed through save/reload. No failed-attempt metrics were promoted.

## Offline and isolation verification

After successful saves, both Hugging Face source caches were deleted. Each model
was loaded from its local tokenizer/checkpoint with `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`, and a new empty isolated Hugging Face home. Both
returned `(1, 2)` logits and valid integer/string inference responses.

Artifacts exist only beneath:

- `models/smoke/transformers/imdb/neobert/`
- `models/smoke/transformers/imdb/mmbert_small/`

The optional raw parquet download caches under `datasets/raw` and the known
broken `.venv` were removed to recover disk space. Processed CSV datasets,
split manifests, final results, legacy joblib models, frontend source, and API
routing were not changed. Raw sources remain recoverable from the exact dataset
revisions already stored in the registry.

## Minimal foundation defect correction

The shared checkpoint loader always forced `trust_remote_code=False`, making
offline reload of a reviewed custom architecture impossible. It now uses the
model registry's validated trust flag; registry validation restricts this to
pinned NeoBERT. A backward-compatible `dependency_versions` metadata map was
also added because the previous schema could not record NeoBERT's required
xFormers version or the full tested dependency set.
