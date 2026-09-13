# Models 1–2 real Transformer smoke validation

This record covers only `EuroBERT/EuroBERT-210m` and
`jhu-clsp/ettin-encoder-150m`. It is integration evidence, not part of the
24-experiment final benchmark.

## Reproducible environment

The repository's pre-existing `.venv` was not used because its interpreter
reference is broken. A clean local `.venv-transformer-smoke` was created with
Python 3.12.14. The real model runs used:

- PyTorch 2.14.0+cpu
- Transformers 5.17.0
- Accelerate 1.15.0
- PEFT 0.20.0
- safetensors 0.8.0
- scikit-learn 1.9.0
- pandas 3.0.5
- NumPy 2.5.3
- huggingface-hub 1.31.0
- tokenizers 0.23.2 (resolved Transformers dependency)
- pytest 9.1.1

No model-specific package was required. Repository dependencies needed by the
full regression suite were installed at the exact versions in
`requirements.txt`. `pip check` reported no broken requirements.

CUDA was unavailable (`torch.cuda.is_available() == False`, CUDA runtime
`None`), so both runs used CPU and FP32. Real GPU precision and OOM recovery
were not validated.

For a fresh environment:

```powershell
py -3.12 -m venv .venv-transformer-smoke
.\.venv-transformer-smoke\Scripts\python.exe -m pip install -r requirements-transformer-smoke.txt
```

This phase-specific file is a compatible exact top-level environment for both
the Transformer smoke runner and the complete legacy/API regression suite.

## Immutable model inputs

| Model | Model revision | Tokenizer revision | Native sequence-classification class |
|---|---|---|---|
| EuroBERT 210M | `39b51e15dd1f1a06f58b5cbf6a8a188cec60bd0e` | same | `EuroBertForSequenceClassification` |
| Ettin Encoder 150M | `45d08642849e5c5701b162671ac811b7654bfd9f` | same | `ModernBertForSequenceClassification` |

Both loaded through Transformers with `trust_remote_code=False`, `num_labels=2`,
`id2label={0: "negative", 1: "positive"}`, and the inverse `label2id` map.

## Locked protocol used

- Dataset: IMDb
- Persisted split manifest: `datasets/splits/imdb.json`
- Manifest hash: `fba6491bca30fa9dcf2a8c48a9d945b3d7042f98f5d61e3415e24664022cb327`
- Full locked memberships: 7,200 train / 800 validation / 2,000 test
- Smoke subsets: 4 balanced train / 4 balanced validation / 2 balanced test
- Preprocessing contract: `1.0`; no model-specific preprocessing
- Seed: 42
- Maximum length: 256; dynamic per-batch padding
- Optimizer: AdamW, learning rate `2e-5`, weight decay `0.01`
- Scheduler: linear, warmup ratio `0.10`
- Physical batch size: 2; gradient accumulation: 1
- Real optimizer steps: 2 (one pass over the four-example training subset)
- Checkpoint selection: validation metrics only
- Test-subset designation: **SMOKE TEST — NOT FINAL BENCHMARK**

## Executed results

| Check | EuroBERT 210M | Ettin Encoder 150M |
|---|---:|---:|
| Real forward/backward/AdamW updates | 2 | 2 |
| Training time | 32.036 s | 31.991 s |
| Trainable / total parameters | 212,358,146 / 212,358,146 | 149,606,402 / 149,606,402 |
| Validation logits shape | `[4, 2]` | `[4, 2]` |
| Validation accuracy | 0.50 | 0.50 |
| Validation weighted F1 | 0.333333 | 0.333333 |
| Validation loss | 0.996343 | 0.762098 |
| Smoke-test logits shape | `[2, 2]` | `[2, 2]` |
| Smoke-test accuracy | 0.50 | 0.50 |
| Safetensors checkpoint | yes | yes |
| Local-only checkpoint reload | yes | yes |
| Local-only tokenizer reload | yes | yes |
| Post-reload inference contract | valid | valid |

The tiny model outputs are not evidence of model quality. Both representative
inference strings happened to receive label 1 after only two updates; the
required result is that logits had exactly two columns and the complete cleaner
→ tokenizer → model → argmax → integer label → sentiment-string contract worked.

## Artifacts and isolation

Smoke artifacts are stored only beneath:

- `models/smoke/transformers/imdb/eurobert_210m/`
- `models/smoke/transformers/imdb/ettin_encoder_150m/`

Each directory contains a safetensors checkpoint, saved tokenizer, config,
validated metadata, and `smoke_report.json`. No `.bin` file exists in either
saved checkpoint. Ettin's source `pytorch_model.bin` was loaded by current
PyTorch/Transformers and re-persisted as safetensors; the source persistence
format was not copied.

The runner rejects output roots outside `models/smoke`. It does not import or
write through the canonical results store, does not alter legacy joblib models,
and does not route the application API to Transformers.

## Foundation defects discovered and minimally corrected

1. Transformer artifact metadata had no mandatory smoke/final designation.
   `experiment_designation` is now required and validated as `smoke` or `final`.
2. The generic model loader set `num_labels=2` but did not explicitly persist
   the academic label names. It now passes the locked `id2label` and `label2id`
   maps when creating a sequence-classification model.

No broader foundation redesign was made.
