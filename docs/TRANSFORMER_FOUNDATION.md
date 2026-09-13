# Shared Transformer foundation

This foundation prepares the academic pipeline for the six locked models without downloading or training them. The frozen scikit-learn results and `.joblib` artifacts remain active until model-specific implementations and real evaluations are complete.

## Locked models and policy

The single source of model policy is `ml/model_registry.py`. All models use binary labels (`0 = negative`, `1 = positive`), cleaned text from preprocessing version 1.0, maximum sequence length 256, dynamic padding, truncation tracking, and effective batch size 32.

NeoBERT, EuroBERT, mmBERT, and Ettin use full fine-tuning. Gemma 3 270M and Qwen3 0.6B use LoRA. Model-specific LoRA target modules are deliberately deferred until each checkpoint is loaded and inspected during model implementation.

Registry revisions are placeholders. A production run must resolve and persist immutable model and tokenizer commit revisions. No trained artifact may be published with a blank or moving revision.

## Evaluation split

`ml/split_manifest.py` first reproduces the existing seeded, stratified 80/20 split. Its test membership is therefore unchanged. It then makes a seeded, stratified 90/10 split of only the existing training pool. For each 10,000-row dataset this produces 7,200 training, 800 validation, and 2,000 test examples.

Membership is persisted by SHA-256 row IDs under `datasets/splits/<dataset>.json`. The manifest includes a dataset fingerprint and its own canonical hash. Loading fails if data, preprocessing version, membership, or overlap checks do not match. Checkpoint selection utilities accept only validation metrics.

## Artifact layout

New artifacts coexist with legacy files:

```text
models/
  transformers/
    <dataset>/
      <model_key>/
        config/
        tokenizer/
        checkpoint/
        metadata.json
```

Metadata records immutable revisions, dataset/split identity, preprocessing, labels, training policy, parameter counts, precision, hardware, package versions, validation results, and history. New weights should use safetensors where supported.

## Remote-code security

Only `chandar-lab/NeoBERT` is allowed to set `trust_remote_code=True`. The adapter refuses to execute its remote code until an exact revision is configured. Before enabling the production run, review the pinned `model.py` and related repository files, validate the NeoBERT/Transformers/PyTorch/xFormers combination in an isolated notebook, and record the commit in the artifact metadata.

`xformers==0.0.28.post3` is model-specific metadata rather than a global dependency because it has tight PyTorch/CUDA compatibility and must not be installed for unrelated models. Gemma access requires free license acceptance and authentication, not a paid endpoint.

## Hardware and recovery

CUDA is reported only when `torch.cuda.is_available()` is true. CPU always uses FP32. CUDA uses BF16 only when PyTorch reports support; otherwise it uses FP16. OOM recovery halves the physical batch and raises gradient accumulation to preserve the effective batch. It never changes split membership or uses test metrics.

The dependency ranges establish the supported architecture floor: Transformers 4.57+ includes the required Gemma 3 and Qwen3 sequence-classification implementations. A reproducible Colab/Kaggle lock file should be captured after the six model-specific smoke tests establish one compatible package set.

## Migration boundary

The existing API, frozen result files, frontend model names, legacy predictor, and VADER/Groq workflow are intentionally unchanged. The Transformer inference facade already returns the same four compatibility fields and can optionally expose logits, confidence, and latency. Wiring it into `/ml/predict` remains model-specific work and must wait for real checkpoints.

Dataset-preparation smoke runs now write under `models/smoke/` and `results/smoke/`, preventing them from replacing canonical Logistic Regression artifacts or benchmark rows.
