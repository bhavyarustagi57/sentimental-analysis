# Phase 4 final modern benchmark

## Scope and integrity rules

This package executes the locked 24-run matrix: six pinned modern models across IMDb, Sentiment140, Amazon Polarity, and Yelp Polarity. It never treats a smoke run as a final result and never uses the final test split for checkpoint selection. A run becomes canonical only after its staged artifact, clean checkpoint reload, 2,000 predictions, hashes, metrics, and metadata all validate.

The local machine used to prepare this package has no CUDA device. Its queue is therefore initialized as `BLOCKED` with failure class `gpu_availability_failure`. Run the package on free Colab or Kaggle GPU capacity; do not use a paid accelerator.

## Files and state

- `ml/modern_benchmark.py`: training, validation, recovery, clean reload, final evaluation, and CLI.
- `ml/modern_benchmark_queue.py`: immutable run configurations, atomic queue state, validation, and canonical promotion.
- `ml/modern_benchmark_analysis.py`: summaries and frontend-ready data, enabled only when all 24 runs pass.
- `scripts/run_phase4_free_gpu.py`: bounded free-GPU session runner.
- `notebooks/phase4_free_colab.ipynb`: Colab entry point.
- `results/modern/staging/<dataset>__<model>/`: isolated configuration, state, checkpoints, predictions, and result.
- `results/modern/queue.json`: reconstructed machine-readable queue.
- `results/modern/results.json` and `results/modern/results.csv`: canonical validated results only.

States are `PENDING`, `RUNNING`, `COMPLETED`, `FAILED`, `BLOCKED`, and `INVALID`. Writes use replacement of a complete temporary file or directory. Each new resume checkpoint replaces older optimizer-bearing resume checkpoints. After canonical promotion, resume checkpoints are removed; the selected best checkpoint remains.

## Locked experiment protocol

- seed 42; max length 256; dynamic padding
- train/validation/test counts 7,200/800/2,000 from persisted manifests
- at most 3 epochs; AdamW; linear scheduler; 10% warm-up
- effective batch size 32 using gradient accumulation
- validation-only checkpoint choice: weighted F1, then accuracy, then lower loss
- full fine-tuning for EuroBERT, Ettin, NeoBERT, and mmBERT
- the Phase 3 validated LoRA configuration for Gemma and Qwen
- GPU BF16 only when supported; otherwise FP16; no CPU final benchmark
- final metrics: accuracy, weighted precision, weighted recall, weighted F1 with `zero_division=0`

Training time is synchronized GPU compute from the first optimization batch through the last validation pass. It excludes base-model download, tokenization, and checkpoint I/O. Inference time is synchronized GPU compute for the complete pre-tokenized 2,000-example test set after a clean checkpoint reload. It excludes checkpoint loading and tokenization. Hardware and numeric precision accompany every timing value.

## Google Colab Free

1. Put a lean copy of the repository in Google Drive. It must contain source, `datasets/processed`, `datasets/splits`, both requirements files, and the notebook. The large Phase 3 smoke checkpoints, `.venv*`, `node_modules`, raw source archives, and caches are not needed in the cloud copy. Do not delete them from the original project.
2. Reserve a separate Drive folder such as `MyDrive/sentiment-phase4-results` for persistent queue state and final checkpoints. Expect roughly 12 GB for the 16 fully fine-tuned encoder checkpoints, plus temporary headroom for the active optimizer checkpoint. Free Drive space should be checked before every session.
3. Open `notebooks/phase4_free_colab.ipynb`, choose **Runtime > Change runtime type > T4 GPU** (or another available free GPU), and set the repository and result paths in the setup cell.
4. Run the CUDA check. If no accelerator is assigned, stop and leave the run `BLOCKED — GPU UNAVAILABLE`; never buy or select paid compute for this benchmark.
5. Install `requirements-transformer-smoke.txt`. Do not upgrade individual packages to make a model load.
6. Authenticate interactively with Hugging Face for official gated Gemma access. Never paste a token into a notebook cell, result, or repository file.
7. Run one experiment per invocation:

   ```text
   python scripts/run_phase4_free_gpu.py --results-root "/content/drive/MyDrive/sentiment-phase4-results" --max-runs 1
   ```

8. Re-run the same command in later sessions. GPU-unavailable blocks are reset to `PENDING` only after CUDA is actually detected. Completed runs are skipped.
9. Inspect state at any time:

   ```text
   python -m ml.modern_benchmark audit --results-root "/content/drive/MyDrive/sentiment-phase4-results"
   ```

10. Generate comparisons only after the audit reports 24 completed and zero failed, blocked, invalid, or missing runs:

   ```text
   python -m ml.modern_benchmark analyze --results-root "/content/drive/MyDrive/sentiment-phase4-results"
   ```

If Drive is close to full, stop before starting another run. Never remove a promoted `best/model` checkpoint. Only `.incomplete` directories from an interrupted atomic write may be discarded after confirming a valid prior `checkpoint-step-*` exists.

## Kaggle free GPU

1. Create a private Kaggle notebook, attach a private dataset containing the lean project, and enable a free GPU under **Notebook options > Accelerator**. Internet must be enabled for immutable Hugging Face downloads.
2. Copy the read-only input project to `/kaggle/working/sentimental-analysis-main`. Put the queue/result directory under `/kaggle/working/sentiment-phase4-results`.
3. Add `HF_TOKEN` as a Kaggle secret for Gemma and expose it only to the process environment. Do not print it or copy it into project files.
4. Install the locked requirements, verify `torch.cuda.is_available()`, then run:

   ```text
   python scripts/run_phase4_free_gpu.py --results-root "/kaggle/working/sentiment-phase4-results" --max-runs 1
   ```

5. At session end, save a notebook version with output so the result directory is preserved. Before the next session, attach that output as a private dataset, copy the entire prior result directory back into `/kaggle/working/sentiment-phase4-results`, and then continue. Do not initialize a fresh queue over a missing prior output.
6. If the free GPU or Internet is unavailable, preserve state and stop. Classify it as GPU availability or transient infrastructure rather than an experiment failure.

Kaggle output quotas may be tighter than the retained full encoder artifacts. Colab plus a lean, dedicated Drive results folder is the preferred zero-cost route when sufficient Drive space is available.

## Recovery and failure handling

The session runner selects only `PENDING` runs. A `RUNNING` run resumes from its latest safe checkpoint when invoked explicitly with its dataset and model. Failed or blocked states remain visible and are never silently skipped as completed. Diagnose the recorded failure before changing state. OOM recovery halves the physical batch and raises accumulation to retain effective batch 32; it does not change splits, sequence length, epochs, or metrics.

The final test-attempt counter prevents an accidental second canonical test evaluation. Use `--allow-test-rerun` only after documenting a proven technical invalidation. Revision, split-hash, checkpoint, prediction-count, prediction-hash, finite-value, and binary-label checks run before promotion.

Useful commands:

```text
python -m ml.modern_benchmark status
python -m ml.modern_benchmark run --dataset imdb --model eurobert_210m
python -m ml.modern_benchmark rebuild
python -m ml.modern_benchmark audit
```

The accepted dataset keys are `imdb`, `sentiment140`, `amazon_polarity`, and `yelp_polarity`. The model keys are `eurobert_210m`, `ettin_encoder_150m`, `neobert`, `mmbert_small`, `gemma3_270m`, and `qwen3_0_6b`.
