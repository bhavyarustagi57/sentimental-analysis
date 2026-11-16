# Phase 4 final modern benchmark

> **Status: Future work (started): transformer extension, 3 of 24 runs complete, not part of the frozen baseline.** The code and tests below are kept for continuation but are not run as part of the submission; the 24-experiment classical comparison is the frozen baseline.

## Scope and integrity rules

This package executes the locked 24-run matrix: six pinned modern models across IMDb, Sentiment140, Amazon Polarity, and Yelp Polarity. It never treats a smoke run as a final result and never uses the final test split for checkpoint selection. A run becomes canonical only after its staged artifact, clean checkpoint reload, 2,000 predictions, hashes, metrics, and metadata all validate.

The local machine used to prepare this package has no CUDA device. Its queue is therefore initialized as `BLOCKED` with failure class `gpu_availability_failure`. Run the package on free Colab or Kaggle GPU capacity; do not use a paid accelerator.

## Files and state

- `ml/modern_benchmark.py`: training, validation, recovery, clean reload, final evaluation, and CLI.
- `ml/modern_benchmark_queue.py`: immutable run configurations, atomic queue state, validation, and canonical promotion.
- `ml/modern_benchmark_analysis.py`: summaries and frontend-ready data, enabled only when all 24 runs pass.
- `scripts/run_phase4_free_gpu.py`: bounded free-GPU session runner.
- `scripts/phase4_colab_orchestrator.py`: complete bootstrap, recovery, dependency, retry, and bounded execution loop.
- `scripts/sync_phase4.py`: GitHub lightweight-state and private Hub checkpoint synchronization.
- `notebooks/phase4_free_colab.ipynb`: Colab entry point.
- `results/modern/staging/<dataset>__<model>/`: isolated configuration, state, checkpoints, predictions, and result.
- `results/modern/queue.json`: reconstructed machine-readable queue.
- `results/modern/results.json` and `results/modern/results.csv`: canonical validated results only.

States are `PENDING`, `RUNNING`, `COMPLETED`, `FAILED`, `BLOCKED`, and `INVALID`. Writes use replacement of a complete temporary file or directory. Each new resume checkpoint replaces older optimizer-bearing resume checkpoints. A selected checkpoint is uploaded to the private repository named by `PHASE4_HF_REPO` and its remote file list is verified before promotion and local resume-checkpoint cleanup. Git tracks lightweight JSON/CSV/JSONL evidence; `.safetensors`, `.pt`, checkpoint, tokenizer, and saved model-config directories are explicitly ignored.

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

1. Store source and lightweight benchmark evidence in the GitHub repository. Add Colab secrets named `GH_TOKEN`, `PHASE4_GITHUB_REPO` (`owner/repository`), `PHASE4_GIT_NAME`, `PHASE4_GIT_EMAIL`, `HF_TOKEN`, and `PHASE4_HF_REPO` (`owner/private-model-repository`). Git name/email are identity strings, not credentials. Grant the minimum GitHub repository permission needed to push and Hugging Face write permission to that one private model repository.
2. Open `notebooks/phase4_free_colab.ipynb` and choose **Runtime > Change runtime type > T4 GPU** (or another available free GPU). The notebook clones a fresh checkout under `/content`; `results/modern` is restored by that clone.
3. The notebook configures Git authentication from the process environment. Tokens are never inserted into a URL, source file, result, log, or committed Git configuration.
4. Run the CUDA check. If no accelerator is assigned, stop and leave the run `BLOCKED — GPU UNAVAILABLE`; never buy or select paid compute for this benchmark.
5. Install `requirements-transformer-smoke.txt`. Do not upgrade individual packages to make a model load.
6. `HF_TOKEN` provides both official gated Gemma access and write access to the configured private checkpoint repository. Never print it.
7. Run one experiment per invocation:

   ```text
   python scripts/run_phase4_free_gpu.py --results-root "results/modern" --max-runs 1 --sync-github
   ```

8. A completed run uploads and verifies only `checkpoints/best/model`, tokenizer/config, and required metadata on the private Hub. It then promotes and pushes lightweight evidence to GitHub. Re-clone GitHub in later sessions; completed runs are skipped.
9. Inspect state at any time:

   ```text
   python -m ml.modern_benchmark audit --results-root "results/modern"
   ```

10. Generate comparisons only after the audit reports 24 completed and zero failed, blocked, invalid, or missing runs:

   ```text
   python -m ml.modern_benchmark analyze --results-root "results/modern"
   ```

If the Git push fails after Hub upload, run `python scripts/sync_phase4.py push-lightweight` before resetting the runtime. The selected checkpoint is already durable on the Hub, but the Git push is what preserves canonical queue/result state.

The preferred automated entry point replaces the manual runner/recovery commands:

```text
python scripts/phase4_colab_orchestrator.py --max-runs 1
python scripts/phase4_colab_orchestrator.py --max-runs 3
```

It validates CUDA, secrets, Git identity/authentication, the private Hub repository, and official Gemma access; fast-forward pulls `main`; discovers completed evidence on the Hub even when Git state is stale; restores and validates completed runs; installs only the locked `xformers==0.0.35` with `--no-deps` when NeoBERT needs it; revalidates the unchanged PyTorch version and CUDA device; conservatively normalizes known pre-test environment failures; and runs the requested bounded number sequentially. Unknown, validation, corruption, and post-test failures stop with `ENGINEERING REVIEW REQUIRED`.

## Kaggle free GPU

1. Create a private Kaggle notebook, attach a private dataset containing the lean project, and enable a free GPU under **Notebook options > Accelerator**. Internet must be enabled for immutable Hugging Face downloads.
2. Clone the GitHub repository into `/kaggle/working/sentimental-analysis-main`; its committed `results/modern` restores lightweight state.
3. Add `HF_TOKEN`, `PHASE4_HF_REPO`, and the minimum GitHub credentials as Kaggle secrets and expose them only to the process environment. Do not print them or copy them into project files.
4. Install the locked requirements, verify `torch.cuda.is_available()`, then run:

   ```text
   python scripts/run_phase4_free_gpu.py --results-root "results/modern" --max-runs 1 --sync-github
   ```

5. Confirm the Git push succeeds before session end. The next fresh clone restores the queue/results; selected checkpoints remain in the private Hugging Face repository.
6. If the free GPU or Internet is unavailable, preserve state and stop. Classify it as GPU availability or transient infrastructure rather than an experiment failure.

Large optimizer-bearing resume checkpoints are intentionally not uploaded because repeated multi-gigabyte synchronization is impractical on free sessions. A runtime reset during the one currently `RUNNING` experiment can therefore require that experiment to restart. Previously `COMPLETED` experiments do not need retraining.

## Recovery and failure handling

The session runner resumes a `RUNNING` run before selecting a `PENDING` run. Failed or blocked states remain visible and are never silently skipped as completed. Diagnose the recorded failure before changing state. OOM recovery halves the physical batch and raises accumulation to retain effective batch 32; it does not change splits, sequence length, epochs, or metrics.

The final test-attempt counter prevents an accidental second canonical test evaluation. Use `--allow-test-rerun` only after documenting a proven technical invalidation. Revision, split-hash, checkpoint, prediction-count, prediction-hash, finite-value, and binary-label checks run before promotion. `restore-checkpoint` is a full completed-run restore: it downloads result, predictions, history, persistence manifest, state, tokenizer/config, and best checkpoint into a temporary directory; validates everything; repairs only ephemeral checkpoint paths; atomically installs the run; and refreshes the queue/canonical output without invoking training or evaluation.

Useful commands:

```text
python -m ml.modern_benchmark status
python -m ml.modern_benchmark run --dataset imdb --model eurobert_210m
python -m ml.modern_benchmark rebuild
python -m ml.modern_benchmark audit
python scripts/sync_phase4.py pull-lightweight
python scripts/sync_phase4.py restore-checkpoint --dataset imdb --model eurobert_210m
python scripts/sync_phase4.py push-lightweight
```

The accepted dataset keys are `imdb`, `sentiment140`, `amazon_polarity`, and `yelp_polarity`. The model keys are `eurobert_210m`, `ettin_encoder_150m`, `neobert`, `mmbert_small`, `gemma3_270m`, and `qwen3_0_6b`.
