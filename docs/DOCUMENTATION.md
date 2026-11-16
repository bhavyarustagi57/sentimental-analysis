# Documentation index

A short map of where everything lives, so you don't have to read every file to find the right one.

## Start here

- **[README.md](../README.md)** — what the project is, screenshots, quick start, features, troubleshooting.
- **[PRD.md](PRD.md)** — what the project is required to do and why (goals, scope, requirements).
- **[ARCHITECTURE.md](ARCHITECTURE.md)** — how it's built: backend/frontend structure, endpoints, data flow.

## Results and methodology

- **[ACADEMIC_RESULTS.md](ACADEMIC_RESULTS.md)** — the full analysis of the 24 frozen experiments.
- **[PROJECT_FREEZE.md](PROJECT_FREEZE.md)** — what "frozen" means and what is locked for submission.
- `results/results.json` / `results.csv` — the raw canonical records everything else is derived from.

## Presenting the project

- **[VIVA_GUIDE.md](VIVA_GUIDE.md)** — talking points and Q&A for the viva.
- **[../viva-prep/preparation.md](../viva-prep/preparation.md)** and its PDF — a longer prep guide with a cheat sheet.
- **[../DESIGN.md](../DESIGN.md)** — the dashboard's visual design system.

## Submission

- **[SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md)** — what to include/exclude in the submitted package.
- **[ROADMAP.md](ROADMAP.md)** — the prompt-by-prompt plan this project was built against, and what's left.

## The unfinished transformer extension (not part of the frozen baseline)

- **[PHASE4_BENCHMARK_EXECUTION.md](PHASE4_BENCHMARK_EXECUTION.md)** — the protocol and current state (3 of 24 runs complete).
- **[MODELS_1_2_SMOKE.md](MODELS_1_2_SMOKE.md)**, **[MODELS_3_4_SMOKE.md](MODELS_3_4_SMOKE.md)**, **[MODELS_5_6_SMOKE.md](MODELS_5_6_SMOKE.md)**, **[TRANSFORMER_FOUNDATION.md](TRANSFORMER_FOUNDATION.md)** — smoke-test records for the individual transformer models.

## Code

- `ml/` — the frozen pipeline: dataset prep, splitting, preprocessing, TF-IDF, the six classifiers, evaluation, prediction.
- `main.py` — the FastAPI app and all routes.
- `src/` — the React dashboard.
- `tests/` — pytest suite covering `ml/`, the API, and (skipped without `torch`) the transformer extension.
