# Submission checklist

## Required ZIP contents

- Application source and configuration files, including `main.py`, `ml/`, `src/`, `requirements.txt`, `package.json`, `package-lock.json`, `vite.config.js`, and `settings.json`
- Frozen evidence: `results/results.json`, `results/results.csv`, and `results/EXPERIMENT_METADATA.md`
- Dataset metadata and all four prepared files in `datasets/processed/`
- All 24 trained files in `models/*.joblib` plus `models/README.md`
- `README.md`, this `docs/` folder, `.env.example`, and tests
- `PRODUCT.md`, `DESIGN.md`, `.impeccable/config.json`, `.impeccable/design.json`, `.impeccable/surfaces/`, and `screenshots/`
- `viva-prep/preparation.md` and `viva-prep/preparation.pdf`

## Exclude from the ZIP

- `.env` (it can contain a private Groq API key)
- `.venv/`, `node_modules/`, `dist/`, `runtime/`, `__pycache__/`, and `.pytest_cache/`
- `datasets/raw/` unless offline dataset regeneration is explicitly required
- `models/smoke/` and any `*.safetensors`/`*.pt` checkpoints (unfinished Phase 4 transformer work — see the README's "Limitations and future scope")
- `.impeccable/mocks/`, `.impeccable/review/`, `.impeccable/brief-body.md` (local design-tool runtime state, not submission evidence)

## Clean-machine run commands

Use Python 3.11 or 3.12 and Node.js 20.19+ or 22.12+.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
npm ci
.\.venv\Scripts\python.exe -m uvicorn main:app --port 8000
# In another terminal:
npm run dev
```

Open `http://localhost:5173`. The academic dashboard and saved-model prediction flow require no Groq key and no retraining.
