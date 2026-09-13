# Academic datasets

The project uses four fixed public binary-sentiment sources:

- IMDb Movie Reviews — `stanfordnlp/imdb`
- Sentiment140 — Stanford archive
- Amazon Polarity — `fancyzhx/amazon_polarity`
- Yelp Polarity — `fancyzhx/yelp_polarity`

Prepare them from the repository root:

```powershell
python -m ml.prepare_datasets
```

The command downloads the required source material, removes empty/invalid rows,
deduplicates cleaned text, uses seed `42`, selects a class-balanced maximum of
10,000 samples per dataset (5,000 negative and 5,000 positive), and writes:

- `processed/<dataset>.csv` — exact columns `text,label`; labels are `0` negative and `1` positive
- `manifest.json` — source URL, original label mapping, sample counts, removals, seed, and sampling method

The configured Hugging Face sources are pinned to explicit repository revisions
in `ml/config.py`; the manifest records those revisions for auditability.

These prepared files are retained with the frozen baseline and must be included
in the submission ZIP. They can be recreated with the same command and should
not be replaced with synthetic or placeholder data.

The command also performs one Logistic Regression compatibility smoke run per
dataset. It does **not** run the final 4 × 6 comparative experiment suite.
