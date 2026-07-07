# Transaction Classifier

An end-to-end ML system that classifies bank transaction descriptions (e.g. `"NETMEDS PHARMACY/ORDER INR 890 TXNe721d278"`) into 9 spending categories, served as a REST API in production.

**Live demo:** [Swagger UI](https://transaction-classifier-ezybyezcpq-lz.a.run.app/docs) · `POST` [`/predict`](https://transaction-classifier-ezybyezcpq-lz.a.run.app/predict)

```bash
curl -X POST "https://transaction-classifier-ezybyezcpq-lz.a.run.app/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Uber ride INR 240 TXN55667788"}'
# {"category": "travel"}
```

## What this project demonstrates

- **Text classification pipeline**: regex cleaning → TF-IDF (word 1-2 grams) → Logistic Regression, with Random Forest as a compared alternative
- **Experiment tracking**: every training run logged to [MLflow on DagsHub](https://dagshub.com/frgr3618/transaction-classifier) (params, metrics, per-class F1), best model registered in the model registry
- **Data versioning**: datasets tracked with DVC, remote storage on DagsHub — every model version is reproducible against the exact data it was trained on
- **CI/CD**: push to `main` triggers GitHub Actions → Docker build → deploy to Google Cloud Run
- **Honest evaluation**: a deliberately hard dataset, a documented val/test robustness check, and a known-limitations section below

## The dataset story (or: why 79% beats 100%)

The first version of this project scored **100% accuracy**. That number was a red flag, not an achievement: the synthetic dataset collapsed to just 45 unique phrases after cleaning, so train, validation, and test all drew from the same closed set — the model was a lookup table.

The dataset was rebuilt from scratch ([`data/generate_dataset.py`](data/generate_dataset.py)) to be realistically hard:

- **~280 merchant phrases** across 9 categories (brands like Swiggy, Zerodha, IRCTC plus generic descriptors), instead of 45 templates
- **Real-world text noise** applied inside the merchant text: typos (`Braodband`), abbreviations (`Catering svc pymt`), POS-terminal truncation, casing and separator variance
- **Genuine ambiguity**: ~25% of rows use brand-free phrases like `"POS purchase"` or `"UPI payment to vendor"` that legitimately map to several categories — the same irreducible uncertainty a real bank-statement classifier faces. Some brands are also cross-category by design (`"Amazon payment"` appears as both `shopping` and `entertainment`)
- **Amount realism**: per-category lognormal amount distributions, with shapes informed by a real 1.3M-row credit card transaction dataset

Results on the rebuilt dataset:

| Model | Validation accuracy | Test accuracy |
|---|---|---|
| Logistic Regression | **0.762** | **0.789** |
| Random Forest | 0.749 | — |

Logistic Regression won the comparison — a real, explainable outcome (linear models generalize better on high-dimensional sparse TF-IDF features, while Random Forest overfits noisy text). Test slightly exceeding validation is sampling noise on 1,000-row evaluation sets; the training notebook includes a **robustness check** that refits across 5 different train/val splits to demonstrate the scores are stable (val spread ~±1.5 points, test stable).

The full progression — 100% (templated) → 79% (realistic) — is visible in the [DagsHub experiment history](https://dagshub.com/frgr3618/transaction-classifier).

## Repository structure

```
├── src/
│   ├── app.py                  # FastAPI serving layer
│   ├── model.joblib            # Trained classifier (current: Logistic Regression)
│   ├── tfidf.joblib            # Fitted TF-IDF vectorizer
│   ├── requirements.txt
│   └── Dockerfile
├── data/
│   ├── generate_dataset.py     # Seeded synthetic dataset generator
│   └── raw/*.csv               # Train/test data (DVC-tracked, pointers in git)
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_model_training.ipynb # Training + MLflow logging + robustness check
└── .github/workflows/
    └── deploy.yml              # CI/CD: build image, push to GCR, deploy to Cloud Run
```

## API usage

`POST /predict` with a JSON body containing `text`:

```bash
curl -X POST "https://transaction-classifier-ezybyezcpq-lz.a.run.app/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Zomato order INR 450 TXNab12cd34"}'
```

Response:

```json
{"category": "food"}
```

Categories: `education`, `emi`, `entertainment`, `food`, `healthcare`, `investment`, `shopping`, `travel`, `utilities`.

Amounts (`INR ...`) and transaction IDs (`TXN...`) are stripped by preprocessing, so bare merchant text works too — `{"text": "Netflix"}` → `entertainment`.

## Reproducing the pipeline

```bash
# environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r src/requirements.txt

# regenerate the dataset (seeded, reproducible)
python data/generate_dataset.py

# version the data
dvc add data/raw/train_transactions.csv data/raw/test_transactions.csv
dvc push

# retrain: run notebooks/02_model_training.ipynb top to bottom
# (logs runs to MLflow, registers the best model, saves joblib artifacts to src/)
```

MLflow tracking requires a `.env` file with `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` (DagsHub token).

## Running the API locally

```bash
cd src
uvicorn app:app --host 0.0.0.0 --port 8080
```

Or with Docker:

```bash
cd src
docker build -t transaction-classifier .
docker run -p 8080:8080 transaction-classifier
```

## Deployment

Handled by CI/CD: every push to `main` builds the Docker image on the GitHub Actions runner, pushes it to Google Container Registry, and deploys to Cloud Run (`europe-north1`). The model and vectorizer are baked into the image at build time.

## Known limitations & future work

- **Vocabulary-bound**: TF-IDF has no world knowledge. `"Google"` classifies as `entertainment` because the training data only contains Google Play — a `"Google Cloud"` charge would be misclassified. Fix: broader training phrases, or embedding/LLM-based features.
- **No confidence in the API response**: the model's `predict_proba` is informative (it drops to ~0.3 on ambiguous inputs) but isn't exposed. Adding it would let callers route low-confidence transactions to human review.
- **Model loading**: the API loads baked-in joblib artifacts; a planned upgrade is pulling the current Production model from the MLflow registry at startup.
- **Synthetic data**: the dataset is generated, not real (real labeled bank data is scarce for privacy reasons). The generator is designed to reproduce the *failure modes* of real data — ambiguity, noise, format variance — rather than its exact distribution.
