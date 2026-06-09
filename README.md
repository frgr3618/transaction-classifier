# Transaction Classifier

A simple end-to-end machine learning service for classifying financial transaction descriptions into categories.

This repository includes a trained scikit-learn model, TF-IDF preprocessing, a FastAPI inference service, and Docker support for containerized deployment.

## Repository Structure

- `src/`
  - `app.py` — FastAPI application for serving prediction requests
  - `requirements.txt` — Python dependencies
  - `Dockerfile` — Container image build instructions
  - `model.joblib` — Trained classification model
  - `tfidf.joblib` — Saved TF-IDF vectorizer

## Features

- Clean transaction text before prediction
- Vectorize text using TF-IDF
- Predict transaction category using a serialized scikit-learn model
- Expose a REST API endpoint for inference
- Buildable as a Docker image for easy deployment

## API Usage

The service exposes a single endpoint:

- `POST /predict`

Request body should include a plain string value for `transaction_text`.

Example with `curl`:

```bash
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{"transaction_text": "Payment to ACME INC INR 1000"}'
```

Example response:

```json
{
  "transaction": "Payment to ACME INC INR 1000",
  "category": "<predicted_category>"
}
```

## Local Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r src/requirements.txt
```

3. Run the FastAPI app from the `src` directory:

```bash
cd src
uvicorn app:app --host 0.0.0.0 --port 8080
```

## Docker

Build the Docker image from the `src` directory:

```bash
cd src
docker build -t transaction-classifier .
```

Run the container:

```bash
docker run -p 8080:8080 transaction-classifier
```

## Deployment

This project is ready for deployment to container platforms such as Google Cloud Run, AWS ECS, or Azure Container Instances.

Example GCP Cloud Run deploy command:

```bash
gcloud run deploy transaction-classifier \
  --image gcr.io/<PROJECT_ID>/transaction-classifier \
  --platform managed \
  --region <REGION> \
  --allow-unauthenticated
```

## Live Deployment

The model is currently deployed and accessible at:

```text
https://transaction-classifier-ezybyezcpq-lz.a.run.app
```

Use the full prediction endpoint for requests:

```text
https://transaction-classifier-ezybyezcpq-lz.a.run.app/predict
```

FastAPI Swagger docs are available at:

```text
https://transaction-classifier-ezybyezcpq-lz.a.run.app/docs
```

## Notes

- The service uses simple regex-based cleaning for transaction text.
- The model and vectorizer are loaded from `model.joblib` and `tfidf.joblib` at startup.
- Customize preprocessing or retrain the model by replacing these serialized artifacts.

