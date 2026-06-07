from fastapi import FastAPI
import joblib
import re

app = FastAPI()

model = joblib.load("model.joblib")
tfidf = joblib.load("tfidf.joblib")

def clean_text(text):
    text = re.sub(r'INR\s*\d+', '', text)
    text = re.sub(r'TXN+\w+', '', text)
    text = re.sub(r'\d+', '', text)
    return text.strip().lower()

@app.post("/predict")
def predict(transaction_text: str):
    cleaned = clean_text(transaction_text)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return {"transaction": transaction_text, "category": prediction}

