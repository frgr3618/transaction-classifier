from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import joblib
import re

app = FastAPI()

model = joblib.load("model.joblib")
tfidf = joblib.load("tfidf.joblib")

class Transaction(BaseModel):
    text: str

def clean_text(text):
    text = re.sub(r'INR\s*\d+', '', text)
    text = re.sub(r'TXN+\w+', '', text)
    text = re.sub(r'\d+', '', text)
    return text.strip().lower()

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.post("/predict")
def predict(transaction: Transaction):
    cleaned = clean_text(transaction.text)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return {"category": prediction}

