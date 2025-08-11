from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import re
import numpy as np
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import unicodedata

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
#     MODEL LOADING
# =========================
mlb = joblib.load('mlb.pkl')

def load_model(model_path, num_labels):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        problem_type="multi_label_classification"
    ).to(device)
    model.eval()
    return tokenizer, model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer_ty_cqbh, model_ty_cqbh = load_model('phobert-base-v2-pretrained-cls-dvct', num_labels=27)

# =========================
#     TEXT CLEANING
# =========================
def normalize_punctuation(text):
    text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("“", "\"").replace("”", "\"")
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("…", "...")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# =========================
#     INFERENCE LOGIC
# =========================
def classify(model, tokenizer, text, max_length):
    inputs = tokenizer(text, 
                       return_tensors="pt", 
                       truncation=True,
                       padding='max_length', 
                       max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    pred_binary = (probs > 0.5).astype(int)
    pred_binary = np.array([pred_binary]) 
    predicted_labels = mlb.inverse_transform(pred_binary)[0]
    return predicted_labels

# =========================
#     REQUEST MODELS
# =========================
class TY_CQBH_Request(BaseModel):
    trich_yeu: str
    co_quan_ban_hanh: str 

@app.post("/cls_ty_cqbh")
async def classify_full(payload: TY_CQBH_Request):
    trich_yeu = payload.trich_yeu
    co_quan_ban_hanh = payload.co_quan_ban_hanh
    context = trich_yeu + " do cơ quan " + co_quan_ban_hanh + " ban hành"
    labels = classify(model_ty_cqbh, tokenizer_ty_cqbh, normalize_punctuation(context), max_length=128)
    return {"output": labels}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("run_api:app", host="0.0.0.0", port=8686, reload=False)