from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import re
import numpy as np
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import unicodedata

# =========================
#     APP INITIALIZATION
# =========================
app = FastAPI(title="Document Classification API", version="1.0")

# CORS configuration
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
# Load MultiLabel Binarizer for decoding predictions
mlb = joblib.load('mlb.pkl')

def load_model(model_path: str, num_labels: int):
    """
    Load tokenizer and model from HuggingFace Transformers.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        torch_dtype=torch.float16,
    ).to(device)
    model.eval()
    return tokenizer, model

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model and tokenizer
MODEL_PATH = 'phobert-base-v2-pretrained-cls-dvct'
NUM_LABELS = 27

tokenizer, model = load_model(MODEL_PATH, NUM_LABELS)

# =========================
#     TEXT PREPROCESSING
# =========================
def normalize_text(text: str) -> str:
    """
    Normalize Vietnamese punctuation and whitespace.
    """
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
def predict_labels(model, tokenizer, input_text: str, max_length: int = 128):
    """
    Predict labels using the loaded model and tokenizer.
    Returns decoded label list.
    """
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding='max_length',
        max_length=max_length
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()

    # Apply threshold to get binary predictions
    predictions = (probabilities > 0.5).astype(int)
    predictions = np.array([predictions])

    # Inverse transform to get actual labels
    predicted_labels = mlb.inverse_transform(predictions)[0]
    return list(predicted_labels)

# =========================
#     REQUEST/RESPONSE MODELS
# =========================
class DocumentRequest(BaseModel):
    summary: str
    issuing_agency: str

class DocumentResponse(BaseModel):
    labels: list[str]

# =========================
#     API ENDPOINT
# =========================
@app.post("/classify", response_model=DocumentResponse)
async def classify_document(payload: DocumentRequest):
    """
    Classify document based on summary and issuing agency.
    """
    summary = payload.summary
    agency = payload.issuing_agency

    # Construct context for classification
    context = f"{summary} do cơ quan {agency} ban hành"
    normalized_context = normalize_text(context)

    # Run inference
    labels = predict_labels(model, tokenizer, normalized_context)

    return DocumentResponse(labels=labels)

# =========================
#     RUN SERVER
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8686, reload=False)