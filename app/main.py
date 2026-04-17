from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "artifacts" / "iris_model.joblib"
LABELS_PATH = BASE_DIR / "artifacts" / "class_names.joblib"

app = FastAPI(title="Iris Classifier API", version="1.0.0")


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Iris Classifier API is running"}


@app.get("/health")
def health() -> dict[str, str]:
    model_ready = MODEL_PATH.exists() and LABELS_PATH.exists()
    return {"status": "ok" if model_ready else "model_not_found"}


@app.post("/predict")
def predict(payload: IrisRequest) -> dict:
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail="Model files not found. Run `python train.py` first.",
        )

    model = joblib.load(MODEL_PATH)
    class_names = joblib.load(LABELS_PATH)

    features = np.array(
        [[payload.sepal_length, payload.sepal_width, payload.petal_length, payload.petal_width]]
    )
    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    return {
        "predicted_class_id": prediction,
        "predicted_class_name": class_names[prediction],
        "class_probabilities": {
            class_names[i]: round(float(probabilities[i]), 6) for i in range(len(class_names))
        },
    }
